#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd -P)"
COMPOSE_FILE="$PROJECT_ROOT/deploy/inference_engines/compose.yaml"
DEFAULT_ENV_FILE="$PROJECT_ROOT/deploy/inference_engines/benchmark.env"
EXAMPLE_ENV_FILE="$PROJECT_ROOT/deploy/inference_engines/benchmark.env.example"

fail() {
    printf 'Benchmark failed: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[benchmark] %s\n' "$*"
}

resolve_env_file() {
    local requested="${1:-}"
    if [[ -n "$requested" ]]; then
        [[ -f "$requested" ]] || fail "environment file not found: $requested"
        (cd -- "$(dirname -- "$requested")" && printf '%s/%s\n' "$PWD" "$(basename -- "$requested")")
    elif [[ -f "$DEFAULT_ENV_FILE" ]]; then
        printf '%s\n' "$DEFAULT_ENV_FILE"
    else
        printf '%s\n' "$EXAMPLE_ENV_FILE"
    fi
}

TARGET="${1:-all}"
ENV_REQUEST="${2:-}"
case "$TARGET" in
    all|sglang|vllm|lmdeploy) ;;
    *)
        ENV_REQUEST="$TARGET"
        TARGET=all
        ;;
esac

ENV_FILE="$(resolve_env_file "$ENV_REQUEST")"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

bash "$SCRIPT_DIR/preflight.sh" "$ENV_FILE"

if [[ "$EXPERIMENT_DIR" = /* ]]; then
    EXPERIMENT_ROOT="$EXPERIMENT_DIR"
else
    EXPERIMENT_ROOT="$PROJECT_ROOT/$EXPERIMENT_DIR"
fi
RUN_ID="$(date -u +'%Y%m%dT%H%M%SZ')"
RUN_DIR="$EXPERIMENT_ROOT/runs/$RUN_ID"
RAW_DIR="$RUN_DIR/raw"
TELEMETRY_DIR="$RUN_DIR/telemetry"
LOG_DIR="$RUN_DIR/logs"
METADATA_FILE="$RUN_DIR/run_metadata.env"
mkdir -p "$RAW_DIR" "$TELEMETRY_DIR" "$LOG_DIR"

COMPOSE=(
    docker compose
    --project-name "$COMPOSE_PROJECT_NAME"
    --env-file "$ENV_FILE"
    --file "$COMPOSE_FILE"
)
ALL_PROFILES=(--profile sglang --profile vllm --profile lmdeploy)
BASE_URL="http://127.0.0.1:$SERVER_PORT"
CURRENT_ENGINE=""
TELEMETRY_PID=""
if [[ "$TARGET" == "all" ]]; then
    SELECTED_ENGINES="$ENGINE_ORDER"
else
    SELECTED_ENGINES="$TARGET"
fi

stop_telemetry() {
    if [[ -n "$TELEMETRY_PID" ]]; then
        kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
        wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
        TELEMETRY_PID=""
    fi
}

capture_server_log() {
    local engine="$1"
    "${COMPOSE[@]}" --profile "$engine" logs --no-color --timestamps "$engine" \
        >"$LOG_DIR/${engine}_server.log" 2>&1 || true
}

stop_engine() {
    local engine="$1"
    [[ -n "$engine" ]] || return 0
    capture_server_log "$engine"
    "${COMPOSE[@]}" --profile "$engine" stop --timeout 30 "$engine" >/dev/null 2>&1 || true
    "${COMPOSE[@]}" --profile "$engine" rm --force "$engine" >/dev/null 2>&1 || true
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    stop_telemetry
    stop_engine "$CURRENT_ENGINE"
    "${COMPOSE[@]}" "${ALL_PROFILES[@]}" down --remove-orphans >/dev/null 2>&1 || true
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

env_line() {
    local key="$1"
    local value="${2//$'\n'/ }"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s="%s"\n' "$key" "$value" >>"$METADATA_FILE"
}

image_for_engine() {
    case "$1" in
        sglang) printf '%s\n' "$SGLANG_IMAGE" ;;
        vllm) printf '%s\n' "$VLLM_IMAGE" ;;
        lmdeploy) printf '%s\n' "$LMDEPLOY_IMAGE" ;;
        *) fail "unknown engine: $1" ;;
    esac
}

backend_for_engine() {
    case "$1" in
        sglang) printf '%s\n' sglang-oai ;;
        vllm) printf '%s\n' vllm ;;
        lmdeploy) printf '%s\n' lmdeploy ;;
        *) fail "unknown engine: $1" ;;
    esac
}

batch_order_for_repeat() {
    local repeat="$1"
    python3 - "$BATCH_SIZES" "$ORDER_SEED" "$repeat" <<'PY'
import random
import sys

batches = sys.argv[1].split()
random.Random(int(sys.argv[2]) + int(sys.argv[3])).shuffle(batches)
print(" ".join(batches))
PY
}

elapsed_seconds() {
    python3 - "$1" "$2" <<'PY'
import sys

print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000_000:.3f}")
PY
}

write_initial_metadata() {
    : >"$METADATA_FILE"
    env_line RUN_ID "$RUN_ID"
    env_line GENERATED_AT_UTC "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    env_line MODEL_ID "$MODEL_ID"
    env_line MODEL_REVISION "$MODEL_REVISION"
    env_line MODEL_PATH "$MODEL_PATH"
    env_line SERVED_MODEL_NAME "$SERVED_MODEL_NAME"
    env_line DTYPE "$DTYPE"
    env_line TP_SIZE "$TP_SIZE"
    env_line CONTEXT_LENGTH "$CONTEXT_LENGTH"
    env_line MAX_SERVER_BATCH "$MAX_SERVER_BATCH"
    env_line ENGINE_ORDER "$SELECTED_ENGINES"
    env_line BATCH_SIZES "$BATCH_SIZES"
    env_line NUM_PROMPTS "$NUM_PROMPTS"
    env_line WARMUP_REQUESTS "$WARMUP_REQUESTS"
    env_line REPEATS "$REPEATS"
    env_line INPUT_LENGTH "$INPUT_LENGTH"
    env_line OUTPUT_LENGTH "$OUTPUT_LENGTH"
    env_line WORKLOAD "$WORKLOAD"
    env_line GSP_NUM_GROUPS "$GSP_NUM_GROUPS"
    env_line GSP_SYSTEM_PROMPT_LENGTH "$GSP_SYSTEM_PROMPT_LENGTH"
    env_line GSP_QUESTION_LENGTH "$GSP_QUESTION_LENGTH"
    env_line GSP_RANGE_RATIO "$GSP_RANGE_RATIO"
    env_line DATASET_SEED_BASE "$DATASET_SEED_BASE"
    env_line ORDER_SEED "$ORDER_SEED"
    env_line GENERATION_SEED "$GENERATION_SEED"
    env_line TEMPERATURE "$TEMPERATURE"
    env_line TOP_P "$TOP_P"
    env_line IGNORE_EOS true
    env_line BATCH_DEFINITION max_concurrency
    env_line PREFIX_CACHING_MODE engine_default
    env_line TELEMETRY_SCOPE benchmark_client_process_window
    env_line UNIQUE_DATASET_SEED_PER_BATCH_REPEAT true
    env_line TELEMETRY_INTERVAL_MS "$TELEMETRY_INTERVAL_MS"
    env_line GPU_DEVICE_ID "$GPU_DEVICE_ID"
    env_line GPU_NAME "$(nvidia-smi --id="$GPU_DEVICE_ID" --query-gpu=name --format=csv,noheader | head -n 1)"
    env_line GPU_MEMORY_TOTAL_MIB "$(nvidia-smi --id="$GPU_DEVICE_ID" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    env_line NVIDIA_DRIVER_VERSION "$(nvidia-smi --id="$GPU_DEVICE_ID" --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
    env_line HOST_KERNEL "$(uname -srmo)"
    env_line DOCKER_VERSION "$(docker version --format '{{.Server.Version}}')"
    env_line COMPOSE_VERSION "$(docker compose version --short)"
    env_line SGLANG_IMAGE "$SGLANG_IMAGE"
    env_line VLLM_IMAGE "$VLLM_IMAGE"
    env_line LMDEPLOY_IMAGE "$LMDEPLOY_IMAGE"
    env_line SGLANG_MEM_FRACTION_STATIC "$SGLANG_MEM_FRACTION_STATIC"
    env_line VLLM_GPU_MEMORY_UTILIZATION "$VLLM_GPU_MEMORY_UTILIZATION"
    env_line LMDEPLOY_CACHE_MAX_ENTRY_COUNT "$LMDEPLOY_CACHE_MAX_ENTRY_COUNT"
    env_line MODEL_PREFETCHED_BEFORE_TIMING true
    local repeat
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        env_line "BATCH_ORDER_R${repeat}" "$(batch_order_for_repeat "$repeat")"
    done
}

pull_images() {
    [[ "$PULL_IMAGES" == "1" ]] || return 0
    local engine
    for engine in $SELECTED_ENGINES; do
        note "pulling pinned image for $engine"
        "${COMPOSE[@]}" --profile "$engine" pull "$engine"
    done
}

prefetch_model() {
    [[ "$PREFETCH_MODEL" == "1" ]] || return 0
    note "prefetching the pinned model revision outside startup and benchmark timings"
    local auth_args=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        auth_args+=(--env "HF_TOKEN=$HF_TOKEN")
    fi
    docker run --rm \
        --volume "$HF_CACHE_VOLUME:/root/.cache/huggingface" \
        --env HF_HOME=/root/.cache/huggingface \
        "${auth_args[@]}" \
        --entrypoint python3 \
        "$BENCHMARK_CLIENT_IMAGE" \
        -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])' \
        "$MODEL_ID" "$MODEL_REVISION"

    docker run --rm \
        --volume "$HF_CACHE_VOLUME:/model-cache:ro" \
        --entrypoint python3 \
        "$BENCHMARK_CLIENT_IMAGE" \
        -c 'from pathlib import Path; import sys; path = Path(sys.argv[1]); assert (path / "tokenizer_config.json").is_file(), f"incomplete snapshot: {path}"' \
        "/model-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/$MODEL_REVISION"
}

record_image_metadata() {
    local engine="$1"
    local image
    image="$(image_for_engine "$engine")"
    local digest
    digest="$(docker image inspect "$image" --format '{{join .RepoDigests ","}}')"
    [[ -n "$digest" ]] || digest="$(docker image inspect "$image" --format '{{.Id}}')"
    env_line "${engine^^}_IMAGE_RESOLVED" "$digest"
}

wait_until_ready() {
    local engine="$1"
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
    until curl --fail --silent --show-error --max-time 5 "$BASE_URL/v1/models" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            capture_server_log "$engine"
            fail "$engine did not become ready within ${SERVER_READY_TIMEOUT_SECONDS}s"
        fi
        if ! "${COMPOSE[@]}" --profile "$engine" ps --status running --services | grep -Fxq "$engine"; then
            capture_server_log "$engine"
            fail "$engine container stopped before readiness"
        fi
        sleep 2
    done
}

start_engine() {
    local engine="$1"
    note "starting $engine"
    local start_ns end_ns startup_seconds
    start_ns="$(date +%s%N)"
    "${COMPOSE[@]}" --profile "$engine" up --detach --no-deps "$engine"
    wait_until_ready "$engine"
    end_ns="$(date +%s%N)"
    startup_seconds="$(elapsed_seconds "$start_ns" "$end_ns")"
    env_line "${engine^^}_STARTUP_TO_READY_SECONDS" "$startup_seconds"
    note "$engine is ready in ${startup_seconds}s"
}

start_telemetry() {
    local output_file="$1"
    printf '%s\n' \
        'timestamp,index,name,pstate,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,power_w,temperature_c,sm_clock_mhz,memory_clock_mhz' \
        >"$output_file"
    nvidia-smi --id="$GPU_DEVICE_ID" \
        --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits \
        --loop-ms="$TELEMETRY_INTERVAL_MS" >>"$output_file" 2>&1 &
    TELEMETRY_PID=$!
}

run_benchmark_client() {
    local engine="$1"
    local batch="$2"
    local request_count="$3"
    local dataset_seed="$4"
    local tag="$5"
    local output_file="$6"
    local log_file="$7"
    local backend extra_body
    backend="$(backend_for_engine "$engine")"
    extra_body="{\"temperature\":${TEMPERATURE},\"top_p\":${TOP_P},\"seed\":${GENERATION_SEED},\"ignore_eos\":true}"

    docker run --rm --network host \
        --user "$(id -u):$(id -g)" \
        --volume "$HF_CACHE_VOLUME:/model-cache:ro" \
        --volume "$RUN_DIR:/results" \
        --env HOME=/tmp \
        --env HF_HOME=/model-cache \
        --env HF_HUB_OFFLINE=1 \
        --entrypoint python3 \
        "$BENCHMARK_CLIENT_IMAGE" \
        -m sglang.benchmark.serving \
        --backend "$backend" \
        --base-url "$BASE_URL" \
        --ready-check-timeout-sec 0 \
        --model "/model-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/$MODEL_REVISION" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tokenizer "/model-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/$MODEL_REVISION" \
        --dataset-name "$WORKLOAD" \
        --num-prompts "$request_count" \
        --gsp-num-groups "$GSP_NUM_GROUPS" \
        --gsp-prompts-per-group "$request_count" \
        --gsp-system-prompt-len "$GSP_SYSTEM_PROMPT_LENGTH" \
        --gsp-question-len "$GSP_QUESTION_LENGTH" \
        --gsp-output-len "$OUTPUT_LENGTH" \
        --gsp-range-ratio "$GSP_RANGE_RATIO" \
        --request-rate inf \
        --max-concurrency "$batch" \
        --warmup-requests 0 \
        --seed "$dataset_seed" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --extra-request-body "$extra_body" \
        --tag "$tag" \
        --output-file "$output_file" \
        --output-details \
        --disable-tqdm 2>&1 | tee "$log_file"
}

run_one_measurement() {
    local engine="$1"
    local batch="$2"
    local repeat="$3"
    local stem dataset_seed warmup_seed
    stem="${engine}_b${batch}_r${repeat}"
    dataset_seed=$((DATASET_SEED_BASE + repeat * 1000 + batch))
    warmup_seed=$((dataset_seed + 500000))

    note "$engine: warming batch=$batch with $WARMUP_REQUESTS requests"
    if ! run_benchmark_client \
        "$engine" "$batch" "$WARMUP_REQUESTS" "$warmup_seed" \
        "${stem}_warmup" "/tmp/${stem}_warmup.jsonl" \
        "$LOG_DIR/${stem}_warmup.log"; then
        capture_server_log "$engine"
        fail "warm-up failed: $stem"
    fi

    note "$engine: measuring batch=$batch repeat=$repeat dataset_seed=$dataset_seed"
    start_telemetry "$TELEMETRY_DIR/${stem}.csv"
    if ! run_benchmark_client \
        "$engine" "$batch" "$NUM_PROMPTS" "$dataset_seed" "$stem" \
        "/results/raw/${stem}.jsonl" "$LOG_DIR/${stem}_client.log"; then
        stop_telemetry
        capture_server_log "$engine"
        fail "measurement failed: $stem"
    fi
    stop_telemetry

    [[ -s "$RAW_DIR/${stem}.jsonl" ]] || fail "benchmark produced no raw result: $stem"
    line_count="$(wc -l <"$RAW_DIR/${stem}.jsonl")"
    [[ "$line_count" == "1" ]] || fail "expected one JSONL row in ${stem}.jsonl, got $line_count"
}

publish_file_atomically() {
    local source="$1"
    local destination="$2"
    local temporary="${destination}.tmp.$$"
    mkdir -p "$(dirname -- "$destination")"
    cp -- "$source" "$temporary"
    mv -- "$temporary" "$destination"
}

aggregate_and_publish() {
    note "aggregating raw results"
    docker run --rm --network none \
        --user "$(id -u):$(id -g)" \
        --volume "$PROJECT_ROOT/src:/workspace/src:ro" \
        --volume "$RUN_DIR:/results" \
        --env HOME=/tmp \
        --entrypoint python3 \
        "$BENCHMARK_CLIENT_IMAGE" \
        /workspace/src/rag_pipeline/benchmark_inference_engines.py aggregate \
        --run-dir /results \
        --report /results/report.md \
        --report-plot-prefix inference_engine_plots

    local artifact
    for artifact in report.md summary.csv summary.json experiment_metadata.json; do
        [[ -s "$RUN_DIR/$artifact" ]] || fail "aggregator did not create $RUN_DIR/$artifact"
    done
    for artifact in throughput_vs_batch.svg p95_ttft_vs_batch.svg p95_e2e_vs_batch.svg peak_vram_vs_batch.svg; do
        [[ -s "$RUN_DIR/plots/$artifact" ]] || fail "aggregator did not create plot: $artifact"
    done

    aggregation_status="$(python3 - "$RUN_DIR/experiment_metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    print(json.load(source).get("status", ""))
PY
)"
    [[ "$aggregation_status" == "complete" ]] || \
        fail "aggregation status is '$aggregation_status'; canonical artifacts were not updated"

    publish_file_atomically "$RUN_DIR/report.md" \
        "$PROJECT_ROOT/reports/inference_engine_comparison.md"
    publish_file_atomically "$RUN_DIR/summary.csv" "$EXPERIMENT_ROOT/summary.csv"
    publish_file_atomically "$RUN_DIR/summary.json" "$EXPERIMENT_ROOT/summary.json"
    publish_file_atomically "$RUN_DIR/experiment_metadata.json" \
        "$EXPERIMENT_ROOT/experiment_metadata.json"
    for artifact in "$RUN_DIR"/plots/*.svg; do
        publish_file_atomically "$artifact" \
            "$PROJECT_ROOT/reports/inference_engine_plots/$(basename -- "$artifact")"
    done
    note "canonical report and summaries updated after successful aggregation"
}

write_initial_metadata
"${COMPOSE[@]}" "${ALL_PROFILES[@]}" down --remove-orphans >/dev/null 2>&1 || true
pull_images
prefetch_model

for engine in $SELECTED_ENGINES; do
    CURRENT_ENGINE="$engine"
    record_image_metadata "$engine"
    start_engine "$engine"
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        order="$(batch_order_for_repeat "$repeat")"
        for batch in $order; do
            run_one_measurement "$engine" "$batch" "$repeat"
        done
    done
    stop_engine "$engine"
    CURRENT_ENGINE=""
    sleep "$ENGINE_COOLDOWN_SECONDS"
done

if [[ "$TARGET" == "all" ]]; then
    aggregate_and_publish
else
    note "single-engine diagnostic completed; canonical artifacts were not updated"
fi
note "completed: $RUN_DIR"
