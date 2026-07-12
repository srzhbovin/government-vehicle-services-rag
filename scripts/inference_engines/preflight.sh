#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd -P)"
COMPOSE_FILE="$PROJECT_ROOT/deploy/inference_engines/compose.yaml"
DEFAULT_ENV_FILE="$PROJECT_ROOT/deploy/inference_engines/benchmark.env"
EXAMPLE_ENV_FILE="$PROJECT_ROOT/deploy/inference_engines/benchmark.env.example"

fail() {
    printf 'Preflight failed: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[preflight] %s\n' "$*"
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

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

require_equal() {
    local name="$1"
    local actual="$2"
    local expected="$3"
    [[ "$actual" == "$expected" ]] || fail "$name must be '$expected', got '$actual'"
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer, got '$value'"
}

ENV_FILE="$(resolve_env_file "${1:-}")"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

[[ "$(uname -s)" == "Linux" ]] || fail "this experiment requires Linux; Windows and macOS are unsupported"

for command_name in docker curl nvidia-smi python3 awk df date grep head tr wc tee cp mv sleep uname id; do
    require_command "$command_name"
done

[[ -f "$COMPOSE_FILE" ]] || fail "Compose file not found: $COMPOSE_FILE"
docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 plugin is unavailable"

require_equal MODEL_ID "${MODEL_ID:-}" "Qwen/Qwen2.5-3B-Instruct"
require_equal MODEL_REVISION "${MODEL_REVISION:-}" "aa8e72537993ba99e69dfaafa59ed015b17504d1"
require_equal MODEL_PATH "${MODEL_PATH:-}" "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
require_equal DTYPE "${DTYPE:-}" "float16"
require_equal TP_SIZE "${TP_SIZE:-}" "1"
require_equal CONTEXT_LENGTH "${CONTEXT_LENGTH:-}" "2048"
require_equal MAX_SERVER_BATCH "${MAX_SERVER_BATCH:-}" "16"
require_equal BATCH_SIZES "${BATCH_SIZES:-}" "1 2 4 8 16"
require_equal NUM_PROMPTS "${NUM_PROMPTS:-}" "80"
require_equal WARMUP_REQUESTS "${WARMUP_REQUESTS:-}" "16"
require_equal REPEATS "${REPEATS:-}" "5"
require_equal INPUT_LENGTH "${INPUT_LENGTH:-}" "512"
require_equal OUTPUT_LENGTH "${OUTPUT_LENGTH:-}" "128"
require_equal WORKLOAD "${WORKLOAD:-}" "generated-shared-prefix"
require_equal GSP_NUM_GROUPS "${GSP_NUM_GROUPS:-}" "1"
require_equal GSP_SYSTEM_PROMPT_LENGTH "${GSP_SYSTEM_PROMPT_LENGTH:-}" "128"
require_equal GSP_QUESTION_LENGTH "${GSP_QUESTION_LENGTH:-}" "384"
require_equal GSP_RANGE_RATIO "${GSP_RANGE_RATIO:-}" "1.0"
require_equal TEMPERATURE "${TEMPERATURE:-}" "0.0"
require_equal TOP_P "${TOP_P:-}" "1.0"
require_equal SGLANG_MEM_FRACTION_STATIC "${SGLANG_MEM_FRACTION_STATIC:-}" "0.85"
require_equal VLLM_GPU_MEMORY_UTILIZATION "${VLLM_GPU_MEMORY_UTILIZATION:-}" "0.85"
require_equal LMDEPLOY_CACHE_MAX_ENTRY_COUNT "${LMDEPLOY_CACHE_MAX_ENTRY_COUNT:-}" "0.80"
require_equal ENGINE_ORDER "${ENGINE_ORDER:-}" "sglang vllm lmdeploy"
require_equal PULL_IMAGES "${PULL_IMAGES:-}" "1"
require_equal PREFETCH_MODEL "${PREFETCH_MODEL:-}" "1"
require_equal HF_HUB_OFFLINE "${HF_HUB_OFFLINE:-}" "1"
require_equal SGLANG_IMAGE "${SGLANG_IMAGE:-}" "lmsysorg/sglang:v0.5.15-cu129-runtime@sha256:c776530cdc5029994d0e3935c6c6d0a3746b9cafd0a19930b5bd3689585d4177"
require_equal VLLM_IMAGE "${VLLM_IMAGE:-}" "vllm/vllm-openai:v0.25.0-cu129@sha256:f7415b0939b32377a1d16429a063977a4ec3224d537aa16af726597cc8e3cdac"
require_equal LMDEPLOY_IMAGE "${LMDEPLOY_IMAGE:-}" "openmmlab/lmdeploy:v0.14.0-cu12.8@sha256:e5f59ec579ec0bc1472ee63705eb04152f30b8236ba97cfc116d21c1fdf3bc76"
require_equal BENCHMARK_CLIENT_IMAGE "${BENCHMARK_CLIENT_IMAGE:-}" "$SGLANG_IMAGE"

for integer_name in DATASET_SEED_BASE ORDER_SEED GENERATION_SEED SERVER_PORT \
    SERVER_READY_TIMEOUT_SECONDS TELEMETRY_INTERVAL_MS MIN_GPU_MEMORY_MIB \
    RECOMMENDED_GPU_MEMORY_MIB MIN_FREE_DISK_GIB ENGINE_COOLDOWN_SECONDS; do
    require_positive_integer "$integer_name" "${!integer_name:-}"
done
[[ "${GPU_DEVICE_ID:-}" =~ ^[0-9]+$ ]] || fail "GPU_DEVICE_ID must be a non-negative integer"

(( INPUT_LENGTH + OUTPUT_LENGTH <= CONTEXT_LENGTH )) || \
    fail "input and output lengths exceed the configured context length"
(( GSP_SYSTEM_PROMPT_LENGTH + GSP_QUESTION_LENGTH == INPUT_LENGTH )) || \
    fail "shared-prefix and question lengths must add up to INPUT_LENGTH"
(( NUM_PROMPTS >= 5 * MAX_SERVER_BATCH )) || \
    fail "NUM_PROMPTS must be at least five times the largest batch size"

docker compose \
    --project-name "$COMPOSE_PROJECT_NAME" \
    --env-file "$ENV_FILE" \
    --file "$COMPOSE_FILE" \
    --profile sglang --profile vllm --profile lmdeploy \
    config --quiet || fail "Compose configuration is invalid"

gpu_memory_mib="$(
    nvidia-smi --id="$GPU_DEVICE_ID" \
        --query-gpu=memory.total --format=csv,noheader,nounits | \
        awk 'NR == 1 {gsub(/ /, "", $1); print int($1)}'
)"
[[ "$gpu_memory_mib" =~ ^[0-9]+$ ]] || fail "cannot read GPU $GPU_DEVICE_ID memory"
(( gpu_memory_mib >= MIN_GPU_MEMORY_MIB )) || \
    fail "GPU has ${gpu_memory_mib} MiB; at least ${MIN_GPU_MEMORY_MIB} MiB is required"
if (( gpu_memory_mib < RECOMMENDED_GPU_MEMORY_MIB )); then
    note "warning: ${gpu_memory_mib} MiB VRAM is below the recommended ${RECOMMENDED_GPU_MEMORY_MIB} MiB"
fi

docker_root="$(docker info --format '{{.DockerRootDir}}')"
free_disk_gib="$(df -Pk "$docker_root" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
[[ "$free_disk_gib" =~ ^[0-9]+$ ]] || fail "cannot determine free space under $docker_root"
(( free_disk_gib >= MIN_FREE_DISK_GIB )) || \
    fail "Docker storage has ${free_disk_gib} GiB free; ${MIN_FREE_DISK_GIB} GiB is required"

note "checking NVIDIA Container Toolkit with $CUDA_PROBE_IMAGE"
docker run --rm --gpus all \
    --env "NVIDIA_VISIBLE_DEVICES=$GPU_DEVICE_ID" \
    "$CUDA_PROBE_IMAGE" nvidia-smi -L >/dev/null || \
    fail "Docker cannot access NVIDIA GPU $GPU_DEVICE_ID"

gpu_name="$(nvidia-smi --id="$GPU_DEVICE_ID" --query-gpu=name --format=csv,noheader | head -n 1)"
note "OK: Linux, Docker Compose, NVIDIA runtime and GPU are ready"
note "GPU: $gpu_name (${gpu_memory_mib} MiB); Docker free space: ${free_disk_gib} GiB"
note "configuration: $ENV_FILE"
