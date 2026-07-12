# Сравнение инференс-движков LLM

**Статус эксперимента: `complete`.**

Матрица запусков заполнена полностью. Все итоговые значения рассчитаны только из raw JSONL и telemetry CSV.

## Что сравнивается

- **SGLang** — движок с RadixAttention/RadixCache, ориентированный на эффективное переиспользование общих префиксов и сложные serving-сценарии.
- **vLLM** — зрелый универсальный LLM-serving стек с PagedAttention, continuous batching и широким OpenAI-compatible API.
- **LMDeploy / TurboMind** — движок с persistent batching, собственными CUDA kernels и KV cache manager.

Внутренние оптимизации движков не отключаются: они являются предметом сравнения. Квантование, speculative decoding, LoRA и multi-GPU tensor parallelism не используются.

## Зафиксированные условия

| Параметр | Значение |
|---|---|
| Модель | `Qwen/Qwen2.5-3B-Instruct` |
| Revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Формат весов | FP16, без квантования |
| GPU / tensor parallelism | Tesla T4, 15360 MiB, compute capability 7.5, TP=1 |
| Workload | `rag-shared-prefix-exact-512` |
| Вход / выход | примерно 512 / 128 токенов |
| Запросов на ячейку | 16 |
| Warm-up запросов перед каждой ячейкой | 4 |
| Sampling | `temperature=0.0`, `top_p=1.0`, `generation_seed=none`, `ignore_eos=false` |

Нагрузка имитирует RAG: у запросов есть общий 128-токенный системный префикс и различающаяся 384-токенная часть с контекстом и вопросом. Seed генератора тестовых запросов меняется между batch/repeat, но остаётся одинаковым для трёх движков в одной ячейке.

`Batch size` здесь означает максимальное число одновременно выполняемых запросов (`max_concurrency`). Это корректнее статического batch для online serving, где все три движка используют continuous/persistent batching.

## Протокол

- Движки: sglang, vllm, lmdeploy.
- Effective batch size / concurrency: 1, 2, 4, 8, 16.
- Повторений для каждой комбинации: 3.
- Bootstrap: 3000 выборок; 95% CI построен для медианы по независимым повторам.
- Mean, median и sample standard deviation считаются между повторами, а не между отдельными запросами внутри одного запуска.
- Error rate рассчитывается только при известном полном числе отправленных и завершённых запросов.

## Полнота данных

- Ожидалось запусков: 45.
- Валидных raw-записей: 45 (45 уникальных ключей).
- Telemetry-записей: 45 (45 уникальных ключей).
- Отсутствуют raw: нет.
- Дубликаты raw: нет.
- Отсутствует telemetry: нет.
- Дубликаты telemetry: нет.
- Некорректные raw-записи: 0.

## Результаты

| Движок | Batch | Запуски | Output tok/s, median [95% CI] | Req/s | p95 TTFT, ms | p95 E2E, ms | Ошибки | Peak VRAM, MiB | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lmdeploy | 1 | 3/3 | 27.45 [27.25; 27.62] | 0.21 | 188.60 | 4693.53 | 0.00% | 13583.0 | 1.000 | 1.000 |
| lmdeploy | 2 | 3/3 | 62.62 [62.53; 62.71] | 0.49 | 357.03 | 4097.98 | 0.00% | 13583.0 | 2.281 | 1.141 |
| lmdeploy | 4 | 3/3 | 114.99 [113.85; 115.37] | 0.90 | 675.27 | 4451.75 | 0.00% | 13583.0 | 4.188 | 1.047 |
| lmdeploy | 8 | 3/3 | 193.85 [193.52; 194.73] | 1.51 | 1331.53 | 5435.24 | 0.00% | 13583.0 | 7.061 | 0.883 |
| lmdeploy | 16 | 3/3 | 295.89 [293.81; 299.55] | 2.31 | 2578.69 | 6910.54 | 0.00% | 13583.0 | 10.778 | 0.674 |
| sglang | 1 | 3/3 | 23.33 [23.26; 23.45] | 0.18 | 415.75 | 5576.98 | 0.00% | 12823.0 | 1.000 | 1.000 |
| sglang | 2 | 3/3 | 55.17 [55.15; 55.25] | 0.43 | 799.89 | 4695.89 | 0.00% | 12823.0 | 2.365 | 1.183 |
| sglang | 4 | 3/3 | 91.96 [91.80; 92.21] | 0.72 | 1579.81 | 5681.32 | 0.00% | 12823.0 | 3.942 | 0.986 |
| sglang | 8 | 3/3 | 136.17 [135.35; 136.48] | 1.06 | 3056.60 | 7610.50 | 0.00% | 12823.0 | 5.837 | 0.730 |
| sglang | 16 | 3/3 | 125.31 [124.72; 125.90] | 0.98 | 7182.07 | 12314.38 | 0.00% | 12823.0 | 5.372 | 0.336 |
| vllm | 1 | 3/3 | 24.27 [24.26; 31.22] | 0.19 | 191.37 | 5296.19 | 0.00% | 11963.0 | 1.000 | 1.000 |
| vllm | 2 | 3/3 | 59.86 [59.86; 60.26] | 0.47 | 361.48 | 4280.55 | 0.00% | 11963.0 | 2.467 | 1.233 |
| vllm | 4 | 3/3 | 107.49 [105.46; 108.04] | 0.84 | 685.01 | 4797.24 | 0.00% | 11963.0 | 4.430 | 1.107 |
| vllm | 8 | 3/3 | 177.31 [176.17; 177.74] | 1.39 | 1243.84 | 5793.63 | 0.00% | 11963.0 | 7.307 | 0.913 |
| vllm | 16 | 3/3 | 264.87 [263.98; 281.57] | 2.07 | 2413.55 | 7721.55 | 0.00% | 11963.0 | 10.915 | 0.682 |

## Графики

- [throughput_vs_batch.svg](../data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/plots/throughput_vs_batch.svg)
- [p95_ttft_vs_batch.svg](../data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/plots/p95_ttft_vs_batch.svg)
- [p95_e2e_vs_batch.svg](../data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/plots/p95_e2e_vs_batch.svg)
- [peak_vram_vs_batch.svg](../data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/plots/peak_vram_vs_batch.svg)

## Вывод

Победители по медианной пропускной способности:

- batch 1: lmdeploy (27.45 output tok/s).
- batch 2: lmdeploy (62.62 output tok/s).
- batch 4: lmdeploy (114.99 output tok/s).
- batch 8: lmdeploy (193.85 output tok/s).
- batch 16: lmdeploy (295.89 output tok/s).

Минимальная интерактивная задержка при batch=1:

- p95 TTFT: lmdeploy (188.60 ms).
- p95 E2E: lmdeploy (4693.53 ms).

Для выбора движка в интерактивном RAG нужно одновременно учитывать output throughput, p95 TTFT, p95 E2E, error rate и расход VRAM. Максимальный throughput сам по себе не означает лучший пользовательский режим.

Результаты относятся только к зафиксированным модели, GPU, версиям ПО и workload из metadata; они не доказывают универсальное превосходство одного движка.

## Интеграция в RAG

По результатам полной матрицы выбран `lmdeploy`. Он подключён в приложение как опциональный OpenAI-compatible provider; Yandex AI Studio остаётся вариантом по умолчанию для запуска без локальной NVIDIA GPU.

## Ограничения

- Peak VRAM относится к окну измеряемого запуска. Средние GPU utilization и power включают короткий запуск benchmark-клиента и считаются диагностическими, а не основой выбора победителя.
- Параметры памяти движков имеют разную семантику: `mem-fraction-static`, `gpu-memory-utilization` и `cache-max-entry-count` нельзя трактовать как один и тот же memory budget. Их значения отдельно сохранены в metadata.
- Ватты показывают среднюю мощность, а не энергию. Для оценки стоимости энергии потребовались бы J/request или J/output-token.
- Результат RAG-like workload нельзя автоматически переносить на другую модель, GPU, длину контекста или совершенно другой профиль запросов.
- На Tesla T4 (SM75) SGLang использовал совместимые Triton/CUDA kernels: быстрый CuTe DSL путь текущей версии рассчитан на более новые GPU. Результат SGLang нельзя напрямую переносить на A100/H100.

## Воспроизведение

Нужен Linux-хост с NVIDIA GPU не менее 12 GB VRAM, Docker Engine, NVIDIA Container Toolkit, Compose v2 и примерно 70 GB свободного места.

```bash
bash scripts/inference_engines/preflight.sh
bash scripts/inference_engines/run_benchmark.sh all
```

## Официальные источники

- [SGLang: benchmark online serving](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md)
- [vLLM: OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [LMDeploy: архитектура TurboMind](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)

## Metadata runner

- `BATCH_SIZES=1,2,4,8,16`
- `ENGINES=sglang,vllm,lmdeploy`
- `EXPERIMENT_PLATFORM=Kaggle`
- `GENERATION_SEED=none`
- `GPU_COMPUTE_CAPABILITY=7.5`
- `GPU_DRIVER_VERSION=580.159.04`
- `GPU_INDEX=0`
- `GPU_MEMORY_MIB=15360`
- `GPU_NAME=Tesla T4`
- `IGNORE_EOS=false`
- `INPUT_TOKENS=512`
- `LMDEPLOY_VERSION=0.14.0`
- `MODEL_ID=Qwen/Qwen2.5-3B-Instruct`
- `MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1`
- `NUM_PROMPTS=16`
- `OUTPUT_TOKENS=128`
- `PROMPT_FILE_SHA256=7f4c29ab0e3d9c965a32cd1aaabd80a94c87f45b842abc79d707f51607a076df`
- `REPEATS=3`
- `SELECTED_ENGINE=lmdeploy`
- `SGLANG_ATTENTION_BACKEND=triton`
- `SGLANG_FLASHINFER_USE_CUDA_NORM=1`
- `SGLANG_VERSION=0.5.15`
- `TEMPERATURE=0.0`
- `TOP_P=1.0`
- `VLLM_VERSION=0.25.0`
- `WARMUP_PROMPTS=4`
- `WORKLOAD=rag-shared-prefix-exact-512`
