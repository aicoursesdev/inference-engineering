# data/

Everything the course's code and lessons read. **Every number is copied from a source named beside it, or measured on a named machine.** The traces are the one exception: they are generated, and say so.

## models/

The real `config.json` of each model, byte for byte as Hugging Face served it (see `sources.json` for each URL and the date fetched). Read them with `servinglab.data.model_config("Qwen3-4B")`.

| File | Model | Used for |
|---|---|---|
| `SmolLM2-135M-Instruct.json`, `SmolLM2-360M-Instruct.json`, `SmolLM2-1.7B-Instruct.json` | HuggingFaceTB/SmolLM2 | the NumPy engine, the KV cache, speculation |
| `Qwen3-0.6B.json`, `Qwen3-1.7B.json`, `Qwen3-4B.json` | Qwen/Qwen3 | local endpoints, vLLM on a T4, quantization |
| `Llama-3.1-8B-Instruct.json`, `Llama-3.1-70B-Instruct.json` | Llama 3.1 | calculations only |
| `DeepSeek-V3.json`, `gpt-oss-20b.json`, `gpt-oss-120b.json` | DeepSeek-V3, gpt-oss | calculations only |

`sources.json` maps each file name (without `.json`) to `{repo, url, fetched}`. The two Llama files come from NousResearch's ungated re-upload, because Meta's repos need a login; their `note` says that the byte-identity with Meta's file could not be checked.

## hardware.json

`{"units": {...}, "hardware": [chip, ...]}`. Each chip:

| Field | Meaning |
|---|---|
| `name` | e.g. `H100-SXM`; the argument to `hardware(name)` |
| `vendor`, `kind` | `kind` is `gpu` or `laptop` |
| `memory_gb` | memory in decimal GB, as vendors quote it (a laptop's is its largest configuration) |
| `bandwidth_gbs` | memory bandwidth in decimal GB/s |
| `fp16_tflops`, `fp8_tflops`, `fp4_tflops` | **dense** FP16/BF16, FP8 and FP4 tensor TFLOPS; `null` where the chip has no such unit or no primary source gives the figure |
| `sources` | for each non-null number: `{url, fetched, note?, derived?}` |
| `notes` | why a field is null, and anything else a reader should know |

`derived: true` means the datasheet figure was "with sparsity" and says dense is half, so the dense number here is half of it; or (B200) a system total divided by its 8 GPUs. The T4's 65 TFLOPS is stored as dense because Turing has no structured sparsity and the page lists no sparsity footnote. NVIDIA publishes the A10, not the A10G, so the A10G's bandwidth and TFLOPS are `null`.

## prices.json

`gpu_hourly`: `{provider, gpu, usd_per_hour, billing, tier?, usd_per_second?, derived?, notes?, url, fetched}`. Where a provider publishes per second only, `usd_per_hour` is that times 3600 and `derived` is true. `api_per_million_tokens`: `{provider, model, input_usd_per_mtok, output_usd_per_mtok, url, fetched}`. These are list prices on the day named; check the page before quoting one.

## traces/

**Synthetic.** Invented customers, invented numbers, made by `make_traces.py` (`uv run python data/make_traces.py`) with a fixed seed, so the files never change unless the script does. One day, times in seconds since midnight IST. One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | `tw-00001` (Tidewater), `kk-` (Kodekart), `ny-` (Nyaya Docs), `nb-` (nightly batch) |
| `arrival_s` | when the request arrives, seconds since midnight IST |
| `customer`, `slo_class` | `tidewater`/`interactive`, `kodekart`/`agent`, `nyaya`/`long`, `batch`/`batch` |
| `input_tokens`, `output_tokens` | prompt length and reply length |
| `prefix_id`, `prefix_tokens` | the shared start of the prompt: which prefix, and how many of the input tokens it covers (`<= input_tokens`) |

The distributions (log-normal means "this median, this spread", clipped to the range):

| File | Requests | Arrivals | Input tokens | Output tokens | Prefix |
|---|---|---|---|---|---|
| `tidewater.jsonl` | about 1,160 | Poisson per minute: 0.025/s from 9:00 to 18:00, 0.018/s in the 13:00 lunch hour, a festival-sale spike at 0.1/s from 12:00 to 12:30, 0.008/s early evening and 07:00-09:00, 0.002/s at night | log-normal, median 350, 240-1,500 | log-normal, median 120, 10-600 | `tw-system`, 220 tokens, for every request |
| `kodekart.jsonl` | about 640 | 40 agent tasks start uniformly 08:00-20:00; each has 5-25 steps; the gap between steps is log-normal, median 4 s | starts at 3,000-3,500 and grows each step by the last reply plus a tool result (log-normal, median 500, 50-3,000) | log-normal, median 120, 20-500 | step 1: `kk-system` (3,000); later steps: `kk-task-NN`, the whole previous prompt |
| `nyaya.jsonl` | 100 | uniform 10:00-18:00 | one of 15 contracts (log-normal, median 35,000, 20,000-58,000) plus a question (median 80, 20-400) | log-normal, median 200, 40-600 | `ny-doc-NN`, the contract's length |
| `batch.jsonl` | 800 | all within two minutes from 02:00 | log-normal, median 1,200, 300-4,000 | log-normal, median 150, 40-400 | `batch-system`, 150 tokens |
| `mixed.jsonl` | the four together | in arrival order | | | |

These are shapes chosen to stress an engine in four different ways, not measurements of any real business.

## text/

Two small text samples for Lessons 07.3 and 07.4, so the activation and calibration runs read the same words on every machine: `wikitext2-train-sample.txt` (40,000 characters, used to calibrate) and `wikitext2-test-sample.txt` (6,000, used to test). They are the opening of the train and test splits of wikitext-2 (Wikipedia articles, with its " @-@ " spacing marks), cut by `make_samples.py`, which downloads the original from Hugging Face's `ggml-org/ci` dataset and rebuilds them exactly.

## recorded/

`llama-server-reply.json`: one real streamed reply, made on the machine named inside it by `record_reply.py` against `llama-server` running `Qwen3-0.6B-Q8_0.gguf` (from `Qwen/Qwen3-0.6B-GGUF`) at temperature 0 with thinking off. Fields: `recorded_at`, `machine` (chip, memory, OS), `llama_cpp` (version, command), `model`, `request` (the exact body sent), and `events`, a list of `{t_ms, data}`: the raw server-sent event payload of each chunk and the milliseconds since the request was sent. `make doctor` plays it back. Later modules add benchmark JSON, metrics snapshots and profiler traces here, each with the same "how and where it was made".

## Not in git

`../models/` holds model files you download (GGUF, safetensors). It is in `.gitignore`.
