# Inference Engineering: the course repo

Code and data for the Inference Engineering course on [aicourses.dev](https://aicourses.dev). You join **Monsoon Cloud**, an imaginary Bengaluru startup that serves open models to other companies, as its first inference engineer, and over the course you build one growing codebase, `servinglab`: an engine, a KV cache, a scheduler, a benchmark harness, a quality gate, a cost model. Monsoon, its people and its four customers (Tidewater Goods, Kodekart, Nyaya Docs and a nightly batch) are invented; the models, GPUs and prices in `data/` are real and each carries its source.

## Set up

You need [uv](https://docs.astral.sh/uv/) (it installs Python for you) and git.

```sh
git clone https://github.com/aicoursesdev/inference-engineering.git
cd inference-engineering
uv sync          # core packages and servinglab, into .venv
make doctor      # checks your machine, then plays a recorded reply
```

Heavier tools are extras, each including the ones before it; install one when a lesson asks:

| Command | Adds | For |
|---|---|---|
| `uv sync --extra tests` | pytest | `make check` (a plain `uv sync` already has it) |
| `uv sync --extra engine` | safetensors, tokenizers, huggingface_hub | the NumPy engine |
| `uv sync --extra bench` | matplotlib, pandas | benchmarks and plots |
| `uv sync --extra quant` | gguf | reading and inspecting quantized files |
| `uv sync --extra service` | fastapi, uvicorn, prometheus-client, opentelemetry-sdk | serving and monitoring |
| `uv sync --extra course` | all of the above | everything light, in one go |

`vllm` and `triton` are never installed by this repo: both ship Linux wheels only. The lessons that need them install them in Colab or on the rented machine and say how.

## Three ways to run

| Path | What you need | What runs |
|---|---|---|
| **Laptop only** | any Mac, Windows or Linux machine with 8 GB or more; [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`, `llama-bench`) from lesson 0.3 | the NumPy engine, simulators, traces, llama.cpp and Ollama, most of the course |
| **Free T4** | a Colab or Kaggle notebook with a T4 | single-process vLLM in float16, quantization runs, profiling |
| **Rented GPU** | an hour or more on a cloud GPU, shut down afterwards | servers under load, deployments, kernels, the capstone |

`make doctor` tells you which one you are on.

## Commands

```sh
make doctor              # your machine, your tools, a recorded reply
make check 03.2          # run lesson 3.2's tests on your code in 03-kv-cache/<lesson>/start/
make check 03.2 SOLUTION=1   # the same tests on the finished solution/
make diff 03.2           # what solution/ has that start/ does not
make test                # every lesson's solution, then the servinglab package
uv run python scripts/check_data.py   # checks every file in data/
```

Each lesson with code has a folder `<module folder>/<lesson file stem>/` holding `start/` (your part as stubs) and `solution/`, with the same offline tests in both: no model, no GPU and no network are needed to run them.

## What is here

- `servinglab/`: the shared package. `servinglab.data` is given (`traces`, `hardware`, `model_config`, `prices`); every other module is finished by the lesson that builds it.
- `data/`: traces, hardware, model configs, prices and recordings. `data/README.md` lists every file and field.
- `scripts/`: `doctor.py`, `lesson.py`, `check_data.py`.
- `models/`: model files you download. Not in git.
