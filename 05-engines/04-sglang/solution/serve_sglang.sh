#!/bin/bash
# Not run here (needs an NVIDIA GPU). SGLang v0.5.23 server arguments checked
# against its docs; whether it runs on a T4 was not checked.
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --dtype float16 --port 30000
