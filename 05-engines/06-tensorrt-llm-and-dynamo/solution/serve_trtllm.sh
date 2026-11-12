#!/bin/bash
# Not run here (needs an NVIDIA GPU with FP8, such as an L4 or H100).
# From TensorRT-LLM's quick-start guide; use the NGC release container.
trtllm-serve "nvidia/Qwen3-8B-FP8"
