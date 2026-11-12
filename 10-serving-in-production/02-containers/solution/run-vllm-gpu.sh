# From vLLM's Docker docs (docs.vllm.ai, deployment/docker, read 2026-11-12). Needs an NVIDIA GPU, the
# driver and the NVIDIA Container Toolkit on the host. Pin a version tag instead of latest in real use.
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
