"""Will it fit? File size plus the KV cache for the context you ask for.

python memory_plan.py Qwen3-4B 2.49 32768
(model name from data/models, file size in GB, context in tokens)
"""
import sys

from servinglab.data import model_config


def kv_bytes_per_token(cfg, bytes_per_value=2):
    """Keys and values, every layer. 2 bytes each at float16."""
    raise NotImplementedError("TODO 20.3: 2 (keys and values) x layers x kv heads x head_dim x bytes per value, see the lesson's 'The flags that matter' section")


def plan(model, file_gb, context, bytes_per_value=2):
    raise NotImplementedError("TODO 20.3: KV GB = bytes per token x context / 1e9; return (KV GB, file GB + KV GB), see the lesson's 'The flags that matter' section")


if __name__ == "__main__":
    model, file_gb, context = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    kv, total = plan(model, file_gb, context)
    print(f"{model}: file {file_gb:.2f} GB + KV cache {kv:.2f} GB "
          f"at {context} tokens = {total:.2f} GB")
