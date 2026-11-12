"""Open a GGUF file and count its tensors by storage type.

    uv run --with gguf python inspect_gguf.py models/mine/Qwen3-0.6B-Q4_K_M.gguf
"""
import collections
import sys

from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
print(f"{len(r.fields)} metadata fields, {len(r.tensors)} tensors")
for key in ("general.architecture", "general.file_type", "qwen3.block_count",
            "qwen3.context_length"):
    f = r.fields[key]
    print(f"  {key} = {f.contents()}")
by_type = collections.defaultdict(lambda: [0, 0])
for t in r.tensors:
    by_type[t.tensor_type.name][0] += 1
    by_type[t.tensor_type.name][1] += int(t.n_bytes)
for name, (n, size) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
    print(f"  {name:6} {n:4} tensors {size / 1e6:8.1f} MB")
big = max(r.tensors, key=lambda t: t.n_bytes)
print(f"largest tensor: {big.name} {list(big.shape)} as {big.tensor_type.name}")
