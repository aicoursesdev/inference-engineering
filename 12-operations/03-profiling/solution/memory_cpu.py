"""Which operators allocate memory in a step? profile_memory=True works on the CPU.

    uv run --with torch python memory_cpu.py

The CUDA memory snapshot (torch.cuda.memory._record_memory_history) is CUDA-only
in the PyTorch docs, so it is shown in the lesson, not run here.
"""
import torch
from torch.profiler import profile, ProfilerActivity
import profile_step as ps

x = torch.randn(16, ps.H)
with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    ps.step(x)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=4, max_name_column_width=34))
