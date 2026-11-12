"""Find where a request's missing milliseconds went, from a client trace and a server log.

    uv run python find_gap.py

Step 1: per request, time to first token minus the server's own prefill time is the unexplained gap.
Step 2: the server log says when its single slot was busy; the gap should sit inside someone else's turn.
Log stamps look like 0.01.531.063: minutes . seconds . milliseconds . microseconds since start.
"""
import json, re
from pathlib import Path

DIR = Path(__file__).resolve().parents[3] / "data" / "recorded" / "incidents"
rows = [json.loads(l) for l in (DIR / "gap-trace.jsonl").read_text().splitlines()]
print("step 1: the unexplained gap")
for r in rows:
    raise NotImplementedError("TODO 12.4: gap_ms is the first-token time minus the server's prefill time, see the lesson's 'Step 1: put every clock on one line' section")
    print(f"  client {r['client']} sent at {r['sent_s']*1000:5.0f} ms: first token after {r['ttft_ms']:6.1f} ms, "
          f"prefill {r['prompt_ms']:5.1f} ms -> gap {r['gap_ms']:6.1f} ms")

stamp = lambda s: (lambda m, sec, ms, us: m * 60 + sec + ms / 1e3 + us / 1e6)(*map(int, s.split(".")))
busy, starts = {}, {}
for line in (DIR / "gap-server.log").read_text().splitlines():
    if m := re.match(r"(\S+) I slot launch_slot_: id +(\d+) \| task (\d+) \| processing task", line):
        starts[m[3]] = stamp(m[1])
    elif m := re.match(r"(\S+) I slot +release: id +(\d+) \| task (\d+)", line):
        busy[m[3]] = (starts[m[3]], stamp(m[1]))
print("\nstep 2: when the one slot was busy (seconds on the server's clock)")
for task, (a, b) in busy.items():
    print(f"  task {task:>2}: {a:6.3f} to {b:6.3f}  ({(b - a) * 1000:5.0f} ms)")
