# Cost report: tidewater on m

Benchmark 2026-11-12 12:44:59 on macOS-26.6.2-arm64-arm-64bit; SLO TTFT 1.0 s, TPOT 50 ms.
Capacity at the SLO: 2 requests/s, 165 output tokens/s.

## Cost at $1.00 per GPU-hour
- input $0.306 and output $1.682 per million tokens (full load)
- output only: $1.682; at 30% utilisation $5.607

## Capacity for 0.17 requests/s peak
- 0.22 GPUs busy -> 2 with 30% headroom and 1 spare (11% busy at peak)

## Versus the API
- API $0.0000138 per request; one GPU always on $730.00 a month
- break-even 52,821,997 requests a month
