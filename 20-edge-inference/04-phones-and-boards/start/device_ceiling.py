"""Decode ceiling and fit for one model on several devices.

ceiling tokens/s = memory bandwidth / file size   (every token reads the file)
fit              = file + KV cache at your context <= usable memory
Phones and boards are not in data/hardware.json, so they are listed here
with the page each number came from.
"""
from memory_plan import plan
from servinglab.data import hardware

DEVICES = {  # name: (bandwidth GB/s, usable memory GB; 8 GB boards: assume 1 GB for the system)
    "M5 Max laptop": (hardware("M5-Max-32c")["bandwidth_gbs"], 30.1),
    # Arm Learning Paths: Pi 5 (2 GB board) sustained 13.8 GB/s, 32-bit bus
    "Raspberry Pi 5 (8 GB)": (13.8, 7.0),
    # NVIDIA Orin Nano Super developer kit page: 128-bit LPDDR5, 102 GB/s
    "Jetson Orin Nano Super": (102.0, 7.0),
}
MODELS = {  # name: (data/models name, file GB)
    "Qwen3-0.6B Q4_K_M": ("Qwen3-0.6B", 0.40),
    "Qwen3-4B Q4_K_M": ("Qwen3-4B", 2.50),
    "Qwen3-4B Q8_0": ("Qwen3-4B", 4.28),
}


def report(context=4096):
    print(f"{'model':20s} {'device':24s} {'ceiling':>8s} {'needs':>7s}")
    for m, (cfg, gb) in MODELS.items():
        _, need = plan(cfg, gb, context)
        for d, (bw, mem) in DEVICES.items():
            verdict = "" if need <= mem else "  does not fit"
            print(f"{m:20s} {d:24s} {bw / gb:6.0f}/s {need:5.1f}GB{verdict}")


if __name__ == "__main__":
    report()
