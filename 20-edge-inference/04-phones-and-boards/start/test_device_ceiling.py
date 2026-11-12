from device_ceiling import DEVICES, MODELS


def test_ceiling_is_bandwidth_over_file_size():
    bw, _ = DEVICES["Jetson Orin Nano Super"]
    assert round(bw / MODELS["Qwen3-4B Q4_K_M"][1]) == 41


def test_the_laptop_is_over_thirty_times_the_pi():
    assert DEVICES["M5 Max laptop"][0] / DEVICES["Raspberry Pi 5 (8 GB)"][0] > 30
