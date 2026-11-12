from ondevice_report import ship_level

# The lesson's worked table: (level, file MB, RSS MB, TTFT ms, tok/s, right, total)
GPU = [("Q2_K", 296, 944, 23, 247, 6, 12), ("Q4_K_M", 397, 1015, 15, 335, 11, 12),
       ("Q8_0", 639, 1253, 19, 267, 11, 12)]
CPU = [("Q2_K", 296, 1072, 35, 110, 5, 12), ("Q4_K_M", 397, 1390, 115, 27, 11, 12),
       ("Q8_0", 639, 1849, 78, 62, 11, 12)]


def test_the_lessons_two_recommendations():
    assert ship_level(GPU, memory_mb=36_000) == "Q4_K_M"
    assert ship_level(CPU, memory_mb=36_000) == "Q8_0"


def test_a_broken_level_never_ships_even_when_it_is_fastest():
    assert ship_level(GPU, 36_000) != "Q2_K"


def test_a_small_device_rules_out_what_does_not_fit_under_60_percent():
    assert ship_level(GPU, memory_mb=1_800) == "Q4_K_M"      # 60% = 1,080 MB
    assert ship_level(GPU, memory_mb=1_500) is None          # 900 MB: none fit
