import ceilings


def test_a_t4_halves_the_decode_bytes_in_8_bits_but_has_no_fp8_prefill():
    row = ceilings.row("T4")
    assert "fp16:    20 dec    4,047 pre" in row
    assert "fp8:    40 dec        - pre" in row  # decode doubles, prefill gets no help


def test_a_b200_speeds_up_both_decode_and_prefill_in_4_bits():
    assert "fp4: 1,771 dec  560,399 pre" in ceilings.row("B200")
