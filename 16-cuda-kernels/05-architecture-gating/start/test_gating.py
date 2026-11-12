import gating as g


def test_the_t4_is_below_every_listed_fast_path_but_flashinfer():
    t4 = g.GPUS["T4"]
    yes = [k for k in g.MATRIX if g.supported(k, t4) == "yes"]
    assert yes == ["FlashInfer (SGLang default)"]


def test_fa3_is_hopper_only_and_fa2_skips_blackwell():
    assert g.supported("FlashAttention 3 (beta)", 9.0) == "yes"
    assert g.supported("FlashAttention 3 (beta)", 8.0) == "not listed"
    assert g.supported("FlashAttention 2 (flash-attn)", 10.0) == "not listed"


def test_every_row_names_its_source():
    assert all(row["source"] for row in g.MATRIX.values())
