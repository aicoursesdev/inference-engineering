import numpy as np

import int4_pack as p


def test_pack_unpack_round_trip_and_first_value_in_low_bits():
    q = np.arange(16, dtype=np.uint8).reshape(2, 8) % 16
    word = p.pack(q)
    assert np.array_equal(p.unpack(word), q)
    assert int(word[0, 0]) & 0xF == q[0, 0]


def test_emulated_kernel_equals_dequantize_then_multiply():
    rng = np.random.default_rng(1)
    w, x = rng.normal(size=(4, 512)), rng.normal(size=512)
    q, s = p.quantize(w)
    deq = ((q.astype(float) - 8).reshape(4, 4, 128) * s[..., None]).reshape(4, 512)
    assert np.allclose(p.matvec_warp_emulation(p.pack(q), s, x), deq @ x)


def test_group_scales_cost_an_eighth_of_a_bit():
    assert p.bits_per_weight(128) == 4.125
    assert round(16 / p.bits_per_weight(128), 3) == 3.879   # why not 4x
