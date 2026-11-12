import pytest

import intensity


def test_vector_add_is_one_twelfth_of_a_flop_per_byte():
    assert intensity.vector_add(1000) == pytest.approx(1 / 12)


def test_a_big_matmul_has_intensity_n_over_three_in_float16():
    assert intensity.matmul(4096) == pytest.approx(4096 / 3)


def test_matrix_times_vector_is_about_one_flop_per_byte_in_float16():
    assert intensity.matvec(4096) == pytest.approx(1.0, abs=0.01)


def test_attainable_is_the_slope_then_the_roof():
    assert intensity.attainable(1, 989.5, 3350) == pytest.approx(3.35)   # memory-bound: 1 FLOP/byte x 3.35 TB/s
    assert intensity.attainable(1000, 989.5, 3350) == 989.5            # past the ridge: the roof
