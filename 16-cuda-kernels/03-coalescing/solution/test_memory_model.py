import numpy as np

import memory_model as mm


def test_contiguous_warp_touches_four_sectors_misaligned_five():
    assert mm.sectors_touched(mm.stride_pattern(1)) == 4
    assert mm.sectors_touched(mm.stride_pattern(1, start=1)) == 5


def test_stride_two_halves_efficiency_and_big_strides_waste_28_of_32():
    assert mm.load_efficiency(mm.stride_pattern(2)) == 0.5
    assert mm.sectors_touched(mm.stride_pattern(8192)) == 32
    assert mm.load_efficiency(mm.stride_pattern(8192)) == 0.125


def test_padding_removes_the_32_way_conflict():
    assert mm.bank_conflict_degree(mm.tile_column_read(0)) == 32
    assert mm.bank_conflict_degree(mm.tile_column_read(1)) == 1


def test_same_word_is_a_broadcast_not_a_conflict():
    assert mm.bank_conflict_degree([5] * 32) == 1


def test_tiled_transpose_indexing_is_right_on_a_ragged_size():
    a = np.arange(70 * 70).reshape(70, 70)
    assert np.array_equal(mm.emulate_tiled_transpose(a), a.T)
