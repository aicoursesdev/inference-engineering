from bytes_per_token import ceiling


def test_ceiling_is_bandwidth_over_file_size():
    assert round(ceiling(460, 0.373)) == 1233
    assert round(ceiling(460, 2.4)) == 192
    assert round(ceiling(460, 4.0)) == 115
