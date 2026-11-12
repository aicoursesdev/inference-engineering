import pytest

import ridge_table


def test_h100_ridge_is_about_295_and_trusting_the_headline_doubles_it():
    h100 = next(r for r in ridge_table.rows() if r["name"] == "H100-SXM")
    assert h100["ridge"] == pytest.approx(295.4, abs=0.1)
    assert h100["headline"] == 1979 and h100["wrong_ridge"] == pytest.approx(590.7, abs=0.1)


def test_chips_the_vendor_prints_dense_are_not_doubled():
    by_name = {r["name"]: r for r in ridge_table.rows()}
    assert by_name["T4"]["headline"] == 65 and by_name["A100-80GB-SXM"]["headline"] == 312


def test_chips_with_no_published_figure_are_left_out():
    names = {r["name"] for r in ridge_table.rows()}
    assert "A10G" not in names and "M5-Max" not in names


def test_every_ridge_sits_in_the_same_broad_band():
    assert all(100 < r["ridge"] < 450 for r in ridge_table.rows())
