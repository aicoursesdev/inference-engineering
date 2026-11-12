import make_table


def test_gate_passes_q8_and_fails_the_damaged_build():
    verdict = {r[0]: r[7] for r in make_table.rows()}
    assert verdict["Q8_0"] and not verdict["Q2_K"]


def test_q2_k_fails_every_rule():
    q2 = next(r for r in make_table.rows() if r[0] == "Q2_K")
    assert len(q2[8]) == 4


def test_ship_rule_picks_the_smallest_passing_build():
    assert make_table.ship_rule(make_table.rows())[0] == "Q8_0"
