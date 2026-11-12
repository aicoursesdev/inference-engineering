import cold_starts as c


def test_same_traffic_the_lessons_692_requests():
    assert len(c.times) == 692


def test_longer_windows_mean_fewer_cold_requests_and_more_billed_hours():
    cold, hours = zip(*(c.simulate(c.times, w, 60) for w in (15, 60, 300, 1800)))
    assert list(cold) == sorted(cold, reverse=True)
    assert hours[-1] > hours[1]
    assert cold[0] == 523 and cold[-1] == 14


def test_a_faster_cold_start_bills_less_at_a_short_window():
    assert c.simulate(c.times, 15, 6)[1] < c.simulate(c.times, 15, 60)[1]
