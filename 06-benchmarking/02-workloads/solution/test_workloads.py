import numpy as np

import arrivals
import trace_rate
import workload_stats
from servinglab.bench import arrival_times
from servinglab.data import traces


def test_a_burstier_stream_has_a_busier_busiest_second_at_the_same_mean_rate():
    even = arrival_times(8, 2400, burstiness=4.0, seed=0)
    bursty = arrival_times(8, 2400, burstiness=0.25, seed=0)
    assert abs(np.diff(even).mean() - 0.125) < 0.01 and abs(np.diff(bursty).mean() - 0.125) < 0.02
    assert arrivals.busiest_second(bursty) > arrivals.busiest_second(even)


def test_tidewaters_day_has_a_spike_above_its_daytime_average():
    counts = trace_rate.per_minute(traces("tidewater"))
    day = counts[9 * 60:18 * 60]
    assert round(day.mean(), 1) == 1.7
    assert counts.max() == 10


def test_summary_reads_percentiles_off_a_list_of_lengths():
    assert workload_stats.summary(range(1, 101)) == {"median": 50, "p95": 95, "p99": 99, "max": 100}
