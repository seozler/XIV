import numpy as np

from refanalyzer.analysis.sync import chance_rate, match


def test_chance_rate_regular_grid():
    beats = [{"t": t} for t in np.arange(0, 60, 0.5)]
    # +-0.1 s windows around beats every 0.5 s cover 40% of the time
    assert abs(chance_rate(beats, 0.1, 60) - 0.4) < 0.01


def test_match_signed_delta():
    hits, n = match([{"t": 1.04, "ref": "a"}, {"t": 3.3, "ref": "b"}], [{"t": 1.0}, {"t": 2.0}], 0.1)
    assert n == 2 and len(hits) == 1
    assert hits[0]["delta_ms"] == np.float64(40.0) or abs(hits[0]["delta_ms"] - 40) < 1e-6
