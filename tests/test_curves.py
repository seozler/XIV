import numpy as np
import pytest

from refanalyzer.analysis import curves as c

T = np.arange(0, 1.5, 1 / 30)


def _make(f, noise=0.8, t0=0.3, dur=0.5, seed=1):
    rng = np.random.default_rng(seed)
    return 100 + 200 * f(np.clip((T - t0) / dur, 0, 1)) + rng.normal(0, noise, len(T))


@pytest.mark.parametrize("name", ["ease_out_cubic", "ease_in_out_sine", "linear", "ease_out_quint", "ease_out_bounce"])
def test_named_easing_recovered(name):
    r = c.analyse_curve(T, _make(c.NAMED[name]))
    assert r["inferred"]["easing"] == name
    assert abs(r["inferred"]["start_s"] - 0.3) < 0.02
    assert abs(r["inferred"]["end_s"] - 0.8) < 0.03
    assert r["inferred"]["fit_rmse"] < 0.01


def test_back_out_overshoot_measured():
    r = c.analyse_curve(T, _make(lambda p: c.back_out(p, 1.70158)))
    assert r["inferred"]["easing"] == "ease_out_back"
    assert r["inferred"]["params"]["overshoot_s"] == pytest.approx(1.70158, abs=0.15)
    assert r["measured"]["overshoot_ratio"] == pytest.approx(0.0996, abs=0.02)


def test_static_signal_has_no_animation():
    assert c.analyse_curve(T, np.full_like(T, 5.0)) is None


def test_cubic_bezier_matches_css_reference():
    f = c.cubic_bezier(0.33, 1, 0.68, 1)  # ~easeOutCubic
    p = np.linspace(0, 1, 50)
    assert np.max(np.abs(f(p) - c.NAMED["ease_out_cubic"](p))) < 0.04
    assert f(np.array([0.0]))[0] == pytest.approx(0, abs=1e-6)
    assert f(np.array([1.0]))[0] == pytest.approx(1, abs=1e-6)


def test_bezier_fit_close_to_curve():
    r = c.analyse_curve(T, _make(c.NAMED["ease_in_out_cubic"], noise=0.3))
    assert r["inferred"]["cubic_bezier_rmse"] < 0.03
