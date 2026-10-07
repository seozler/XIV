"""Motion-curve fitting: measured samples -> nearest easing function.

The rendered video never reveals the original CSS/GSAP easing, so this
module measures the progress curve and reports the closest mathematical
match together with its fit error. Everything returned here is
``inferred`` except duration and the raw kinematics, which are measured.

Input: timestamps ``t`` and a scalar property ``v`` (e.g. distance along
the path, scale, opacity). The curve is normalised to progress
``p = (v - v0) / (v1 - v0)`` where ``v1`` is the settled value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------- easings
Easing = Callable[[np.ndarray], np.ndarray]


def _poly_in(k: float) -> Easing:
    return lambda p: p ** k


def _poly_out(k: float) -> Easing:
    return lambda p: 1 - (1 - p) ** k


def _poly_inout(k: float) -> Easing:
    return lambda p: np.where(p < 0.5, 2 ** (k - 1) * p ** k, 1 - (-2 * p + 2) ** k / 2)


def _expo_out(p: np.ndarray) -> np.ndarray:
    return np.where(p >= 1, 1.0, 1 - 2.0 ** (-10 * p))


def _expo_in(p: np.ndarray) -> np.ndarray:
    return np.where(p <= 0, 0.0, 2.0 ** (10 * p - 10))


def _expo_inout(p: np.ndarray) -> np.ndarray:
    return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, np.where(
        p < 0.5, 2.0 ** (20 * p - 10) / 2, (2 - 2.0 ** (-20 * p + 10)) / 2)))


def back_out(p: np.ndarray, s: float = 1.70158) -> np.ndarray:
    q = p - 1
    return 1 + (s + 1) * q ** 3 + s * q ** 2


def back_in(p: np.ndarray, s: float = 1.70158) -> np.ndarray:
    return (s + 1) * p ** 3 - s * p ** 2


def elastic_out(p: np.ndarray, period: float = 0.3) -> np.ndarray:
    # period=0.3 reproduces the standard easeOutElastic (c4 = 2*pi/3).
    c4 = (2 * math.pi / 3) * (0.3 / max(period, 1e-3))
    return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, 2.0 ** (-10 * p) * np.sin((p * 10 - 0.75) * c4) + 1))


def bounce_out(p: np.ndarray) -> np.ndarray:
    n1, d1 = 7.5625, 2.75
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    a = p < 1 / d1
    b = (~a) & (p < 2 / d1)
    c = (~a) & (~b) & (p < 2.5 / d1)
    d = ~(a | b | c)
    out[a] = n1 * p[a] ** 2
    q = p[b] - 1.5 / d1
    out[b] = n1 * q ** 2 + 0.75
    q = p[c] - 2.25 / d1
    out[c] = n1 * q ** 2 + 0.9375
    q = p[d] - 2.625 / d1
    out[d] = n1 * q ** 2 + 0.984375
    return out


def spring(p: np.ndarray, zeta: float, omega: float) -> np.ndarray:
    """Damped harmonic oscillator step response; ``p`` in units of duration."""
    t = np.asarray(p, dtype=float)
    zeta = float(np.clip(zeta, 0.02, 0.999))
    wd = omega * math.sqrt(1 - zeta ** 2)
    return 1 - np.exp(-zeta * omega * t) * (np.cos(wd * t) + zeta * omega / wd * np.sin(wd * t))


def cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> Easing:
    """CSS ``cubic-bezier(x1, y1, x2, y2)`` evaluated by Newton + bisection."""

    def bez(u, a1, a2):
        return 3 * a1 * u * (1 - u) ** 2 + 3 * a2 * u ** 2 * (1 - u) + u ** 3

    def dbez(u, a1, a2):
        return 3 * a1 * (1 - u) ** 2 + 6 * (a2 - a1) * u * (1 - u) + 3 * (1 - a2) * u ** 2

    def f(p: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(p, dtype=float), 0, 1)
        u = x.copy()
        for _ in range(8):
            d = dbez(u, x1, x2)
            ok = np.abs(d) > 1e-6
            u = np.where(ok, u - (bez(u, x1, x2) - x) / np.where(ok, d, 1), u)
            u = np.clip(u, 0, 1)
        lo, hi = np.zeros_like(x), np.ones_like(x)
        bad = np.abs(bez(u, x1, x2) - x) > 1e-4
        if bad.any():
            for _ in range(30):
                mid = (lo + hi) / 2
                gt = bez(mid, x1, x2) > x
                hi = np.where(gt, mid, hi)
                lo = np.where(gt, lo, mid)
            u = np.where(bad, (lo + hi) / 2, u)
        return bez(u, y1, y2)

    return f


NAMED: dict[str, Easing] = {
    "linear": lambda p: p,
    "ease_in_sine": lambda p: 1 - np.cos(p * np.pi / 2),
    "ease_out_sine": lambda p: np.sin(p * np.pi / 2),
    "ease_in_out_sine": lambda p: -(np.cos(np.pi * p) - 1) / 2,
    "ease_in_quad": _poly_in(2), "ease_out_quad": _poly_out(2), "ease_in_out_quad": _poly_inout(2),
    "ease_in_cubic": _poly_in(3), "ease_out_cubic": _poly_out(3), "ease_in_out_cubic": _poly_inout(3),
    "ease_in_quart": _poly_in(4), "ease_out_quart": _poly_out(4), "ease_in_out_quart": _poly_inout(4),
    "ease_in_quint": _poly_in(5), "ease_out_quint": _poly_out(5), "ease_in_out_quint": _poly_inout(5),
    "ease_in_expo": _expo_in, "ease_out_expo": _expo_out, "ease_in_out_expo": _expo_inout,
    "ease_out_circ": lambda p: np.sqrt(np.clip(1 - (p - 1) ** 2, 0, 1)),
    "ease_in_circ": lambda p: 1 - np.sqrt(np.clip(1 - p ** 2, 0, 1)),
    "ease_out_bounce": bounce_out,
}

# Equivalent names in common animation stacks (all approximate).
EQUIVALENTS: dict[str, dict[str, str]] = {
    "linear": {"css": "linear", "gsap": "none", "remotion": "Easing.linear"},
    "ease_in_sine": {"css": "cubic-bezier(0.12, 0, 0.39, 0)", "gsap": "sine.in", "remotion": "Easing.in(Easing.sin)"},
    "ease_out_sine": {"css": "cubic-bezier(0.61, 1, 0.88, 1)", "gsap": "sine.out", "remotion": "Easing.out(Easing.sin)"},
    "ease_in_out_sine": {"css": "cubic-bezier(0.37, 0, 0.63, 1)", "gsap": "sine.inOut", "remotion": "Easing.inOut(Easing.sin)"},
    "ease_in_quad": {"css": "cubic-bezier(0.11, 0, 0.5, 0)", "gsap": "power1.in", "remotion": "Easing.in(Easing.quad)"},
    "ease_out_quad": {"css": "cubic-bezier(0.5, 1, 0.89, 1)", "gsap": "power1.out", "remotion": "Easing.out(Easing.quad)"},
    "ease_in_out_quad": {"css": "cubic-bezier(0.45, 0, 0.55, 1)", "gsap": "power1.inOut", "remotion": "Easing.inOut(Easing.quad)"},
    "ease_in_cubic": {"css": "cubic-bezier(0.32, 0, 0.67, 0)", "gsap": "power2.in", "remotion": "Easing.in(Easing.cubic)"},
    "ease_out_cubic": {"css": "cubic-bezier(0.33, 1, 0.68, 1)", "gsap": "power2.out", "remotion": "Easing.out(Easing.cubic)"},
    "ease_in_out_cubic": {"css": "cubic-bezier(0.65, 0, 0.35, 1)", "gsap": "power2.inOut", "remotion": "Easing.inOut(Easing.cubic)"},
    "ease_in_quart": {"css": "cubic-bezier(0.5, 0, 0.75, 0)", "gsap": "power3.in", "remotion": "Easing.in(Easing.poly(4))"},
    "ease_out_quart": {"css": "cubic-bezier(0.25, 1, 0.5, 1)", "gsap": "power3.out", "remotion": "Easing.out(Easing.poly(4))"},
    "ease_in_out_quart": {"css": "cubic-bezier(0.76, 0, 0.24, 1)", "gsap": "power3.inOut", "remotion": "Easing.inOut(Easing.poly(4))"},
    "ease_in_quint": {"css": "cubic-bezier(0.64, 0, 0.78, 0)", "gsap": "power4.in", "remotion": "Easing.in(Easing.poly(5))"},
    "ease_out_quint": {"css": "cubic-bezier(0.22, 1, 0.36, 1)", "gsap": "power4.out", "remotion": "Easing.out(Easing.poly(5))"},
    "ease_in_out_quint": {"css": "cubic-bezier(0.83, 0, 0.17, 1)", "gsap": "power4.inOut", "remotion": "Easing.inOut(Easing.poly(5))"},
    "ease_in_expo": {"css": "cubic-bezier(0.7, 0, 0.84, 0)", "gsap": "expo.in", "remotion": "Easing.in(Easing.exp)"},
    "ease_out_expo": {"css": "cubic-bezier(0.16, 1, 0.3, 1)", "gsap": "expo.out", "remotion": "Easing.out(Easing.exp)"},
    "ease_in_out_expo": {"css": "cubic-bezier(0.87, 0, 0.13, 1)", "gsap": "expo.inOut", "remotion": "Easing.inOut(Easing.exp)"},
    "ease_out_circ": {"css": "cubic-bezier(0, 0.55, 0.45, 1)", "gsap": "circ.out", "remotion": "Easing.out(Easing.circle)"},
    "ease_in_circ": {"css": "cubic-bezier(0.55, 0, 1, 0.45)", "gsap": "circ.in", "remotion": "Easing.in(Easing.circle)"},
    "ease_out_back": {"css": "cubic-bezier(0.34, 1.56, 0.64, 1)", "gsap": "back.out(s)", "remotion": "Easing.out(Easing.back(s))"},
    "ease_in_back": {"css": "cubic-bezier(0.36, 0, 0.66, -0.56)", "gsap": "back.in(s)", "remotion": "Easing.in(Easing.back(s))"},
    "ease_out_elastic": {"css": "(no cubic-bezier equivalent; use keyframes)", "gsap": "elastic.out(1, period)", "remotion": "Easing.out(Easing.elastic(b))"},
    "ease_out_bounce": {"css": "(no cubic-bezier equivalent; use keyframes)", "gsap": "bounce.out", "remotion": "Easing.out(Easing.bounce)"},
    "spring": {"css": "(no cubic-bezier equivalent; use linear() or keyframes)", "gsap": "CustomEase / physics", "remotion": "spring({damping, stiffness, mass})"},
}

# Named curves with a free shape parameter: (function, initial, lower, upper)
PARAMETRIC: dict[str, tuple[Callable[..., np.ndarray], list[float], list[float], list[float]]] = {
    "ease_out_back": (back_out, [1.70158], [0.05], [6.0]),
    "ease_in_back": (back_in, [1.70158], [0.05], [6.0]),
    "ease_out_elastic": (elastic_out, [0.3], [0.08], [1.0]),
    "spring": (spring, [0.5, 10.0], [0.05, 2.0], [0.99, 60.0]),
}


# ---------------------------------------------------------------- results
@dataclass
class FitResult:
    name: str
    rmse: float
    t0: float
    t1: float
    params: dict[str, float]
    bic: float

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _progress(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.clip((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)


def _bic(resid: np.ndarray, k: int) -> float:
    n = max(len(resid), 1)
    rss = float(np.sum(resid ** 2)) + 1e-12
    return n * math.log(rss / n) + k * math.log(n)


def _fit_named(name: str, f: Easing, t: np.ndarray, p: np.ndarray, t0: float, t1: float, dt: float) -> FitResult:
    """Fit a fixed-shape easing allowing t0/t1 to shift by up to ~2 frames."""

    def resid(x):
        return f(_progress(t, x[0], x[1])) - p

    span = max(2 * dt, 0.15 * (t1 - t0))
    lo = [t0 - span, t1 - span]
    hi = [t0 + span, t1 + max(span, 0.6 * (t1 - t0))]
    x0 = [t0, t1]
    try:
        r = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale=[dt, dt], max_nfev=200)
        x = r.x
    except ValueError:
        x = np.array(x0)
    res = resid(x)
    return FitResult(name, float(np.sqrt(np.mean(res ** 2))), float(x[0]), float(x[1]), {}, _bic(res, 2))


def _fit_parametric(name: str, t: np.ndarray, p: np.ndarray, t0: float, t1: float, dt: float) -> FitResult:
    f, init, plo, phi = PARAMETRIC[name]
    span = max(2 * dt, 0.15 * (t1 - t0))

    def resid(x):
        tt = _progress(t, x[0], x[1])
        if name == "spring":
            # spring is defined on [0, inf); scale so the curve settles by t1
            return f(tt * 1.0, x[2], x[3]) - p
        return f(tt, *x[2:]) - p

    x0 = [t0, t1, *init]
    lo = [t0 - span, t1 - span, *plo]
    hi = [t0 + span, t1 + max(span, 0.6 * (t1 - t0)), *phi]
    try:
        r = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=400)
        x = r.x
    except ValueError:
        x = np.array(x0)
    res = resid(x)
    keys = {"ease_out_back": ["overshoot_s"], "ease_in_back": ["overshoot_s"],
            "ease_out_elastic": ["period"], "spring": ["damping_ratio", "omega"]}[name]
    params = {k: float(v) for k, v in zip(keys, x[2:])}
    return FitResult(name, float(np.sqrt(np.mean(res ** 2))), float(x[0]), float(x[1]), params,
                     _bic(res, 2 + len(keys)))


def fit_bezier(t: np.ndarray, p: np.ndarray, t0: float, t1: float) -> tuple[list[float], float]:
    """Best CSS cubic-bezier control points for the normalised curve."""
    tt = _progress(t, t0, t1)

    def resid(x):
        return cubic_bezier(*x)(tt) - p

    best = None
    for init in ([0.25, 0.1, 0.25, 1.0], [0.33, 1.0, 0.68, 1.0], [0.42, 0.0, 0.58, 1.0], [0.34, 1.56, 0.64, 1.0]):
        try:
            r = least_squares(resid, init, bounds=([0, -1.0, 0, -1.0], [1, 2.5, 1, 2.5]), max_nfev=300)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return [0.0, 0.0, 1.0, 1.0], float("nan")
    return [float(round(v, 3)) for v in best.x], float(np.sqrt(np.mean(resid(best.x) ** 2)))


# ---------------------------------------------------------------- analysis
def detect_window(t: np.ndarray, v: np.ndarray, settle_tol: float | None = None) -> tuple[float, float, float, float] | None:
    """Find the animated window of a 1-D property.

    Returns ``(t_start, t_end, v_start, v_end)`` or ``None`` if the property
    does not change meaningfully. The end is the moment the value settles
    (stays within ``settle_tol`` of its final value), so overshoot and
    bounce are inside the window. The tolerance adapts to the measurement
    noise of the settled tail.
    """
    if len(t) < 3:
        return None
    v0 = float(np.median(v[:2]))
    v1 = float(np.median(v[-3:]))
    amp = v1 - v0
    if abs(amp) < 1e-9:
        return None
    p = (v - v0) / amp
    if settle_tol is None:
        tail = p[-max(3, len(p) // 6):]
        head = p[:max(3, len(p) // 6)]
        noise = max(float(np.std(np.diff(tail))), float(np.std(np.diff(head)))) / np.sqrt(2)
        settle_tol = float(np.clip(3 * noise, 0.008, 0.06))
    moving = np.where(np.abs(p) > settle_tol)[0]
    if not len(moving):
        return None
    i0 = max(0, moving[0] - 1)
    unsettled = np.where(np.abs(p - 1) > settle_tol)[0]
    i1 = min(len(t) - 1, (unsettled[-1] + 1) if len(unsettled) else len(t) - 1)
    if i1 <= i0:
        return None
    return float(t[i0]), float(t[i1]), v0, v1


def _count_crossings(x: np.ndarray, deadband: float) -> int:
    """Sign changes of ``x`` ignoring excursions inside +-deadband."""
    state, n = 0, 0
    for val in x:
        s = 1 if val > deadband else (-1 if val < -deadband else 0)
        if s and state and s != state:
            n += 1
        if s:
            state = s
    return n


_CHARACTER = [
    ("spring", "spring (damped oscillation)"),
    ("elastic", "elastic (oscillating settle)"),
    ("bounce", "bounce"),
    ("back", "overshoot then settle (back)"),
    ("in_out", "ease-in-out (symmetric)"),
    ("_out", "ease-out (fast start, slow settle)"),
    ("_in", "ease-in (slow start, fast finish)"),
    ("linear", "linear (constant velocity)"),
]


def analyse_curve(t: np.ndarray, v: np.ndarray, *, kind: str = "position") -> dict | None:
    """Measure kinematics and fit easings to one animated property."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 4:
        return None
    win = detect_window(t, v)
    if win is None:
        return None
    ts, te, v0, v1 = win
    amp = v1 - v0
    p = (v - v0) / amp
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    sel = (t >= ts - 3 * dt) & (t <= te + 0.6 * (te - ts) + 3 * dt)
    tw, pw = t[sel], p[sel]
    if len(tw) < 4:
        return None

    fits = [_fit_named(n, f, tw, pw, ts, te, dt) for n, f in NAMED.items()]
    fits += [_fit_parametric(n, tw, pw, ts, te, dt) for n in PARAMETRIC]
    fits.sort(key=lambda r: r.bic)
    best = fits[0]
    bezier, bez_rmse = fit_bezier(tw, pw, best.t0, best.t1)

    # Kinematics in the property's own units per second.
    vel = np.gradient(v, t) if len(t) > 2 else np.zeros_like(v)
    acc = np.gradient(vel, t) if len(t) > 2 else np.zeros_like(v)
    inw = (t >= ts) & (t <= te)
    peak_i = int(np.argmax(np.abs(vel) * inw)) if inw.any() else 0
    tail = p[t > te] if (t > te).sum() >= 3 else p[-3:]
    noise = float(np.std(np.diff(tail)) / np.sqrt(2)) if len(tail) > 2 else 0.0
    floor = max(0.01, 2.5 * noise)
    overshoot = float(max(0.0, np.max(p[t >= ts]) - 1.0))
    undershoot = float(max(0.0, -np.min(p[t >= ts])))
    overshoot = overshoot if overshoot > floor else 0.0
    undershoot = undershoot if undershoot > floor else 0.0
    span = te - ts
    reached = np.where(p >= 0.97)[0]
    after = p[reached[0]:] if len(reached) else np.array([])
    crossings = _count_crossings(after - 1.0, max(0.015, 2.5 * noise)) if len(after) > 2 else 0
    peak_frac = (t[peak_i] - ts) / span if span > 0 else 0.5
    character = next((c for key, c in _CHARACTER if key in best.name), "custom")

    alts = [{"name": r.name, "rmse": r.rmse, **({"params": r.params} if r.params else {})} for r in fits[1:4]]
    family = best.name if best.name in EQUIVALENTS else "spring"
    return {
        "property": kind,
        "measured": {
            "window_start_s": ts, "window_end_s": te, "window_ms": span * 1000.0,
            "from": v0, "to": v1, "delta": amp,
            "peak_value": float(np.max(v[t >= ts])) if amp > 0 else float(np.min(v[t >= ts])),
            # Overshoot relative to the final value (e.g. scale 1.10 -> 0.10);
            # overshoot_ratio is relative to the travelled distance.
            "overshoot_of_final": (float((np.max(v[t >= ts]) - v1) / abs(v1)) if amp > 0 and v1 != 0 and overshoot > 0
                                   else 0.0),
            "peak_velocity": float(vel[peak_i]), "peak_velocity_at_s": float(t[peak_i]),
            "peak_velocity_time_fraction": float(peak_frac),
            "max_abs_acceleration": float(np.max(np.abs(acc[inw]))) if inw.any() else 0.0,
            "overshoot_ratio": overshoot, "undershoot_ratio": undershoot,
            "settle_oscillations": crossings,
            "samples": int(inw.sum()), "noise_level": noise,
        },
        "inferred": {
            "easing": best.name, "fit_rmse": best.rmse, "params": best.params,
            "start_s": best.t0, "end_s": best.t1, "duration_ms": best.duration * 1000.0,
            "character": character,
            "cubic_bezier": bezier, "cubic_bezier_rmse": bez_rmse,
            "equivalents": EQUIVALENTS.get(family, {}),
            "alternatives": alts,
            "confidence": float(np.clip(1.0 - best.rmse * 6, 0.05, 0.98)) * min(1.0, inw.sum() / 6),
        },
    }
