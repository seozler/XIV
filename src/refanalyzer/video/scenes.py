"""Stage: scenes - cuts, fades and dissolves.

Built-in detector (CPU, uses the visual_scan metrics):
  * cut: HSV-histogram distance spike that is large in absolute terms and
    relative to its local neighbourhood (adaptive threshold)
  * fade: monotonic luma ramp into / out of a black or white hold
  * dissolve: frame ``i`` is well explained as a linear blend of frames
    ``i-k`` and ``i+k`` while no single step is a cut
Optional cross-check with PySceneDetect's ContentDetector.
"""

from __future__ import annotations

import numpy as np

from ..models import INFERRED, OBSERVED, prov
from ..pipeline import Context


def _rolling_median(x: np.ndarray, half: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(len(x)):
        a, b = max(0, i - half), min(len(x), i + half + 1)
        nb = np.concatenate([x[a:i], x[i + 1:b]])
        out[i] = np.median(nb) if len(nb) else 0.0
    return out


def detect_cuts(t: np.ndarray, hist: np.ndarray, pix: np.ndarray, thr: float, min_gap: float) -> list[dict]:
    if len(t) < 3:
        return []
    base = _rolling_median(hist, 8)
    pbase = _rolling_median(pix, 8)
    cuts: list[dict] = []
    for i in range(1, len(t)):
        ratio = hist[i] / (base[i] + 0.02)
        # The pixel change must also spike: during a dissolve or a colour
        # fade every step changes a little, so no single step stands out.
        pratio = pix[i] / (pbase[i] + 0.005)
        if hist[i] >= thr and ratio >= 3.0 and pix[i] >= 0.02 and pratio >= 3.0:
            if cuts and t[i] - cuts[-1]["t"] < min_gap:
                if hist[i] > cuts[-1]["score"]:
                    cuts[-1] = {"t": float(t[i]), "index": i, "score": float(hist[i]), "ratio": float(ratio)}
                continue
            cuts.append({"t": float(t[i]), "index": i, "score": float(hist[i]), "ratio": float(ratio)})
    return cuts


def detect_fades(t: np.ndarray, luma: np.ndarray, min_dur: float = 0.15) -> list[dict]:
    """Fades through black/white: ramps adjacent to near-uniform holds."""
    out: list[dict] = []
    for kind, hold in (("black", luma < 0.04), ("white", luma > 0.96)):
        i = 0
        n = len(luma)
        while i < n:
            if not hold[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and hold[j + 1]:
                j += 1
            # fade-out: ramp before the hold
            a = i
            sign = 1 if kind == "black" else -1
            while a - 1 >= 0 and sign * (luma[a - 1] - luma[a]) > 0.003:
                a -= 1
            if a < i and t[i] - t[a] >= min_dur and abs(luma[a] - luma[i]) > 0.1:
                out.append({"type": f"fade_out_to_{kind}", "start": float(t[a]), "end": float(t[i]),
                            "duration_ms": float((t[i] - t[a]) * 1000)})
            b = j
            while b + 1 < n and sign * (luma[b + 1] - luma[b]) > 0.003:
                b += 1
            if b > j and t[b] - t[j] >= min_dur and abs(luma[b] - luma[j]) > 0.1:
                out.append({"type": f"fade_in_from_{kind}", "start": float(t[j]), "end": float(t[b]),
                            "duration_ms": float((t[b] - t[j]) * 1000)})
            if t[j] - t[i] >= 0.05:
                out.append({"type": f"{kind}_hold", "start": float(t[i]), "end": float(t[j]),
                            "duration_ms": float((t[j] - t[i]) * 1000)})
            i = j + 1
    return sorted(out, key=lambda f: f["start"])


def detect_dissolves(t: np.ndarray, color: np.ndarray, cut_idx: set[int], k: int) -> list[dict]:
    """Frame i explained as a blend of frames i-k and i+k, change spread evenly."""
    if len(color) < 2 * k + 1:
        return []
    g = color.reshape(len(color), -1).astype(np.float32)
    cand = np.zeros(len(g), bool)
    for i in range(k, len(g) - k):
        a, b, c = g[i - k], g[i + k], g[i]
        span = np.abs(a - b)
        total = span.mean()
        if total < 6:  # endpoints too similar to be a transition
            continue
        if any(j in cut_idx for j in range(i - k, i + k + 1)):
            continue
        # A cut concentrates the change in one step; a dissolve spreads it.
        steps = [np.abs(g[j + 1] - g[j]).mean() for j in range(i - k, i + k)]
        if max(steps) > 0.5 * total:
            continue
        d = a - b
        w = float(np.dot(c - b, d) / (np.dot(d, d) + 1e-6))
        if not 0.1 < w < 0.9:
            continue
        pred = b + w * d
        r2 = 1 - np.sum((c - pred) ** 2) / (np.sum((c - c.mean()) ** 2) + 1e-6)
        # A blend changes most of the frame; a local animation does not.
        changed = np.mean(span.reshape(-1, 3).max(axis=1) > 12)
        if r2 > 0.97 and changed > 0.35:
            cand[i] = True
    out = []
    i = 0
    while i < len(cand):
        if cand[i]:
            j = i
            while j + 1 < len(cand) and cand[j + 1]:
                j += 1
            if j - i + 1 >= 3:
                s, e = max(0, i - k), min(len(t) - 1, j + k)
                # Refine the bounds with the blend weight between the two
                # settled endpoints: start = last frame still ~A, end = first ~B.
                a, b = g[s], g[e]
                d = a - b
                wts = np.array([float(np.dot(g[m] - b, d) / (np.dot(d, d) + 1e-6)) for m in range(s, e + 1)])
                in_a = np.where(wts >= 0.97)[0]
                in_b = np.where(wts <= 0.03)[0]
                s2 = s + (int(in_a[-1]) if len(in_a) else 0)
                e2 = s + (int(in_b[0]) if len(in_b) else e - s)
                if e2 <= s2:
                    s2, e2 = s, e
                out.append({"type": "dissolve", "start": float(t[s2]), "end": float(t[e2]),
                            "duration_ms": float((t[e2] - t[s2]) * 1000), "mid": float(t[(i + j) // 2])})
            i = j + 1
        else:
            i += 1
    return out


def _pyscenedetect_cuts(path: str) -> list[float] | None:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        return None
    scenes = detect(path, ContentDetector())
    return [float(s[0].get_seconds()) for s in scenes[1:]]


def run(ctx: Context) -> dict:
    fm = ctx.ws.read("visual/frame_metrics.json")["columns"]
    vol = np.load(ctx.ws.path("cache/motion_volume.npz"))
    meta = ctx.ws.read("metadata/video.json")
    t = np.asarray(fm["t"], float)
    hist = np.asarray(fm["hist_dist"], float)
    pix = np.asarray(fm["pixel_diff"], float)
    luma = np.asarray(fm["luma"], float)
    thr = float(ctx.config.get("analysis.cut_threshold"))
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    notes = []

    cuts = detect_cuts(t, hist, pix, thr, min_gap=max(0.25, 4 * dt))
    method = "builtin HSV-histogram adaptive threshold"
    if ctx.config.get("analysis.scene_detector") == "pyscenedetect":
        ps = _pyscenedetect_cuts(str(ctx.ws.video_path))
        if ps is None:
            notes.append("PySceneDetect not installed; builtin detector used")
        else:
            method = "PySceneDetect ContentDetector"
            cuts = []
            for c in ps:
                i = int(np.argmin(np.abs(t - c)))
                cuts.append({"t": float(t[i]), "index": i, "score": float(hist[i]), "ratio": None})
    fades = detect_fades(t, luma)
    # A fade's last step into the black/white hold looks like a cut; drop it.
    ramps = [f for f in fades if f["type"].startswith("fade")]
    cuts = [c for c in cuts if not any(f["start"] - dt <= c["t"] <= f["end"] + 2 * dt for f in ramps)
            and not (c["index"] > 0 and max(luma[c["index"]], luma[c["index"] - 1]) < 0.06)]
    k = max(2, int(round(0.15 / dt)))
    color = vol["color"] if "color" in vol.files else np.repeat(vol["gray"][..., None], 3, axis=3)
    dissolves = detect_dissolves(t, color, {c["index"] for c in cuts}, k)
    # Fading to/from black or white is a fade, not a dissolve.
    dissolves = [d for d in dissolves if not any(d["start"] < f["end"] and f["start"] < d["end"] for f in fades)]

    duration = float(meta["duration_s"])
    bounds = [0.0] + [c["t"] for c in cuts] + [duration]
    scenes = []
    for n, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        sel = (t >= a) & (t < b)
        scenes.append({
            "id": f"scene_{n + 1:02d}", "start": a, "end": b, "duration_s": b - a,
            "frame_range": [int(fm["frame"][int(np.argmax(sel))]), int(fm["frame"][len(sel) - 1 - int(np.argmax(sel[::-1]))])] if sel.any() else None,
            "in_transition": "cut" if n > 0 else "start",
            "mean_luma": float(luma[sel].mean()) if sel.any() else None,
        })
    for tr in fades + dissolves:
        for sc in scenes:
            if sc["start"] - 0.05 <= tr["start"] < sc["end"]:
                sc.setdefault("transitions", []).append(tr["type"])
    lengths = np.array([s["duration_s"] for s in scenes])
    stats = {
        "scene_count": len(scenes), "cut_count": len(cuts),
        "cuts_per_minute": len(cuts) / (duration / 60.0) if duration > 0 else 0.0,
        "mean_scene_s": float(lengths.mean()) if len(lengths) else None,
        "median_scene_s": float(np.median(lengths)) if len(lengths) else None,
        "min_scene_s": float(lengths.min()) if len(lengths) else None,
        "max_scene_s": float(lengths.max()) if len(lengths) else None,
        "fade_count": sum(1 for f in fades if f["type"].startswith("fade")),
        "dissolve_count": len(dissolves),
    }
    ctx.ws.write("metadata/scenes.json", {
        "scenes": scenes, "cuts": cuts, "fades": fades, "dissolves": dissolves, "stats": stats,
        "provenance": {
            "cuts": prov("opencv" if "builtin" in method else "pyscenedetect", method, 0.9, OBSERVED),
            "fades": prov("numpy", "luma ramps adjacent to uniform black/white holds", 0.8, OBSERVED),
            "dissolves": prov("numpy", "linear-blend fit of frame i from frames i-k and i+k (R2>0.97)", 0.65, INFERRED),
        },
    })
    return {"outputs": ["metadata/scenes.json"], "notes": notes}
