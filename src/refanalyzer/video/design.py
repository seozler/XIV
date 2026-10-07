"""Stage: design - colour, palette and composition measurements.

Per scene (on stored keyframes) the stage measures dominant colours
(k-means in Lab), brightness, contrast, saturation, colourfulness, whether
the background is a gradient (plane fit), negative space, visual balance
(ink centre of mass), safe margins, and alignment between the elements and
text measured by earlier stages. Numbers here are what the vision model
later interprets; the model is never asked to guess them.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from ..models import INFERRED, OBSERVED, prov
from ..pipeline import Context


def palette(img: np.ndarray, k: int = 6) -> list[dict]:
    small = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = min(k, len(np.unique(lab, axis=0)))
    if k < 1:
        return []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.reshape(-1), minlength=k) / len(labels)
    out = []
    for c, share in sorted(zip(centers, counts), key=lambda x: -x[1]):
        bgr = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_LAB2BGR)[0, 0]
        r, g, b = [int(v) for v in bgr[::-1]]
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        out.append({"hex": "#%02x%02x%02x" % (r, g, b), "share": float(share),
                    "hue_deg": float(h * 360), "lightness": float(l), "saturation": float(s)})
    return out


def gradient_fit(img: np.ndarray) -> dict:
    """Fit a linear plane to luma; strong fit + range = gradient background."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:36, 0:64]
    A = np.c_[xx.ravel() / 63, yy.ravel() / 35, np.ones(xx.size)]
    coef, *_ = np.linalg.lstsq(A, g.ravel(), rcond=None)
    pred = A @ coef
    ss_tot = np.sum((g.ravel() - g.mean()) ** 2) + 1e-6
    r2 = 1 - np.sum((g.ravel() - pred) ** 2) / ss_tot
    span = float(np.hypot(coef[0], coef[1]))
    radial = None
    # Radial (vignette-like) gradient: luma vs distance from centre.
    d = np.hypot(xx / 63 - 0.5, yy / 35 - 0.5).ravel()
    rc = np.corrcoef(d, g.ravel())[0, 1] if g.std() > 1 else 0.0
    if abs(rc) > 0.8 and g.std() > 6:
        radial = "darker edges (vignette/radial)" if rc < 0 else "brighter edges"
    return {"linear_r2": float(r2), "linear_span_levels": span,
            "angle_deg": float(np.degrees(np.arctan2(coef[1], coef[0]))),
            "is_linear_gradient": bool(r2 > 0.85 and span > 12), "radial": radial}


def composition(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    edges = cv2.Canny(gray, 60, 140)
    bg = np.median(img.reshape(-1, 3), axis=0)
    ink = (np.abs(img.astype(np.int16) - bg).max(axis=2) > 30)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    out = {"edge_density": float((edges > 0).mean()), "negative_space": float(1 - ink.mean())}
    if ink.any():
        ys, xs = np.nonzero(ink)
        cx, cy = xs.mean() / W, ys.mean() / H
        out.update({
            "ink_center": [float(cx), float(cy)],
            "balance_offset": float(np.hypot(cx - 0.5, cy - 0.5)),
            "content_bbox_norm": [float(xs.min() / W), float(ys.min() / H), float((xs.max() - xs.min()) / W),
                                  float((ys.max() - ys.min()) / H)],
            "margins_norm": {"left": float(xs.min() / W), "right": float(1 - xs.max() / W),
                             "top": float(ys.min() / H), "bottom": float(1 - ys.max() / H)},
            "left_right_ink_ratio": float(ink[:, :W // 2].sum() / max(ink[:, W // 2:].sum(), 1)),
        })
    return out


def thirds_alignment(points: list[tuple[float, float]]) -> float | None:
    """Share of element centres within 5% of a rule-of-thirds line."""
    if not points:
        return None
    hits = 0
    for x, y in points:
        if min(abs(x - 1 / 3), abs(x - 2 / 3), abs(x - 0.5)) < 0.05 or min(abs(y - 1 / 3), abs(y - 2 / 3), abs(y - 0.5)) < 0.05:
            hits += 1
    return hits / len(points)


def shared_edges(boxes: list[list[float]], tol: float = 0.01) -> dict:
    """How often boxes share left edges / centres / tops (grid discipline)."""
    if len(boxes) < 2:
        return {}
    keys = {"left": [b[0] for b in boxes], "center_x": [b[0] + b[2] / 2 for b in boxes],
            "top": [b[1] for b in boxes], "center_y": [b[1] + b[3] / 2 for b in boxes]}
    out = {}
    for k, vals in keys.items():
        vals = sorted(vals)
        pairs = sum(1 for a, b in zip(vals, vals[1:]) if b - a < tol)
        out[k] = pairs / (len(vals) - 1)
    return out


def run(ctx: Context) -> dict:
    kfs = ctx.ws.read("visual/keyframes.json")["keyframes"]
    scenes = ctx.ws.read("metadata/scenes.json")["scenes"]
    objects = (ctx.ws.read("visual/objects.json") or {}).get("elements", [])
    fm = ctx.ws.read("visual/frame_metrics.json")["columns"]
    t = np.asarray(fm["t"])
    out_scenes = []
    all_pal: dict[str, float] = {}
    for sc in scenes:
        in_sc = [k for k in kfs if sc["start"] <= k["t"] < sc["end"]]
        if not in_sc:
            continue
        # Prefer settled frames: they show the finished layout.
        settled = [k for k in in_sc if k["reason"] == "settled"] or in_sc
        rep = settled[-1]
        img = cv2.imread(str(ctx.ws.path(rep["path"])))
        if img is None:
            continue
        pal = palette(img)
        for p in pal:
            all_pal[p["hex"]] = all_pal.get(p["hex"], 0) + p["share"] * sc["duration_s"]
        sel = (t >= sc["start"]) & (t < sc["end"])
        els = [e for e in objects if e["scene"] == sc["id"]]
        centers = [(e["rest_bbox_norm"][0] + e["rest_bbox_norm"][2] / 2, e["rest_bbox_norm"][1] + e["rest_bbox_norm"][3] / 2) for e in els]
        out_scenes.append({
            "scene": sc["id"], "start": sc["start"], "end": sc["end"], "keyframe": rep["path"], "keyframe_t": rep["t"],
            "palette": pal,
            "background_color_hex": pal[0]["hex"] if pal else None,
            "brightness_mean": float(np.mean(np.asarray(fm["luma"])[sel])) if sel.any() else None,
            "contrast_mean": float(np.mean(np.asarray(fm["contrast"])[sel])) if sel.any() else None,
            "saturation_mean": float(np.mean(np.asarray(fm["saturation"])[sel])) if sel.any() else None,
            "colourfulness_mean": float(np.mean(np.asarray(fm["colourfulness"])[sel])) if sel.any() else None,
            "gradient": gradient_fit(img),
            "composition": composition(img),
            "element_thirds_alignment": thirds_alignment(centers),
            "element_shared_edges": shared_edges([e["rest_bbox_norm"] for e in els]),
        })
    total = sum(all_pal.values()) or 1.0
    global_pal = [{"hex": k, "weighted_share": v / total} for k, v in sorted(all_pal.items(), key=lambda kv: -kv[1])[:12]]
    lum = [s["brightness_mean"] for s in out_scenes if s["brightness_mean"] is not None]
    summary = {
        "global_palette": global_pal,
        "theme": None if not lum else "dark" if np.mean(lum) < 0.35 else "light" if np.mean(lum) > 0.65 else "mid",
        "mean_negative_space": float(np.mean([s["composition"]["negative_space"] for s in out_scenes])) if out_scenes else None,
        "gradient_background_share": float(np.mean([s["gradient"]["is_linear_gradient"] for s in out_scenes])) if out_scenes else None,
        "mean_saturation": float(np.mean(fm["saturation"])) if fm["saturation"] else None,
    }
    ctx.ws.write("visual/colors.json", {
        "summary": summary, "scenes": out_scenes,
        "timeline": {"t": fm["t"], "luma": fm["luma"], "saturation": fm["saturation"],
                     "mean_rgb": [[r, g, b] for r, g, b in zip(fm["mean_r"], fm["mean_g"], fm["mean_b"])]},
        "provenance": {
            "palette": prov("opencv", "k-means (k=6) in CIELAB on settled keyframe", 0.85, OBSERVED),
            "gradient": prov("numpy", "plane fit to luma (R2) + radial correlation", 0.7, OBSERVED),
            "composition": prov("opencv", "ink mask vs median background", 0.65, INFERRED),
        },
    }, ndigits=4, indent=None)
    return {"outputs": ["visual/colors.json"]}
