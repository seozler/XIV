"""Stage: text - OCR, text tracks and measured typography.

OCR runs on the keyframes chosen by visual_scan (fixed interval +
change-triggered + settled-after-motion), re-decoded at ``ocr_width`` so
small type stays legible. Detections are linked over time into text tracks;
appearance times are refined with the motion stage (a text track that
overlaps an animated element inherits its measured entrance animation).

Measured per track: content, first/last seen, bbox, text/background colour,
WCAG contrast, box height, stroke-width ratio, case style, alignment,
line spacing, plate / outline cues. Font family is left to the vision model
(inferred) because it cannot be measured reliably from pixels.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

import cv2
import numpy as np

from .. import media
from ..models import INFERRED, OBSERVED, prov
from ..pipeline import Context
from ..providers.ocr import make_engine


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^\w]+", "", s)


def _box_xywh(box) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


def _luminance(rgb: np.ndarray) -> float:
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast_ratio(a_rgb, b_rgb) -> float:
    la, lb = _luminance(a_rgb), _luminance(b_rgb)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _hex(bgr) -> str:
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def measure_typography(img: np.ndarray, xywh, text: str) -> dict:
    """Pixel measurements of one text line in a full-resolution frame."""
    H, W = img.shape[:2]
    x, y, w, h = [int(round(v)) for v in xywh]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    out: dict = {"box_height_px": float(h), "box_width_px": float(w)}
    if w < 4 or h < 4:
        return out
    crop = img[y:y + h, x:x + w]
    px = crop.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(px, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg_k = int(np.bincount(border, minlength=2).argmax())
    fg_k = 1 - bg_k
    text_bgr, bg_bgr = centers[fg_k], centers[bg_k]
    mask = (labels == fg_k).astype(np.uint8)
    out["text_color_hex"] = _hex(text_bgr)
    out["background_color_hex"] = _hex(bg_bgr)
    out["contrast_ratio"] = contrast_ratio(text_bgr[::-1], bg_bgr[::-1])
    out["ink_coverage"] = float(mask.mean())

    # Stroke width ~ 2 * area / perimeter of the ink.
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    stroke = 2.0 * mask.sum() / perim if perim > 0 else None
    rows = np.where(mask.sum(axis=1) > 0)[0]
    ink_h = float(rows[-1] - rows[0] + 1) if len(rows) else float(h)
    out["ink_height_px"] = ink_h
    if stroke:
        ratio = stroke / max(ink_h, 1)
        out["stroke_width_px"] = float(stroke)
        out["stroke_to_height_ratio"] = float(ratio)
        out["weight_estimate"] = ("light" if ratio < 0.07 else "regular" if ratio < 0.10 else
                                  "medium/semibold" if ratio < 0.13 else "bold" if ratio < 0.17 else "black/heavy")
    letters = [c for c in text if c.isalpha()]
    if letters:
        up = sum(c.isupper() for c in letters) / len(letters)
        words = [wd for wd in text.split() if wd[:1].isalpha()]
        title = words and all(wd[0].isupper() for wd in words)
        out["case_style"] = "ALL CAPS" if up > 0.9 else "Title Case" if title else "sentence/lower"
        # Caps height ~ ink height for caps; x-height lines need ~1.45x.
        out["font_size_px_estimate"] = ink_h * (1.0 / 0.70 if up > 0.9 else 1.0 / 0.95)
        n = len(text.replace(" ", ""))
        if n:
            out["advance_per_char_em"] = float(w / n / out["font_size_px_estimate"])

    # Plate behind the text and outline/shadow cues.
    pad = max(4, h // 3)
    ring = img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)].copy()
    rmask = np.ones(ring.shape[:2], bool)
    rmask[min(pad, y):min(pad, y) + h, min(pad, x):min(pad, x) + w] = False
    if rmask.any():
        ring_med = np.median(ring[rmask], axis=0)
        out["on_plate"] = bool(np.abs(ring_med - bg_bgr).max() > 35)
        out["surrounding_color_hex"] = _hex(ring_med)
    halo = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(1, int(round((stroke or 2) / 2)))) - mask
    if halo.sum() > 10:
        halo_l = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[halo.astype(bool)].mean())
        bg_l = float(np.dot(bg_bgr, [0.114, 0.587, 0.299]))
        txt_l = float(np.dot(text_bgr, [0.114, 0.587, 0.299]))
        out["outline_or_shadow_likely"] = bool(txt_l > bg_l and halo_l < bg_l - 25)
    return out


def _alignment(boxes: list[tuple[float, float, float, float]], W: float) -> dict:
    """Alignment of text lines that share a frame."""
    if len(boxes) < 2:
        b = boxes[0]
        cx = (b[0] + b[2] / 2) / W
        return {"frame_position": "center" if abs(cx - 0.5) < 0.04 else "left" if cx < 0.5 else "right"}
    tol = 0.012 * W
    lefts = np.array([b[0] for b in boxes])
    rights = np.array([b[0] + b[2] for b in boxes])
    centers = np.array([b[0] + b[2] / 2 for b in boxes])
    spread = {"left": np.ptp(lefts), "center": np.ptp(centers), "right": np.ptp(rights)}
    best = min(spread, key=spread.get)
    out = {"block_alignment": best if spread[best] < tol else "mixed"}
    ys = sorted(boxes, key=lambda b: b[1])
    gaps = [(b2[1] - b1[1]) / max(b1[3], 1) for b1, b2 in zip(ys, ys[1:]) if abs(b2[3] - b1[3]) < 0.35 * b1[3]]
    if gaps:
        out["line_pitch_to_box_height"] = float(np.median(gaps))
    return out


def run(ctx: Context) -> dict:
    cfg = ctx.config
    engine = make_engine(cfg.get("analysis.ocr_engine"), cfg.get("analysis.ocr_languages"))
    meta = ctx.ws.read("metadata/video.json")
    kfs = ctx.ws.read("visual/keyframes.json")["keyframes"]
    if engine is None:
        ctx.ws.write("visual/ocr.json", {"engine": None, "detections": [], "tracks": [],
                                         "note": "no OCR engine installed (pip install rapidocr_onnxruntime)"})
        return {"outputs": ["visual/ocr.json"], "notes": ["OCR skipped: no engine available"]}

    ocr_w = min(int(cfg.get("analysis.ocr_width", 1280) or 1280), meta["width"])
    frames = media.extract_frames(ctx.ws.video_path, [k["frame"] for k in kfs], ocr_w, ctx.ws.path("cache/ocr_frames"))
    sw = meta["width"] / ocr_w  # scale to source pixels
    W, H = float(meta["width"]), float(meta["height"])

    detections = []
    images: dict[int, np.ndarray] = {}
    for kf in kfs:
        p = frames.get(kf["frame"])
        if p is None:
            continue
        img = cv2.imread(str(p))
        for d in engine.detect(img):
            if d["score"] < 0.5 or not norm_text(d["text"]):
                continue
            x, y, w, h = _box_xywh(d["box"])
            detections.append({"t": kf["t"], "frame": kf["frame"], "text": d["text"], "score": d["score"],
                               "bbox_px": [x * sw, y * sw, w * sw, h * sw],
                               "bbox_norm": [x * sw / W, y * sw / H, w * sw / W, h * sw / H]})
        images[kf["frame"]] = img
    ctx.log("%d OCR detections on %d keyframes", len(detections), len(kfs))

    # ---- link detections into tracks
    kf_times = [k["t"] for k in kfs]
    tracks: list[dict] = []
    active: list[dict] = []
    by_frame: dict[int, list[dict]] = {}
    for d in detections:
        by_frame.setdefault(d["frame"], []).append(d)
    for kf in kfs:
        dets = by_frame.get(kf["frame"], [])
        matched = set()
        for d in dets:
            best, best_s = None, 0.0
            for tr in active:
                last = tr["observations"][-1]
                iou = _iou(d["bbox_norm"], last["bbox_norm"])
                a, b = norm_text(d["text"]), norm_text(last["text"])
                sim = difflib.SequenceMatcher(None, a, b).ratio()
                grows = bool(a and b and (a.startswith(b) or b.startswith(a)))
                s = iou + sim + (0.5 if grows else 0)
                if (iou > 0.3 and (sim > 0.6 or grows)) or (sim > 0.9 and iou > 0.05):
                    if s > best_s:
                        best, best_s = tr, s
            if best is None or id(best) in matched:
                best = {"observations": []}
                tracks.append(best)
                active.append(best)
            best["observations"].append(d)
            matched.add(id(best))
        for tr in list(active):
            if id(tr) not in matched:
                active.remove(tr)
                tr["next_absent_t"] = kf["t"]

    elements = (ctx.ws.read("visual/objects.json") or {}).get("elements", [])
    out_tracks = []
    for n, tr in enumerate(tracks, start=1):
        obs = tr["observations"]
        texts = [o["text"] for o in obs]
        final = max(obs, key=lambda o: (len(norm_text(o["text"])), o["score"]))
        first, last = obs[0], obs[-1]
        i_first = kf_times.index(first["t"])
        prev_t = kf_times[i_first - 1] if i_first > 0 else None
        reveal = []
        for o in obs:
            if not reveal or norm_text(o["text"]) != norm_text(reveal[-1]["text"]):
                reveal.append({"t": o["t"], "text": o["text"]})
        progressive = len(reveal) > 1 and all(
            norm_text(b["text"]).startswith(norm_text(a["text"])) for a, b in zip(reveal, reveal[1:]))
        # Inherit a measured animation from an overlapping element.
        link = None
        for el in elements:
            if el["role"] not in ("entrance", "move", "transient"):
                continue
            if _iou(el["rest_bbox_norm"], final["bbox_norm"]) > 0.25 or _contains(el["rest_bbox_norm"], final["bbox_norm"]):
                fv = el.get("first_visible_s")
                if fv is not None and (prev_t is None or fv >= prev_t - 0.5) and fv <= first["t"] + 0.05:
                    link = el
                    break
        img = images.get(final["frame"])
        typo = measure_typography(img, [v / sw for v in final["bbox_px"]], final["text"]) if img is not None else {}
        for k in ("box_height_px", "box_width_px", "ink_height_px", "stroke_width_px", "font_size_px_estimate"):
            if k in typo:
                typo[k] = typo[k] * sw
        crop_path = ctx.ws.path(f"visual/evidence/text_{n:03d}.png")
        if img is not None:
            x, y, w, h = [int(round(v / sw)) for v in final["bbox_px"]]
            pad = max(4, h // 2)
            cv2.imwrite(str(crop_path), img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad])
        appear = {"first_seen_s": first["t"], "appear_after_s": prev_t}
        if link is not None:
            appear["appear_s"] = link["animation"]["start_s"] if link["animation"]["start_s"] is not None else link["first_visible_s"]
            appear["source"] = "linked element animation"
        out_tracks.append({
            "id": f"tx_{n:03d}", "content": final["text"], "content_variants": sorted(set(texts)),
            "start": appear.get("appear_s", first["t"]), "end": tr.get("next_absent_t", last["t"]),
            "last_seen_s": last["t"], "appearance": appear,
            "bbox_norm": final["bbox_norm"], "bbox_px": final["bbox_px"], "ocr_score": final["score"],
            "typography": typo, "progressive_reveal": reveal if progressive else None,
            "linked_element": link["id"] if link else None,
            "animation": link["animation"] if link else None,
            "provenance": {
                "content": prov(engine.name, "OCR on keyframes", float(final["score"]), OBSERVED,
                                frame_range=[first["frame"], last["frame"]], evidence=ctx.ws.rel(crop_path)),
                "timing": prov(engine.name, "keyframe sampling (+ element animation link)",
                               0.85 if link else 0.6, OBSERVED),
                "colors_and_size": prov("opencv", "k-means ink/background split on full-res crop", 0.8, OBSERVED),
                "weight_case_font_size": prov("opencv", "stroke-width ratio, case heuristics", 0.55, INFERRED),
            },
        })

    # Frame-level layout (alignment, line spacing) on the busiest keyframes.
    layout = []
    for frame, dets in sorted(by_frame.items(), key=lambda kv: -len(kv[1]))[:20]:
        boxes = [tuple(d["bbox_px"]) for d in dets]
        layout.append({"frame": frame, "t": dets[0]["t"], "lines": len(dets), **_alignment(boxes, W)})
    sizes = [t["typography"].get("font_size_px_estimate") for t in out_tracks if t["typography"].get("font_size_px_estimate")]
    summary = {
        "text_tracks": len(out_tracks),
        "font_size_px_levels": _levels(sizes),
        "case_styles": _count(t["typography"].get("case_style") for t in out_tracks),
        "weights": _count(t["typography"].get("weight_estimate") for t in out_tracks),
        "text_colors": _count(t["typography"].get("text_color_hex") for t in out_tracks),
        "min_contrast_ratio": min((t["typography"].get("contrast_ratio", 99) for t in out_tracks), default=None),
        "on_plate_share": float(np.mean([bool(t["typography"].get("on_plate")) for t in out_tracks])) if out_tracks else None,
        "progressive_reveals": sum(1 for t in out_tracks if t["progressive_reveal"]),
        "animated_text_share": float(np.mean([t["animation"] is not None for t in out_tracks])) if out_tracks else None,
    }
    ctx.ws.write("visual/ocr.json", {"engine": engine.name, "ocr_width": ocr_w, "summary": summary,
                                     "tracks": out_tracks, "layout": layout, "detections": detections})
    return {"outputs": ["visual/ocr.json"], "notes": [f"{len(out_tracks)} text tracks ({engine.name})"]}


def _contains(outer, inner) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    cx, cy = ix + iw / 2, iy + ih / 2
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _levels(sizes: list[float]) -> list[dict]:
    """Cluster font sizes into a type scale (levels within 12% merge)."""
    levels: list[list[float]] = []
    for s in sorted(sizes, reverse=True):
        if levels and abs(s - np.mean(levels[-1])) / np.mean(levels[-1]) < 0.12:
            levels[-1].append(s)
        else:
            levels.append([s])
    return [{"size_px": float(np.mean(l)), "count": len(l)} for l in levels]
