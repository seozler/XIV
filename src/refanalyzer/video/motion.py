"""Stage: motion - element animations and camera moves.

1. Motion events: 3-D connected components of the camera-compensated
   residual-motion volume (time x grid). Global transitions (cuts, fades,
   dissolves) are masked out first.
2. Element tracking per event at analysis resolution (second decode pass):
   the frames just before and just after the event are compared; every
   region that differs is an element at rest (entrance) or the place an
   element left (exit). Each element is then followed frame by frame via
   its foreground mask against the clean plate, with
     * border completion for elements entering from off-screen,
     * scale from the bbox relative to the settled size,
     * opacity by least-squares regression of the frame against the
       template over the clean plate (works for fades).
3. Per animated property (position / scale / opacity) the curve module
   measures duration, velocity, acceleration, overshoot and fits easings.
4. Camera moves (pan/tilt/zoom/roll) from the per-frame similarity
   transforms, plus a parallax indicator from residual dense flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .. import media
from ..analysis.curves import analyse_curve
from ..models import INFERRED, OBSERVED, prov
from ..pipeline import Context

DIFF_THR = 28  # colour difference (max over channels) treated as foreground


@dataclass
class Event:
    id: str
    scene: str
    r0: int  # first active row in the volume
    r1: int
    rows_scene: tuple[int, int]
    grid_bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive) in grid cells
    voxels: int
    mean_area: float
    kind: str
    path: list[dict] = field(default_factory=list)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def find_events(active: np.ndarray, t: np.ndarray, scenes: list[dict], max_event_s: float) -> list[Event]:
    events: list[Event] = []
    gh, gw = active.shape[1:]
    st = np.ones((3, 3, 3), bool)
    for sc in scenes:
        rows = np.where((t >= sc["start"]) & (t < sc["end"]))[0]
        if len(rows) < 3:
            continue
        a, b = int(rows[0]), int(rows[-1])
        sub = active[a:b + 1]
        if not sub.any():
            continue
        # Join gaps of up to 2 frames / 1 cell so one animation stays one event.
        joined = ndimage.binary_dilation(sub, structure=st, iterations=1)
        labels, n = ndimage.label(joined, structure=st)
        for k, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            comp = (labels[sl] == k) & sub[sl]
            vox = int(comp.sum())
            if vox < 4:
                continue
            r0, r1 = a + sl[0].start, a + sl[0].stop - 1
            area = comp.reshape(comp.shape[0], -1).sum(axis=1)
            frame_area = gh * gw
            dur = t[r1] - t[r0]
            kind = "animation"
            if dur > max_event_s or area.mean() / frame_area > 0.45:
                kind = "continuous"
            path = []
            for i in range(comp.shape[0]):
                ys, xs = np.nonzero(comp[i])
                if len(xs):
                    path.append({"t": float(t[sl[0].start + a + i]),
                                 "x": float((xs.mean() + sl[2].start + 0.5) / gw),
                                 "y": float((ys.mean() + sl[1].start + 0.5) / gh),
                                 "area": float(len(xs) / frame_area)})
            events.append(Event(
                id=f"ev_{len(events) + 1:03d}", scene=sc["id"], r0=r0, r1=r1, rows_scene=(a, b),
                grid_bbox=(sl[2].start, sl[1].start, sl[2].stop - 1, sl[1].stop - 1),
                voxels=vox, mean_area=float(area.mean() / frame_area), kind=kind, path=path))
    return events


# --------------------------------------------------------------- tracking
def _fg_mask(frame: np.ndarray, plate: np.ndarray, roi: tuple[int, int, int, int], close_k: int) -> np.ndarray:
    x0, y0, x1, y1 = roi
    d = cv2.absdiff(frame[y0:y1, x0:x1], plate[y0:y1, x0:x1]).max(axis=2)
    m = (d > DIFF_THR).astype(np.uint8)
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)))
    full = np.zeros(frame.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = m
    return full


def _components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(int(v) for v in stats[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _overlaps(a, b, pad: float) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return not (ax0 + aw + pad < bx0 or bx0 + bw + pad < ax0 or ay0 + ah + pad < by0 or by0 + bh + pad < ay0)


def _opacity(frame: np.ndarray, plate: np.ndarray, template: np.ndarray, box) -> float | None:
    x, y, w, h = [int(round(v)) for v in box]
    if w < 2 or h < 2:
        return None
    H, W = frame.shape[:2]
    tpl = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if xb - xa < 2 or yb - ya < 2:
        return None
    p = frame[ya:yb, xa:xb].astype(np.float32)
    b = plate[ya:yb, xa:xb].astype(np.float32)
    tt = tpl[ya - y:yb - y, xa - x:xb - x]
    d = tt - b
    sel = np.abs(d).max(axis=2) > 20
    if sel.sum() < 12:
        return None
    return float(np.clip(np.sum((p - b)[sel] * d[sel]) / np.sum(d[sel] ** 2), 0.0, 1.5))


def track_element(frames: list[np.ndarray], times: list[float], frame_ids: list[int], ref_i: int,
                  plate: np.ndarray, ref_box, roi, close_k: int, min_area: int, order: list[int]) -> list[dict]:
    """Follow one element through ``frames`` in ``order`` starting near ``ref_i``.

    ``plate`` is the clean background the element is segmented against.
    """
    H, W = frames[0].shape[:2]
    rx, ry, rw, rh = ref_box
    template = frames[ref_i][ry:ry + rh, rx:rx + rw].copy()
    pred = ref_box
    samples = []
    for i in order:
        fg = _fg_mask(frames[i], plate, roi, close_k)
        comps = _components(fg, max(4, min_area // 6))
        pad = max(30.0, 0.5 * max(pred[2], pred[3]))
        chosen = [c[:4] for c in comps if _overlaps(c[:4], pred, pad)]
        box, found = None, False
        if chosen:
            bx, by, bw, bh = _union(chosen)
            found = True
            touch_l, touch_r = bx <= 1, bx + bw >= W - 1
            touch_t, touch_b = by <= 1, by + bh >= H - 1
            sx = bw / rw if not (touch_l or touch_r) else None
            sy = bh / rh if not (touch_t or touch_b) else None
            s = math.sqrt(sx * sy) if sx and sy else (sx or sy or 1.0)
            ew, eh = rw * s, rh * s
            # Complete the box for elements partly outside the frame.
            if touch_l and not touch_r:
                bx = bx + bw - ew
            if touch_t and not touch_b:
                by = by + bh - eh
            if touch_l or touch_r:
                bw = ew
            if touch_t or touch_b:
                bh = eh
            box = (bx, by, bw, bh)
            visible = float(np.clip((min(W, bx + bw) - max(0, bx)) * (min(H, by + bh) - max(0, by)) / max(bw * bh, 1), 0, 1))
        else:
            box = pred
            visible = None
        alpha = _opacity(frames[i], plate, template, box)
        scale = math.sqrt(max(box[2], 1) * max(box[3], 1) / (rw * rh)) if found else None
        if alpha is not None and alpha < 0.6:
            scale = None  # partially transparent masks shrink; do not read scale from them
        if not found and (alpha is None or alpha < 0.03):
            samples.append({"i": i, "t": times[i], "frame": frame_ids[i], "visible": False})
            continue
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        samples.append({"i": i, "t": times[i], "frame": frame_ids[i], "visible": True, "mask_found": found,
                        "cx": cx / W, "cy": cy / H, "w": box[2] / W, "h": box[3] / H, "scale": scale,
                        "opacity": alpha, "in_frame_fraction": visible})
        if found:
            pred = tuple(box)
    samples.sort(key=lambda s: s["t"])
    return samples


def _contrast_to_ring(img: np.ndarray, box, sel: np.ndarray) -> float:
    """How much the masked region differs from the ring of pixels around it.

    High in the frame where an element occupies the region, low where the
    region shows the same background as its surroundings.
    """
    H, W = img.shape[:2]
    x, y, w, h = box
    pad = max(4, min(w, h) // 4)
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    if not ring.any() or not sel.any():
        return 0.0
    bg = np.median(img[y0:y1, x0:x1][ring], axis=0)
    region = img[y:y + h, x:x + w][sel].astype(np.float32)
    return float(np.abs(region - bg).max(axis=1).mean())


def _direction(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(dy, dx)) % 360
    names = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


def _shape(mask: np.ndarray) -> dict:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    peri = cv2.arcLength(c, True)
    rect = area / max(w * h, 1)
    circ = 4 * math.pi * area / max(peri ** 2, 1)
    approx = len(cv2.approxPolyDP(c, 0.02 * peri, True))
    if circ > 0.85 and 0.8 < w / max(h, 1) < 1.25:
        guess = "circle"
    elif rect > 0.9:
        guess = "rectangle/panel"
    elif mask.sum() / max(w * h, 1) < 0.35:
        guess = "text or line-art (sparse)"
    else:
        guess = f"polygon/graphic ({approx} vertices)"
    return {"rectangularity": rect, "circularity": circ, "aspect_ratio": w / max(h, 1), "shape_guess": guess}


def analyse_samples(samples: list[dict], direction: str, W: int, H: int) -> dict:
    vis = [s for s in samples if s.get("visible")]
    out: dict = {"curves": {}, "labels": []}
    if len(vis) < 3:
        return out
    t = np.array([s["t"] for s in vis])
    cx = np.array([s["cx"] for s in vis])
    cy = np.array([s["cy"] for s in vis])
    # Path from the first to the last visible position (entrance: off-rest ->
    # rest; exit: rest -> last seen), measured as progress along the chord.
    start_c = np.array([cx[0], cy[0]])
    end_c = np.array([cx[-1], cy[-1]])
    disp = (end_c - start_c) * np.array([W, H])
    dist = float(np.hypot(*disp))
    diag = math.hypot(W, H)
    if dist > 0.01 * diag:
        u = disp / dist
        prog = (cx - start_c[0]) * W * u[0] + (cy - start_c[1]) * H * u[1]
        c = analyse_curve(t, prog, kind="position_px")
        if c:
            perp = -(cx - start_c[0]) * W * u[1] + (cy - start_c[1]) * H * u[0]
            c["measured"]["path_length_px"] = dist
            c["measured"]["path_deviation_px"] = float(np.max(np.abs(perp)))
            c["measured"]["direction"] = _direction(*disp)
            c["measured"]["from_xy_norm"] = start_c.tolist()
            c["measured"]["to_xy_norm"] = end_c.tolist()
            out["curves"]["position"] = c
    sv = [(s["t"], s["scale"]) for s in vis if s.get("scale") is not None]
    if len(sv) >= 4:
        st, ss = np.array(sv).T
        if ss.max() - ss.min() > 0.08:
            c = analyse_curve(st, ss, kind="scale")
            if c:
                out["curves"]["scale"] = c
    ov = [(s["t"], s.get("opacity")) for s in samples if s.get("opacity") is not None or not s.get("visible")]
    if len(ov) >= 4:
        ot = np.array([o[0] for o in ov])
        oo = np.clip(np.array([o[1] if o[1] is not None else 0.0 for o in ov]), 0, 1.2)
        # A real fade has intermediate opacities; an element that pops into
        # view (or slides in from off-screen) jumps 0 -> 1 between frames.
        partial = int(np.sum((oo > 0.12) & (oo < 0.88)))
        if oo.max() - oo.min() > 0.15 and partial >= 2:
            c = analyse_curve(ot, oo, kind="opacity")
            if c:
                out["curves"]["opacity"] = c

    labels = []
    edge = vis[0] if direction != "exit" else vis[-1]
    offscreen = (edge.get("in_frame_fraction") or 1.0) < 0.95
    if "position" in out["curves"]:
        d = out["curves"]["position"]["measured"]["direction"]
        if direction == "entrance":
            opposite = {"right": "left", "left": "right", "up": "bottom", "down": "top"}.get(d, d)
            labels.append(f"slide_in_from_{opposite}" + ("_offscreen" if offscreen else ""))
        elif direction == "exit":
            labels.append(f"slide_out_to_{d}")
        else:
            labels.append(f"move_{d}")
    if "scale" in out["curves"]:
        m = out["curves"]["scale"]["measured"]
        os_ = m["overshoot_ratio"] > 0.02
        if direction == "entrance" and m["from"] < 0.6:
            labels.append("pop_in (scale with overshoot)" if os_ else "scale_in")
        elif direction == "exit":
            labels.append("scale_out")
        else:
            labels.append("scale_change" + (" with overshoot" if os_ else ""))
    if "opacity" in out["curves"] and direction != "move":
        labels.append({"entrance": "fade_in", "exit": "fade_out"}.get(direction, "opacity_change"))
    if not labels:
        labels.append({"entrance": "cut_in (appears without measurable animation)",
                       "exit": "cut_out (disappears without measurable animation)"}.get(direction, "static"))
    out["labels"] = labels
    out["start_s"] = min((c["inferred"]["start_s"] for c in out["curves"].values()), default=float(t[0]))
    out["end_s"] = max((c["inferred"]["end_s"] for c in out["curves"].values()), default=float(t[-1]))
    return out


def _contact_sheet(frames, samples, path, W, H) -> None:
    vis = [s for s in samples if s.get("visible")]
    if not vis:
        return
    picks = [vis[int(round(k))] for k in np.linspace(0, len(vis) - 1, min(6, len(vis)))]
    tiles = []
    for s in picks:
        img = frames[s["i"]].copy()
        x, y = int((s["cx"] - s["w"] / 2) * W), int((s["cy"] - s["h"] / 2) * H)
        cv2.rectangle(img, (x, y), (x + int(s["w"] * W), y + int(s["h"] * H)), (0, 255, 255), 2)
        cv2.putText(img, f"{s['t']:.3f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (W // 2, H // 2)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def process_event(ev: Event, frames, times, frame_ids, ctx: Context, W: int, H: int, src_w: int,
                  min_area: int, counter: list[int]) -> list[dict]:
    gw, gh = ctx.config.get("analysis.motion_grid")
    cw, ch = W / gw, H / gh
    x0, y0, x1, y1 = ev.grid_bbox
    roi = (max(0, int((x0 - 2) * cw)), max(0, int((y0 - 2) * ch)), min(W, int((x1 + 3) * cw)), min(H, int((y1 + 3) * ch)))
    close_k = max(3, W // 80)
    before, after = frames[0], frames[-1]
    d = cv2.absdiff(after, before).max(axis=2)
    m = np.zeros((H, W), np.uint8)
    m[roi[1]:roi[3], roi[0]:roi[2]] = (d[roi[1]:roi[3], roi[0]:roi[2]] > DIFF_THR)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    comps = _components(m, min_area)

    elements: list[dict] = []
    hyps = []
    for (bx, by, bw, bh, _area) in comps:
        sel = m[by:by + bh, bx:bx + bw].astype(bool)
        sa = _contrast_to_ring(after, (bx, by, bw, bh), sel)
        sb = _contrast_to_ring(before, (bx, by, bw, bh), sel)
        direction = "entrance" if sa >= sb else "exit"
        hyps.append((direction, (bx, by, bw, bh), m[by:by + bh, bx:bx + bw]))
    if not hyps:
        # Element animates and returns to rest (pulse/wiggle) or passes through.
        peak = np.zeros((H, W), np.uint8)
        for f in frames[1:-1]:
            peak |= _fg_mask(f, before, roi, close_k)
        for (bx, by, bw, bh, _area) in _components(peak, min_area):
            hyps.append(("transient", (bx, by, bw, bh), peak[by:by + bh, bx:bx + bw]))

    # Clean plates: the "before" frame with places that elements vacated
    # filled from "after" (and vice versa). Without this, an element moving
    # A->B is segmented together with the hole it leaves at A.
    plate_in, plate_out = before.copy(), after.copy()
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # include anti-aliased / compressed edges
    for direction, (bx, by, bw, bh), _mask in hyps:
        if direction not in ("entrance", "exit"):
            continue
        sel = np.zeros((H, W), np.uint8)
        sel[by:by + bh, bx:bx + bw] = m[by:by + bh, bx:bx + bw]
        sel = cv2.dilate(sel, grow).astype(bool)
        if direction == "exit":
            plate_in[sel] = after[sel]
        else:
            plate_out[sel] = before[sel]
    n = len(frames)
    tracked = []
    for direction, box, mask in hyps:
        if direction == "entrance":
            samples = track_element(frames, times, frame_ids, n - 1, plate_in, box, roi, close_k, min_area, list(range(n - 1, -1, -1)))
        elif direction == "exit":
            samples = track_element(frames, times, frame_ids, 0, plate_out, box, roi, close_k, min_area, list(range(n)))
        else:
            mid = n // 2
            samples = track_element(frames, times, frame_ids, mid, before, box, roi, close_k, min_area, list(range(n)))
        tracked.append([direction, box, mask, samples])

    # An element that moves A->B shows up as an exit at A and an entrance at
    # B. Keep the entrance track (it carries the full path) and call it a move.
    for e in tracked:
        if e[0] != "exit":
            continue
        last = next((s for s in reversed(e[3]) if s.get("visible")), None)
        for f in tracked:
            if f[0] != "entrance" or last is None:
                continue
            fx, fy, fw, fh = f[1]
            lx, ly = last["cx"] * W, last["cy"] * H
            if fx <= lx <= fx + fw and fy <= ly <= fy + fh:
                f[0] = "move"
                e[0] = "merged"
    for direction, box, mask, samples in tracked:
        if direction == "merged":
            continue
        # Curves are measured in source-video pixels.
        ana = analyse_samples(samples, direction, src_w, src_w * H / W)
        counter[0] += 1
        eid = f"el_{counter[0]:03d}"
        ev_path = ctx.ws.path(f"visual/evidence/{eid}.jpg")
        _contact_sheet(frames, samples, ev_path, W, H)
        bx, by, bw, bh = box
        ref = frames[-1] if direction in ("entrance", "move") else frames[0]
        pix = ref[by:by + bh, bx:bx + bw][mask.astype(bool)] if mask.size else ref[by:by + bh, bx:bx + bw].reshape(-1, 3)
        col = np.median(pix, axis=0) if len(pix) else np.array([0, 0, 0])
        vis = [s for s in samples if s.get("visible")]
        scale_src = src_w / W
        conf = 0.85 if vis and np.mean([s.get("mask_found", False) for s in vis]) > 0.6 else 0.5
        if ev.mean_area > 0.3:
            conf *= 0.7
        elements.append({
            "id": eid, "event": ev.id, "scene": ev.scene, "role": direction,
            "rest_bbox_norm": [bx / W, by / H, bw / W, bh / H],
            "rest_bbox_px": [bx * scale_src, by * scale_src, bw * scale_src, bh * scale_src],
            "color_hex": "#%02x%02x%02x" % (int(col[2]), int(col[1]), int(col[0])),
            "geometry": _shape(mask.astype(np.uint8)) if mask.size else {},
            "first_visible_s": vis[0]["t"] if vis else None,
            "last_visible_s": vis[-1]["t"] if vis else None,
            "animation": {"labels": ana.get("labels", []), "start_s": ana.get("start_s"), "end_s": ana.get("end_s"),
                          "curves": ana.get("curves", {})},
            "samples": [{k: v for k, v in s.items() if k != "i"} for s in samples],
            "provenance": {
                "samples": prov("opencv", "foreground mask vs clean plate, border completion, regression opacity",
                                conf, OBSERVED, frame_range=[frame_ids[0], frame_ids[-1]], evidence=ctx.ws.rel(ev_path)),
                "labels_and_easing": prov("scipy.least_squares", "easing library + cubic-bezier fit (BIC)",
                                          conf * 0.85, INFERRED),
                "geometry": prov("opencv", "contour shape descriptors", 0.6, INFERRED),
            },
            "needs_segmentation": conf < 0.6,
        })
    return elements


# ----------------------------------------------------------------- camera
def camera_moves(fm: dict, scenes: list[dict], src_w: int, src_h: int) -> list[dict]:
    t = np.asarray(fm["t"])
    valid = np.asarray(fm["cam_valid"], bool)
    comps = {
        "pan": np.asarray(fm["cam_dx"]) / src_w,
        "tilt": np.asarray(fm["cam_dy"]) / src_h,
        "zoom": np.log(np.asarray(fm["cam_scale"])),
        "roll": np.asarray(fm["cam_rot_deg"]),
    }
    thr = {"pan": 0.0008, "tilt": 0.0008, "zoom": 0.0006, "roll": 0.03}
    out = []
    k = np.ones(5) / 5
    for name, v in comps.items():
        v = np.where(valid, v, 0.0)
        sm = np.convolve(v, k, mode="same")
        for a, b in _segments(np.abs(sm) > thr[name]):
            if t[b] - t[a] < 0.3:
                continue
            seg = v[a:b + 1]
            if np.sign(seg).mean() * np.sign(seg.sum()) < 0.6:
                continue  # jitter, not a move
            cum = np.cumsum(v[max(0, a - 3):min(len(v), b + 4)])
            tc = t[max(0, a - 3):min(len(v), b + 4)]
            total = float(seg.sum())
            desc = {
                "pan": "right" if total < 0 else "left",  # content moving left = camera panning right
                "tilt": "down" if total < 0 else "up",
                "zoom": "in" if total > 0 else "out",
                "roll": "clockwise" if total > 0 else "counter-clockwise",
            }[name]
            units = {"pan": "frame widths", "tilt": "frame heights", "zoom": "log scale", "roll": "degrees"}[name]
            scene = next((s["id"] for s in scenes if s["start"] <= t[a] < s["end"]), None)
            out.append({
                "type": name, "direction": desc, "start": float(t[a]), "end": float(t[b]),
                "duration_ms": float((t[b] - t[a]) * 1000), "total": abs(total), "units": units,
                "zoom_factor": float(math.exp(total)) if name == "zoom" else None, "scene": scene,
                "curve": analyse_curve(tc, np.abs(cum), kind=f"camera_{name}"),
                "provenance": prov("opencv", "accumulated per-frame similarity transforms", 0.7, OBSERVED),
            })
    return sorted(out, key=lambda m: m["start"])


def parallax_score(frames_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """Residual dense-flow spread after removing the global transform."""
    if not frames_pairs:
        return None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    scores = []
    for a, b in frames_pairs:
        ga = cv2.cvtColor(cv2.resize(a, (320, int(320 * a.shape[0] / a.shape[1]))), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, (320, int(320 * b.shape[0] / b.shape[1]))), cv2.COLOR_BGR2GRAY)
        flow = dis.calc(ga, gb, None)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        tex = cv2.Laplacian(ga, cv2.CV_32F)
        sel = np.abs(tex) > 8
        if sel.sum() < 200 or np.median(mag[sel]) < 0.3:
            continue
        fx, fy = flow[..., 0][sel], flow[..., 1][sel]
        rx, ry = fx - np.median(fx), fy - np.median(fy)
        scores.append(float(np.percentile(np.hypot(rx, ry), 90) / (np.median(mag[sel]) + 1e-6)))
    return float(np.median(scores)) if scores else None


# -------------------------------------------------------------------- run
def run(ctx: Context) -> dict:
    cfg = ctx.config
    meta = ctx.ws.read("metadata/video.json")
    fmj = ctx.ws.read("visual/frame_metrics.json")
    fm = fmj["columns"]
    sc = ctx.ws.read("metadata/scenes.json")
    vol = np.load(ctx.ws.path("cache/motion_volume.npz"))
    t = vol["t"]
    frame_of_row = vol["frame"]
    W, H = meta["analysis_size"]
    src_w, src_h = meta["width"], meta["height"]
    min_area = int(cfg.get("analysis.min_element_area") * W * H)

    active = vol["residual"] > int(0.02 * 255)
    # Mask global transitions so they do not become "elements".
    for c in sc["cuts"]:
        i = c["index"]
        active[max(0, i - 1):i + 2] = False
    for tr in sc["fades"] + sc["dissolves"]:
        if tr["type"].endswith("_hold"):
            continue
        active[(t >= tr["start"] - 0.05) & (t <= tr["end"] + 0.05)] = False

    events = find_events(active, t, sc["scenes"], float(cfg.get("analysis.max_event_seconds")))
    anim = [e for e in events if e.kind == "animation"]
    ctx.log("%d motion events (%d element animations)", len(events), len(anim))

    # Windows of frames needed per event: one settled frame before and after.
    windows = {}
    for e in anim:
        a, b = e.rows_scene
        r_before = max(a, e.r0 - 2)
        r_after = min(b, e.r1 + 2)
        windows[e.id] = (int(frame_of_row[r_before]), int(frame_of_row[r_after]))
    by_id = {e.id: e for e in anim}
    buffers: dict[str, list] = {eid: [] for eid in windows}
    elements: list[dict] = []
    counter = [0]

    cam_rows = [i for i, v in enumerate(fm["cam_valid"]) if v and (abs(fm["cam_dx"][i]) + abs(fm["cam_dy"][i])) > 1.0]
    cam_frames = set(int(fm["frame"][i]) for i in cam_rows[::5][:40])
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    prev = None

    need_decode = bool(windows) or bool(cam_frames)
    if need_decode:
        ts_all = np.asarray(ctx.ws.read("metadata/frames.json")["timestamps"])
        lo = min([w[0] for w in windows.values()] + [min(cam_frames)] if cam_frames else [w[0] for w in windows.values()])
        hi = max([w[1] for w in windows.values()] + [max(cam_frames)] if cam_frames else [w[1] for w in windows.values()])
        for idx, tt, frame in media.iter_frames(ctx.ws.video_path, W, H, ts_all, start=float(ts_all[lo]), end=float(ts_all[hi])):
            if idx in cam_frames and prev is not None:
                pairs.append((prev, frame.copy()))
            prev = frame
            for eid, (fa, fb) in list(windows.items()):
                if fa <= idx <= fb:
                    buffers[eid].append((idx, tt, frame.copy()))
                if idx == fb:
                    buf = buffers.pop(eid)
                    windows.pop(eid)
                    if len(buf) >= 3:
                        elements += process_event(by_id[eid], [b[2] for b in buf], [b[1] for b in buf],
                                                  [b[0] for b in buf], ctx, W, H, src_w, min_area, counter)
            if not windows and (not cam_frames or idx >= max(cam_frames)):
                break

    moves = camera_moves(fm, sc["scenes"], src_w, src_h)
    plx = parallax_score(pairs)
    cont = [{"id": e.id, "scene": e.scene, "start": float(t[e.r0]), "end": float(t[e.r1]),
             "mean_area_fraction": e.mean_area, "path": e.path[:: max(1, len(e.path) // 60)],
             "provenance": prov("opencv", "residual-motion volume component (long or large)", 0.6, OBSERVED)}
            for e in events if e.kind == "continuous"]

    # Summary statistics over entrance/exit animations.
    durs = [c["inferred"]["duration_ms"] for el in elements for c in el["animation"]["curves"].values()]
    easings: dict[str, int] = {}
    for el in elements:
        for c in el["animation"]["curves"].values():
            easings[c["inferred"]["easing"]] = easings.get(c["inferred"]["easing"], 0) + 1
    overshoots = [c["measured"]["overshoot_ratio"] for el in elements for c in el["animation"]["curves"].values()]
    labels: dict[str, int] = {}
    for el in elements:
        for lb in el["animation"]["labels"]:
            key = lb.split(" (")[0]
            labels[key] = labels.get(key, 0) + 1
    entr = sorted(el["animation"]["start_s"] for el in elements if el["role"] in ("entrance", "move") and el["animation"]["start_s"] is not None)
    stagger = [b - a for a, b in zip(entr, entr[1:]) if 0 < b - a < 0.6]
    duration = meta["duration_s"]
    summary = {
        "element_count": len(elements),
        "animated_elements_per_minute": len(elements) / (duration / 60) if duration else None,
        "animation_duration_ms": {"median": float(np.median(durs)) if durs else None,
                                  "p10": float(np.percentile(durs, 10)) if durs else None,
                                  "p90": float(np.percentile(durs, 90)) if durs else None},
        "easing_histogram": dict(sorted(easings.items(), key=lambda kv: -kv[1])),
        "animation_types": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "share_with_overshoot": float(np.mean([o > 0.02 for o in overshoots])) if overshoots else None,
        "stagger_intervals_ms": {"count": len(stagger), "median": float(np.median(stagger) * 1000) if stagger else None},
        "camera_moves": len(moves),
        "parallax_score": plx,
        "parallax_interpretation": (None if plx is None else
                                    "layered depth motion likely (parallax)" if plx > 0.6 else "planar/global motion"),
        "continuous_motion_regions": len(cont),
        "motion_energy_mean": float(np.mean(fm["residual_frac"])) if fm["residual_frac"] else None,
    }
    ctx.ws.write("visual/motion.json", {
        "summary": summary, "camera_moves": moves, "continuous_motion": cont,
        "events": [{"id": e.id, "scene": e.scene, "start": float(t[e.r0]), "end": float(t[e.r1]), "kind": e.kind,
                    "grid_bbox": e.grid_bbox, "mean_area_fraction": e.mean_area} for e in events],
        "provenance": {"events": prov("scipy.ndimage", "3-D components of residual-motion volume", 0.8, OBSERVED),
                       "parallax": prov("opencv DIS flow", "90th pct residual flow / median flow", 0.45, INFERRED)},
    })
    ctx.ws.write("visual/objects.json", {"elements": elements})
    seg_candidates = [el["id"] for el in elements if el["needs_segmentation"]]
    return {"outputs": ["visual/motion.json", "visual/objects.json"],
            "notes": [f"{len(elements)} elements tracked; {len(seg_candidates)} flagged for segmentation"]}
