"""Stage: visual_scan - one decode pass over every frame.

Per frame (numeric only, no ML):
  * luma / contrast / saturation / colourfulness / edge density
  * HSV histogram distance and pixel difference to the previous frame
  * global camera motion: sparse LK features + RANSAC similarity transform
    (pan = dx, tilt = dy, zoom = scale, roll = rotation)
  * camera-compensated residual-motion mask on a coarse grid; stacked over
    time this is the spatio-temporal motion volume used to find animations
Also stores keyframes (fixed interval, change-triggered and "settled after
motion") for OCR, colour and vision-model analysis.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import media
from ..models import OBSERVED, prov
from ..pipeline import Context

HIST_BINS = [16, 4, 4]


def _hist(hsv: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def _colourfulness(bgr: np.ndarray) -> float:
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


class CameraEstimator:
    """Global similarity transform between consecutive frames."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.prev_pts = None

    def estimate(self, prev: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray | None, dict]:
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
        info = {"features": 0, "inliers": 0, "valid": False}
        if pts is None or len(pts) < 12:
            return None, info
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
        good = st.reshape(-1) == 1
        p0, p1 = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
        info["features"] = int(len(p0))
        if len(p0) < 12:
            return None, info
        m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.5,
                                             maxIters=2000, confidence=0.99)
        if m is None:
            return None, info
        inl = inl.reshape(-1).astype(bool)
        info["inliers"] = int(inl.sum())
        q = p0[inl]
        spread = 0.0
        if len(q):
            spread = min((q[:, 0].max() - q[:, 0].min()) / self.w, (q[:, 1].max() - q[:, 1].min()) / self.h)
        info["inlier_ratio"] = float(inl.mean())
        info["spread"] = float(spread)
        # Only trust the transform when it is supported by points spread over
        # the frame; otherwise a single moving element would masquerade as
        # camera motion (common on flat motion-graphics backgrounds).
        info["valid"] = bool(info["inliers"] >= 15 and info["inlier_ratio"] >= 0.5 and spread >= 0.4)
        return (m if info["valid"] else None), info


def run(ctx: Context) -> dict:
    cfg = ctx.config
    meta = ctx.ws.read("metadata/video.json")
    ts = np.asarray(ctx.ws.read("metadata/frames.json")["timestamps"])
    aw, ah = meta["analysis_size"]
    gw, gh = cfg.get("analysis.motion_grid")
    stride = max(1, int(cfg.get("analysis.frame_stride")))
    res_thr = float(cfg.get("analysis.residual_threshold"))
    ocr_dt = float(cfg.get("analysis.ocr_interval_s"))
    change_thr = float(cfg.get("analysis.ocr_change_threshold"))

    n_expected = (len(ts) + stride - 1) // stride
    cols = {k: [] for k in ["frame", "t", "luma", "contrast", "saturation", "colourfulness", "edge_density",
                            "mean_b", "mean_g", "mean_r", "hist_dist", "pixel_diff", "residual_frac",
                            "cam_dx", "cam_dy", "cam_scale", "cam_rot_deg", "cam_valid", "cam_inlier_ratio"]}
    vol_gray = np.zeros((n_expected, gh, gw), np.uint8)
    vol_res = np.zeros((n_expected, gh, gw), np.uint8)  # fraction of moving pixels per cell, 0..255
    tw, th = 48, max(2, int(round(48 * ah / aw)))
    vol_color = np.zeros((n_expected, th, tw, 3), np.uint8)  # colour thumbnails (dissolves, colour cuts)
    keyframes: list[dict] = []

    cam = CameraEstimator(aw, ah)
    prev_gray = prev_hist = prev_small_bgr = None
    last_key_small = None
    next_key_t = 0.0
    was_moving = False
    row = 0
    for idx, t, frame in media.iter_frames(ctx.ws.video_path, aw, ah, ts):
        if idx % stride:
            continue
        if row >= n_expected:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
        small_bgr = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(small_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        hist = _hist(hsv)
        edges = cv2.Canny(gray, 80, 160)

        m = None
        cinfo = {"valid": False, "inlier_ratio": 0.0}
        if prev_gray is not None:
            m, cinfo = cam.estimate(prev_gray, gray)
            ref = cv2.warpAffine(prev_gray, m, (aw, ah), borderMode=cv2.BORDER_REPLICATE) if m is not None else prev_gray
            moving = (cv2.absdiff(gray, ref) > res_thr).astype(np.float32)
            if m is not None:  # ignore the strip uncovered by the warp
                moving[:3, :] = moving[-3:, :] = 0
                moving[:, :3] = moving[:, -3:] = 0
            cell = cv2.resize(moving, (gw, gh), interpolation=cv2.INTER_AREA)
            vol_res[row] = np.clip(cell * 255, 0, 255).astype(np.uint8)
            res_frac = float(moving.mean())
            hist_d = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            # Colour-aware: hue changes at equal luma still count.
            pix_d = float(cv2.absdiff(small_bgr, prev_small_bgr).max(axis=2).mean() / 255.0)
        else:
            res_frac, hist_d, pix_d = 0.0, 0.0, 0.0

        if m is not None:
            dx, dy = float(m[0, 2]), float(m[1, 2])
            scale = float(math.hypot(m[0, 0], m[1, 0]))
            rot = float(math.degrees(math.atan2(m[1, 0], m[0, 0])))
        else:
            dx = dy = rot = 0.0
            scale = 1.0
        mean_bgr = small_bgr.reshape(-1, 3).mean(axis=0)
        for k, v in (("frame", idx), ("t", t), ("luma", gray.mean() / 255.0), ("contrast", gray.std() / 255.0),
                     ("saturation", hsv[..., 1].mean() / 255.0), ("colourfulness", _colourfulness(small_bgr)),
                     ("edge_density", float((edges > 0).mean())), ("mean_b", mean_bgr[0]), ("mean_g", mean_bgr[1]),
                     ("mean_r", mean_bgr[2]), ("hist_dist", hist_d), ("pixel_diff", pix_d), ("residual_frac", res_frac),
                     ("cam_dx", dx * meta["width"] / aw), ("cam_dy", dy * meta["width"] / aw), ("cam_scale", scale),
                     ("cam_rot_deg", rot), ("cam_valid", bool(cinfo.get("valid"))),
                     ("cam_inlier_ratio", cinfo.get("inlier_ratio", 0.0))):
            cols[k].append(v)
        vol_gray[row] = small
        vol_color[row] = thumb

        # Keyframes for OCR / colour / vision analysis.
        reason = None
        if last_key_small is None:
            reason = "first"
        elif t >= next_key_t:
            reason = "interval"
        elif cv2.absdiff(small, last_key_small).mean() / 255.0 > change_thr:
            reason = "change"
        moving_now = res_frac > 0.002
        if reason is None and was_moving and not moving_now:
            reason = "settled"
        was_moving = moving_now
        if reason:
            path = ctx.ws.path(f"visual/frames/f{idx:06d}.jpg")
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            keyframes.append({"frame": idx, "t": t, "reason": reason, "path": ctx.ws.rel(path)})
            last_key_small = small
            next_key_t = t + ocr_dt

        prev_gray, prev_hist, prev_small_bgr = gray, hist, small_bgr
        row += 1
        if row % 500 == 0:
            ctx.log("scanned %d/%d frames", row, n_expected)

    vol_gray, vol_res, vol_color = vol_gray[:row], vol_res[:row], vol_color[:row]
    np.savez_compressed(ctx.ws.path("cache/motion_volume.npz"), gray=vol_gray, residual=vol_res, color=vol_color,
                        frame=np.asarray(cols["frame"], np.int32), t=np.asarray(cols["t"]))
    ctx.ws.write("visual/frame_metrics.json", {
        "layout": "columnar; one entry per analysed frame",
        "frame_stride": stride, "analysis_size": [aw, ah], "motion_grid": [gw, gh],
        "units": {"luma": "0..1", "cam_dx": "source px/frame", "cam_dy": "source px/frame",
                  "cam_scale": "ratio/frame", "cam_rot_deg": "deg/frame", "residual_frac": "share of pixels"},
        "columns": cols,
        "provenance": {
            "pixel_metrics": prov("opencv", "per-frame pixel statistics", 0.97, OBSERVED),
            "camera": prov("opencv", "goodFeaturesToTrack + PyrLK + RANSAC similarity; accepted only with "
                           "spatially spread inliers", 0.7, OBSERVED),
        },
    }, ndigits=5, indent=None)
    ctx.ws.write("visual/keyframes.json", {"keyframes": keyframes})
    return {"outputs": ["visual/frame_metrics.json", "visual/keyframes.json", "cache/motion_volume.npz"],
            "notes": [f"{row} frames analysed, {len(keyframes)} keyframes stored"]}
