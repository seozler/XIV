"""Managed SAM 2 video segmentation (GPU runs on the provider's side).

* ``replicate`` - meta/sam-2-video (default)
* ``fal``       - fal-ai/sam2/video (queue API)

Both take a short clip plus a click point on the element and return per-frame
masks, from which centroid, visible area, bbox and occlusion are measured.
Input field names follow the providers' published schemas at the time of
writing; they are kept in one place so they are easy to adjust.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from ..config import api_key
from ..pipeline import Context
from .http import ProviderError, check, poll, session


def _data_uri(path: Path, mime: str = "video/mp4") -> str:
    data = path.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        raise ProviderError("clip too large for inline upload (>20 MB)")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def replicate_sam2(ctx: Context, clip: Path, click_xy: tuple[int, int], click_frame: int) -> list:
    cfg = ctx.config.get("replicate")
    s = session()
    s.headers.update({"Authorization": f"Bearer {api_key('replicate')}", "Prefer": "wait"})
    inp = {"input_video": _data_uri(clip), "click_coordinates": f"[{click_xy[0]},{click_xy[1]}]",
           "click_frames": str(click_frame), "click_labels": "1", "click_object_ids": "element",
           "mask_type": "binary", "output_video": False, "output_format": "png", "output_frame_interval": 1}
    if cfg.get("sam2_version"):
        url, body = f"{cfg['base_url']}/predictions", {"version": cfg["sam2_version"], "input": inp}
    else:
        url, body = f"{cfg['base_url']}/models/{cfg['sam2_model']}/predictions", {"input": inp}
    pred = check(s.post(url, json=body, timeout=120), "Replicate SAM 2")
    if pred.get("status") not in ("succeeded", "failed", "canceled"):
        pred = poll(lambda: check(s.get(pred["urls"]["get"], timeout=60), "Replicate status"),
                    lambda d: d.get("status") == "succeeded",
                    lambda d: d.get("error") or (d["status"] if d.get("status") in ("failed", "canceled") else None),
                    3, 900, "Replicate SAM 2")
    if pred.get("status") != "succeeded":
        raise ProviderError(f"Replicate SAM 2 {pred.get('status')}: {pred.get('error')}")
    ctx.costs.add("replicate", "sam-2-video run", 1, "run", ctx.config.get("pricing.replicate_sam2_per_run"),
                  note="list price example; billed by inference time")
    out = pred.get("output")
    return out if isinstance(out, list) else [out]


def fal_sam2(ctx: Context, clip: Path, click_xy: tuple[int, int], click_frame: int) -> list:
    cfg = ctx.config.get("fal")
    s = session()
    s.headers["Authorization"] = f"Key {api_key('fal')}"
    body = {"video_url": _data_uri(clip),
            "prompts": [{"x": click_xy[0], "y": click_xy[1], "frame_index": click_frame, "label": 1}],
            "apply_mask": False}
    sub = check(s.post(f"{cfg['base_url']}/{cfg['sam2_endpoint']}", json=body, timeout=120), "fal SAM 2")
    res = poll(lambda: check(s.get(sub["status_url"], timeout=60), "fal status"),
               lambda d: d.get("status") == "COMPLETED", lambda d: d.get("error"), 3, 900, "fal SAM 2")
    result = check(s.get(res.get("response_url") or sub["response_url"], timeout=60), "fal result")
    ctx.costs.add("fal", "sam2 video run", 1, "run", ctx.config.get("pricing.fal_sam2_per_run"))
    vid = result.get("video") or {}
    return [vid.get("url")] if vid.get("url") else []


def masks_from_output(outputs: list, work: Path) -> list[np.ndarray]:
    """Download provider output (mask images or a mask video) into arrays."""
    s = session()
    masks: list[np.ndarray] = []
    work.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(outputs):
        if not isinstance(url, str):
            continue
        data = s.get(url, timeout=300).content
        if url.split("?")[0].lower().endswith((".mp4", ".webm", ".mov")):
            p = work / f"masks_{i}.mp4"
            p.write_bytes(data)
            cap = cv2.VideoCapture(str(p))
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                masks.append((cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8))
            cap.release()
        else:
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                masks.append((arr > 127).astype(np.uint8))
    return masks


def mask_stats(masks: list[np.ndarray]) -> list[dict]:
    out = []
    peak = max((int(m.sum()) for m in masks), default=0) or 1
    for m in masks:
        area = int(m.sum())
        if area == 0:
            out.append({"visible": False})
            continue
        ys, xs = np.nonzero(m)
        H, W = m.shape
        out.append({"visible": True, "cx": float(xs.mean() / W), "cy": float(ys.mean() / H),
                    "bbox_norm": [float(xs.min() / W), float(ys.min() / H), float((xs.max() - xs.min() + 1) / W),
                                  float((ys.max() - ys.min() + 1) / H)],
                    "area_fraction": area / (W * H), "visible_ratio_vs_peak": area / peak})
    return out
