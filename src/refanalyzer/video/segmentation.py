"""Stage: segmentation - SAM 2 refinement for hard-to-track elements.

Only elements the motion stage flagged (low tracking confidence: moving
background, occlusion, large overlapping regions) are sent, and only their
event window is uploaded, so managed-GPU cost stays proportional to the
problem cases.
"""

from __future__ import annotations

from .. import media
from ..models import OBSERVED, prov
from ..pipeline import Context
from ..providers import segmentation as seg


def run(ctx: Context) -> dict:
    provider = ctx.config.provider("segmentation")
    objects = (ctx.ws.read("visual/objects.json") or {}).get("elements", [])
    todo = [e for e in objects if e.get("needs_segmentation")]
    if provider == "none" or not todo:
        reason = "no segmentation provider configured" if provider == "none" else "no element needs segmentation"
        ctx.ws.write("visual/segmentation.json", {"provider": provider, "elements": [], "note": reason,
                                                  "candidates": [e["id"] for e in todo]})
        return {"outputs": ["visual/segmentation.json"], "notes": [reason] if todo else []}
    meta = ctx.ws.read("metadata/video.json")
    fps = meta.get("measured_fps") or 30.0
    limit = int(ctx.config.get(f"{provider}.max_elements", 6))
    results = []
    for el in todo[:limit]:
        vis = [s for s in el["samples"] if s.get("visible")]
        if not vis:
            continue
        t0, t1 = el["samples"][0]["t"], el["samples"][-1]["t"]
        clip = ctx.ws.path(f"cache/seg_{el['id']}.mp4")
        width = min(1280, meta["width"])
        media.run([media.require("ffmpeg"), "-v", "error", "-nostdin", "-y", "-ss", f"{max(0, t0):.3f}", "-to",
                   f"{t1 + 1 / fps:.3f}", "-i", str(ctx.ws.video_path), "-an", "-vf", f"scale={width}:-2,fps={fps:.4f}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(clip)])
        height = int(round(meta["height"] * width / meta["width"]))
        rest = vis[-1] if el["role"] in ("entrance", "move") else vis[0]
        click = (int(rest["cx"] * width), int(rest["cy"] * height))
        frame_i = int(round((rest["t"] - t0) * fps))
        try:
            fn = seg.replicate_sam2 if provider == "replicate" else seg.fal_sam2
            outputs = fn(ctx, clip, click, frame_i)
            masks = seg.masks_from_output(outputs, ctx.ws.path(f"cache/seg_{el['id']}"))
        except Exception as exc:  # noqa: BLE001 - one failing element should not stop the rest
            ctx.warn("SAM 2 failed for %s: %s", el["id"], exc)
            results.append({"element": el["id"], "error": str(exc)})
            continue
        stats = seg.mask_stats(masks)
        samples = [{"t": t0 + k / fps, **st} for k, st in enumerate(stats)]
        occl = [s for s in samples if s.get("visible") and s["visible_ratio_vs_peak"] < 0.7]
        results.append({
            "id": el["id"], "provider": provider, "frames": len(samples), "samples": samples,
            "partially_occluded_s": [s["t"] for s in occl][:200],
            "provenance": prov(f"sam2 ({provider})", "click-prompted video segmentation masks", 0.85, OBSERVED,
                               evidence=ctx.ws.rel(clip)),
        })
    ctx.ws.write("visual/segmentation.json", {"provider": provider, "elements": results})
    return {"outputs": ["visual/segmentation.json"], "notes": [f"{len(results)} elements segmented with {provider}"]}
