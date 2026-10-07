"""Stage: semantics - interpretation layer (vision LLM, narration LLM,
optional Twelve Labs Pegasus).

Inputs are measured data plus evidence images; outputs are stored as
``inferred`` and never overwrite measurements.
"""

from __future__ import annotations

import json

from ..models import INFERRED, prov
from ..pipeline import Context
from ..providers import twelvelabs
from ..providers.llm import extract_json, make_llm
from . import prompts


def _scene_payload(ws, sc: dict) -> tuple[dict, list[str]]:
    elements = [e for e in (ws.read("visual/objects.json") or {}).get("elements", []) if e["scene"] == sc["id"]]
    texts = [t for t in (ws.read("visual/ocr.json") or {}).get("tracks", []) if sc["start"] <= t["start"] < sc["end"]]
    colors = next((c for c in (ws.read("visual/colors.json") or {}).get("scenes", []) if c["scene"] == sc["id"]), {})
    cams = [m for m in (ws.read("visual/motion.json") or {}).get("camera_moves", []) if m.get("scene") == sc["id"]]
    images = []
    payload = {
        "scene": {k: sc[k] for k in ("id", "start", "end", "duration_s")},
        "camera_moves": [{k: m[k] for k in ("type", "direction", "start", "end", "total", "units")} for m in cams],
        "colors": {k: colors.get(k) for k in ("palette", "background_color_hex", "brightness_mean", "saturation_mean",
                                               "gradient", "composition")},
        "elements": [],
        "text": [],
    }
    for e in elements[:12]:
        payload["elements"].append({
            "id": e["id"], "role": e["role"], "rest_bbox_norm": e["rest_bbox_norm"], "color_hex": e["color_hex"],
            "shape_guess": e["geometry"].get("shape_guess"), "labels": e["animation"]["labels"],
            "animation_window_s": [e["animation"]["start_s"], e["animation"]["end_s"]],
            "curves": {k: {"easing_fit": c["inferred"]["easing"], "duration_ms": c["inferred"]["duration_ms"],
                           "cubic_bezier": c["inferred"]["cubic_bezier"], "overshoot_ratio": c["measured"]["overshoot_ratio"],
                           "from": c["measured"]["from"], "to": c["measured"]["to"]}
                       for k, c in e["animation"]["curves"].items()},
            "evidence_image": e["provenance"]["samples"].get("evidence_reference"),
        })
        ev = e["provenance"]["samples"].get("evidence_reference")
        if ev and len(images) < 3:
            images.append(ev)
    for t in texts[:15]:
        payload["text"].append({"id": t["id"], "content": t["content"], "start": t["start"], "end": t["end"],
                                "bbox_norm": t["bbox_norm"], "typography_measured": t["typography"],
                                "linked_element": t["linked_element"]})
    return payload, images


def run(ctx: Context) -> dict:
    ws = ctx.ws
    notes = []
    llm_v = make_llm(ctx, "vision")
    llm_r = make_llm(ctx, "reasoning")
    if llm_v is None:
        notes.append("no LLM provider configured: semantic interpretation skipped")
    else:
        scenes = ws.read("metadata/scenes.json")["scenes"]
        kfs = ws.read("visual/keyframes.json")["keyframes"]
        per_scene = int(ctx.config.get("analysis.llm_keyframes_per_scene"))
        budget = int(ctx.config.get("analysis.llm_max_images"))
        # Longest / busiest scenes first when the image budget is tight.
        objects = (ws.read("visual/objects.json") or {}).get("elements", [])
        busy = {s["id"]: sum(1 for e in objects if e["scene"] == s["id"]) for s in scenes}
        order = sorted(scenes, key=lambda s: -(s["duration_s"] + 2 * busy[s["id"]]))
        results = []
        for sc in order:
            if budget <= 0:
                break
            in_sc = [k for k in kfs if sc["start"] <= k["t"] < sc["end"]]
            if not in_sc:
                continue
            settled = [k for k in in_sc if k["reason"] == "settled"]
            pick = (settled[-per_scene:] if settled else []) or in_sc[-per_scene:]
            payload, ev_imgs = _scene_payload(ws, sc)
            imgs = [k["path"] for k in pick] + ev_imgs
            imgs = imgs[:max(1, budget)]
            budget -= len(imgs)
            parts = [{"type": "text", "text": "Measured data for this scene (deterministic tools):\n" +
                      json.dumps(payload, ensure_ascii=False, default=str)}]
            for p in imgs:
                parts.append({"type": "text", "text": f"Image: {p}"})
                parts.append({"type": "image", "path": str(ws.path(p))})
            parts.append({"type": "text", "text": prompts.VISUAL_JSON_CONTRACT})
            try:
                text = llm_v.complete(prompts.VISUAL_SYSTEM, parts, max_tokens=6000)
                parsed = extract_json(text)
                results.append({"scene": sc["id"], "images": imgs, "analysis": parsed, "raw": None if parsed else text})
            except Exception as exc:  # noqa: BLE001
                ctx.warn("vision analysis failed for %s: %s", sc["id"], exc)
                results.append({"scene": sc["id"], "error": str(exc)})
        ws.write("visual/design_llm.json", {
            "provider": llm_v.provider, "model": llm_v.model, "scenes": results,
            "provenance": prov(f"{llm_v.provider}:{llm_v.model}", "vision model interpreting measured data + keyframes",
                               0.6, INFERRED)})

        sp = ws.read("audio/speech_style.json") or {}
        if sp.get("available"):
            sync = ws.read("combined/synchronization.json") or {}
            tm = sp["text_metrics"]
            payload = {"language": sp.get("language"), "metrics": {k: v for k, v in tm.items() if k != "sentence_index"},
                       "prosody": sp.get("prosody"), "speakers": sp.get("speakers"),
                       "sentences": tm.get("sentence_index", [])[:400],
                       "text_speech_sync": sync.get("text_speech"),
                       "visual_events": [p for p in sync.get("patterns", []) if p.get("pattern_detected")]}
            parts = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)},
                     {"type": "text", "text": prompts.NARRATION_JSON_CONTRACT}]
            try:
                text = llm_r.complete(prompts.NARRATION_SYSTEM, parts, max_tokens=8000)
                parsed = extract_json(text)
                ws.write("audio/speech_style_llm.json", {
                    "provider": llm_r.provider, "model": llm_r.model, "profile": parsed,
                    "raw": None if parsed else text,
                    "provenance": prov(f"{llm_r.provider}:{llm_r.model}", "LLM on transcript + acoustic metrics", 0.6,
                                       INFERRED)})
            except Exception as exc:  # noqa: BLE001
                ctx.warn("narration analysis failed: %s", exc)
                notes.append(f"narration LLM failed: {exc}")

    vu = ctx.config.provider("video_understanding")
    if vu == "twelvelabs":
        meta = ws.read("metadata/video.json")
        try:
            res = twelvelabs.analyze(ctx, ws.video_path, prompts.PEGASUS_PROMPT, meta["duration_s"] / 3600)
            res["provenance"] = prov("twelvelabs pegasus", "video-native semantic description", 0.55, INFERRED)
            ws.write("metadata/video_semantics.json", res)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Twelve Labs failed: {exc}")
    return {"outputs": [p for p in ("visual/design_llm.json", "audio/speech_style_llm.json",
                                     "metadata/video_semantics.json") if ws.path(p).exists()], "notes": notes}
