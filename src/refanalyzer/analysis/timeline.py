"""Stage: timeline - merge every analysis into one time system.

Outputs
  combined/timeline.json  collections on one clock (seconds, container PTS)
  combined/slices.json    fixed 0.5 s slices: what is on screen / heard
  combined/evidence.json  every provenance block with its source file
  combined/profile.json   compact, comparable metrics per subsystem
"""

from __future__ import annotations

import numpy as np

from ..pipeline import Context


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _curve_at(curve: dict | None, times: np.ndarray) -> list[float | None]:
    if not curve or not curve.get("db"):
        return [None] * len(times)
    db = np.asarray(curve["db"], float)
    t = curve["t0"] + np.arange(len(db)) * curve["hop_s"]
    return [float(v) for v in np.interp(times, t, db)]


def build_profile(ws) -> dict:
    meta = ws.read("metadata/video.json") or {}
    sc = ws.read("metadata/scenes.json") or {}
    mo = ws.read("visual/motion.json") or {}
    ocr = ws.read("visual/ocr.json") or {}
    col = ws.read("visual/colors.json") or {}
    beats = ws.read("audio/beats.json") or {}
    sfx = ws.read("audio/sfx.json") or {}
    sp = ws.read("audio/speech_style.json") or {}
    duck = ws.read("audio/ducking.json") or {}
    loud = ws.read("audio/loudness.json") or {}
    stems = ws.read("audio/stems.json") or {}
    sync = ws.read("combined/synchronization.json") or {}
    elements = (ws.read("visual/objects.json") or {}).get("elements", [])
    seg = ws.read("visual/segmentation.json") or {}
    tm = sp.get("text_metrics") or {}
    return {
        "video": {"duration_s": meta.get("duration_s"), "resolution": [meta.get("width"), meta.get("height")],
                  "fps": meta.get("measured_fps"), "variable_frame_rate": meta.get("variable_frame_rate")},
        "editing": {**(sc.get("stats") or {}), "transition_types": sorted({f["type"] for f in sc.get("fades", []) + sc.get("dissolves", [])
                                                                          if not f["type"].endswith("_hold")})},
        "motion": {**(mo.get("summary") or {}),
                   "camera_move_types": sorted({f"{m['type']}_{m['direction']}" for m in mo.get("camera_moves", [])}),
                   "segmented_elements": len(seg.get("elements", []))},
        "elements": [{"id": e["id"], "role": e["role"], "labels": e["animation"]["labels"],
                      "start_s": e["animation"]["start_s"],
                      "curves": {k: {"easing": c["inferred"]["easing"], "duration_ms": c["inferred"]["duration_ms"],
                                     "cubic_bezier": c["inferred"]["cubic_bezier"],
                                     "overshoot_ratio": c["measured"]["overshoot_ratio"],
                                     "fit_rmse": c["inferred"]["fit_rmse"]}
                                 for k, c in e["animation"]["curves"].items()}} for e in elements],
        "typography": ocr.get("summary") or {"text_tracks": 0},
        "color": col.get("summary") or {},
        "audio": {
            "separation": {"provider": stems.get("provider"), "kind": stems.get("kind"),
                           "quality": stems.get("quality")},
            "mix_loudness": _get(loud, "layers", "mix", "summary"),
            "layer_loudness": {k: v.get("summary") for k, v in (loud.get("layers") or {}).items() if k != "mix"},
            "music": {"tempo_bpm": beats.get("tempo_bpm"), "beat_regularity": beats.get("beat_regularity"),
                      "key": _get(beats, "key_estimate", "key"), "character": beats.get("character"),
                      "sections": len(beats.get("sections") or []), "source": beats.get("source"),
                      "present_segments": beats.get("music_present_segments")},
            "ducking": duck.get("summary") if duck.get("measurable") else {"measurable": False, "reason": duck.get("reason")},
            "sfx": {"counts": sfx.get("counts"), "per_minute": sfx.get("per_minute"), "source": sfx.get("source")},
        },
        "narration": {
            "available": sp.get("available", False), "language": sp.get("language"),
            "words_per_minute": tm.get("words_per_minute"), "articulation_rate_wpm": tm.get("articulation_rate_wpm"),
            "pauses": tm.get("pauses"), "sentences": {k: v for k, v in (tm.get("sentences") or {}).items()},
            "vocabulary": tm.get("vocabulary"), "address": tm.get("address"),
            "fillers_per_100w": tm.get("fillers_per_100w"), "discourse_markers_per_100w": tm.get("discourse_markers_per_100w"),
            "pitch": _get(sp, "prosody", "pitch"), "emphasis_per_100w": _get(sp, "prosody", "emphasis", "per_100w"),
            "sentence_final_contour": _get(sp, "prosody", "sentence_final_contour"), "speakers": sp.get("speakers"),
            "speech_share_vad": sp.get("speech_share"),
        },
        "sync": {"patterns": [{k: p.get(k) for k in ("rule", "description", "hit_rate", "chance_rate", "lift", "p_value",
                                                       "delta_ms", "pattern_detected", "visual_events", "matched")}
                              for p in sync.get("patterns", [])],
                 "pacing": sync.get("pacing")},
    }


def _collect_evidence(ws) -> list[dict]:
    files = ["metadata/video.json", "metadata/frames.json", "metadata/scenes.json", "visual/frame_metrics.json",
             "visual/motion.json", "visual/objects.json", "visual/ocr.json", "visual/colors.json",
             "visual/segmentation.json", "visual/design_llm.json", "audio/stems.json", "audio/transcript.json",
             "audio/loudness.json", "audio/beats.json", "audio/speech_style.json", "audio/sfx.json",
             "audio/ducking.json", "combined/synchronization.json"]
    out: list[dict] = []

    def walk(node, path, src):
        if isinstance(node, dict):
            if "source_tool" in node and "observed_or_inferred" in node:
                out.append({"id": f"ev_{len(out) + 1:04d}", "file": src, "path": path, **node})
                return
            for k, v in node.items():
                if k in ("samples", "detections", "columns", "timestamps", "rms_curve", "r128_curve", "words"):
                    continue
                walk(v, f"{path}.{k}" if path else k, src)
        elif isinstance(node, list):
            for i, v in enumerate(node[:2000]):
                walk(v, f"{path}[{i}]", src)

    for f in files:
        data = ws.read(f)
        if data is not None:
            walk(data, "", f)
    return out


def run(ctx: Context) -> dict:
    ws = ctx.ws
    meta = ws.read("metadata/video.json")
    dur = float(meta["duration_s"])
    sc = ws.read("metadata/scenes.json")
    mo = ws.read("visual/motion.json") or {}
    elements = (ws.read("visual/objects.json") or {}).get("elements", [])
    seg = {e["id"]: e for e in (ws.read("visual/segmentation.json") or {}).get("elements", [])}
    ocr = ws.read("visual/ocr.json") or {}
    tr = ws.read("audio/transcript.json") or {}
    beats = ws.read("audio/beats.json") or {}
    sfx = ws.read("audio/sfx.json") or {}
    loud = ws.read("audio/loudness.json") or {}
    duck = ws.read("audio/ducking.json") or {}
    sync = ws.read("combined/synchronization.json") or {}
    fm = ws.read("visual/frame_metrics.json")["columns"]

    objects = []
    for e in elements:
        samples = [{"t": s["t"], "x": s.get("cx"), "y": s.get("cy"), "w": s.get("w"), "h": s.get("h"),
                    "scale": s.get("scale"), "opacity": s.get("opacity"), "visible": s.get("visible")}
                   for s in e["samples"]]
        objects.append({"id": e["id"], "type": "graphic", "role": e["role"], "scene": e["scene"],
                        "start": e["first_visible_s"], "end": e["last_visible_s"],
                        "rest_bbox_norm": e["rest_bbox_norm"], "color_hex": e["color_hex"],
                        "shape": e["geometry"].get("shape_guess"), "motion": e["animation"],
                        "position_samples": samples, "segmentation": seg.get(e["id"]),
                        "provenance": e["provenance"]})
    for t in ocr.get("tracks", []):
        objects.append({"id": t["id"], "type": "text", "content": t["content"], "start": t["start"], "end": t["end"],
                        "bbox_norm": t["bbox_norm"], "typography": t["typography"], "linked_element": t["linked_element"],
                        "motion": t["animation"], "progressive_reveal": t["progressive_reveal"],
                        "provenance": t["provenance"]})

    music_curve = None
    layers = loud.get("layers") or {}
    for name in ("music", "music_and_effects", "mix"):
        if name in layers:
            music_curve = (name, layers[name]["rms_curve"])
            break
    t1 = np.arange(0, dur, 1.0)
    timeline = {
        "video_id": ws.video_id, "timebase": "seconds", "clock": "container-relative presentation time",
        "duration": dur, "fps": meta.get("measured_fps"), "resolution": [meta["width"], meta["height"]],
        "scenes": sc["scenes"], "transitions": {"cuts": sc["cuts"], "fades": sc["fades"], "dissolves": sc["dissolves"]},
        "camera": mo.get("camera_moves", []),
        "objects": objects,
        "speech": {"provider": tr.get("provider"), "language": tr.get("language"), "words": tr.get("words") or [],
                   "utterances": tr.get("utterances") or [], "vad_segments": tr.get("vad_segments") or []},
        "music": {"tempo_bpm": beats.get("tempo_bpm"), "beats": beats.get("beats") or [],
                  "downbeats": beats.get("downbeats_4_4") or [], "sections": beats.get("sections") or [],
                  "present": beats.get("music_present_segments") or [],
                  "level_1s": {"layer": music_curve[0] if music_curve else None, "t": t1.tolist(),
                               "rms_db": _curve_at(music_curve[1] if music_curve else None, t1)}},
        "sfx": sfx.get("events") or [],
        "ducking": duck.get("events") or [],
        "sync_events": sync.get("pairs") or [],
        "sync_patterns": [p for p in sync.get("patterns", []) if p.get("pattern_detected")],
    }
    ws.write("combined/timeline.json", timeline, ndigits=4)

    # 0.5 s slices: a joined view of every layer.
    step = 0.5
    t = np.asarray(fm["t"])
    res = np.asarray(fm["residual_frac"])
    mix_curve = layers.get("mix", {}).get("rms_curve")
    edges = np.arange(0, dur + 1e-9, step)
    starts = edges[:-1] if len(edges) > 1 else np.array([0.0])
    mix_lvl = _curve_at(mix_curve, starts + step / 2)
    mus_lvl = _curve_at(music_curve[1] if music_curve else None, starts + step / 2)
    beat_t = np.asarray(beats.get("beats") or [])
    words = tr.get("words") or []
    slices = []
    for i, a in enumerate(starts):
        b = a + step
        scn = next((s["id"] for s in sc["scenes"] if s["start"] <= a < s["end"]), None)
        active = []
        for o in objects:
            if o["start"] is None or o["end"] is None or not (o["start"] < b and a <= o["end"]):
                continue
            state = "visible"
            m = o.get("motion") or {}
            if m.get("start_s") is not None and m.get("end_s") is not None and m["start_s"] < b and a <= m["end_s"]:
                state = "animating"
            active.append({"id": o["id"], "state": state, **({"text": o["content"]} if o["type"] == "text" else {})})
        sel = (t >= a) & (t < b)
        slices.append({
            "t": float(a), "scene": scn, "objects": active,
            "words": " ".join(w["word"] for w in words if a <= w["start"] < b),
            "sfx": [e["type"] for e in sfx.get("events") or [] if a <= e["start"] < b],
            "beats": int(((beat_t >= a) & (beat_t < b)).sum()) if len(beat_t) else 0,
            "mix_rms_db": mix_lvl[i], "music_rms_db": mus_lvl[i],
            "motion_energy": float(res[sel].mean()) if sel.any() else None,
            "cut": any(a <= c["t"] < b for c in sc["cuts"]),
        })
    ws.write("combined/slices.json", {"step_s": step, "slices": slices}, ndigits=3)
    ws.write("combined/profile.json", build_profile(ws))
    ev = _collect_evidence(ws)
    ws.write("combined/evidence.json", {"count": len(ev), "records": ev,
                                         "legend": {"observed": "directly measured from pixels/samples",
                                                    "inferred": "estimated / classified from measurements"}})
    return {"outputs": ["combined/timeline.json", "combined/slices.json", "combined/profile.json",
                        "combined/evidence.json"]}
