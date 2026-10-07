"""Stage: compare - reference findings vs the existing video system.

Two layers:

1. Deterministic gap analysis (always runs, works offline). A catalogue of
   capabilities, each with
     * an evidence function over the measured reference profile (is the
       technique used in the reference, how strongly, at which timecodes),
     * keyword patterns (EN + TR) searched in the system description to
       estimate whether the current system supports it,
     * a concrete engineering recommendation and measurable acceptance
       criteria derived from the reference numbers.
2. LLM architect review (when an LLM provider is configured): the full
   system description, the profile, the deterministic table and evidence go
   to the comparison prompt, which returns a report and a JSON task list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..models import INFERRED, prov
from ..pipeline import Context
from ..providers.llm import extract_json, make_llm
from . import prompts

NEGATION = re.compile(r"\b(no|not|without|lacks?|missing|unsupported|todo|planned|future|"
                      r"yok|değil|eksik|ileride|henüz)\b"
                      # Turkish negative verb forms: kullanılmıyor, desteklenmiyor, desteklemez, ...
                      r"|\w+m[ıiuü]yor\b|\w+[^\W\d_]m[ae]z\b", re.I)


@dataclass
class Capability:
    id: str
    subsystem: str
    title: str
    keywords: list[str]
    evidence: Callable[[dict, "Data"], tuple[float, str, list[str]] | None]
    recommendation: str
    acceptance: Callable[[dict, "Data"], list[str]]
    impact: float = 1.0
    support_min: int = 2  # distinct keywords needed to call it "supported"


class Data:
    """Lazy access to the raw analysis files for evidence functions."""

    def __init__(self, ws):
        self.ws = ws
        self._c: dict = {}

    def __getattr__(self, name):
        files = {"objects": "visual/objects.json", "ocr": "visual/ocr.json", "motion": "visual/motion.json",
                 "scenes": "metadata/scenes.json", "sync": "combined/synchronization.json", "sfx": "audio/sfx.json",
                 "beats": "audio/beats.json", "duck": "audio/ducking.json", "speech": "audio/speech_style.json"}
        if name not in files:
            raise AttributeError(name)
        if name not in self._c:
            self._c[name] = self.ws.read(files[name]) or {}
        return self._c[name]


def _fmt_t(t: float | None) -> str:
    if t is None:
        return "?"
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:06.3f}"


def _curves(d: Data, prop: str | None = None):
    for e in d.objects.get("elements", []):
        for k, c in e["animation"]["curves"].items():
            if prop is None or k == prop:
                yield e, k, c


def _pattern(d: Data, rule: str) -> dict | None:
    return next((p for p in d.sync.get("patterns", []) if p["rule"] == rule), None)


def _label_hits(d: Data, prefix: str) -> list[dict]:
    return [e for e in d.objects.get("elements", []) if any(l.startswith(prefix) for l in e["animation"]["labels"])]


# ------------------------------------------------------------ evidence fns
def ev_easing(p, d):
    cs = [(e, k, c) for e, k, c in _curves(d) if k in ("position", "scale")]
    if not cs:
        return None
    nonlin = [x for x in cs if x[2]["inferred"]["easing"] != "linear"]
    refs = [f"{_fmt_t(c['inferred']['start_s'])} {e['id']} {k}: {c['inferred']['easing']} "
            f"{c['inferred']['duration_ms']:.0f} ms cubic-bezier{tuple(c['inferred']['cubic_bezier'])}" for e, k, c in nonlin[:6]]
    return len(nonlin) / len(cs), f"{len(nonlin)}/{len(cs)} measured motion curves are non-linear", refs


def acc_easing(p, d):
    durs = [c["inferred"]["duration_ms"] for _, k, c in _curves(d) if k in ("position", "scale")]
    hist = p["motion"].get("easing_histogram") or {}
    top = next(iter(hist), "ease_out_cubic")
    out = [f"Animation engine exposes named easings incl. `{top}` and arbitrary cubic-bezier(x1,y1,x2,y2)"]
    if durs:
        out.append(f"Generated entrances: median duration within ±15% of {np.median(durs):.0f} ms "
                   f"(re-measured with this analyzer)")
    return out


def ev_overshoot(p, d):
    hits = [(e, k, c) for e, k, c in _curves(d) if c["measured"]["overshoot_ratio"] > 0.02]
    if not hits:
        return None
    total = sum(1 for _ in _curves(d))
    refs = [f"{_fmt_t(c['inferred']['start_s'])} {e['id']} {k}: overshoot {c['measured']['overshoot_ratio']:.0%}, "
            f"fit {c['inferred']['easing']} {c['inferred'].get('params') or ''}" for e, k, c in hits[:6]]
    return len(hits) / max(total, 1), f"{len(hits)} curves overshoot their rest value", refs


def acc_overshoot(p, d):
    os_ = [c["measured"]["overshoot_ratio"] for _, _, c in _curves(d) if c["measured"]["overshoot_ratio"] > 0.02]
    return ["Spring/back easing available per element (damping/stiffness or back overshoot parameter)",
            f"Re-measured overshoot ratio of generated pop-ins ≈ {np.median(os_):.0%} (±5 pp)" if os_ else
            "Overshoot measurable on generated output"]


def ev_stagger(p, d):
    st = (p["motion"].get("stagger_intervals_ms") or {})
    if not st.get("count"):
        return None
    starts = sorted(e["animation"]["start_s"] for e in d.objects.get("elements", []) if e["animation"]["start_s"] is not None)
    return min(1.0, st["count"] / 4), f"{st['count']} staggered entrances, median offset {st['median']:.0f} ms", \
        [f"entrances at {', '.join(_fmt_t(s) for s in starts[:8])}"]


def ev_label(prefix, what):
    def f(p, d):
        hits = _label_hits(d, prefix)
        if not hits:
            return None
        return min(1.0, len(hits) / 3), f"{len(hits)} element(s) {what}", \
            [f"{_fmt_t(e['animation']['start_s'])} {e['id']} {', '.join(e['animation']['labels'])}" for e in hits[:6]]
    return f


def ev_camera(p, d):
    mv = d.motion.get("camera_moves", [])
    if not mv:
        return None
    return min(1.0, len(mv) / 3), f"{len(mv)} camera moves ({', '.join(sorted({m['type'] for m in mv}))})", \
        [f"{_fmt_t(m['start'])}-{_fmt_t(m['end'])} {m['type']} {m['direction']} ({m['total']:.3f} {m['units']})" for m in mv[:6]]


def ev_parallax(p, d):
    s = p["motion"].get("parallax_score")
    if s is None or s < 0.6:
        return None
    return min(1.0, s), f"parallax score {s:.2f} during camera moves (layered depth)", []


def ev_continuous(p, d):
    c = d.motion.get("continuous_motion", [])
    if not c:
        return None
    return 0.5, f"{len(c)} regions with continuous/ambient motion", [f"{_fmt_t(x['start'])}-{_fmt_t(x['end'])} {x['id']}" for x in c[:5]]


def ev_pacing(p, d):
    e = p["editing"]
    if not e.get("cut_count"):
        return None
    return 0.4, (f"{e['cuts_per_minute']:.1f} cuts/min, median scene {e['median_scene_s']:.2f} s "
                 f"(min {e['min_scene_s']:.2f} s, max {e['max_scene_s']:.2f} s)"), \
        [f"cuts at {', '.join(_fmt_t(c['t']) for c in d.scenes.get('cuts', [])[:10])}"]


def ev_transitions(p, d):
    tr = p["editing"].get("transition_types") or []
    if not tr:
        return None
    items = [f for f in d.scenes.get("fades", []) + d.scenes.get("dissolves", []) if not f["type"].endswith("_hold")]
    return 0.6, f"transition types used: {', '.join(tr)}", [f"{_fmt_t(f['start'])} {f['type']} {f['duration_ms']:.0f} ms" for f in items[:6]]


def ev_pattern(rule, min_hit=0.4):
    def f(p, d):
        pt = _pattern(d, rule)
        if not pt or not pt.get("pattern_detected") or pt["hit_rate"] < min_hit:
            return None
        lift = f", {pt['lift']:.1f}x chance" if pt.get("lift") else ""
        delta = pt.get("delta_ms") or {}
        refs = [f"{_fmt_t(x.get('visual_t', x.get('text_appear_s')))} {x.get('visual', x.get('text_track'))} "
                f"Δ {x['delta_ms']:+.0f} ms" for x in pt.get("examples", [])[:6]]
        return pt["hit_rate"], (f"{pt['matched']}/{pt['visual_events']} matched ({pt['hit_rate']:.0%}{lift}), "
                                f"median Δ {delta.get('median', 0):+.0f} ms"), refs
    return f


def ev_text_anim(p, d):
    t = p["typography"]
    share = t.get("animated_text_share")
    if not share:
        return None
    tracks = [x for x in d.ocr.get("tracks", []) if x.get("animation")]
    return share, f"{share:.0%} of text tracks animate in", \
        [f"{_fmt_t(x['start'])} '{x['content'][:30]}' {', '.join(x['animation']['labels'])}" for x in tracks[:6]]


def ev_reveal(p, d):
    n = p["typography"].get("progressive_reveals")
    if not n:
        return None
    tr = [x for x in d.ocr.get("tracks", []) if x.get("progressive_reveal")]
    return min(1.0, n / 2), f"{n} progressive (word/letter) text reveals", \
        [f"{_fmt_t(x['start'])} '{x['content'][:40]}' in {len(x['progressive_reveal'])} steps" for x in tr[:5]]


def ev_typescale(p, d):
    lv = p["typography"].get("font_size_px_levels") or []
    if len(lv) < 2:
        return None
    return 0.5, "type scale levels (px): " + ", ".join(f"{l['size_px']:.0f}×{l['count']}" for l in lv), []


def ev_plate(p, d):
    t = p["typography"]
    out = [x for x in d.ocr.get("tracks", []) if x["typography"].get("on_plate") or x["typography"].get("outline_or_shadow_likely")]
    if not out:
        return None
    return min(1.0, len(out) / max(1, t.get("text_tracks", 1))), f"{len(out)} text tracks on plates or with outline/shadow", \
        [f"{_fmt_t(x['start'])} '{x['content'][:30]}' plate={x['typography'].get('on_plate')} "
         f"outline={x['typography'].get('outline_or_shadow_likely')}" for x in out[:5]]


def ev_gradient(p, d):
    g = p["color"].get("gradient_background_share")
    if not g:
        return None
    return g, f"{g:.0%} of scenes use gradient backgrounds", []


def ev_palette(p, d):
    pal = p["color"].get("global_palette") or []
    if len(pal) < 3:
        return None
    top3 = sum(x["weighted_share"] for x in pal[:3])
    return min(1.0, top3), f"top-3 colours cover {top3:.0%}: {', '.join(x['hex'] for x in pal[:5])}", []


def ev_music(p, d):
    segs = d.beats.get("music_present_segments") or []
    dur = p["video"].get("duration_s") or 1
    cov = sum(s["end"] - s["start"] for s in segs) / dur
    if cov < 0.3 or not d.beats.get("tempo_bpm"):
        return None
    return min(1.0, cov), f"music present {cov:.0%} of runtime, {d.beats['tempo_bpm']:.1f} BPM, " \
                          f"{(d.beats.get('character') or {}).get('energy')} energy", []


def ev_ducking(p, d):
    s = d.duck.get("summary") or {}
    if not s.get("ducking_used"):
        return None
    return 0.9, (f"music ducks {s['median_attenuation_db']:.1f} dB under speech, attack {s.get('median_attack_s') or 0:.2f} s, "
                 f"release {s.get('median_release_s') or 0:.2f} s"), \
        [f"{_fmt_t(e['speech_start'])} −{e['attenuation_db']:.1f} dB" for e in d.duck.get("events", [])[:6] if e["attenuation_db"] > 2]


def ev_sfx(types, what):
    def f(p, d):
        ev = [e for e in d.sfx.get("events", []) if e["type"] in types]
        if not ev:
            return None
        return min(1.0, len(ev) / 4) * (1.0 if "stem" in (d.sfx.get("source") or "") else 0.7), \
            f"{len(ev)} {what} ({d.sfx.get('source')})", [f"{_fmt_t(e['start'])} {e['type']} ({e['confidence']:.2f})" for e in ev[:6]]
    return f


def ev_loudness(p, d):
    s = (p["audio"].get("mix_loudness") or {})
    i = s.get("integrated_lufs")
    # Only a mix mastered near a delivery target (-11..-18 LUFS) is evidence
    # of deliberate loudness normalisation worth copying.
    if i is None or not -18 <= i <= -11:
        return None
    return 0.6, f"integrated {i:.1f} LUFS, LRA {s.get('loudness_range_lu')}, true peak {s.get('true_peak_dbfs')} dBFS", []


def ev_wpm(p, d):
    n = p["narration"]
    if not n.get("words_per_minute"):
        return None
    pa = n.get("pauses") or {}
    return 0.7, (f"{n['words_per_minute']:.0f} WPM (articulation {n.get('articulation_rate_wpm') or 0:.0f}), "
                 f"{pa.get('per_minute') or 0:.1f} pauses/min, median pause {pa.get('p50_s', 0):.2f} s"), []


def ev_emphasis(p, d):
    n = p["narration"]
    e = n.get("emphasis_per_100w")
    pitch = n.get("pitch") or {}
    if not e or e < 3:
        return None
    return 0.6, f"{e:.1f} emphasised words /100w, pitch range {pitch.get('range_semitones', 0):.1f} st", \
        [f"{_fmt_t(w['start'])} '{w['word']}' z={w['prominence_z']:.1f}" for w in
         ((d.speech.get('prosody') or {}).get('emphasis') or {}).get('words', [])[:6]]


def ev_questions(p, d):
    s = (p["narration"].get("sentences") or {})
    if not s.get("questions"):
        return None
    return min(1.0, s["question_share"] * 4), f"{s['questions']} questions ({s['question_share']:.0%} of sentences)", []


def ev_address(p, d):
    a = p["narration"].get("address") or {}
    v = a.get("second_person_per_100w", 0)
    if v < 1:
        return None
    return min(1.0, v / 4), f"direct address {v:.1f}/100 words, inclusive 'we' {a.get('inclusive_we_per_100w', 0):.1f}/100", []


CATALOG: list[Capability] = [
    Capability("easing_library", "animation", "Non-linear easing curves", [
        r"easing", r"ease[-_ ]?(in|out)", r"cubic[-_ ]?bezier", r"bezier", r"(?-i:Easing\.[a-z])", r"power[1-4]\.(in|out)",
        r"yumuşatma", r"geçiş eğrisi"], ev_easing,
        "Expose named easings + arbitrary cubic-bezier per animated property; default entrances to the measured family.",
        acc_easing, 1.0),
    Capability("spring_overshoot", "animation", "Spring / overshoot motion", [
        r"spring", r"overshoot", r"back\.?out", r"elastic", r"bounce", r"damping", r"stiffness", r"yay", r"esneme"],
        ev_overshoot, "Add spring() / back-out easing with damping & stiffness (or overshoot s) parameters.",
        acc_overshoot, 0.9),
    Capability("stagger", "animation", "Staggered (sequential) entrances", [
        r"stagger", r"sequential", r"one (after|by) (another|one)", r"delay between", r"kademeli", r"sırayla", r"ardışık"],
        ev_stagger, "Support stagger offsets for groups (list items, words, icons) with configurable delay.",
        lambda p, d: [f"Grouped entrances offset by ≈{(p['motion'].get('stagger_intervals_ms') or {}).get('median') or 0:.0f} ms"], 0.8),
    Capability("slide_offscreen", "animation", "Slide-in from off-screen", [
        r"slide", r"translate[XY]?", r"off[- ]?screen", r"kayarak", r"kaydır"], ev_label("slide_in", "slide in"),
        "Allow entrances from outside the frame with eased position tweens.",
        lambda p, d: ["Element starts fully outside frame and settles with eased motion"], 0.6),
    Capability("scale_pop", "animation", "Scale / pop-in entrances", [
        r"\bscale\b", r"\bpop\b", r"zoom[- ]?in", r"ölçek", r"büyü"], ev_label("pop_in", "pop in with scale"),
        "Support scale-from-zero entrances with spring/back easing (transform-origin aware).",
        lambda p, d: ["Scale 0→1 entrance with measurable overshoot"], 0.7),
    Capability("fade", "animation", "Opacity fades on elements", [
        r"opacity", r"fade", r"alpha", r"saydam", r"belir"], ev_label("fade", "fade"),
        "Opacity tweens per element, combinable with position/scale.", lambda p, d: ["Per-element opacity curve"], 0.4, 1),
    Capability("camera", "camera", "Virtual camera moves (pan/zoom/roll)", [
        r"camera", r"\bpan\b", r"zoom", r"ken[- ]?burns", r"dolly", r"kamera"], ev_camera,
        "Add a scene-level camera transform track (pan/zoom/roll) with easing, independent of element animation.",
        lambda p, d: [f"{len(d.motion.get('camera_moves', []))} camera moves reproducible with eased transforms"], 0.8),
    Capability("parallax", "camera", "Parallax / layered depth", [
        r"parallax", r"depth", r"layer(s|ed)? .*speed", r"derinlik", r"katman"], ev_parallax,
        "Assign depth per layer and derive per-layer camera offsets (parallax).",
        lambda p, d: ["Background/midground/foreground move at different rates under camera moves"], 0.6),
    Capability("ambient_motion", "animation", "Continuous / ambient background motion", [
        r"ambient", r"background animation", r"particles?", r"\bloop", r"idle", r"arka plan animasyon"], ev_continuous,
        "Support looping background layers (particles, gradients, subtle drift) so frames are never fully static.",
        lambda p, d: ["Background motion energy > 0 during static narration"], 0.5),
    Capability("pacing", "editing", "Shot pacing / cut rhythm", [
        r"\bcut", r"scene (length|duration)", r"pacing", r"sahne süresi", r"kesme", r"tempo"], ev_pacing,
        "Drive scene durations from the script/narration with the measured pacing as a target distribution.",
        lambda p, d: [f"Median scene length ≈ {p['editing'].get('median_scene_s') or 0:.2f} s (±20%)"], 0.6),
    Capability("transitions", "editing", "Scene transitions (fade/dissolve)", [
        r"transition", r"dissolve", r"crossfade", r"dip to (black|white)", r"geçiş efekt", r"sahne geçiş"], ev_transitions,
        "Add a transition library (dip-to-black/white, dissolve) with duration control.",
        lambda p, d: ["Transition types and durations match the reference table"], 0.5),
    Capability("beat_sync_cuts", "sync", "Cuts on music beats", [
        r"\bbeat", r"\bbpm\b", r"on[- ]beat", r"ritim", r"vuruş", r"music sync"], ev_pattern("cut~beat"),
        "Run beat tracking on the chosen music and snap scene boundaries to the nearest beat (±1 frame).",
        lambda p, d: ["≥70% of generated cuts within 40 ms of a beat"], 0.9),
    Capability("beat_sync_motion", "sync", "Element entrances on beats", [
        r"\bbeat", r"\bbpm\b", r"ritim", r"vuruş"], ev_pattern("entrance~beat"),
        "Quantise entrance start times to beat grid when music is present.",
        lambda p, d: ["Entrance~beat hit rate on generated output ≥ reference hit rate"], 0.7),
    Capability("animated_text", "typography", "Animated text entrances", [
        r"text animation", r"kinetic", r"animated (text|title|caption)", r"yazı animasyon", r"metin animasyon"],
        ev_text_anim, "Give every text layer an entrance animation preset (measured easing/duration).",
        lambda p, d: ["All titles/labels enter with an animation (no hard cut-ins)"], 0.7),
    Capability("progressive_reveal", "typography", "Word-by-word / typewriter reveal", [
        r"typewriter", r"word[- ]by[- ]word", r"per[- ]word", r"letter[- ]by[- ]letter", r"kelime kelime", r"harf harf"],
        ev_reveal, "Split text into words/letters with per-unit timing (optionally from word timestamps).",
        lambda p, d: ["Text reveals in ≥2 steps with per-word timing"], 0.7),
    Capability("text_speech_sync", "sync", "On-screen text synced to spoken words", [
        r"word[- ]level timestamp", r"word timestamp", r"forced alignment", r"karaoke", r"subtitle", r"caption",
        r"altyazı", r"kelime zaman"], ev_pattern("text_appear~spoken_word", 0.0),
        "Use TTS/STT word timestamps to schedule text appearance relative to the spoken word.",
        lambda p, d: [f"Median text-vs-word offset ≈ {((_pattern(d, 'text_appear~spoken_word') or {}).get('delta_ms') or {}).get('median', 0):+.0f} ms (±120 ms)"], 0.9),
    Capability("type_scale", "typography", "Typographic hierarchy (type scale)", [
        r"font size", r"type scale", r"hierarchy", r"heading", r"başlık", r"yazı boyut", r"punto"], ev_typescale,
        "Define a type scale (title/subtitle/label/caption) in layout templates instead of ad-hoc sizes.",
        lambda p, d: ["Generated frames show the same number of size levels and ratios (±10%)"], 0.5),
    Capability("text_plate", "typography", "Text plates / outlines / shadows", [
        r"text background", r"\bplate", r"outline", r"stroke", r"drop[- ]?shadow", r"text[- ]shadow", r"gölge", r"kontur"],
        ev_plate, "Add text background plates and outline/shadow styles for legibility.",
        lambda p, d: [f"Minimum text contrast ratio ≥ {max(4.5, (p['typography'].get('min_contrast_ratio') or 4.5)):.1f}:1"], 0.4),
    Capability("gradients", "color", "Gradient backgrounds", [r"gradient", r"gradyan", r"renk geçiş"], ev_gradient,
        "Support linear/radial gradient backgrounds in scene templates.", lambda p, d: ["Background luma plane-fit R² > 0.85"], 0.4, 1),
    Capability("palette", "color", "Consistent colour palette / theme", [
        r"palette", r"brand colou?r", r"theme", r"design token", r"renk palet", r"tema"], ev_palette,
        "Centralise a palette (tokens) and constrain generated scenes to it.",
        lambda p, d: ["Top-3 colours cover a similar share of runtime as the reference"], 0.5, 1),
    Capability("music_bed", "audio", "Background music bed", [
        r"music", r"soundtrack", r"\bbgm\b", r"müzi[kğ]"], ev_music,
        "Select/generate a music bed per video with tempo & energy matched to the script.",
        lambda p, d: [f"Music tempo ≈ {d.beats.get('tempo_bpm') or 0:.0f} BPM class, present ≥ reference coverage"], 0.8, 1),
    Capability("ducking", "audio", "Music ducking under narration", [
        r"duck", r"sidechain", r"volume automation", r"müzi[kğ]\w* kıs", r"ses kısma"], ev_ducking,
        "Automate music gain from narration timestamps (attack/release envelope).",
        lambda p, d: [f"Music attenuation under speech ≈ {(d.duck.get('summary') or {}).get('median_attenuation_db') or 0:.1f} dB, "
                      f"attack ≈ {(d.duck.get('summary') or {}).get('median_attack_s') or 0:.2f} s"], 0.9),
    Capability("sfx_transitions", "audio", "Whoosh / transition sound effects", [
        r"sound effect", r"\bsfx\b", r"whoosh", r"swoosh", r"ses efekt"], ev_sfx({"whoosh"}, "whoosh effects"),
        "Add an SFX layer that places whooshes on entrances/transitions automatically.",
        lambda p, d: ["Whoosh starts within ±80 ms of the matching visual event"], 0.7),
    Capability("sfx_accents", "audio", "Impact / pop / click accents", [
        r"sound effect", r"\bsfx\b", r"impact", r"\bpop\b", r"click", r"ses efekt"],
        ev_sfx({"impact", "pop", "click", "typing"}, "accent effects"),
        "Map animation events (landings, pops, typing) to SFX cues.", lambda p, d: ["Accent SFX aligned to element settle/pop"], 0.6),
    Capability("riser", "audio", "Risers before transitions", [r"riser", r"build[- ]?up", r"swell"],
        ev_sfx({"riser"}, "risers"), "Add risers leading into scene changes.", lambda p, d: ["Riser ends within 100 ms of the cut"], 0.4),
    Capability("loudness", "audio", "Loudness normalisation / mastering", [
        r"\blufs\b", r"loudness", r"loudnorm", r"normali[sz]", r"ses seviyesi"], ev_loudness,
        "Master the final mix to a LUFS target with true-peak limiting (ffmpeg loudnorm two-pass).",
        lambda p, d: [f"Integrated loudness within ±1 LU of {(p['audio'].get('mix_loudness') or {}).get('integrated_lufs') or -14:.1f} LUFS"], 0.6),
    Capability("narration_pace", "narration", "Narration pace & pauses", [
        r"speech rate", r"speaking rate", r"\bwpm\b", r"ssml", r"<break", r"pause", r"konuşma hız", r"duraklama", r"tts"],
        ev_wpm, "Control TTS rate and inject pauses (SSML breaks) at sentence/section boundaries.",
        lambda p, d: [f"Generated narration ≈ {p['narration'].get('words_per_minute') or 0:.0f} WPM (±8%), pause p50 ≈ "
                      f"{(p['narration'].get('pauses') or {}).get('p50_s', 0):.2f} s"], 0.8),
    Capability("prosody", "narration", "Emphasis & intonation control", [
        r"emphasis", r"prosody", r"pitch", r"intonation", r"vurgu", r"tonlama", r"<emphasis"], ev_emphasis,
        "Mark emphasised words in the script and render them with prosody controls (SSML emphasis/pitch).",
        lambda p, d: ["Emphasised words per 100 words within ±30% of reference"], 0.6),
    Capability("questions", "narration", "Rhetorical questions in script", [
        r"question", r"hook", r"soru", r"merak"], ev_questions,
        "Script generator: use question-led hooks/transitions at the measured rate.",
        lambda p, d: ["Question share of sentences similar to reference"], 0.4),
    Capability("address", "narration", "Direct audience address", [
        r"second person", r"\byou\b", r"conversational", r"hitap", r"samimi"], ev_address,
        "Script generator: conversational second-person tone at the measured rate.",
        lambda p, d: ["Second-person markers per 100 words similar to reference"], 0.4),
    Capability("speech_visual_sync", "sync", "Visuals timed to narration phrases", [
        r"word[- ]level timestamp", r"timestamps", r"alignment", r"senkron", r"narration[- ]driven", r"zamanlama"],
        ev_pattern("entrance~speech_onset"),
        "Schedule element entrances from narration phrase starts (word timestamps).",
        lambda p, d: ["Entrance~phrase-start hit rate ≥ reference"], 0.8),
    Capability("whoosh_entrance_sync", "sync", "SFX timed to element entrances", [
        r"sound effect", r"\bsfx\b", r"whoosh", r"ses efekt"], ev_pattern("entrance~sfx_whoosh"),
        "Emit SFX cues from the animation timeline so sound and motion share one clock.",
        lambda p, d: ["Whoosh–entrance Δ within ±60 ms"], 0.6),
]


def system_support(text: str, cap: Capability) -> dict:
    """Estimate support from the system description (keyword evidence)."""
    lines = [l.strip() for l in re.split(r"[\n\r]+|(?<=[.;])\s", text) if l.strip()]
    hits, neg = [], []
    for kw in cap.keywords:
        rx = re.compile(kw, re.I)
        for l in lines:
            if rx.search(l):
                (neg if NEGATION.search(l) else hits).append(l[:200])
    pos_kws = {kw for kw in cap.keywords if any(re.search(kw, h, re.I) for h in hits)}
    if len(pos_kws) >= cap.support_min or (pos_kws and len(hits) >= 3):
        status = "supported"
    elif pos_kws:
        status = "partial"
    else:
        status = "missing"
    return {"status": status, "matched_keywords": sorted(pos_kws), "negated_mentions": neg[:3],
            "supporting_lines": list(dict.fromkeys(hits))[:4]}


def deterministic_gaps(ws, system_text: str | None) -> list[dict]:
    profile = ws.read("combined/profile.json") or {}
    d = Data(ws)
    rows = []
    for cap in CATALOG:
        try:
            ev = cap.evidence(profile, d)
        except Exception as exc:  # noqa: BLE001 - one bad evidence fn must not stop the comparison
            ev = None
            rows.append({"capability": cap.id, "error": str(exc)})
            continue
        sup = system_support(system_text, cap) if system_text else {"status": "unknown"}
        used = ev is not None
        strength, summary, refs = ev if ev else (0.0, "not observed in reference", [])
        gap = used and sup["status"] in ("missing", "partial")
        score = strength * cap.impact * (1.0 if sup["status"] == "missing" else 0.5 if sup["status"] == "partial" else 0.0)
        rows.append({
            "capability": cap.id, "subsystem": cap.subsystem, "title": cap.title,
            "used_in_reference": used, "reference_strength": strength, "reference_evidence": summary,
            "timecoded_examples": refs, "current_system": sup, "gap": gap, "priority_score": score if gap else 0.0,
            "recommendation": cap.recommendation if gap else None,
            "acceptance_criteria": cap.acceptance(profile, d) if gap else [],
        })
    rows.sort(key=lambda r: -r.get("priority_score", 0))
    return rows


def _tasks_from_rows(rows: list[dict]) -> list[dict]:
    tasks = []
    for r in rows:
        if not r.get("gap"):
            continue
        s = r["priority_score"]
        tasks.append({
            "id": f"T{len(tasks) + 1:02d}", "title": r["title"], "subsystem": r["subsystem"],
            "priority": "P0" if s >= 0.6 else "P1" if s >= 0.3 else "P2",
            "status_in_current_system": r["current_system"]["status"],
            "reference_behavior": r["reference_evidence"], "evidence": r["timecoded_examples"],
            "required_changes": [r["recommendation"]], "acceptance_criteria": r["acceptance_criteria"],
            "expected_benefit": "closes a measured gap with the reference", "complexity": None,
            "confidence": round(min(0.9, 0.4 + 0.5 * r["reference_strength"]), 2), "source": "deterministic",
        })
    return tasks


def _deterministic_markdown(rows, system_path, has_llm) -> str:
    L = ["# Sistem Karşılaştırması (deterministik)", ""]
    if system_path is None:
        L += ["> Mevcut sistem açıklaması (`current-video-system.txt`) verilmedi; yalnızca referansta kullanılan "
              "teknikler listelendi. `--system` ile tekrar çalıştırın.", ""]
    L += ["Bu tablo, referans videoda **ölçülen** tekniklerle sistem açıklamasındaki **anahtar kelime** "
          "kanıtlarını eşleştirir. Destek durumu bir tahmindir (inferred); ölçüm sütunları gözlemdir (observed).", ""]
    if not has_llm:
        L += ["> LLM sağlayıcısı yapılandırılmadığı için mimari yorum katmanı çalışmadı.", ""]
    L += ["| Öncelik | Yetenek | Alt sistem | Referansta | Mevcut sistem | Kanıt |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            continue
        pri = f"{r['priority_score']:.2f}" if r["gap"] else "-"
        used = "evet" if r["used_in_reference"] else "hayır"
        L.append(f"| {pri} | {r['title']} | {r['subsystem']} | {used} | {r['current_system']['status']} | "
                 f"{r['reference_evidence']} |")
    L += ["", "## Açık boşluklar", ""]
    for r in [r for r in rows if r.get("gap")]:
        L += [f"### {r['title']} ({r['subsystem']}) — skor {r['priority_score']:.2f}", "",
              f"- **Referans davranışı:** {r['reference_evidence']}",
              *[f"  - `{x}`" for x in r["timecoded_examples"]],
              f"- **Mevcut sistem:** {r['current_system']['status']}"
              + (f" (eşleşen: {', '.join(r['current_system'].get('matched_keywords', []))})" if r['current_system'].get('matched_keywords') else ""),
              *[f"  - satır: “{x}”" for x in r["current_system"].get("supporting_lines", [])[:2]],
              f"- **Öneri:** {r['recommendation']}",
              "- **Kabul kriterleri:**", *[f"  - {a}" for a in r["acceptance_criteria"]], ""]
    keep = [r for r in rows if r.get("used_in_reference") and r["current_system"].get("status") == "supported"]
    if keep:
        L += ["## Zaten desteklenenler (değiştirmeyin)", ""] + [f"- {r['title']}: {r['reference_evidence']}" for r in keep]
    return "\n".join(L) + "\n"


def run(ctx: Context) -> dict:
    ws = ctx.ws
    spath = ws.system_spec_path
    system_text = spath.read_text(encoding="utf-8", errors="replace") if spath else None
    rows = deterministic_gaps(ws, system_text)
    ws.write("reports/gap-analysis.json", {
        "system_spec": ws.rel(spath) if spath else None, "rows": rows,
        "provenance": prov("refanalyzer.compare", "capability catalogue: measured evidence × keyword support", 0.6, INFERRED)})
    tasks = _tasks_from_rows(rows)
    llm = make_llm(ctx, "reasoning")
    md_det = _deterministic_markdown(rows, spath, llm is not None)
    notes = []
    report = md_det
    if llm is not None and system_text:
        profile = ws.read("combined/profile.json") or {}
        design = ws.read("visual/design_llm.json") or {}
        narr = ws.read("audio/speech_style_llm.json") or {}
        semantics = ws.read("metadata/video_semantics.json") or {}
        lang = {"tr": "Turkish", "en": "English"}.get(ctx.config.get("llm.report_language"), "Turkish")
        parts = [
            {"type": "text", "text": "## 1. Existing system specification\n\n" + system_text},
            {"type": "text", "text": "## 2. Reference measured profile (JSON)\n\n" + json.dumps(profile, ensure_ascii=False, default=str)},
            {"type": "text", "text": "## 3. Deterministic gap table (keyword-based starting point)\n\n" + json.dumps(
                [{k: r.get(k) for k in ("capability", "title", "subsystem", "used_in_reference", "reference_evidence",
                                         "timecoded_examples", "current_system", "gap", "priority_score")} for r in rows],
                ensure_ascii=False, default=str)},
            {"type": "text", "text": "## 4. Interpretation layer (inferred)\n\n" + json.dumps({
                "design": [s.get("analysis") for s in design.get("scenes", [])],
                "narration": narr.get("profile"), "video_semantics": semantics.get("text")}, ensure_ascii=False, default=str)},
            {"type": "text", "text": prompts.COMPARE_OUTPUT_CONTRACT.format(language=lang)},
        ]
        try:
            text = llm.complete(prompts.COMPARE_SYSTEM, parts)
            data = extract_json(text[text.rfind("```json"):] if "```json" in text else text)
            llm_tasks = (data or {}).get("tasks") if isinstance(data, dict) else None
            body = text[:text.rfind("```json")].rstrip() if "```json" in text else text
            report = (f"# Sistem Karşılaştırma Raporu\n\n_Model: {llm.provider}:{llm.model} — yorum katmanı (inferred). "
                      f"Ölçümler: `combined/profile.json`, kanıtlar: `combined/evidence.json`._\n\n{body}\n\n---\n\n"
                      + md_det.replace("# Sistem Karşılaştırması (deterministik)", "## Ek: Deterministik boşluk tablosu"))
            if llm_tasks:
                for t in llm_tasks:
                    t["source"] = "llm"
                tasks = llm_tasks + [t for t in tasks if t["title"] not in {x.get("title") for x in llm_tasks}]
        except Exception as exc:  # noqa: BLE001
            notes.append(f"LLM comparison failed, deterministic report kept: {exc}")
    elif system_text is None:
        notes.append("no system spec provided: comparison lists reference techniques only")
    (ws.path("reports/system-comparison.md")).write_text(report, encoding="utf-8")
    ws.write("reports/improvement-tasks.json", {"tasks": tasks, "count": len(tasks)})
    return {"outputs": ["reports/system-comparison.md", "reports/improvement-tasks.json", "reports/gap-analysis.json"],
            "notes": notes}
