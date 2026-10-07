"""Audio stages: separation, transcription, loudness, music, speech style,
sound effects and ducking.

Every stage prefers the matching separated stem (dialogue / music /
effects) and falls back to the original mix with reduced confidence, so the
pipeline still produces results when no separation provider is configured.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import media
from ..models import INFERRED, OBSERVED, prov
from ..pipeline import Context
from ..providers import separation as sep
from ..providers import transcription as stt

EPS = 1e-10


# ---------------------------------------------------------------- helpers
def _duration(ctx: Context) -> float:
    return float(ctx.ws.read("metadata/video.json")["duration_s"])


def _has_audio(ctx: Context) -> bool:
    return ctx.ws.path("audio/original.wav").exists()


def stems(ctx: Context) -> dict[str, Path]:
    info = ctx.ws.read("audio/stems.json") or {}
    return {k: ctx.ws.path(v) for k, v in (info.get("stems") or {}).items() if ctx.ws.path(v).exists()}


def source_for(ctx: Context, stem: str) -> tuple[Path, bool]:
    """(path, is_isolated_stem) for the requested layer."""
    s = stems(ctx)
    if stem in s:
        return s[stem], True
    if stem in ("music", "effects") and "music_and_effects" in s:
        return s["music_and_effects"], False
    return ctx.ws.path("audio/original.wav"), False


def rms_db(y: np.ndarray, sr: int, hop_s: float, win_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * (win_s or 2 * hop_s)))
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)
    t = (np.arange(len(db)) * hop + win / 2) / sr
    return t, db


def speech_segments(ctx: Context, gap: float = 0.35) -> list[dict]:
    tr = ctx.ws.read("audio/transcript.json") or {}
    words = tr.get("words") or []
    if words:
        segs: list[dict] = []
        for w in words:
            if segs and w["start"] - segs[-1]["end"] <= gap and w.get("speaker") == segs[-1].get("speaker"):
                segs[-1]["end"] = w["end"]
                segs[-1]["words"] += 1
            else:
                segs.append({"start": w["start"], "end": w["end"], "speaker": w.get("speaker"), "words": 1})
        return segs
    return list(tr.get("vad_segments") or [])


# ------------------------------------------------------------- separation
def run_separation(ctx: Context) -> dict:
    if not _has_audio(ctx):
        ctx.ws.write("audio/stems.json", {"provider": None, "stems": {}, "note": "no audio stream"})
        return {"outputs": ["audio/stems.json"], "notes": ["no audio"]}
    provider = ctx.config.provider("separation")
    mix = ctx.ws.path("audio/original.wav")
    dur = _duration(ctx)
    res = sep.separate(ctx, provider, mix, dur / 60.0)
    y, sr = media.load_audio(mix)
    quality: dict = {"mix_rms_db": float(10 * np.log10(np.mean(y ** 2) + EPS))}
    if res["stems"]:
        total = np.zeros_like(y)
        for name, p in res["stems"].items():
            ys, srs = media.load_audio(p)
            n = min(len(ys), len(y))
            total[:n] += ys[:n]
            quality[f"{name}_energy_share"] = float(np.sum(ys[:n] ** 2) / (np.sum(y[:n] ** 2) + EPS))
        err = y - total
        quality["reconstruction_snr_db"] = float(10 * np.log10(np.sum(y ** 2) / (np.sum(err ** 2) + EPS)))
        quality["note"] = ("Stems are model estimates: expect leakage between layers. Low reconstruction SNR "
                           "(< 10 dB) means the stems do not add back up to the mix.")
    ctx.ws.write("audio/stems.json", {
        "provider": provider, "kind": res["kind"],
        "stems": {k: ctx.ws.rel(v) for k, v in res["stems"].items()},
        "missing_layers": res["missing"], "quality": quality,
        "provenance": prov(provider, "source separation", 0.7 if res["stems"] else 0.0, INFERRED),
    })
    notes = [] if res["stems"] else ["no separation provider: audio layers analysed on the mix"]
    return {"outputs": ["audio/stems.json"] + [ctx.ws.rel(p) for p in res["stems"].values()], "notes": notes}


# ---------------------------------------------------------- transcription
def vad(y: np.ndarray, sr: int) -> list[dict]:
    """Energy + spectral-shape voice activity (fallback without STT)."""
    import librosa

    hop = int(0.02 * sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = S[(freqs >= 300) & (freqs <= 3400)].sum(axis=0) / (S.sum(axis=0) + EPS)
    flat = librosa.feature.spectral_flatness(S=S)[0]
    e = 10 * np.log10(S.sum(axis=0) + EPS)
    floor = np.percentile(e, 10)
    speech = (e > floor + 12) & (band > 0.5) & (flat < 0.3)
    k = 10
    speech = np.convolve(speech.astype(float), np.ones(k) / k, mode="same") > 0.5
    segs, i = [], 0
    while i < len(speech):
        if speech[i]:
            j = i
            while j + 1 < len(speech) and speech[j + 1]:
                j += 1
            if (j - i) * 0.02 >= 0.3:
                segs.append({"start": i * 0.02, "end": (j + 1) * 0.02, "speaker": None})
            i = j + 1
        else:
            i += 1
    return segs


def run_transcription(ctx: Context) -> dict:
    if not _has_audio(ctx):
        ctx.ws.write("audio/transcript.json", {"provider": None, "words": [], "note": "no audio"})
        return {"outputs": ["audio/transcript.json"]}
    provider = ctx.config.provider("transcription")
    src, isolated = source_for(ctx, "dialogue")
    up = ctx.ws.path("cache/speech_16k.wav")
    media.run([media.require("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", str(up)])
    dur = _duration(ctx)
    if provider == "assemblyai":
        res = stt.assemblyai(ctx, up, dur / 3600)
    elif provider == "deepgram":
        res = stt.deepgram(ctx, up, dur / 60)
    elif provider == "faster_whisper":
        res = stt.faster_whisper(ctx, up)
    else:
        y, sr = media.load_audio(up)
        res = {"provider": "vad-only", "model": None, "language": None, "text": "", "words": [], "utterances": [],
               "vad_segments": vad(y, sr)}
    res["input"] = "dialogue stem" if isolated else "original mix"
    res["provenance"] = prov(res["provider"], "speech-to-text with word timestamps" if res["words"] else
                             "energy/spectral voice activity only (no transcript)",
                             0.9 if res["words"] else 0.35, OBSERVED if res["words"] else INFERRED)
    ctx.ws.write("audio/transcript.json", res)
    notes = [] if res["words"] else ["no transcription provider configured: only voice-activity segments"]
    return {"outputs": ["audio/transcript.json"], "notes": notes}


# ---------------------------------------------------------------- loudness
def run_loudness(ctx: Context) -> dict:
    if not _has_audio(ctx):
        ctx.ws.write("audio/loudness.json", {"layers": {}})
        return {"outputs": ["audio/loudness.json"]}
    hop = float(ctx.config.get("analysis.loudness_hop_ms")) / 1000
    layers = {"mix": ctx.ws.path("audio/original.wav"), **stems(ctx)}
    out = {}
    for name, p in layers.items():
        y, sr = media.load_audio(p, mono=False)
        mono = y.mean(axis=1)
        t, db = rms_db(mono, sr, hop, 0.05)
        peak = 20 * np.log10(np.max(np.abs(y)) + EPS)
        r128 = media.ebur128(p)
        out[name] = {
            "summary": {**r128["summary"], "rms_dbfs": float(10 * np.log10(np.mean(mono ** 2) + EPS)),
                        "sample_peak_dbfs": float(peak),
                        "rms_p10_db": float(np.percentile(db, 10)), "rms_p90_db": float(np.percentile(db, 90)),
                        "dynamic_range_db": float(np.percentile(db, 95) - np.percentile(db, 10))},
            "rms_curve": {"hop_s": hop, "window_s": 0.05, "t0": float(t[0]), "db": db.astype(np.float32)},
            "r128_curve": {"t": r128["t"], "momentary_lufs": r128["momentary_lufs"],
                           "short_term_lufs": r128["short_term_lufs"]},
        }
    ctx.ws.write("audio/loudness.json", {
        "layers": out,
        "provenance": prov("ffmpeg ebur128 + numpy", "EBU R128 (M 400 ms, S 3 s) and windowed RMS", 0.95, OBSERVED),
    }, ndigits=2, indent=None)
    return {"outputs": ["audio/loudness.json"]}


# ------------------------------------------------------------------- music
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(chroma_mean: np.ndarray) -> dict:
    best = (-2.0, "C", "major")
    for i in range(12):
        for prof, mode in ((KS_MAJOR, "major"), (KS_MINOR, "minor")):
            r = np.corrcoef(chroma_mean, np.roll(prof, i))[0, 1]
            if r > best[0]:
                best = (r, NOTES[i], mode)
    return {"key": f"{best[1]} {best[2]}", "correlation": float(best[0])}


def novelty_sections(feat: np.ndarray, times: np.ndarray, min_gap: float = 4.0) -> list[float]:
    """Checkerboard-kernel novelty on a self-similarity matrix."""
    if feat.shape[1] < 16:
        return []
    f = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + EPS)
    f = f / (np.linalg.norm(f, axis=0, keepdims=True) + EPS)
    ssm = f.T @ f
    L = 8
    k = np.kron(np.array([[1, -1], [-1, 1]]), np.ones((L, L)))
    nov = np.zeros(ssm.shape[0])
    for i in range(L, ssm.shape[0] - L):
        nov[i] = np.sum(ssm[i - L:i + L, i - L:i + L] * k)
    nov = np.maximum(nov, 0)
    if nov.max() <= 0:
        return []
    thr = nov.mean() + 1.5 * nov.std()
    peaks = [i for i in range(1, len(nov) - 1) if nov[i] > thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    out: list[float] = []
    for i in peaks:
        if not out or times[i] - out[-1] >= min_gap:
            out.append(float(times[i]))
    return out


def run_music(ctx: Context) -> dict:
    import librosa

    if not _has_audio(ctx):
        ctx.ws.write("audio/beats.json", {"present": False})
        return {"outputs": ["audio/beats.json"]}
    src, isolated = source_for(ctx, "music")
    y, sr = media.load_audio(src)
    y = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    sr = 22050
    hop = 256  # ~11.6 ms beat resolution
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, float)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, units="time")
    # Beat-strength regularity: how well beats line up with onsets.
    ibi = np.diff(beats)
    regularity = float(1 - np.std(ibi) / np.mean(ibi)) if len(ibi) > 2 else None
    tempo_tracker = tempo
    if len(ibi) >= 3:
        # The median inter-beat interval is finer than the tracker's tempo grid.
        tempo = float(60.0 / np.median(ibi))
    # Downbeat phase assuming 4/4: phase with the strongest accumulated onsets.
    downbeats = []
    if len(beats) >= 8:
        strength = np.interp(beats, librosa.times_like(oenv, sr=sr, hop_length=hop), oenv)
        phase = int(np.argmax([strength[p::4].sum() for p in range(4)]))
        downbeats = beats[phase::4].tolist()
    t_e, e_db = rms_db(y, sr, 0.25, 0.5)
    present_thr = max(np.percentile(e_db, 95) - 30, -60)
    presence = e_db > present_thr
    segs = []
    i = 0
    while i < len(presence):
        if presence[i]:
            j = i
            while j + 1 < len(presence) and presence[j + 1]:
                j += 1
            if t_e[j] - t_e[i] >= 1.0:
                segs.append({"start": float(t_e[i] - 0.125), "end": float(t_e[j] + 0.125)})
            i = j + 1
        else:
            i += 1
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop * 4) if len(y) > sr else None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop * 4, n_mfcc=13)
    feat_t = librosa.times_like(mfcc, sr=sr, hop_length=hop * 4)
    sections = novelty_sections(np.vstack([mfcc, chroma]) if chroma is not None else mfcc, feat_t)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    key = estimate_key(chroma.mean(axis=1)) if chroma is not None else None
    energy_level = float(np.percentile(e_db, 75))
    character = {
        "tempo_class": "slow" if tempo < 85 else "moderate" if tempo < 115 else "upbeat" if tempo < 140 else "fast",
        "brightness": "dark/warm" if centroid < 1500 else "balanced" if centroid < 3000 else "bright",
        "energy": "low" if energy_level < -30 else "medium" if energy_level < -18 else "high",
    }
    ctx.ws.write("audio/beats.json", {
        "source": "music stem" if isolated else "original mix (no music stem: beats include all layers)",
        "tempo_bpm": tempo, "tempo_tracker_bpm": tempo_tracker, "beat_regularity": regularity, "beats": beats.tolist(), "downbeats_4_4": downbeats,
        "onsets": onsets.tolist(), "music_present_segments": segs, "sections": sections,
        "energy_curve": {"hop_s": 0.25, "t0": float(t_e[0]), "db": e_db},
        "spectral_centroid_hz": centroid, "key_estimate": key, "character": character,
        "provenance": {
            "tempo_beats": prov("librosa", "onset-strength dynamic-programming beat tracker", 0.8 if isolated else 0.6, OBSERVED),
            "downbeats": prov("librosa", "4/4 phase with strongest onset accumulation", 0.45, INFERRED),
            "key": prov("librosa", "Krumhansl-Schmuckler on mean chroma", 0.5, INFERRED),
            "character": prov("heuristic", "tempo / centroid / energy thresholds", 0.5, INFERRED),
            "sections": prov("numpy", "checkerboard novelty on MFCC+chroma SSM", 0.5, INFERRED),
        },
    }, ndigits=3, indent=None)
    return {"outputs": ["audio/beats.json"]}


# ------------------------------------------------------------ speech style
EN_SECOND = {"you", "your", "you're", "yours", "yourself"}
TR_SECOND = {"sen", "sana", "seni", "senin", "siz", "size", "sizi", "sizin", "sizce"}
EN_WE = {"we", "us", "our", "let's", "lets"}
TR_WE = {"biz", "bize", "bizi", "bizim", "hadi", "gelin"}
FILLERS = {"um", "uh", "erm", "like", "basically", "actually", "yani", "şey", "hani", "ee", "ıı", "işte"}
MARKERS = {"so", "now", "first", "next", "then", "finally", "because", "but", "imagine", "remember",
           "şimdi", "peki", "önce", "sonra", "ilk", "son", "çünkü", "ama", "düşünün", "hatırlayın"}


def text_metrics(words: list[dict]) -> dict:
    if not words:
        return {}
    toks = [re.sub(r"[^\w'çğıöşü]", "", w["word"].lower()) for w in words]
    toks = [t for t in toks if t]
    span = words[-1]["end"] - words[0]["start"]
    gaps = np.array([b["start"] - a["end"] for a, b in zip(words, words[1:])])
    pauses = gaps[gaps > 0.15]
    long_pauses = gaps[gaps > 0.3]
    speaking = span - long_pauses.sum()
    sentences, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?…]$", w["word"]):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    lens = [len(s) for s in sentences]
    questions = sum(1 for s in sentences if s[-1]["word"].endswith("?"))
    n = len(toks)
    return {
        "word_count": n, "speech_span_s": span,
        "words_per_minute": n / (span / 60) if span > 0 else None,
        "articulation_rate_wpm": n / (speaking / 60) if speaking > 0 else None,
        "pauses": {"count_over_150ms": int(len(pauses)), "count_over_300ms": int(len(long_pauses)),
                   "mean_s": float(pauses.mean()) if len(pauses) else 0.0,
                   "p50_s": float(np.median(pauses)) if len(pauses) else 0.0,
                   "p90_s": float(np.percentile(pauses, 90)) if len(pauses) else 0.0,
                   "per_minute": len(pauses) / (span / 60) if span > 0 else None},
        "sentences": {"count": len(sentences), "words_mean": float(np.mean(lens)), "words_median": float(np.median(lens)),
                      "words_p90": float(np.percentile(lens, 90)), "questions": questions,
                      "question_share": questions / len(sentences)},
        "vocabulary": {"type_token_ratio": len(set(toks)) / n, "mean_word_length": float(np.mean([len(t) for t in toks])),
                       "long_word_share": float(np.mean([len(t) >= 7 for t in toks])),
                       "numbers_share": float(np.mean([any(c.isdigit() for c in t) for t in toks]))},
        "address": {"second_person_per_100w": 100 * sum(t in EN_SECOND or t in TR_SECOND for t in toks) / n,
                    "inclusive_we_per_100w": 100 * sum(t in EN_WE or t in TR_WE for t in toks) / n},
        "fillers_per_100w": 100 * sum(t in FILLERS for t in toks) / n,
        "discourse_markers_per_100w": 100 * sum(t in MARKERS for t in toks) / n,
        "sentence_index": [{"start": s[0]["start"], "end": s[-1]["end"], "words": len(s),
                            "text": " ".join(w["word"] for w in s)} for s in sentences],
    }


def prosody(path: Path, words: list[dict], sentences: list[dict]) -> dict:
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {"note": "parselmouth not installed (pip install praat-parselmouth)"}
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    tp = pitch.xs()
    inten = snd.to_intensity(time_step=0.01)
    ti, iv = inten.xs(), inten.values[0]
    voiced = f0 > 0
    out: dict = {}
    if voiced.sum() > 10:
        st = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
        out["pitch"] = {"median_hz": float(np.median(f0[voiced])), "p5_hz": float(np.percentile(f0[voiced], 5)),
                        "p95_hz": float(np.percentile(f0[voiced], 95)),
                        "range_semitones": float(np.percentile(st, 95) - np.percentile(st, 5)),
                        "variability_semitones_sd": float(np.std(st)), "voiced_share": float(voiced.mean())}
    out["intensity"] = {"mean_db": float(np.mean(iv)), "sd_db": float(np.std(iv))}
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 70, 500)
        out["voice_quality"] = {
            "jitter_local": float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)),
            "shimmer_local": float(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)),
            "hnr_db": float(call(snd.to_harmonicity_cc(), "Get mean", 0, 0)),
        }
    except Exception:  # noqa: BLE001 - optional measurement
        pass
    # Word-level prominence: z-scores of pitch, loudness and duration.
    feats = []
    for w in words:
        sel = (tp >= w["start"]) & (tp <= w["end"]) & (f0 > 0)
        seli = (ti >= w["start"]) & (ti <= w["end"])
        feats.append([np.mean(f0[sel]) if sel.any() else np.nan, np.max(iv[seli]) if seli.any() else np.nan,
                      w["end"] - w["start"]])
    emph = []
    if feats:
        F = np.array(feats, float)
        Z = (F - np.nanmean(F, axis=0)) / (np.nanstd(F, axis=0) + EPS)
        score = np.nanmean(Z, axis=1)
        for w, s in zip(words, score):
            if np.isfinite(s) and s > 1.0:
                emph.append({"word": w["word"], "start": w["start"], "prominence_z": float(s)})
    out["emphasis"] = {"count": len(emph), "per_100w": 100 * len(emph) / max(len(words), 1), "words": emph[:200]}
    finals = {"rising": 0, "falling": 0, "flat": 0}
    for s in sentences:
        sel = (tp >= s["end"] - 0.5) & (tp <= s["end"]) & (f0 > 0)
        if sel.sum() < 5:
            continue
        stv = 12 * np.log2(f0[sel] / np.median(f0[voiced]))
        slope = np.polyfit(tp[sel], stv, 1)[0]
        finals["rising" if slope > 3 else "falling" if slope < -3 else "flat"] += 1
    out["sentence_final_contour"] = finals
    return out


def run_speech_style(ctx: Context) -> dict:
    tr = ctx.ws.read("audio/transcript.json") or {}
    words = tr.get("words") or []
    if not words:
        segs = tr.get("vad_segments") or []
        ctx.ws.write("audio/speech_style.json", {
            "available": False, "speech_segments": segs,
            "speech_share": sum(s["end"] - s["start"] for s in segs) / max(_duration(ctx), 1e-6) if segs else 0.0,
            "note": "no word-level transcript; configure a transcription provider for narration metrics"})
        return {"outputs": ["audio/speech_style.json"]}
    tm = text_metrics(words)
    src, isolated = source_for(ctx, "dialogue")
    pro = prosody(src, words, tm.get("sentence_index", []))
    speakers: dict = {}
    for u in tr.get("utterances") or []:
        s = speakers.setdefault(str(u.get("speaker")), {"talk_time_s": 0.0, "turns": 0})
        s["talk_time_s"] += u["end"] - u["start"]
        s["turns"] += 1
    total_talk = sum(s["talk_time_s"] for s in speakers.values()) or 1.0
    for s in speakers.values():
        s["share"] = s["talk_time_s"] / total_talk
    ctx.ws.write("audio/speech_style.json", {
        "available": True, "language": tr.get("language"), "text_metrics": tm, "prosody": pro, "speakers": speakers,
        "segments": speech_segments(ctx),
        "provenance": {
            "text_metrics": prov(tr.get("provider", "?"), "word timestamps + punctuation", 0.85, OBSERVED),
            "prosody": prov("parselmouth/praat", "F0 (AC), intensity, jitter/shimmer/HNR on " +
                            ("dialogue stem" if isolated else "mix"), 0.8 if isolated else 0.5, OBSERVED),
            "emphasis": prov("numpy", "word z-score of pitch+intensity+duration > 1", 0.55, INFERRED),
        },
    })
    return {"outputs": ["audio/speech_style.json"]}


# --------------------------------------------------------------------- SFX
PANN_MAP = {
    "whoosh": ["Whoosh, swoosh, swish"], "impact": ["Thump, thud", "Bang", "Slam", "Boom", "Smash, crash"],
    "click": ["Click", "Tick"], "pop": ["Pop", "Bouncing"], "typing": ["Typing", "Computer keyboard", "Typewriter"],
    "riser": ["Rumble", "Sine wave"], "ambience": ["Wind", "Rain", "Crowd", "Inside, large room or hall", "Outside, urban or manmade"],
}


def classify_event(f: dict) -> tuple[str, float]:
    d, att, low, high, flat, slope = f["duration_s"], f["attack_s"], f["low_ratio"], f["high_ratio"], f["flatness"], f["centroid_slope_hz_s"]
    if d < 0.07 and (high > 0.25 or f["centroid_hz"] > 2500):
        return "click", 0.6
    if att < 0.03 and low > 0.35 and d >= 0.06:
        return "impact", 0.6
    if d < 0.18 and flat < 0.3 and 400 < f["centroid_hz"] < 3500:
        return "pop", 0.5
    if 0.15 <= d <= 2.0 and flat > 0.08 and att >= 0.04:
        return "whoosh", 0.55
    if d > 1.0 and slope > 300:
        return "riser", 0.5
    return "transient", 0.3


def _event_features(S: np.ndarray, freqs: np.ndarray, db: np.ndarray, i0: int, i1: int, hop_s: float) -> dict:
    import librosa

    seg = S[:, i0:i1 + 1] + EPS
    tot = seg.sum(axis=0)
    cent = (freqs[:, None] * seg).sum(axis=0) / tot
    peak = i0 + int(np.argmax(db[i0:i1 + 1]))
    tt = np.arange(len(cent)) * hop_s
    return {
        "duration_s": (i1 - i0 + 1) * hop_s, "attack_s": (peak - i0) * hop_s, "peak_db": float(db[peak]),
        "centroid_hz": float(np.mean(cent)),
        "centroid_slope_hz_s": float(np.polyfit(tt, cent, 1)[0]) if len(tt) > 2 else 0.0,
        "low_ratio": float(seg[freqs < 200].sum() / tot.sum()), "high_ratio": float(seg[freqs > 4000].sum() / tot.sum()),
        "flatness": float(np.mean(librosa.feature.spectral_flatness(S=np.sqrt(seg)))),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
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


def beat_grid(beats: np.ndarray, duration: float) -> np.ndarray:
    """Regular beat grid (period = median IBI) covering the whole duration."""
    if len(beats) < 4:
        return beats
    period = float(np.median(np.diff(beats)))
    phase = float(np.median(np.mod(beats, period)))
    return np.arange(phase, duration + period, period)


def run_sfx(ctx: Context) -> dict:
    import librosa

    if not _has_audio(ctx):
        ctx.ws.write("audio/sfx.json", {"events": []})
        return {"outputs": ["audio/sfx.json"]}
    src, isolated = source_for(ctx, "effects")
    y, sr = media.load_audio(src)
    y = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    sr, hop = 22050, 256
    hop_s = hop / sr
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    db = 10 * np.log10(S.sum(axis=0) + EPS)
    oenv = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop)
    on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, backtrack=True, delta=0.2)
    floor = np.percentile(db, 20)
    beat_info = ctx.ws.read("audio/beats.json") or {}
    beats = np.asarray(beat_info.get("beats") or [])
    grid = beat_grid(beats, len(y) / sr)
    speech = speech_segments(ctx)
    events = []
    for k, i0 in enumerate(on):
        nxt = on[k + 1] if k + 1 < len(on) else len(db) - 1
        peak = i0 + int(np.argmax(db[i0:max(i0 + 1, min(nxt, i0 + int(2.0 / hop_s)))]))
        if db[peak] < floor + 10:
            continue
        i1 = peak
        while i1 + 1 < min(nxt, len(db)) and db[i1 + 1] > db[peak] - 18:
            i1 += 1
        f = _event_features(S, freqs, db, int(i0), int(i1), hop_s)
        cls, conf = classify_event(f)
        t0 = float(i0 * hop_s)
        on_beat = bool(len(grid) and np.min(np.abs(grid - t0)) < 0.045)
        in_speech = any(s["start"] - 0.05 <= t0 <= s["end"] for s in speech)
        if not isolated:
            # On the full mix, drum hits and syllables look like SFX: keep
            # them only when they stand out and lower the confidence.
            if (on_beat and cls in ("impact", "click", "transient")) or (in_speech and cls != "whoosh"):
                continue
            conf *= 0.6
        events.append({"start": t0, "end": float((i1 + 1) * hop_s), "type": cls, "confidence": conf,
                       "intensity_db": f["peak_db"] - floor, "features": f, "on_beat": on_beat})
    # Whooshes: broadband high-frequency noise swells. Detected separately
    # because their slow attack hides them from onset detection, and a
    # coincident drum hit would otherwise mask them.
    hf = freqs > 1500
    hf_db = 10 * np.log10(S[hf].sum(axis=0) + EPS)
    k = max(1, int(0.05 / hop_s))
    hf_db = np.convolve(hf_db, np.ones(k) / k, mode="same")
    hf_floor = np.percentile(hf_db, 20)
    flat_hf = librosa.feature.spectral_flatness(S=np.sqrt(S[hf] + EPS))[0]
    for a, b in _runs(hf_db > hf_floor + 12):
        d = (b - a + 1) * hop_s
        if not 0.15 <= d <= 2.5 or np.mean(flat_hf[a:b + 1]) < 0.05:
            continue
        t0, t1 = float(a * hop_s), float((b + 1) * hop_s)
        if any(e["type"] == "whoosh" and e["start"] < t1 and t0 < e["end"] for e in events):
            continue
        events = [e for e in events if not (e["type"] in ("transient", "impact") and t0 <= e["start"] <= t1
                                            and e["on_beat"])]
        f = _event_features(S, freqs, db, a, b, hop_s)
        events.append({"start": t0, "end": t1, "type": "whoosh", "confidence": 0.6 if isolated else 0.45,
                       "intensity_db": f["peak_db"] - floor, "features": f,
                       "on_beat": bool(len(grid) and np.min(np.abs(grid - t0)) < 0.045), "detector": "hf-noise swell"})
    events.sort(key=lambda e: e["start"])
    # Typing: runs of clicks 60-250 ms apart.
    clicks = [e for e in events if e["type"] == "click"]
    i = 0
    while i < len(clicks):
        j = i
        while j + 1 < len(clicks) and 0.06 <= clicks[j + 1]["start"] - clicks[j]["start"] <= 0.25:
            j += 1
        if j - i + 1 >= 4:
            for e in clicks[i:j + 1]:
                e["type"] = "typing"
                e["group"] = f"typing_{i}"
        i = j + 1
    # Risers: >=1 s of rising energy and brightness.
    t_s, e_s = rms_db(y, sr, 0.05, 0.2)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=int(0.05 * sr))[0][:len(e_s)]
    win = int(1.0 / 0.05)
    for a in range(0, len(e_s) - win, win // 2):
        es, cs = e_s[a:a + win], cent[a:a + win]
        if es[-1] - es[0] > 8 and np.polyfit(np.arange(win), es, 1)[0] > 0.3 and np.polyfit(np.arange(win), cs, 1)[0] > 20:
            if not any(e["type"] == "riser" and abs(e["end"] - t_s[a]) < 0.5 for e in events):
                events.append({"start": float(t_s[a]), "end": float(t_s[a + win - 1]), "type": "riser",
                               "confidence": 0.45 * (1 if isolated else 0.6), "intensity_db": float(es[-1] - es[0]),
                               "features": {}, "on_beat": False})
    events.sort(key=lambda e: e["start"])
    tagger = _panns_tag(y, sr, events)
    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    dur = _duration(ctx)
    ctx.ws.write("audio/sfx.json", {
        "source": "effects stem" if isolated else "original mix (no effects stem)",
        "events": events, "counts": counts, "per_minute": len(events) / (dur / 60) if dur else None,
        "tagger": tagger,
        "provenance": prov("librosa + heuristics" + (" + PANNs" if tagger else ""),
                           "spectral-flux onsets, envelope segmentation, rule-based class from attack/centroid/"
                           "flatness/band ratios", 0.55 if isolated else 0.35, INFERRED),
    }, ndigits=3)
    return {"outputs": ["audio/sfx.json"]}


def _panns_tag(y: np.ndarray, sr: int, events: list[dict]) -> str | None:
    """Optional AudioSet tagging of each event with PANNs (if installed)."""
    import importlib.util

    if not events or not importlib.util.find_spec("panns_inference"):
        return None
    import librosa
    from panns_inference import AudioTagging
    from panns_inference import labels as pl

    at = AudioTagging(checkpoint_path=None, device="cpu")
    y32 = librosa.resample(y, orig_sr=sr, target_sr=32000)
    for e in events:
        a = int(max(0, e["start"] - 0.1) * 32000)
        b = int(max(e["end"] + 0.1, e["start"] + 1.0) * 32000)
        clip, _ = at.inference(y32[a:b][None, :])
        probs = clip[0]
        top = np.argsort(probs)[::-1][:5]
        e["audioset_top"] = [{"label": pl[i], "p": float(probs[i])} for i in top]
        for cls, names in PANN_MAP.items():
            p = max((probs[pl.index(n)] for n in names if n in pl), default=0.0)
            if p > 0.3 and p > e["confidence"]:
                e["type"], e["confidence"] = cls, float(p)
    return "panns_inference Cnn14"


# ----------------------------------------------------------------- ducking
def run_ducking(ctx: Context) -> dict:
    if not _has_audio(ctx):
        ctx.ws.write("audio/ducking.json", {"measurable": False})
        return {"outputs": ["audio/ducking.json"]}
    s = stems(ctx)
    segs = [g for g in speech_segments(ctx) if g["end"] - g["start"] >= 0.8]
    music_src, isolated = source_for(ctx, "music")
    if not isolated or not segs:
        reason = "no isolated music stem" if not isolated else "no speech segments"
        ctx.ws.write("audio/ducking.json", {"measurable": False, "reason": reason,
                                            "note": "ducking needs separated music and dialogue layers"})
        return {"outputs": ["audio/ducking.json"], "notes": [f"ducking not measured: {reason}"]}
    hop = 0.02
    ym, sr = media.load_audio(music_src)
    t, mdb = rms_db(ym, sr, hop, 0.1)
    yx, srx = media.load_audio(ctx.ws.path("audio/original.wav"))
    _, xdb = rms_db(yx, srx, hop, 0.1)
    rdb = None
    if "dialogue" in s:  # cross-check: mix minus dialogue (independent of the music model)
        yd, _ = media.load_audio(s["dialogue"])
        n = min(len(yd), len(yx))
        _, rdb = rms_db(yx[:n] - yd[:n], srx, hop, 0.1)
    k = int(0.2 / hop)
    sm = np.convolve(mdb, np.ones(k) / k, mode="same")
    events = []
    for g in segs:
        pre = (t >= g["start"] - 1.5) & (t <= g["start"] - 0.15)
        if any(o is not g and o["end"] > g["start"] - 1.5 and o["start"] < g["start"] for o in segs):
            pre = (t >= g["start"] - 0.6) & (t <= g["start"] - 0.1)
        dur = (t >= g["start"] + 0.4) & (t <= g["end"] - 0.1)
        post = (t >= g["end"] + 0.8) & (t <= g["end"] + 2.0)
        if pre.sum() < 5 or dur.sum() < 5:
            continue
        pre_l, dur_l = float(np.median(sm[pre])), float(np.median(sm[dur]))
        att = pre_l - dur_l
        ev = {"speech_start": g["start"], "speech_end": g["end"], "music_pre_db": pre_l, "music_during_db": dur_l,
              "attenuation_db": att}
        if att > 2.0:
            target = pre_l - 0.9 * att
            after = np.where((t >= g["start"] - 0.3) & (sm <= target))[0]
            if len(after):
                ev["attack_s"] = float(t[after[0]] - g["start"])
            if post.sum() >= 5:
                rel_t = pre_l - 0.1 * att
                back = np.where((t >= g["end"]) & (sm >= rel_t))[0]
                if len(back):
                    ev["release_s"] = float(t[back[0]] - g["end"])
            if rdb is not None:
                rp, rd = float(np.median(rdb[pre])), float(np.median(rdb[dur]))
                ev["mix_minus_dialogue_attenuation_db"] = rp - rd
                ev["separation_artifact_suspected"] = bool(att - (rp - rd) > 4)
        ev["mix_pre_db"], ev["mix_during_db"] = float(np.median(xdb[pre])), float(np.median(xdb[dur]))
        events.append(ev)
    ducked = [e for e in events if e["attenuation_db"] > 2.0 and not e.get("separation_artifact_suspected")]
    summary = {
        "speech_segments_checked": len(events), "ducked_segments": len(ducked),
        "ducking_used": bool(events and len(ducked) / len(events) >= 0.5),
        "median_attenuation_db": float(np.median([e["attenuation_db"] for e in ducked])) if ducked else None,
        "median_attack_s": _med([e.get("attack_s") for e in ducked]),
        "median_release_s": _med([e.get("release_s") for e in ducked]),
    }
    ctx.ws.write("audio/ducking.json", {
        "measurable": True, "summary": summary, "events": events,
        "provenance": prov("numpy", "music-stem RMS before vs during speech; mix-minus-dialogue cross-check",
                           0.7, OBSERVED),
    })
    return {"outputs": ["audio/ducking.json"]}


def _med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None
