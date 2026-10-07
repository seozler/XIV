"""Speech-to-text adapters with word timestamps and speaker labels.

All adapters return the same structure::

    {"provider", "model", "language", "text",
     "words": [{"word", "start", "end", "confidence", "speaker"}],
     "utterances": [{"speaker", "start", "end", "text"}]}
"""

from __future__ import annotations

from pathlib import Path

from ..config import api_key
from ..pipeline import Context
from .http import check, poll, session


def assemblyai(ctx: Context, audio: Path, duration_h: float) -> dict:
    cfg = ctx.config.get("assemblyai")
    s = session()
    s.headers["authorization"] = api_key("assemblyai")
    with audio.open("rb") as fh:
        up = check(s.post(f"{cfg['base_url']}/v2/upload", data=fh, timeout=600), "AssemblyAI upload")
    body = {"audio_url": up["upload_url"], "speaker_labels": bool(cfg["speaker_labels"]), "punctuate": True,
            "format_text": True}
    if cfg.get("speech_model"):
        body["speech_model"] = cfg["speech_model"]
    if cfg.get("language_code"):
        body["language_code"] = cfg["language_code"]
    else:
        body["language_detection"] = True
    job = check(s.post(f"{cfg['base_url']}/v2/transcript", json=body, timeout=60), "AssemblyAI submit")
    res = poll(lambda: check(s.get(f"{cfg['base_url']}/v2/transcript/{job['id']}", timeout=60), "AssemblyAI status"),
               lambda d: d.get("status") == "completed",
               lambda d: d.get("error") if d.get("status") == "error" else None, 3, 3600, "AssemblyAI transcript")
    price = ctx.config.get("pricing.assemblyai_per_hour")
    ctx.costs.add("assemblyai", "transcription", duration_h, "hour", price)
    if cfg["speaker_labels"]:
        ctx.costs.add("assemblyai", "speaker diarization", duration_h, "hour",
                      ctx.config.get("pricing.assemblyai_diarization_per_hour"))
    words = [{"word": w["text"], "start": w["start"] / 1000, "end": w["end"] / 1000,
              "confidence": w.get("confidence"), "speaker": w.get("speaker")} for w in res.get("words") or []]
    utts = [{"speaker": u.get("speaker"), "start": u["start"] / 1000, "end": u["end"] / 1000, "text": u["text"]}
            for u in res.get("utterances") or []]
    return {"provider": "assemblyai", "model": res.get("speech_model") or cfg.get("speech_model") or "default",
            "language": res.get("language_code"), "text": res.get("text", ""), "words": words, "utterances": utts}


def deepgram(ctx: Context, audio: Path, duration_min: float) -> dict:
    cfg = ctx.config.get("deepgram")
    s = session()
    s.headers.update({"Authorization": f"Token {api_key('deepgram')}", "Content-Type": "audio/wav"})
    params = {"model": cfg["model"], "smart_format": "true", "punctuate": "true", "diarize": "true", "utterances": "true"}
    if cfg.get("language"):
        params["language"] = cfg["language"]
    else:
        params["detect_language"] = "true"
    with audio.open("rb") as fh:
        res = check(s.post(f"{cfg['base_url']}/v1/listen", params=params, data=fh, timeout=1800), "Deepgram")
    ctx.costs.add("deepgram", f"transcription {cfg['model']}", duration_min, "minute", ctx.config.get("pricing.deepgram_per_min"))
    ch = res["results"]["channels"][0]
    alt = ch["alternatives"][0]
    words = [{"word": w.get("punctuated_word") or w["word"], "start": w["start"], "end": w["end"],
              "confidence": w.get("confidence"), "speaker": w.get("speaker")} for w in alt.get("words", [])]
    utts = [{"speaker": u.get("speaker"), "start": u["start"], "end": u["end"], "text": u["transcript"]}
            for u in res["results"].get("utterances") or []]
    return {"provider": "deepgram", "model": cfg["model"], "language": ch.get("detected_language") or cfg.get("language"),
            "text": alt.get("transcript", ""), "words": words, "utterances": utts}


def faster_whisper(ctx: Context, audio: Path) -> dict:
    from faster_whisper import WhisperModel

    cfg = ctx.config.get("faster_whisper")
    model = WhisperModel(cfg["model"], device="cpu", compute_type=cfg["compute_type"])
    segs, info = model.transcribe(str(audio), word_timestamps=True, vad_filter=True)
    words, utts, text = [], [], []
    for seg in segs:
        text.append(seg.text.strip())
        utts.append({"speaker": None, "start": seg.start, "end": seg.end, "text": seg.text.strip()})
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end, "confidence": w.probability, "speaker": None})
    return {"provider": "faster_whisper", "model": cfg["model"], "language": info.language, "text": " ".join(text),
            "words": words, "utterances": utts}
