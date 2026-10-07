"""Stage: ingest - metadata, PTS frame index and audio extraction."""

from __future__ import annotations

import numpy as np

from .. import media
from ..models import OBSERVED, prov
from ..pipeline import Context


def run(ctx: Context) -> dict:
    video = ctx.ws.video_path
    info = media.video_info(video)
    ts = media.frame_timestamps(video, info.start_time)
    if len(ts) == 0:
        raise media.MediaError("no video frames found")
    diffs = np.diff(ts) if len(ts) > 1 else np.array([1 / 30])
    median_dt = float(np.median(diffs))
    vfr = bool(len(diffs) > 2 and (np.std(diffs) / max(np.mean(diffs), 1e-9) > 0.05))
    dw, dh = info.display_size
    aw, ah = media.analysis_size(info, ctx.config.get("analysis.analysis_width"))

    notes = []
    if vfr:
        notes.append("variable frame rate detected: all timing uses real PTS")
    meta = {
        "video_id": ctx.ws.video_id,
        "sha256": ctx.video_hash,
        "input_file": video.name,
        "width": dw, "height": dh,
        "coded_width": info.width, "coded_height": info.height, "rotation": info.rotation,
        "duration_s": info.duration if info.duration > 0 else float(ts[-1] + median_dt),
        "frame_count": int(len(ts)),
        "avg_fps": info.avg_fps, "r_fps": info.r_fps, "measured_fps": 1.0 / median_dt if median_dt > 0 else None,
        "variable_frame_rate": vfr,
        "frame_interval": {"median_s": median_dt, "min_s": float(diffs.min()), "max_s": float(diffs.max())},
        "codec": info.codec, "pix_fmt": info.pix_fmt, "bit_rate": info.bit_rate,
        "container_start_time": info.start_time,
        "analysis_size": [aw, ah],
        "audio": {
            "present": info.has_audio, "codec": info.audio_codec,
            "sample_rate": info.audio_sample_rate, "channels": info.audio_channels,
        },
        "provenance": prov("ffprobe", "container/stream metadata + packet PTS", 0.99, OBSERVED),
    }
    ctx.ws.write("metadata/video.json", meta)
    ctx.ws.write("metadata/frames.json", {
        "timebase": "seconds (container-relative PTS)",
        "timestamps": ts.tolist(),
        "provenance": prov("ffprobe", "packet pts_time sorted to presentation order", 0.98, OBSERVED),
    }, ndigits=6, indent=None)

    outputs = ["metadata/video.json", "metadata/frames.json"]
    if info.has_audio:
        media.extract_audio(video, ctx.ws.path("audio/original.wav"), 44100, 2)
        media.extract_audio(video, ctx.ws.path("audio/mix_16k.wav"), 16000, 1)
        outputs += ["audio/original.wav", "audio/mix_16k.wav"]
    else:
        notes.append("no audio stream: audio stages will be skipped")
    return {"outputs": outputs, "notes": notes}
