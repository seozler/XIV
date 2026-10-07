"""FFmpeg / FFprobe helpers: probing, PTS-accurate frame decoding, audio IO.

Frame times are never derived from ``frame_number / fps``. Presentation
timestamps come from the container (packet PTS, sorted into presentation
order) so variable-frame-rate sources keep their real timing. All times are
relative to the container start so audio and video share one clock.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


class MediaError(RuntimeError):
    pass


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise MediaError(f"{binary} not found on PATH; install FFmpeg")
    return path


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise MediaError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])}...\n"
                         f"{proc.stderr.decode(errors='replace')[-2000:]}")
    return proc


@functools.lru_cache(maxsize=1)
def _fps_mode_flag() -> list[str]:
    out = subprocess.run([require("ffmpeg"), "-hide_banner", "-h", "full"], capture_output=True).stdout
    return ["-fps_mode", "passthrough"] if b"fps_mode" in out else ["-vsync", "0"]


def probe(path: str | Path) -> dict:
    proc = run([require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                "-show_streams", str(path)])
    return json.loads(proc.stdout)


def _ratio(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    avg_fps: float | None
    r_fps: float | None
    codec: str
    pix_fmt: str | None
    rotation: int
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    start_time: float
    bit_rate: int | None
    raw: dict

    @property
    def display_size(self) -> tuple[int, int]:
        if abs(self.rotation) % 180 == 90:
            return self.height, self.width
        return self.width, self.height


def video_info(path: str | Path) -> VideoInfo:
    data = probe(path)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise MediaError("no video stream found")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if "rotate" in (v.get("tags") or {}):
        rotation = int(v["tags"]["rotate"])
    duration = float(fmt.get("duration") or v.get("duration") or 0.0)
    return VideoInfo(
        width=int(v["width"]), height=int(v["height"]), duration=duration,
        avg_fps=_ratio(v.get("avg_frame_rate")), r_fps=_ratio(v.get("r_frame_rate")),
        codec=v.get("codec_name", "?"), pix_fmt=v.get("pix_fmt"), rotation=rotation,
        has_audio=a is not None, audio_codec=a.get("codec_name") if a else None,
        audio_sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        audio_channels=int(a["channels"]) if a and a.get("channels") else None,
        start_time=float(fmt.get("start_time") or 0.0),
        bit_rate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None, raw=data,
    )


def frame_timestamps(path: str | Path, start_time: float = 0.0) -> np.ndarray:
    """Presentation timestamps (s) of every video frame, container-relative.

    Uses packet PTS (no decode, fast) sorted into presentation order. If
    packets lack PTS the slower decoded-frame probe is used.
    """
    proc = run([require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                "packet=pts_time,flags", "-of", "csv=p=0", str(path)])
    pts: list[float] = []
    for line in proc.stdout.decode().splitlines():
        parts = line.strip().split(",")
        if not parts or parts[0] in ("", "N/A"):
            continue
        try:
            pts.append(float(parts[0]))
        except ValueError:
            continue
    if not pts:
        proc = run([require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)])
        pts = [float(x) for x in proc.stdout.decode().split() if x not in ("", "N/A")]
    arr = np.unique(np.asarray(pts, dtype=np.float64))  # sorts + drops duplicate PTS
    return arr - start_time


def analysis_size(info: VideoInfo, width: int) -> tuple[int, int]:
    w0, h0 = info.display_size
    width = min(width, w0)
    height = int(round(h0 * width / w0 / 2) * 2)
    return int(width // 2 * 2), max(2, height)


def iter_frames(path: str | Path, width: int, height: int, timestamps: np.ndarray | None = None,
                start: float | None = None, end: float | None = None,
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t, bgr_frame)`` for every decoded frame.

    Frames are decoded without rate conversion so frame ``i`` lines up with
    ``timestamps[i]``. ``start``/``end`` restrict the yielded range (the
    decoder still seeks from the nearest earlier keyframe).
    """
    cmd = [require("ffmpeg"), "-v", "error", "-nostdin"]
    first_index = 0
    if start is not None and timestamps is not None and start > 1.0:
        # Accurate seek: decode from 1s earlier and discard by timestamp.
        seek = max(0.0, start - 1.0)
        cmd += ["-ss", f"{seek:.6f}"]
        first_index = int(np.searchsorted(timestamps, seek - 1e-4))
    cmd += ["-i", str(path), "-map", "0:v:0", *_fps_mode_flag(),
            "-vf", f"scale={width}:{height}:flags=area", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
    idx = first_index
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if timestamps is not None and idx < len(timestamps):
                t = float(timestamps[idx])
            elif timestamps is not None and len(timestamps) > 1:
                t = float(timestamps[-1] + (idx - len(timestamps) + 1) * np.median(np.diff(timestamps)))
            else:
                t = float("nan")
            if end is not None and t > end + 1e-6:
                break
            if start is None or t >= start - 1e-6:
                yield idx, t, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()


def extract_audio(path: str | Path, out: str | Path, sample_rate: int, channels: int) -> Path:
    """Decode the first audio stream to PCM WAV, aligned to container t=0."""
    out = Path(out)
    run([require("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(path), "-map", "0:a:0",
         "-af", "aresample=async=1:first_pts=0", "-ac", str(channels), "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(out)])
    return out


def load_audio(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        data = data.mean(axis=1)
    return data, int(sr)


_EBUR_RE = re.compile(r"t:\s*([\d.]+)\s+TARGET:.*?M:\s*(-?[\d.inf]+)\s+S:\s*(-?[\d.inf]+)\s+I:\s*(-?[\d.]+)\s+LUFS")


def ebur128(path: str | Path) -> dict:
    """EBU R128 loudness: momentary/short-term curves (100 ms) and summary."""
    proc = subprocess.run([require("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path),
                           "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True)
    text = proc.stderr.decode(errors="replace")
    t, m, s = [], [], []
    for mt in _EBUR_RE.finditer(text):
        t.append(float(mt.group(1)))
        m.append(_lufs(mt.group(2)))
        s.append(_lufs(mt.group(3)))
    summary: dict[str, float | None] = {}
    tail = text[text.rfind("Summary:"):] if "Summary:" in text else ""
    for key, pat in {
        "integrated_lufs": r"I:\s*(-?[\d.]+)\s*LUFS",
        "loudness_range_lu": r"LRA:\s*(-?[\d.]+)\s*LU",
        "true_peak_dbfs": r"Peak:\s*(-?[\d.inf]+)\s*dBFS",
    }.items():
        mt = re.search(pat, tail)
        summary[key] = _lufs(mt.group(1)) if mt else None
    return {"t": t, "momentary_lufs": m, "short_term_lufs": s, "summary": summary}


def _lufs(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return None if not np.isfinite(v) or v < -120 else v


def extract_frames(path: str | Path, indices: list[int], width: int, out_dir: str | Path,
                   prefix: str = "hr") -> dict[int, Path]:
    """Decode selected frames (by decode index) as PNG at ``width``.

    FFmpeg still decodes the stream, but only the selected frames are
    scaled and written, which is far cheaper than piping every frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(int(i) for i in indices))
    result: dict[int, Path] = {}
    todo = [i for i in indices if not (out_dir / f"{prefix}_{i:06d}.png").exists()]
    for i in indices:
        if i not in todo:
            result[i] = out_dir / f"{prefix}_{i:06d}.png"
    for chunk_start in range(0, len(todo), 200):
        chunk = todo[chunk_start:chunk_start + 200]
        expr = "+".join(f"eq(n\\,{i})" for i in chunk)
        tmp = out_dir / f".tmp_{prefix}_%06d.png"
        run([require("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(path), "-map", "0:v:0",
             "-vf", f"select='{expr}',scale={width}:-2:flags=lanczos", *_fps_mode_flag(), str(tmp)])
        produced = sorted(out_dir.glob(f".tmp_{prefix}_*.png"))
        for i, p in zip(chunk, produced):
            dst = out_dir / f"{prefix}_{i:06d}.png"
            p.replace(dst)
            result[i] = dst
        for p in out_dir.glob(f".tmp_{prefix}_*.png"):
            p.unlink()
    return result
