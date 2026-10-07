"""Synthetic reference video with known ground truth for end-to-end tests.

Timeline (30 fps, 1280x720, 6.0 s):
  scene 1 (0.0-3.0 s): dark-blue background
    0.50-1.00 s  orange box slides in from the left, easeOutCubic,
                 x: -0.20 -> 0.15 (normalised left edge)
    1.50-1.90 s  text "GRAVITY" fades in (linear opacity)
  hard cut at 3.0 s
  scene 2 (3.0-6.0 s): light background
    3.50-4.10 s  green circle scales 0 -> 1 with easeOutBack overshoot
Audio: 120 BPM kick clicks (beat every 0.5 s), a noise whoosh at
0.45-0.95 s (box entrance) and a 220 Hz bed tone as "music".
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, DUR = 1280, 720, 30, 6.0
SR = 44100

TRUTH = {
    "cut": 3.0,
    "box": {"start": 0.5, "end": 1.0, "easing": "ease_out_cubic", "x0": -0.20, "x1": 0.15, "y": 0.30},
    "text": {"content": "GRAVITY", "start": 1.5, "end": 1.9},
    "circle": {"start": 3.5, "end": 4.1, "easing": "ease_out_back"},
    "bpm": 120.0,
    "whoosh": 0.45,
}


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def ease_out_back(p, s=1.70158):
    q = p - 1
    return 1 + (s + 1) * q ** 3 + s * q ** 2


def _progress(t, a, b):
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def render_frame(t: float) -> np.ndarray:
    if t < TRUTH["cut"]:
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (70, 30, 20)  # BGR dark blue
        b = TRUTH["box"]
        if t >= b["start"]:
            p = ease_out_cubic(_progress(t, b["start"], b["end"]))
            x = (b["x0"] + (b["x1"] - b["x0"]) * p) * W
            y = b["y"] * H
            cv2.rectangle(img, (int(round(x)), int(y)), (int(round(x + 0.18 * W)), int(y + 0.16 * H)),
                          (40, 140, 250), -1)
        tx = TRUTH["text"]
        if t >= tx["start"]:
            alpha = _progress(t, tx["start"], tx["end"])
            overlay = img.copy()
            cv2.putText(overlay, tx["content"], (int(0.42 * W), int(0.78 * H)), cv2.FONT_HERSHEY_DUPLEX,
                        3.0, (255, 255, 255), 6, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    else:
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (235, 235, 240)
        c = TRUTH["circle"]
        if t >= c["start"]:
            s = ease_out_back(_progress(t, c["start"], c["end"]))
            r = max(0, int(round(110 * s)))
            if r > 0:
                cv2.circle(img, (W // 2, H // 2), r, (60, 170, 60), -1, cv2.LINE_AA)
    return img


def render_audio() -> np.ndarray:
    n = int(DUR * SR)
    t = np.arange(n) / SR
    y = 0.08 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 330 * t)
    rng = np.random.default_rng(0)
    for beat in np.arange(0, DUR, 60.0 / TRUTH["bpm"]):
        i = int(beat * SR)
        k = np.arange(int(0.12 * SR))
        env = np.exp(-k / (0.025 * SR))
        seg = 0.6 * env * np.sin(2 * np.pi * 70 * k / SR)
        y[i:i + len(k)] += seg[: max(0, n - i)]
    a, b = int(TRUTH["whoosh"] * SR), int((TRUTH["whoosh"] + 0.5) * SR)
    noise = rng.standard_normal(b - a)
    sos = __import__("scipy.signal", fromlist=["butter"]).butter(4, [800, 6000], "bandpass", fs=SR, output="sos")
    noise = __import__("scipy.signal", fromlist=["sosfilt"]).sosfilt(sos, noise)
    env = np.sin(np.linspace(0, np.pi, b - a)) ** 2
    y[a:b] += 0.5 * noise / np.abs(noise).max() * env
    return np.clip(y, -1, 1)


def build(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.avi"
    wav = out_dir / "audio.wav"
    final = out_dir / "synthetic_reference.mp4"
    if final.exists():
        return final
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(DUR * FPS)):
        vw.write(render_frame(i / FPS))
    vw.release()
    pcm = (render_audio() * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(wav), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", str(final)], check=True)
    raw.unlink()
    return final


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "."))
