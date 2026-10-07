"""Second synthetic clip: camera pan, element move A->B, fade to black, dissolve.

Ground truth (30 fps, 1280x720, 7.4 s):
  0.0-2.5  textured background pans (content moves left by 400 px),
           ease-in-out-sine between 0.5 and 2.0 s  -> camera pan right
  2.5      hard cut
  3.0-3.6  white square moves from x=0.2W to x=0.6W, ease-out-quad
  4.5-5.0  fade to black, black hold until 5.4
  5.4      cut from black
  6.0-6.6  dissolve between two flat compositions
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, DUR = 1280, 720, 30, 7.4
TRUTH = {"cuts": [2.5, 5.4], "pan": (0.5, 2.0, 400 / W), "move": (3.0, 3.6), "fade": (4.5, 5.0),
         "dissolve": (6.0, 6.6)}


def _frame(t, tex):
    if t < 2.5:
        p = np.clip((t - 0.5) / 1.5, 0, 1)
        p = -(np.cos(np.pi * p) - 1) / 2
        x = int(round(200 + 400 * p))
        return tex[90:90 + H, x:x + W].copy()
    if t < 5.0:
        img = np.full((H, W, 3), (40, 120, 40), np.uint8)
        p = np.clip((t - 3.0) / 0.6, 0, 1)
        p = 1 - (1 - p) ** 2
        x = int(0.2 * W + 0.4 * W * p)
        cv2.rectangle(img, (x, 300), (x + 150, 450), (240, 240, 240), -1)
        if t >= 4.5:
            img = (img * max(0.0, 1 - (t - 4.5) / 0.5)).astype(np.uint8)
        return img
    if t < 5.4:
        return np.zeros((H, W, 3), np.uint8)
    a = np.full((H, W, 3), (200, 80, 30), np.uint8)
    cv2.circle(a, (400, 360), 150, (30, 200, 250), -1)
    b = np.full((H, W, 3), (30, 60, 200), np.uint8)
    cv2.rectangle(b, (700, 200), (1100, 500), (250, 250, 250), -1)
    w = float(np.clip((t - 6.0) / 0.6, 0, 1))
    return cv2.addWeighted(b, w, a, 1 - w, 0)


def build(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "synthetic_hard.mp4"
    if final.exists():
        return final
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur((rng.random((900, 2600, 3)) * 255).astype(np.uint8), (0, 0), 3)
    raw = out_dir / "raw_hard.avi"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(DUR * FPS)):
        vw.write(_frame(i / FPS, tex))
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-f", "lavfi", "-i", f"sine=f=440:d={DUR}",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)],
                   check=True)
    raw.unlink()
    return final
