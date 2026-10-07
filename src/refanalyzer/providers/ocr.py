"""OCR engine adapters (all CPU).

* ``rapidocr``  - PaddleOCR detection/recognition models via ONNX Runtime
                  (no Paddle install needed; default)
* ``paddleocr`` - the PaddleOCR package itself (2.x and 3.x APIs)
* ``tesseract`` - pytesseract + system tesseract

Every engine returns ``[{"text", "score", "box": [[x, y] * 4]}]`` in the
pixel coordinates of the image passed in.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


class OCREngine:
    name = "none"

    def detect(self, img: np.ndarray) -> list[dict[str, Any]]:
        raise NotImplementedError


class RapidOCREngine(OCREngine):
    name = "rapidocr"

    def __init__(self, languages: list[str]):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def detect(self, img):
        result, _ = self.engine(img)
        out = []
        for box, text, score in result or []:
            out.append({"text": str(text), "score": float(score), "box": [[float(x), float(y)] for x, y in box]})
        return out


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, languages: list[str]):
        from paddleocr import PaddleOCR

        lang = languages[0] if languages else "en"
        try:
            self.engine = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
            self.v3 = False
        except TypeError:
            self.engine = PaddleOCR(lang=lang, use_textline_orientation=False)
            self.v3 = True

    def detect(self, img):
        out = []
        if not self.v3:
            res = self.engine.ocr(img, cls=False)
            for line in (res[0] if res else None) or []:
                box, (text, score) = line
                out.append({"text": text, "score": float(score), "box": [[float(x), float(y)] for x, y in box]})
        else:
            for page in self.engine.predict(img):
                for text, score, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
                    out.append({"text": text, "score": float(score), "box": [[float(x), float(y)] for x, y in poly]})
        return out


class TesseractEngine(OCREngine):
    name = "tesseract"

    LANG = {"en": "eng", "tr": "tur", "de": "deu", "fr": "fra", "es": "spa"}

    def __init__(self, languages: list[str]):
        import pytesseract

        self.pt = pytesseract
        self.lang = "+".join(self.LANG.get(l, l) for l in (languages or ["en"]))

    def detect(self, img):
        d = self.pt.image_to_data(img, lang=self.lang, output_type=self.pt.Output.DICT)
        lines: dict[tuple, list[int]] = {}
        for i, txt in enumerate(d["text"]):
            if txt.strip() and float(d["conf"][i]) > 0:
                lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
        out = []
        for idxs in lines.values():
            x0 = min(d["left"][i] for i in idxs)
            y0 = min(d["top"][i] for i in idxs)
            x1 = max(d["left"][i] + d["width"][i] for i in idxs)
            y1 = max(d["top"][i] + d["height"][i] for i in idxs)
            out.append({"text": " ".join(d["text"][i] for i in idxs),
                        "score": float(np.mean([float(d["conf"][i]) for i in idxs]) / 100.0),
                        "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})
        return out


ENGINES = {"rapidocr": RapidOCREngine, "paddleocr": PaddleOCREngine, "tesseract": TesseractEngine}
MODULES = {"rapidocr": "rapidocr_onnxruntime", "paddleocr": "paddleocr", "tesseract": "pytesseract"}


def make_engine(choice: str, languages: list[str]) -> OCREngine | None:
    order = ["rapidocr", "paddleocr", "tesseract"] if choice == "auto" else [choice]
    for name in order:
        if name == "none":
            return None
        if importlib.util.find_spec(MODULES[name]):
            return ENGINES[name](languages)
        if choice != "auto":
            raise RuntimeError(f"OCR engine {name!r} is not installed (pip install {MODULES[name]})")
    return None
