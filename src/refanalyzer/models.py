"""Shared data conventions: provenance records and JSON serialisation.

Every measurement the system writes carries a provenance block so that a
directly measured X position is never confused with an inferred easing
function. The fields mirror the spec:

    source_tool, confidence, observed_or_inferred, frame_range, method,
    evidence_reference
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

OBSERVED = "observed"
INFERRED = "inferred"


def prov(
    source_tool: str,
    method: str,
    confidence: float,
    kind: str = OBSERVED,
    frame_range: Iterable[int] | None = None,
    evidence: str | list[str] | None = None,
) -> dict[str, Any]:
    """Build a provenance block attached to a measurement or inference."""
    if kind not in (OBSERVED, INFERRED):
        raise ValueError(f"kind must be observed or inferred, got {kind!r}")
    out: dict[str, Any] = {
        "source_tool": source_tool,
        "method": method,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 3),
        "observed_or_inferred": kind,
    }
    if frame_range is not None:
        fr = list(frame_range)
        out["frame_range"] = [int(fr[0]), int(fr[-1])] if fr else None
    if evidence is not None:
        out["evidence_reference"] = evidence
    return out


def _clean(obj: Any, ndigits: int) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, ndigits) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _clean(dataclasses.asdict(obj), ndigits)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist(), ndigits)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def to_jsonable(obj: Any, ndigits: int = 4) -> Any:
    return _clean(obj, ndigits)


def write_json(path: str | os.PathLike, data: Any, ndigits: int = 4, indent: int | None = 2) -> Path:
    """Atomically write JSON (numpy-aware, NaN -> null, rounded floats)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(data, ndigits), ensure_ascii=False, indent=indent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
