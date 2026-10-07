"""Configuration: defaults, TOML overrides, analysis modes and API keys.

Every external service is selected per task through ``[providers]`` so a
provider can be swapped (AssemblyAI -> Deepgram, Replicate -> fal, ...)
without touching the analysis code. ``auto`` picks the first provider whose
credentials are present, in the preference order below. Google is never an
``auto`` choice; it has to be named explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "analysis": {
        # Width (px) frames are decoded at for numeric analysis. Coordinates
        # are always reported normalised (0..1) and in source pixels.
        "analysis_width": 640,
        # Coarse grid for the spatio-temporal motion volume.
        "motion_grid": [96, 54],
        # 1 = every decoded frame enters numeric analysis.
        "frame_stride": 1,
        # OCR sampling: fixed interval plus change-triggered refreshes.
        "ocr_interval_s": 0.5,
        "ocr_change_threshold": 0.04,
        "ocr_width": 1280,
        "ocr_engine": "auto",  # rapidocr | paddleocr | tesseract | none
        "ocr_languages": ["en"],
        # Element tracking.
        "max_event_seconds": 8.0,
        "min_element_area": 0.0008,  # fraction of frame area
        "residual_threshold": 18,  # grey-level difference for motion masks
        # Scene detection.
        "scene_detector": "builtin",  # builtin | pyscenedetect
        "cut_threshold": 0.35,
        # Audio.
        "loudness_hop_ms": 20,
        "sync_tolerance_ms": 120,
        # Keyframes per scene sent to vision models.
        "llm_keyframes_per_scene": 2,
        "llm_max_images": 24,
    },
    "providers": {
        "separation": "auto",  # musicai | demucs | none
        "transcription": "auto",  # assemblyai | deepgram | faster_whisper | none
        "segmentation": "auto",  # replicate | fal | none
        "video_understanding": "none",  # twelvelabs | none
        "llm": "auto",  # anthropic | openai | google | none
    },
    "llm": {
        "anthropic_vision_model": "claude-sonnet-5-5",
        "anthropic_reasoning_model": "claude-sonnet-5-5",
        "anthropic_effort": "high",
        "openai_vision_model": "gpt-5.4-mini",
        "openai_reasoning_model": "gpt-5.4",
        "google_model": "gemini-2.5-pro",
        "max_output_tokens": 16000,
        "report_language": "tr",
    },
    "musicai": {
        "base_url": "https://api.music.ai/api",
        # Workflow slugs are created in the Music AI dashboard.
        "workflow_dme": "",
        "workflow_beats": "",
        "workflow_tagging": "",
        "stem_keys": {"dialogue": "dialogue", "music": "music", "effects": "effects"},
        "poll_seconds": 5,
        "timeout_seconds": 1800,
    },
    "assemblyai": {
        "base_url": "https://api.assemblyai.com",
        "speech_model": "",  # empty = provider default
        "language_code": "",  # empty = automatic language detection
        "speaker_labels": True,
    },
    "deepgram": {"base_url": "https://api.deepgram.com", "model": "nova-3", "language": ""},
    "faster_whisper": {"model": "small", "compute_type": "int8"},
    "replicate": {
        "base_url": "https://api.replicate.com/v1",
        "sam2_model": "meta/sam-2-video",
        "sam2_version": "",
        "max_elements": 6,
    },
    "fal": {"base_url": "https://queue.fal.run", "sam2_endpoint": "fal-ai/sam2/video", "max_elements": 6},
    "twelvelabs": {
        "base_url": "https://api.twelvelabs.io/v1.3",
        "index_name": "reference-analyzer",
        "model": "pegasus1.2",
    },
    "storage": {"backend": "local", "s3_bucket": "", "s3_prefix": "reference-analysis", "s3_endpoint": ""},
    # Public list prices used for cost accounting (USD unless noted).
    # Update when providers change their price lists.
    "pricing": {
        "musicai_dme_per_min": 0.05,
        "musicai_beats_per_min": 0.03,
        "musicai_tagging_per_min": 0.08,
        "assemblyai_per_hour": 0.21,
        "assemblyai_diarization_per_hour": 0.02,
        "deepgram_per_min": 0.0043,
        "twelvelabs_per_video_hour": 1.75,
        "twelvelabs_output_per_mtok": 7.50,
        "replicate_sam2_per_run": 0.039,
        "fal_sam2_per_run": 0.05,
        "anthropic": {
            "claude-sonnet-5-5": [2.0, 10.0],
            "claude-opus-5-5": [4.0, 20.0],
            "claude-haiku-5-5": [0.10, 0.50],
            "claude-sonnet-4-6": [3.0, 15.0],
        },
        "openai": {"gpt-5.4-mini": [0.75, 4.50], "gpt-5.4": [2.50, 15.0]},
    },
}

MODES: dict[str, dict[str, Any]] = {
    # Every frame, dense OCR, all configured providers.
    "full": {},
    # Faster pass for iteration: every 2nd frame, sparser OCR, no SAM 2.
    "quick": {
        "analysis": {"frame_stride": 2, "ocr_interval_s": 1.0, "analysis_width": 480},
        "providers": {"segmentation": "none", "video_understanding": "none"},
    },
}

API_KEY_ENV = {
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "musicai": ["MUSICAI_API_KEY", "MUSIC_AI_API_KEY"],
    "assemblyai": ["ASSEMBLYAI_API_KEY"],
    "deepgram": ["DEEPGRAM_API_KEY"],
    "replicate": ["REPLICATE_API_TOKEN"],
    "fal": ["FAL_KEY"],
    "twelvelabs": ["TWELVELABS_API_KEY", "TWELVE_LABS_API_KEY"],
}

# Preference order for ``auto``. Local fallbacks come last.
AUTO_ORDER = {
    "separation": ["musicai", "demucs", "none"],
    "transcription": ["assemblyai", "deepgram", "faster_whisper", "none"],
    "segmentation": ["replicate", "fal", "none"],
    "video_understanding": ["none"],
    "llm": ["anthropic", "openai", "none"],  # Google only when named explicitly
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def api_key(provider: str) -> str | None:
    for name in API_KEY_ENV.get(provider, []):
        val = os.environ.get(name)
        if val:
            return val
    return None


class Config:
    def __init__(self, data: dict[str, Any], mode: str = "full", offline: bool = False):
        self.data = data
        self.mode = mode
        self.offline = offline

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, mode: str = "full", offline: bool = False,
             overrides: dict[str, Any] | None = None) -> "Config":
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
        data = deep_merge(DEFAULTS, MODES[mode])
        candidates = [Path(path)] if path else [Path("refanalyzer.toml"), Path("config/refanalyzer.toml")]
        for cand in candidates:
            if cand.exists():
                with cand.open("rb") as fh:
                    data = deep_merge(data, tomllib.load(fh))
                break
            if path:
                raise FileNotFoundError(cand)
        if overrides:
            data = deep_merge(data, overrides)
        return cls(data, mode=mode, offline=offline)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def section_hash(self, *sections: str) -> str:
        """Stable hash of config sections, used for stage cache fingerprints."""
        payload = {s: self.get(s) for s in sections}
        payload["__offline"] = self.offline
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def provider(self, task: str) -> str:
        """Resolve the provider for a task, honouring ``auto`` and ``--offline``."""
        choice = self.get(f"providers.{task}", "none") or "none"
        candidates = AUTO_ORDER.get(task, ["none"]) if choice == "auto" else [choice]
        for cand in candidates:
            if cand == "none":
                return "none"
            if cand in API_KEY_ENV:
                if self.offline:
                    continue
                if api_key(cand) or (cand == "anthropic" and _anthropic_profile_available()):
                    return cand
                if choice != "auto":
                    raise RuntimeError(
                        f"provider {cand!r} selected for {task} but none of {API_KEY_ENV[cand]} is set"
                    )
                continue
            if _local_available(cand):
                return cand
            if choice != "auto":
                raise RuntimeError(f"local provider {cand!r} for {task} is not installed")
        return "none"


def _local_available(name: str) -> bool:
    import importlib.util

    module = {"demucs": "demucs", "faster_whisper": "faster_whisper"}.get(name)
    return bool(module and importlib.util.find_spec(module))


def _anthropic_profile_available() -> bool:
    # The Anthropic SDK can also authenticate from an `ant auth login` profile.
    return Path.home().joinpath(".config", "anthropic").exists()
