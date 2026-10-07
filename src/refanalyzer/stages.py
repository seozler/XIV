"""Registry of pipeline stages and their dependencies."""

from __future__ import annotations

from .analysis import compare, semantics, sync, timeline
from .audio import stages as audio
from .pipeline import Pipeline, Stage
from .report import markdown
from .video import design, ingest, motion, scan, scenes, segmentation, text

A = ("analysis",)


def build_pipeline() -> Pipeline:
    stages = [
        # -- visual
        Stage("ingest", ingest.run, (), A, description="metadata, PTS index, audio extraction"),
        Stage("visual_scan", scan.run, ("ingest",), A, description="per-frame metrics, camera, motion volume, keyframes"),
        Stage("scenes", scenes.run, ("visual_scan",), A, description="cuts, fades, dissolves"),
        Stage("motion", motion.run, ("scenes",), A, description="element animations, easing, camera moves"),
        Stage("text", text.run, ("motion",), A, description="OCR, text tracks, typography"),
        Stage("design", design.run, ("motion",), A, description="palette, gradients, composition"),
        # -- audio
        Stage("separation", audio.run_separation, ("ingest",), ("musicai", "pricing"),
              providers=("separation",), description="dialogue / music / effects stems"),
        Stage("transcription", audio.run_transcription, ("separation",),
              ("assemblyai", "deepgram", "faster_whisper", "pricing"), providers=("transcription",),
              description="words, timestamps, speakers"),
        Stage("loudness", audio.run_loudness, ("separation",), A, description="EBU R128 + RMS per layer"),
        Stage("music", audio.run_music, ("separation",), A, description="tempo, beats, sections, key, energy"),
        Stage("speech_style", audio.run_speech_style, ("transcription",), A, description="pace, pauses, prosody"),
        Stage("sfx", audio.run_sfx, ("music", "transcription"), A, description="sound-effect events"),
        Stage("ducking", audio.run_ducking, ("transcription",), A, description="music ducking under speech"),
        # -- advanced / multimodal
        Stage("segmentation", segmentation.run, ("motion",), ("replicate", "fal", "pricing"),
              providers=("segmentation",), description="SAM 2 masks for hard-to-track elements"),
        Stage("sync", sync.run, ("text", "design", "speech_style", "sfx", "ducking", "loudness"), A,
              description="audio-visual correlation patterns"),
        Stage("timeline", timeline.run, ("sync", "segmentation"), A, description="master timeline, slices, evidence"),
        # -- reasoning
        Stage("semantics", semantics.run, ("timeline",), ("analysis", "llm", "twelvelabs", "pricing"),
              providers=("llm", "video_understanding"), description="vision/narration LLM, Pegasus"),
        Stage("compare", compare.run, ("semantics",), ("llm", "pricing"), uses_system_spec=True,
              providers=("llm",), description="gap analysis vs current-video-system.txt"),
        Stage("report", markdown.run, ("compare",), ("llm",), description="reference-analysis.md"),
    ]
    return Pipeline(stages)
