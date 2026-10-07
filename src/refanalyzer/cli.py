"""Command-line interface.

    reference-analyzer analyze --video reference.mp4 --system current-video-system.txt --mode full
    reference-analyzer estimate --video reference.mp4
    reference-analyzer compare --id ref_xxxxxxxxxxxx --system new-spec.txt
    reference-analyzer status --id ref_xxxxxxxxxxxx
    reference-analyzer library
    reference-analyzer stages
    reference-analyzer doctor
    reference-analyzer export --id ref_xxxxxxxxxxxx --s3 | --postgres DSN
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import sys
from pathlib import Path

from . import __version__, media
from .config import API_KEY_ENV, AUTO_ORDER, Config, api_key
from .models import write_json
from .pipeline import Context
from .stages import build_pipeline
from .workspace import Library, Workspace, sha256_file

log = logging.getLogger("refanalyzer")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def _config(args) -> Config:
    return Config.load(args.config, mode=getattr(args, "mode", "full"), offline=getattr(args, "offline", False))


def _context(ws: Workspace, cfg: Config, video_hash: str) -> Context:
    spec = ws.system_spec_path
    return Context(ws=ws, config=cfg, video_hash=video_hash, system_hash=sha256_file(spec) if spec else None)


def _finish(ws: Workspace, results: dict, root: Path) -> int:
    ws.write("reports/costs.json", ws.costs())
    meta = ws.read("metadata/video.json") or {}
    profile = ws.read("combined/profile.json") or {}
    Library(root).register(ws.video_id, {
        "input_file": meta.get("input_file"), "duration_s": meta.get("duration_s"),
        "resolution": [meta.get("width"), meta.get("height")], "path": str(ws.dir),
        "stages": {k: v.get("status") for k, v in results.items()},
        "headline": {"cuts_per_minute": (profile.get("editing") or {}).get("cuts_per_minute"),
                     "elements": (profile.get("motion") or {}).get("element_count"),
                     "tempo_bpm": ((profile.get("audio") or {}).get("music") or {}).get("tempo_bpm"),
                     "wpm": (profile.get("narration") or {}).get("words_per_minute")},
    })
    failed = [k for k, v in results.items() if v.get("status") != "done"]
    print(f"\nWorkspace: {ws.dir}")
    for rel in ("reports/reference-analysis.md", "reports/system-comparison.md", "reports/improvement-tasks.json",
                "combined/timeline.json"):
        if ws.path(rel).exists():
            print(f"  {rel}")
    costs = ws.costs()
    print(f"Known API cost: ${costs['known_total_usd']:.4f} ({costs['unpriced_items']} unpriced items)")
    if failed:
        print("Stages not completed: " + ", ".join(f"{k} ({results[k].get('error') or results[k].get('reason')})" for k in failed))
        return 1
    return 0


def cmd_analyze(args) -> int:
    video = Path(args.video)
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    try:
        media.video_info(video)
    except media.MediaError as exc:
        print(f"not a readable video: {exc}", file=sys.stderr)
        return 2
    cfg = _config(args)
    root = Path(args.out)
    ws, digest = Workspace.for_video(root, video)
    ws.import_input(video)
    if args.system:
        spec = Path(args.system)
        if not spec.is_file():
            print(f"system description not found: {spec}", file=sys.stderr)
            return 2
        shutil.copyfile(spec, ws.path("input/current-video-system.txt"))
    log.info("workspace %s (mode=%s, offline=%s)", ws.dir, cfg.mode, cfg.offline)
    for task in AUTO_ORDER:
        try:
            log.info("provider %-20s -> %s", task, cfg.provider(task))
        except RuntimeError as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            return 2
    ctx = _context(ws, cfg, digest)
    targets = args.stages.split(",") if args.stages else None
    force = set(args.force.split(",")) if args.force else set()
    results = build_pipeline().run(ctx, targets=targets, force=force, force_all=args.force_all, workers=args.workers)
    return _finish(ws, results, root)


def cmd_compare(args) -> int:
    cfg = _config(args)
    root = Path(args.out)
    ws = Workspace.open(root, args.id)
    if args.system:
        shutil.copyfile(args.system, ws.path("input/current-video-system.txt"))
    ctx = _context(ws, cfg, ws.read("metadata/video.json")["sha256"])
    results = build_pipeline().run(ctx, targets=["report"], force={"compare", "report"}, workers=args.workers)
    return _finish(ws, results, root)


def cmd_estimate(args) -> int:
    cfg = _config(args)
    info = media.video_info(args.video)
    minutes = info.duration / 60
    pr = cfg["pricing"]
    lines = []
    total = 0.0

    def add(label, usd):
        nonlocal total
        total += usd
        lines.append((label, usd))

    sep = cfg.provider("separation")
    if sep == "musicai" and info.has_audio:
        add("Music AI separation", minutes * pr["musicai_dme_per_min"])
    stt = cfg.provider("transcription")
    if stt == "assemblyai" and info.has_audio:
        add("AssemblyAI transcription + diarization", minutes / 60 * (pr["assemblyai_per_hour"] + pr["assemblyai_diarization_per_hour"]))
    elif stt == "deepgram" and info.has_audio:
        add("Deepgram transcription", minutes * pr["deepgram_per_min"])
    if cfg.provider("video_understanding") == "twelvelabs":
        add("Twelve Labs Pegasus (+ output tokens)", minutes / 60 * pr["twelvelabs_per_video_hour"])
    llm = cfg.provider("llm")
    print(f"Video: {info.duration:.1f} s, {info.width}x{info.height}, audio={info.has_audio}")
    print(f"Providers: separation={sep}, transcription={stt}, llm={llm}, segmentation={cfg.provider('segmentation')}")
    for label, usd in lines:
        print(f"  {label:<45} ${usd:.4f}")
    print(f"  {'Known per-minute API total':<45} ${total:.4f}")
    print("Not included: LLM tokens (depends on scenes/images; capped by analysis.llm_max_images), SAM 2 runs "
          "(only for flagged elements), CPU time, storage, retries.")
    return 0


def cmd_status(args) -> int:
    ws = Workspace.open(args.out, args.id)
    man = ws.manifest()
    for name in build_pipeline().order:
        rec = man.get("stages", {}).get(name, {})
        print(f"{name:<14} {rec.get('status', '-'):<8} {str(rec.get('seconds', '')):>8}s  {'; '.join(rec.get('notes') or [])} {rec.get('error') or ''}")
    return 0


def cmd_library(args) -> int:
    for v in Library(args.out).list():
        h = v.get("headline") or {}
        print(f"{v['video_id']}  {v.get('input_file')}  {v.get('duration_s', 0):.1f}s  cuts/min={h.get('cuts_per_minute')} "
              f"elements={h.get('elements')} bpm={h.get('tempo_bpm')} wpm={h.get('wpm')}  {v.get('updated_at')}")
    return 0


def cmd_stages(args) -> int:
    p = build_pipeline()
    for name in p.order:
        s = p.stages[name]
        print(f"{name:<14} deps={','.join(s.deps) or '-':<45} {s.description}")
    return 0


def cmd_doctor(args) -> int:
    ok = True
    for b in ("ffmpeg", "ffprobe"):
        path = shutil.which(b)
        print(f"{b:<22} {'OK ' + path if path else 'MISSING'}")
        ok &= bool(path)
    for mod, purpose in [("cv2", "OpenCV (required)"), ("librosa", "audio analysis (required)"),
                         ("rapidocr_onnxruntime", "OCR (recommended)"), ("paddleocr", "OCR alternative"),
                         ("pytesseract", "OCR alternative"), ("parselmouth", "prosody (recommended)"),
                         ("scenedetect", "PySceneDetect cross-check"), ("anthropic", "Claude LLM"),
                         ("faster_whisper", "local STT fallback"), ("demucs", "local separation"),
                         ("panns_inference", "SFX tagging"), ("boto3", "S3/R2 export"), ("psycopg", "PostgreSQL export")]:
        print(f"{mod:<22} {'installed' if importlib.util.find_spec(mod) else '-'}  {purpose}")
    for prov, envs in API_KEY_ENV.items():
        print(f"key {prov:<18} {'set' if api_key(prov) else '-'}  ({' / '.join(envs)})")
    cfg = Config.load(args.config)
    for task in AUTO_ORDER:
        try:
            print(f"provider {task:<15} -> {cfg.provider(task)}")
        except RuntimeError as exc:
            print(f"provider {task:<15} -> ERROR {exc}")
            ok = False
    return 0 if ok else 1


def cmd_export(args) -> int:
    from . import export

    ws = Workspace.open(args.out, args.id)
    cfg = Config.load(args.config)
    if args.s3:
        n = export.to_s3(ws, cfg)
        print(f"uploaded {n} files")
    if args.postgres:
        export.to_postgres(ws, args.postgres)
        print("exported to PostgreSQL")
    if args.json:
        write_json(args.json, export.bundle(ws))
        print(f"wrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reference-analyzer", description="Reference Video Reverse Analysis System")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--config", help="TOML config (default: ./refanalyzer.toml if present)")
    ap.add_argument("--out", default="reference-analysis", help="analysis root directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse a reference video")
    a.add_argument("--video", required=True)
    a.add_argument("--system", help="current-video-system.txt describing the existing generator")
    a.add_argument("--mode", choices=["full", "quick"], default="full")
    a.add_argument("--offline", action="store_true", help="never call paid/network APIs")
    a.add_argument("--stages", help="comma-separated targets (dependencies run automatically)")
    a.add_argument("--force", help="comma-separated stages to re-run even if cached")
    a.add_argument("--force-all", action="store_true")
    a.add_argument("--workers", type=int, default=4)
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("compare", help="re-run the comparison for an analysed video (e.g. with a new spec)")
    c.add_argument("--id", required=True)
    c.add_argument("--system")
    c.add_argument("--offline", action="store_true")
    c.add_argument("--workers", type=int, default=2)
    c.set_defaults(func=cmd_compare)

    e = sub.add_parser("estimate", help="estimate per-minute API cost before running")
    e.add_argument("--video", required=True)
    e.add_argument("--mode", choices=["full", "quick"], default="full")
    e.add_argument("--offline", action="store_true")
    e.set_defaults(func=cmd_estimate)

    s = sub.add_parser("status", help="stage status of an analysis")
    s.add_argument("--id", required=True)
    s.set_defaults(func=cmd_status)

    sub.add_parser("library", help="list analysed references").set_defaults(func=cmd_library)
    sub.add_parser("stages", help="list pipeline stages").set_defaults(func=cmd_stages)
    sub.add_parser("doctor", help="check tools, optional modules and API keys").set_defaults(func=cmd_doctor)

    x = sub.add_parser("export", help="export results to object storage / PostgreSQL / single JSON")
    x.add_argument("--id", required=True)
    x.add_argument("--s3", action="store_true", help="upload workspace to S3/R2 (storage.* config)")
    x.add_argument("--postgres", metavar="DSN")
    x.add_argument("--json", metavar="FILE", help="write a single JSON bundle")
    x.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (FileNotFoundError, media.MediaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
