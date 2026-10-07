"""Export analysis results: S3-compatible storage (R2), PostgreSQL, JSON bundle."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .workspace import Workspace

BUNDLE_FILES = ["metadata/video.json", "metadata/scenes.json", "visual/motion.json", "visual/objects.json",
                "visual/ocr.json", "visual/colors.json", "audio/beats.json", "audio/sfx.json", "audio/ducking.json",
                "audio/speech_style.json", "combined/profile.json", "combined/synchronization.json",
                "reports/improvement-tasks.json", "reports/gap-analysis.json", "reports/costs.json"]


def bundle(ws: Workspace) -> dict:
    return {"video_id": ws.video_id, **{f: ws.read(f) for f in BUNDLE_FILES}}


def to_s3(ws: Workspace, cfg: Config) -> int:
    """Upload the workspace (minus caches) to S3 or Cloudflare R2.

    Credentials come from the usual AWS environment variables; for R2 set
    ``storage.s3_endpoint`` to ``https://<account>.r2.cloudflarestorage.com``.
    """
    import boto3

    bucket = cfg.get("storage.s3_bucket")
    if not bucket:
        raise RuntimeError("storage.s3_bucket is not configured")
    s3 = boto3.client("s3", endpoint_url=cfg.get("storage.s3_endpoint") or None)
    prefix = cfg.get("storage.s3_prefix").strip("/")
    n = 0
    for p in ws.dir.rglob("*"):
        if p.is_file() and "cache" not in p.relative_to(ws.dir).parts:
            s3.upload_file(str(p), bucket, f"{prefix}/{ws.video_id}/{p.relative_to(ws.dir).as_posix()}")
            n += 1
    return n


def to_postgres(ws: Workspace, dsn: str) -> None:
    """Upsert the analysis into PostgreSQL (schema in schema.sql)."""
    import psycopg

    schema = (Path(__file__).parent / "schema.sql").read_text()
    meta = ws.read("metadata/video.json")
    profile = ws.read("combined/profile.json") or {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(schema)
        cur.execute("""INSERT INTO ref_videos (video_id, sha256, input_file, duration_s, width, height, fps, profile)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (video_id) DO UPDATE SET profile = EXCLUDED.profile, analysed_at = now()""",
                    (ws.video_id, meta["sha256"], meta["input_file"], meta["duration_s"], meta["width"], meta["height"],
                     meta.get("measured_fps"), json.dumps(profile)))
        for table in ("ref_scenes", "ref_elements", "ref_text_tracks", "ref_audio_events", "ref_words", "ref_sync_patterns",
                      "ref_tasks"):
            cur.execute(f"DELETE FROM {table} WHERE video_id = %s", (ws.video_id,))
        for s in (ws.read("metadata/scenes.json") or {}).get("scenes", []):
            cur.execute("INSERT INTO ref_scenes VALUES (%s,%s,%s,%s,%s)", (ws.video_id, s["id"], s["start"], s["end"], json.dumps(s)))
        for e in (ws.read("visual/objects.json") or {}).get("elements", []):
            cur.execute("INSERT INTO ref_elements VALUES (%s,%s,%s,%s,%s,%s)",
                        (ws.video_id, e["id"], e["role"], e["animation"]["start_s"], e["animation"]["end_s"], json.dumps(e)))
        for t in (ws.read("visual/ocr.json") or {}).get("tracks", []):
            cur.execute("INSERT INTO ref_text_tracks VALUES (%s,%s,%s,%s,%s,%s)",
                        (ws.video_id, t["id"], t["content"], t["start"], t["end"], json.dumps(t)))
        for e in (ws.read("audio/sfx.json") or {}).get("events", []):
            cur.execute("INSERT INTO ref_audio_events VALUES (%s,%s,%s,%s,%s)",
                        (ws.video_id, e["type"], e["start"], e["end"], json.dumps(e)))
        for w in (ws.read("audio/transcript.json") or {}).get("words", []):
            cur.execute("INSERT INTO ref_words VALUES (%s,%s,%s,%s,%s)", (ws.video_id, w["word"], w["start"], w["end"], w.get("speaker")))
        for p in (ws.read("combined/synchronization.json") or {}).get("patterns", []):
            cur.execute("INSERT INTO ref_sync_patterns VALUES (%s,%s,%s,%s,%s)",
                        (ws.video_id, p["rule"], p.get("hit_rate"), p.get("pattern_detected"), json.dumps(p)))
        for t in (ws.read("reports/improvement-tasks.json") or {}).get("tasks", []):
            cur.execute("INSERT INTO ref_tasks VALUES (%s,%s,%s,%s,%s,%s)",
                        (ws.video_id, t.get("id"), t.get("title"), t.get("subsystem"), t.get("priority"), json.dumps(t)))
