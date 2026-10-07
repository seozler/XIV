"""Stage DAG with content-addressed caching and parallel execution.

A stage is re-run only when its fingerprint changes. The fingerprint covers
the stage code version, the config sections it reads, the input video hash
(and system spec hash when relevant) and the fingerprints of its
dependencies. Independent stages run concurrently; API-bound stages
(separation, transcription, LLM calls) overlap with CPU-bound visual work.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .config import Config
from .workspace import CostTracker, Workspace

log = logging.getLogger("refanalyzer")


@dataclass
class Context:
    ws: Workspace
    config: Config
    video_hash: str
    system_hash: str | None = None
    costs: CostTracker = field(default_factory=CostTracker)
    stage: str = ""

    def log(self, msg: str, *args: Any) -> None:
        log.info(f"[{self.stage}] " + msg, *args)

    def warn(self, msg: str, *args: Any) -> None:
        log.warning(f"[{self.stage}] " + msg, *args)


@dataclass
class Stage:
    name: str
    func: Callable[[Context], dict[str, Any] | None]
    deps: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ("analysis",)
    version: str = "1"
    uses_system_spec: bool = False
    description: str = ""
    providers: tuple[str, ...] = ()  # resolved provider names join the fingerprint


class StageError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = {s.name: s for s in stages}
        for s in stages:
            for d in s.deps:
                if d not in self.stages:
                    raise ValueError(f"stage {s.name} depends on unknown stage {d}")
        self.order = self._toposort()

    def _toposort(self) -> list[str]:
        seen: dict[str, int] = {}
        order: list[str] = []

        def visit(n: str) -> None:
            state = seen.get(n, 0)
            if state == 1:
                raise ValueError(f"dependency cycle at {n}")
            if state == 2:
                return
            seen[n] = 1
            for d in self.stages[n].deps:
                visit(d)
            seen[n] = 2
            order.append(n)

        for n in self.stages:
            visit(n)
        return order

    def closure(self, targets: list[str] | None) -> list[str]:
        if not targets:
            return list(self.order)
        need: set[str] = set()

        def add(n: str) -> None:
            if n not in self.stages:
                raise ValueError(f"unknown stage {n!r}; known: {', '.join(self.order)}")
            if n in need:
                return
            need.add(n)
            for d in self.stages[n].deps:
                add(d)

        for t in targets:
            add(t)
        return [n for n in self.order if n in need]

    def fingerprint(self, stage: Stage, ctx: Context, dep_fps: dict[str, str]) -> str:
        h = hashlib.sha256()
        h.update(f"{stage.name}:{stage.version}:{_code_hash(stage.func)}".encode())
        h.update(ctx.video_hash.encode())
        h.update(ctx.config.section_hash(*stage.config_sections).encode())
        if stage.uses_system_spec:
            h.update((ctx.system_hash or "no-system-spec").encode())
        for task in stage.providers:
            try:
                h.update(f"{task}={ctx.config.provider(task)}".encode())
            except RuntimeError as exc:
                h.update(f"{task}=error:{exc}".encode())
        for d in sorted(stage.deps):
            h.update(dep_fps[d].encode())
        return h.hexdigest()[:20]

    def run(self, ctx: Context, targets: list[str] | None = None, force: set[str] | None = None,
            force_all: bool = False, workers: int = 4, keep_going: bool = True) -> dict[str, dict[str, Any]]:
        force = force or set()
        names = self.closure(targets)
        manifest = ctx.ws.manifest().get("stages", {})
        fps: dict[str, str] = {}
        results: dict[str, dict[str, Any]] = {}
        pending = list(names)
        running: dict[Future, str] = {}
        failed: set[str] = set()

        def ready(n: str) -> bool:
            return all(d in results for d in self.stages[n].deps)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            while pending or running:
                progressed = False
                for n in list(pending):
                    stage = self.stages[n]
                    if any(d in failed for d in stage.deps):
                        pending.remove(n)
                        failed.add(n)
                        results[n] = {"status": "skipped", "reason": "dependency failed"}
                        fps[n] = "skipped"
                        ctx.ws.update_manifest(n, {**results[n], "at": _now()})
                        progressed = True
                        continue
                    if not ready(n):
                        continue
                    pending.remove(n)
                    progressed = True
                    fp = self.fingerprint(stage, ctx, fps)
                    fps[n] = fp
                    prev = manifest.get(n, {})
                    if (not force_all and n not in force and prev.get("fingerprint") == fp
                            and prev.get("status") == "done"):
                        log.info("[%s] cached (fingerprint %s)", n, fp)
                        results[n] = {**prev, "cached": True}
                        continue
                    running[pool.submit(self._run_one, stage, ctx, fp)] = n
                if not running:
                    if pending and not progressed:
                        raise StageError(f"unschedulable stages: {pending}")
                    continue
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    n = running.pop(fut)
                    rec = fut.result()
                    results[n] = rec
                    if rec["status"] != "done":
                        failed.add(n)
                        if not keep_going:
                            raise StageError(f"stage {n} failed: {rec.get('error')}")
        return results

    def _run_one(self, stage: Stage, base: Context, fp: str) -> dict[str, Any]:
        ctx = Context(ws=base.ws, config=base.config, video_hash=base.video_hash,
                      system_hash=base.system_hash, costs=CostTracker(), stage=stage.name)
        t0 = time.time()
        ctx.log("start")
        try:
            out = stage.func(ctx) or {}
            rec = {"status": "done", "fingerprint": fp, "seconds": round(time.time() - t0, 2),
                   "outputs": out.get("outputs", []), "notes": out.get("notes", []),
                   "costs": ctx.costs.items, "at": _now()}
            ctx.log("done in %.1fs", rec["seconds"])
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest
            rec = {"status": "failed", "fingerprint": fp, "seconds": round(time.time() - t0, 2),
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8),
                   "costs": ctx.costs.items, "at": _now()}
            log.error("[%s] failed: %s", stage.name, rec["error"])
        base.ws.update_manifest(stage.name, rec)
        return rec


@functools.lru_cache(maxsize=None)
def _code_hash(func: Callable) -> str:
    """Hash of the stage's module source so code changes invalidate the cache."""
    try:
        src = inspect.getsource(inspect.getmodule(func))
    except (OSError, TypeError):
        return "nosrc"
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
