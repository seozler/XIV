from refanalyzer.config import Config
from refanalyzer.pipeline import Context, Pipeline, Stage
from refanalyzer.workspace import Workspace

CALLS = []


def a(ctx):
    CALLS.append("a")
    ctx.ws.write("cache/a.json", {"v": 1})


def b(ctx):
    CALLS.append("b")
    assert ctx.ws.read("cache/a.json")["v"] == 1


def boom(ctx):
    raise ValueError("nope")


def test_dag_cache_and_failure(tmp_path):
    ws = Workspace(tmp_path, "vid")
    ctx = Context(ws=ws, config=Config.load(offline=True), video_hash="h")
    p = Pipeline([Stage("a", a), Stage("b", b, ("a",)), Stage("c", boom, ("a",)), Stage("d", b, ("c",))])
    res = p.run(ctx)
    assert res["a"]["status"] == "done" and res["b"]["status"] == "done"
    assert res["c"]["status"] == "failed" and "nope" in res["c"]["error"]
    assert res["d"]["status"] == "skipped"
    CALLS.clear()
    res = p.run(ctx, targets=["b"])
    assert CALLS == [] and res["b"].get("cached")
    p.run(ctx, targets=["b"], force={"b"})
    assert CALLS == ["b"]
