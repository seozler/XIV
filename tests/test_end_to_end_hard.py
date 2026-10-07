"""Camera pan, move, fade and dissolve against known ground truth (offline)."""

import pytest

import synth_hard as sh
from refanalyzer.cli import main
from refanalyzer.workspace import Library, Workspace


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("hard")
    video = sh.build(d / "src")
    assert main(["--out", str(d / "out"), "analyze", "--video", str(video), "--offline",
                 "--stages", "motion"]) == 0
    return Workspace.open(d / "out", Library(d / "out").list()[0]["video_id"])


def test_cuts_fade_dissolve(ws):
    sc = ws.read("metadata/scenes.json")
    assert [round(c["t"], 2) for c in sc["cuts"]] == pytest.approx(sh.TRUTH["cuts"], abs=0.04)
    fade = next(f for f in sc["fades"] if f["type"] == "fade_out_to_black")
    assert fade["start"] == pytest.approx(sh.TRUTH["fade"][0], abs=0.05)
    assert len(sc["dissolves"]) == 1
    d = sc["dissolves"][0]
    assert (d["start"], d["end"]) == pytest.approx(sh.TRUTH["dissolve"], abs=0.07)


def test_camera_pan(ws):
    moves = ws.read("visual/motion.json")["camera_moves"]
    pan = next(m for m in moves if m["type"] == "pan")
    assert pan["direction"] == "right"
    assert pan["total"] == pytest.approx(sh.TRUTH["pan"][2], rel=0.08)
    assert pan["start"] == pytest.approx(sh.TRUTH["pan"][0], abs=0.15)
    assert pan["end"] == pytest.approx(sh.TRUTH["pan"][1], abs=0.15)


def test_move_a_to_b(ws):
    els = ws.read("visual/objects.json")["elements"]
    mv = [e for e in els if e["role"] == "move"]
    assert len(mv) == 1
    pos = mv[0]["animation"]["curves"]["position"]
    assert mv[0]["animation"]["labels"] == ["move_right"]
    assert pos["inferred"]["easing"] in ("ease_out_quad", "ease_out_sine", "ease_out_cubic")
    assert (pos["inferred"]["start_s"], pos["inferred"]["end_s"]) == pytest.approx(sh.TRUTH["move"], abs=0.07)
