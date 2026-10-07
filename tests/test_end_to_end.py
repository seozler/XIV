"""Full pipeline on a synthetic video with known ground truth (offline)."""

import importlib.util

import pytest

import synth
from refanalyzer.cli import main
from refanalyzer.workspace import Library, Workspace


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    video = synth.build(d / "src")
    spec = d / "spec.txt"
    spec.write_text("Animation: linear interpolate() opacity fades only. Spring kullanılmıyor.\n"
                    "Music: fixed-level background music. Ses efekti desteği yok.\n", encoding="utf-8")
    rc = main(["--out", str(d / "out"), "analyze", "--video", str(video), "--system", str(spec), "--offline"])
    assert rc == 0
    vid = Library(d / "out").list()[0]["video_id"]
    return Workspace.open(d / "out", vid)


def test_cut_detected(analysed):
    cuts = analysed.read("metadata/scenes.json")["cuts"]
    assert len(cuts) == 1 and abs(cuts[0]["t"] - synth.TRUTH["cut"]) < 0.04


def test_box_slide_in(analysed):
    els = analysed.read("visual/objects.json")["elements"]
    box = next(e for e in els if "position" in e["animation"]["curves"])
    pos = box["animation"]["curves"]["position"]
    assert box["animation"]["labels"][0].startswith("slide_in_from_left")
    assert pos["inferred"]["easing"].startswith("ease_out")
    assert abs(pos["inferred"]["end_s"] - synth.TRUTH["box"]["end"]) < 0.07


def test_text_fade(analysed):
    els = analysed.read("visual/objects.json")["elements"]
    fade = next(e for e in els if "opacity" in e["animation"]["curves"])
    op = fade["animation"]["curves"]["opacity"]["inferred"]
    assert abs(op["start_s"] - synth.TRUTH["text"]["start"]) < 0.05
    assert abs(op["end_s"] - synth.TRUTH["text"]["end"]) < 0.07


def test_circle_overshoot(analysed):
    els = analysed.read("visual/objects.json")["elements"]
    circ = next(e for e in els if "scale" in e["animation"]["curves"])
    sc = circ["animation"]["curves"]["scale"]
    assert sc["measured"]["overshoot_ratio"] > 0.03
    assert "back" in sc["inferred"]["easing"] or "spring" in sc["inferred"]["easing"] or "elastic" in sc["inferred"]["easing"]


@pytest.mark.skipif(not importlib.util.find_spec("rapidocr_onnxruntime"), reason="OCR engine not installed")
def test_ocr_text(analysed):
    tracks = analysed.read("visual/ocr.json")["tracks"]
    assert any(t["content"].upper() == "GRAVITY" for t in tracks)


def test_tempo_and_whoosh(analysed):
    beats = analysed.read("audio/beats.json")
    assert abs(beats["tempo_bpm"] - synth.TRUTH["bpm"]) < 2.5
    sfx = analysed.read("audio/sfx.json")["events"]
    assert any(e["type"] == "whoosh" and abs(e["start"] - synth.TRUTH["whoosh"]) < 0.06 for e in sfx)


def test_outputs_and_comparison(analysed):
    for rel in ("combined/timeline.json", "combined/slices.json", "combined/evidence.json",
                "reports/reference-analysis.md", "reports/system-comparison.md", "reports/improvement-tasks.json"):
        assert analysed.path(rel).exists(), rel
    rows = {r["capability"]: r for r in analysed.read("reports/gap-analysis.json")["rows"]}
    assert rows["easing_library"]["gap"]
    assert rows["spring_overshoot"]["current_system"]["status"] == "missing"
    ev = analysed.read("combined/evidence.json")
    assert ev["count"] > 10
    assert {r["observed_or_inferred"] for r in ev["records"]} == {"observed", "inferred"}
