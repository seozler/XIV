from refanalyzer.analysis.compare import CATALOG, system_support

CAP = {c.id: c for c in CATALOG}


def test_turkish_negation_is_not_support():
    st = system_support("Animasyon: Spring kullanılmıyor.\nDucking yok.", CAP["spring_overshoot"])
    assert st["status"] == "missing"
    assert st["negated_mentions"]


def test_supported_with_multiple_keywords():
    text = "Animations use spring() physics with configurable damping and stiffness."
    assert system_support(text, CAP["spring_overshoot"])["status"] == "supported"


def test_single_keyword_partial_and_single_feature_support():
    assert system_support("Animations use easing.", CAP["easing_library"])["status"] == "partial"
    assert system_support("We use cubic-bezier curves.", CAP["easing_library"])["status"] == "supported"
    assert system_support("Arka plan müziği eklenir.", CAP["music_bed"])["status"] == "supported"


def test_missing_when_absent():
    assert system_support("Remotion renders scenes.", CAP["ducking"])["status"] == "missing"
