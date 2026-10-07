"""Stage: report - reports/reference-analysis.md (human-readable).

Built only from stored JSON so it can be regenerated without re-analysis.
Measured values and inferred interpretations are labelled separately.
"""

from __future__ import annotations

from ..pipeline import Context


def _t(t) -> str:
    if t is None:
        return "–"
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:06.3f}"


def _n(v, fmt="{:.2f}", none="–"):
    if v is None:
        return none
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def build(ws) -> str:
    meta = ws.read("metadata/video.json") or {}
    p = ws.read("combined/profile.json") or {}
    sc = ws.read("metadata/scenes.json") or {}
    objs = (ws.read("visual/objects.json") or {}).get("elements", [])
    mo = ws.read("visual/motion.json") or {}
    ocr = ws.read("visual/ocr.json") or {}
    col = ws.read("visual/colors.json") or {}
    stems = ws.read("audio/stems.json") or {}
    beats = ws.read("audio/beats.json") or {}
    sfx = ws.read("audio/sfx.json") or {}
    duck = ws.read("audio/ducking.json") or {}
    sp = ws.read("audio/speech_style.json") or {}
    sync = ws.read("combined/synchronization.json") or {}
    design = ws.read("visual/design_llm.json") or {}
    narr = ws.read("audio/speech_style_llm.json") or {}
    sem = ws.read("metadata/video_semantics.json") or {}
    man = ws.manifest().get("stages", {})

    L = [f"# Referans Video Analizi — `{meta.get('input_file')}`", "",
         f"_video_id `{ws.video_id}` · sha256 `{(meta.get('sha256') or '')[:16]}…` · "
         f"{meta.get('width')}×{meta.get('height')} · {_n(meta.get('measured_fps'))} fps"
         f"{' (VFR)' if meta.get('variable_frame_rate') else ''} · {_n(meta.get('duration_s'))} s · "
         f"{meta.get('frame_count')} kare_", "",
         "> **Ölçülen** (observed) değerler doğrudan piksel/örneklerden gelir. **Tahmin** (inferred) değerler "
         "ölçümlerden türetilmiştir (easing ailesi, font kategorisi, SFX sınıfı…). Tüm kayıtların kaynağı "
         "`combined/evidence.json` içindedir.", ""]

    # --- editing
    st = sc.get("stats") or {}
    L += ["## 1. Kurgu ve sahne yapısı", "",
          f"- Sahne: **{st.get('scene_count', 0)}**, kesme: **{st.get('cut_count', 0)}** "
          f"({_n(st.get('cuts_per_minute'), '{:.1f}')}/dk), medyan sahne **{_n(st.get('median_scene_s'))} s** "
          f"(min {_n(st.get('min_scene_s'))}, max {_n(st.get('max_scene_s'))})",
          f"- Fade: {st.get('fade_count', 0)}, dissolve: {st.get('dissolve_count', 0)}", "",
          "| Sahne | Başlangıç | Bitiş | Süre (s) | Geçişler |", "|---|---|---|---|---|"]
    for s in sc.get("scenes", [])[:80]:
        L.append(f"| {s['id']} | {_t(s['start'])} | {_t(s['end'])} | {_n(s['duration_s'])} | "
                 f"{', '.join(s.get('transitions', [])) or s.get('in_transition')} |")
    L.append("")

    # --- motion
    ms = mo.get("summary") or {}
    L += ["## 2. Hareket ve animasyon", "",
          f"- İzlenen element: **{ms.get('element_count', 0)}** ({_n(ms.get('animated_elements_per_minute'), '{:.1f}')}/dk)",
          f"- Animasyon süresi (ms): medyan **{_n((ms.get('animation_duration_ms') or {}).get('median'), '{:.0f}')}**, "
          f"p10 {_n((ms.get('animation_duration_ms') or {}).get('p10'), '{:.0f}')}, p90 {_n((ms.get('animation_duration_ms') or {}).get('p90'), '{:.0f}')}",
          f"- Easing dağılımı (tahmin): {', '.join(f'{k}×{v}' for k, v in (ms.get('easing_histogram') or {}).items()) or '–'}",
          f"- Overshoot içeren eğri oranı: {_n(ms.get('share_with_overshoot'), '{:.0%}')}; "
          f"stagger aralığı medyan {_n((ms.get('stagger_intervals_ms') or {}).get('median'), '{:.0f}')} ms",
          f"- Parallax skoru: {_n(ms.get('parallax_score'))} ({ms.get('parallax_interpretation') or 'kamera hareketi yok/ölçülemedi'})", "",
          "| Element | Rol | Başlangıç | Etiketler | Özellik | Easing (tahmin) | Süre ms | cubic-bezier | Overshoot | Fit RMSE | Kanıt |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in objs[:120]:
        curves = e["animation"]["curves"]
        ev = e["provenance"]["samples"].get("evidence_reference", "")
        if not curves:
            L.append(f"| {e['id']} | {e['role']} | {_t(e['animation']['start_s'])} | {', '.join(e['animation']['labels'])} "
                     f"| – | – | – | – | – | – | `{ev}` |")
        for k, c in curves.items():
            L.append(f"| {e['id']} | {e['role']} | {_t(c['inferred']['start_s'])} | {', '.join(e['animation']['labels'])} | {k} | "
                     f"{c['inferred']['easing']} | {_n(c['inferred']['duration_ms'], '{:.0f}')} | "
                     f"`{tuple(c['inferred']['cubic_bezier'])}` | {_n(c['measured']['overshoot_ratio'], '{:.0%}')} | "
                     f"{_n(c['inferred']['fit_rmse'], '{:.3f}')} | `{ev}` |")
    if mo.get("camera_moves"):
        L += ["", "**Kamera hareketleri**", "", "| Tür | Yön | Başlangıç | Bitiş | Toplam | Easing |", "|---|---|---|---|---|---|"]
        for m in mo["camera_moves"]:
            L.append(f"| {m['type']} | {m['direction']} | {_t(m['start'])} | {_t(m['end'])} | {_n(m['total'], '{:.3f}')} {m['units']} | "
                     f"{(m.get('curve') or {}).get('inferred', {}).get('easing', '–')} |")
    L.append("")

    # --- typography
    ts = ocr.get("summary") or {}
    L += ["## 3. Metin ve tipografi", ""]
    if not ocr.get("engine"):
        L += ["OCR çalışmadı (motor kurulu değil).", ""]
    else:
        L += [f"- Metin izi: **{ts.get('text_tracks', 0)}** ({ocr.get('engine')}), animasyonlu metin oranı "
              f"{_n(ts.get('animated_text_share'), '{:.0%}')}, kademeli açılım: {ts.get('progressive_reveals', 0)}",
              f"- Font boyutu seviyeleri (px, tahmin): {', '.join(_n(l['size_px'], '{:.0f}') + '×' + str(l['count']) for l in ts.get('font_size_px_levels', [])) or '–'}",
              f"- Harf stili: {ts.get('case_styles')}, ağırlık (tahmin): {ts.get('weights')}, min. kontrast {_n(ts.get('min_contrast_ratio'), '{:.1f}')}:1", "",
              "| Metin | Görünme | Kaybolma | Boyut px (tahmin) | Renk | Kontrast | Ağırlık | Animasyon |", "|---|---|---|---|---|---|---|---|"]
        for t in ocr.get("tracks", [])[:80]:
            ty = t["typography"]
            L.append(f"| {t['content'][:40]} | {_t(t['start'])} | {_t(t['end'])} | {_n(ty.get('font_size_px_estimate'), '{:.0f}')} | "
                     f"{ty.get('text_color_hex', '–')} on {ty.get('background_color_hex', '–')} | {_n(ty.get('contrast_ratio'), '{:.1f}')} | "
                     f"{ty.get('weight_estimate', '–')} | {', '.join((t.get('animation') or {}).get('labels', [])) or '–'} |")
        L.append("")

    # --- colour
    cs = col.get("summary") or {}
    L += ["## 4. Renk ve kompozisyon", "",
          f"- Tema: **{cs.get('theme')}**, ortalama negatif alan {_n(cs.get('mean_negative_space'), '{:.0%}')}, "
          f"gradient arka plan oranı {_n(cs.get('gradient_background_share'), '{:.0%}')}",
          f"- Global palet: {' '.join('`' + c['hex'] + '`' for c in (cs.get('global_palette') or [])[:8])}", ""]

    # --- audio
    q = stems.get("quality") or {}
    L += ["## 5. Ses katmanları", "",
          f"- Ayrıştırma: **{stems.get('provider') or 'yok'}** ({stems.get('kind')}); eksik katmanlar: {', '.join(stems.get('missing_layers') or []) or '–'}; "
          f"yeniden birleştirme SNR {_n(q.get('reconstruction_snr_db'), '{:.1f}')} dB"]
    ml = (p.get("audio") or {}).get("mix_loudness") or {}
    L += [f"- Mix: entegre **{_n(ml.get('integrated_lufs'), '{:.1f}')} LUFS**, LRA {_n(ml.get('loudness_range_lu'), '{:.1f}')} LU, "
          f"true peak {_n(ml.get('true_peak_dbfs'), '{:.1f}')} dBFS"]
    for name, s in ((p.get("audio") or {}).get("layer_loudness") or {}).items():
        L.append(f"  - {name}: {_n((s or {}).get('integrated_lufs'), '{:.1f}')} LUFS")
    L += [f"- Müzik: **{_n(beats.get('tempo_bpm'), '{:.1f}')} BPM**, düzenlilik {_n(beats.get('beat_regularity'))}, "
          f"ton (tahmin) {(beats.get('key_estimate') or {}).get('key', '–')}, karakter (tahmin) {beats.get('character')}, "
          f"bölüm sınırları {', '.join(_t(x) for x in beats.get('sections') or []) or '–'} — kaynak: {beats.get('source')}",
          f"- SFX ({sfx.get('source')}): {sfx.get('counts')}"]
    for e in (sfx.get("events") or [])[:25]:
        L.append(f"  - {_t(e['start'])}–{_t(e['end'])} **{e['type']}** (güven {e['confidence']:.2f}, +{e['intensity_db']:.0f} dB)")
    if duck.get("measurable"):
        s = duck["summary"]
        L.append(f"- Ducking: kullanılıyor = **{s['ducking_used']}**, medyan zayıflatma {_n(s['median_attenuation_db'], '{:.1f}')} dB, "
                 f"attack {_n(s['median_attack_s'])} s, release {_n(s['median_release_s'])} s")
    else:
        L.append(f"- Ducking ölçülemedi: {duck.get('reason', '–')}")
    L.append("")

    # --- narration
    L += ["## 6. Konuşma ve anlatım", ""]
    if sp.get("available"):
        tm = sp["text_metrics"]
        pr = sp.get("prosody") or {}
        L += [f"- Dil: {sp.get('language')}, kelime {tm['word_count']}, **{_n(tm['words_per_minute'], '{:.0f}')} WPM** "
              f"(artikülasyon {_n(tm['articulation_rate_wpm'], '{:.0f}')})",
              f"- Duraklama: {tm['pauses']['count_over_300ms']} adet >300 ms, medyan {_n(tm['pauses']['p50_s'])} s, p90 {_n(tm['pauses']['p90_s'])} s",
              f"- Cümle: {tm['sentences']['count']} (ortalama {_n(tm['sentences']['words_mean'], '{:.1f}')} kelime), soru oranı {_n(tm['sentences']['question_share'], '{:.0%}')}",
              f"- Hitap: 2. tekil/çoğul {_n(tm['address']['second_person_per_100w'], '{:.1f}')}/100k, kapsayıcı 'biz' {_n(tm['address']['inclusive_we_per_100w'], '{:.1f}')}/100k",
              f"- Pitch: medyan {_n((pr.get('pitch') or {}).get('median_hz'), '{:.0f}')} Hz, aralık {_n((pr.get('pitch') or {}).get('range_semitones'), '{:.1f}')} st; "
              f"vurgu {_n((pr.get('emphasis') or {}).get('per_100w'), '{:.1f}')}/100k; cümle sonu {pr.get('sentence_final_contour')}",
              f"- Konuşmacılar: {sp.get('speakers') or '–'}"]
        prof = narr.get("profile") or {}
        if prof:
            L += [f"- **Anlatım profili (LLM, tahmin):** {prof.get('style_label')}"]
            L += [f"  - kural: {r}" for r in (prof.get("reusable_rules") or [])[:8]]
    else:
        L += [f"Kelime düzeyinde transkript yok. {sp.get('note', '')}"]
    L.append("")

    # --- sync
    L += ["## 7. Ses–görüntü senkronizasyonu", "", f"Tolerans ±{_n(sync.get('tolerance_ms'), '{:.0f}')} ms. "
          "Desen yalnızca isabet oranı rastlantı olasılığını anlamlı biçimde aştığında (binom testi p<0.05) işaretlenir.", "",
          "| Kural | Açıklama | Eşleşme | İsabet | Rastlantı | p | Medyan Δ ms | Desen |", "|---|---|---|---|---|---|---|---|"]
    for pt in sync.get("patterns", []):
        L.append(f"| `{pt['rule']}` | {pt['description']} | {pt['matched']}/{pt['visual_events']} | {_n(pt['hit_rate'], '{:.0%}')} | "
                 f"{_n(pt.get('chance_rate'), '{:.0%}')} | {_n(pt.get('p_value'), '{:.3f}')} | {_n((pt.get('delta_ms') or {}).get('median'), '{:+.0f}')} | "
                 f"{'✅' if pt.get('pattern_detected') else '–'} |")
    L.append("")

    # --- interpretation
    if design.get("scenes") or sem.get("text"):
        L += ["## 8. Yorum katmanı (LLM / video understanding — tahmin)", ""]
        for s in design.get("scenes", []):
            a = s.get("analysis") or {}
            if a:
                L.append(f"- **{s['scene']}**: {a.get('semantic_summary', '')} — stil: {(a.get('visual_style') or {}).get('category')}")
                for t in (a.get("typography") or [])[:4]:
                    L.append(f"  - '{t.get('text')}': {t.get('font_category')} ({', '.join(t.get('closest_fonts') or [])}), güven {t.get('confidence')}")
        if sem.get("text"):
            L += ["", "**Twelve Labs Pegasus:**", "", sem["text"][:4000]]
        L.append("")

    # --- reliability
    L += ["## 9. Güvenilirlik ve sınırlamalar", "",
          "- Orijinal CSS/GSAP/Three.js kodu, kesin font dosyası ve gerçek easing fonksiyonu render edilmiş videodan "
          "geri elde edilemez; easing alanları ölçülen eğriye en yakın matematiksel karşılıktır.",
          "- Ekran dışından giren elementlerde animasyon başlangıcı, elementin ilk görünür olduğu kareye göre raporlanır.",
          "- Separation çıktıları model tahminidir; katmanlar arası sızıntı olabilir. Ayrıştırma yoksa ses analizleri mix "
          "üzerinde ve düşük güvenle yapılır.",
          "- OCR anahtar karelerde çalışır; metin görünme zamanı, bağlı bir element animasyonu yoksa iki anahtar kare arasındadır.", ""]
    L += ["| Aşama | Durum | Süre (s) | Not |", "|---|---|---|---|"]
    for name, rec in man.items():
        L.append(f"| {name} | {rec.get('status')} | {rec.get('seconds', '–')} | {'; '.join(rec.get('notes') or []) or rec.get('error', '') or ''} |")
    costs = ws.costs()
    if costs:
        L += ["", f"**Bilinen API maliyeti:** ${costs.get('known_total_usd', 0):.4f} (CPU, depolama, ağ hariç; "
              f"fiyatı bilinmeyen kalem: {costs.get('unpriced_items', 0)})"]
    return "\n".join(L) + "\n"


def run(ctx: Context) -> dict:
    ctx.ws.path("reports/reference-analysis.md").write_text(build(ctx.ws), encoding="utf-8")
    return {"outputs": ["reports/reference-analysis.md"]}
