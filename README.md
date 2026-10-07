# ONLEARN — Reference Video Reverse Analysis System

`reference-analyzer`, örnek alınan bir videoyu kare kare **ölçer**, ses katmanlarını ayrı ayrı inceler, bütün
bulguları tek bir zaman çizelgesinde birleştirir ve sonuçları mevcut video üretim sisteminizi anlatan
`current-video-system.txt` ile karşılaştırarak **kanıta dayalı geliştirme görevleri** üretir.

> Önce ölç. Sonra yorumla. Sonra mevcut sistemle karşılaştır. Son olarak somut teknik geliştirmelere dönüştür.

- Varsayılan çalışma: **normal CPU sunucusu + isteğe bağlı managed API'ler**. GPU kiralamak gerekmez.
- LLM varsayılanı Anthropic Claude'dur (OpenAI alternatif); **Google yalnızca açıkça seçilirse** kullanılır.
- Her sağlayıcı bir adapter arkasındadır (AssemblyAI ↔ Deepgram, Replicate ↔ fal, Music AI ↔ Demucs…).
- Hiçbir API anahtarı olmadan da (`--offline`) tam bir ölçüm + deterministik karşılaştırma raporu üretir.
- Sonuçlar video içeriğinin hash'ine göre saklanır; değişmeyen aşamalar tekrar çalışmaz.

## Hızlı başlangıç

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[full]"            # OpenCV, librosa, RapidOCR, Parselmouth, PySceneDetect, anthropic
reference-analyzer doctor           # ffmpeg, modüller, API anahtarları, seçilen sağlayıcılar

reference-analyzer analyze \
  --video reference.mp4 \
  --system current-video-system.txt \
  --mode full                       # veya: --mode quick, --offline
```

FFmpeg/FFprobe PATH üzerinde olmalıdır. API anahtarları yalnızca ortam değişkenlerinden okunur:

| Sağlayıcı | Değişken | Kullanım |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | görsel/anlatım yorumu, nihai karşılaştırma (varsayılan LLM) |
| OpenAI | `OPENAI_API_KEY` | alternatif LLM (GPT-5.4 / 5.4 Mini) |
| Music AI | `MUSICAI_API_KEY` | dialogue/music/effects ayrıştırma (workflow slug'ı config'te) |
| AssemblyAI | `ASSEMBLYAI_API_KEY` | transkripsiyon, kelime zamanları, konuşmacı ayrımı |
| Deepgram | `DEEPGRAM_API_KEY` | alternatif transkripsiyon |
| Replicate / fal | `REPLICATE_API_TOKEN` / `FAL_KEY` | SAM 2 video segmentasyonu |
| Twelve Labs | `TWELVELABS_API_KEY` | Pegasus video-semantik analiz (isteğe bağlı) |
| Google | `GEMINI_API_KEY` | yalnızca `providers.llm = "google"` ile |

Maliyeti çalıştırmadan önce görmek için: `reference-analyzer estimate --video reference.mp4`.

## Komutlar

| Komut | Açıklama |
|---|---|
| `analyze --video V [--system S] [--mode full\|quick] [--offline] [--stages a,b] [--force a,b] [--force-all]` | Tam analiz (bağımlılıklar otomatik). |
| `compare --id ref_xxx [--system yeni.txt]` | Analizi tekrarlamadan karşılaştırmayı yeni sistem açıklamasıyla yeniden üretir. |
| `estimate --video V` | Dakika bazlı API maliyet tahmini. |
| `status --id ref_xxx` | Aşama durumu, süreler, notlar. |
| `library` | Daha önce analiz edilmiş referanslar (yeniden kullanım indeksi). |
| `stages` | Aşama DAG'ı. |
| `export --id ref_xxx [--s3] [--postgres DSN] [--json dosya]` | R2/S3, PostgreSQL veya tek JSON paketi. |
| `doctor` | Ortam kontrolü. |

## Mimari

```
ingest ─► visual_scan ─► scenes ─► motion ─┬─► text ─────┐
   │                                       ├─► design ───┤
   │                                       └─► segmentation (SAM 2, yalnız işaretli elementler)
   └─► separation ─┬─► transcription ─┬─► speech_style ─┤
                   ├─► loudness       ├─► ducking ──────┤
                   └─► music ─────────┴─► sfx ──────────┴─► sync ─► timeline ─► semantics ─► compare ─► report
```

| Katman | Aşamalar | Çalıştığı yer |
|---|---|---|
| Deterministik ölçüm | ingest, visual_scan, scenes, motion, text (OCR), design, loudness, music, sfx, sync | CPU |
| AI analiz | separation, transcription, segmentation, (Pegasus) | Managed API (GPU sağlayıcıda) veya yerel yedek |
| Yorum & karşılaştırma | semantics, compare | LLM API |

Bağımsız aşamalar paralel çalışır (`--workers`). Her aşamanın parmak izi; kod sürümü, ilgili config bölümleri,
seçilen sağlayıcı, video hash'i (ve karşılaştırmada sistem açıklamasının hash'i) ile bağımlılıklarının parmak
izlerinden oluşur. Bir şey değişmedikçe aşama önbellekten gelir.

### Görsel analiz nasıl ölçülür?

- **Zaman:** Kare zamanları `frame_number / fps` ile değil, konteynerdeki gerçek **PTS** değerlerinden alınır (VFR desteği).
- **visual_scan (her kare):** parlaklık, kontrast, doygunluk, renklilik, kenar yoğunluğu, HSV histogram farkı;
  LK öznitelikleri + RANSAC ile **global kamera dönüşümü** (pan/tilt/zoom/roll); kamera telafili
  **artık hareket hacmi** (zaman × ızgara); OCR/tasarım için anahtar kareler (aralıklı, değişim tetiklemeli,
  hareket sonrası "oturmuş" kareler).
- **scenes:** uyarlamalı eşikli kesme tespiti (histogram + renk farkı sıçraması), siyah/beyaza fade, **dissolve**
  (karenin iki uç karenin doğrusal karışımı olarak açıklanması).
- **motion:** hareket hacminin 3B bağlantılı bileşenleri = animasyon olayları. Her olay için öncesi/sonrası kareler
  karşılaştırılır; element temiz arka plana (clean plate) göre kare kare izlenir: ekran dışından girişte kutu
  tamamlama, yerleşik boyuta göre ölçek, şablon regresyonuyla **opaklık**. Her özellik (konum/ölçek/opaklık) için
  süre, hız, ivme, overshoot, salınım ölçülür; **easing kütüphanesi + CSS cubic-bezier** en küçük kareler ile
  uydurulur (BIC ile seçim) ve CSS/GSAP/Remotion karşılıkları verilir. Kamera hareketleri ve parallax skoru da buradadır.
- **text:** RapidOCR (PaddleOCR modelleri, ONNX, CPU) / PaddleOCR / Tesseract; metin izleri, kademeli açılım
  (typewriter/kelime kelime), metin rengi ve arka planı, WCAG kontrastı, kontur kalınlığı → ağırlık tahmini,
  büyük/küçük harf stili, hizalama, satır aralığı, plaka/kontur ipuçları. Metin bir animasyonlu elementle
  çakışırsa o elementin ölçülmüş animasyonunu devralır.
- **design:** sahne başına Lab k-means palet, gradient (düzlem uydurma) ve radyal vinyet, negatif alan,
  görsel denge, kenar boşlukları, üçler kuralı ve ortak kenar hizalaması.

### Ses analizi

- **separation:** Music AI (Cinematic Stems iş akışı) → `dialogue/music/effects`; yoksa Demucs (yalnız
  vokal/eşlik) veya ayrıştırmasız mix. Stem'lerin mix'i yeniden oluşturma SNR'ı raporlanır.
- **transcription:** AssemblyAI / Deepgram / faster-whisper; hiçbiri yoksa yalnız konuşma etkinliği (VAD).
- **speech_style:** WPM, artikülasyon hızı, duraklama dağılımı, cümle uzunluğu, soru oranı, hitap (sen/siz/you),
  dolgu kelimeleri, söylem belirteçleri; Parselmouth ile F0, aralık (yarım ton), yoğunluk, jitter/shimmer/HNR,
  kelime vurgusu (pitch+yoğunluk+süre z-skoru), cümle sonu tonlaması.
- **loudness:** her katman için EBU R128 (M/S/I, LRA, true peak) ve 20 ms RMS eğrisi.
- **music:** tempo (IBI medyanı), beat'ler, 4/4 vuruş başı tahmini, onset'ler, enerji eğrisi, bölüm sınırları
  (novelty), ton tahmini, karakter.
- **sfx:** spektral akı onset'leri + zarf segmentasyonu + kural tabanlı sınıf (impact/click/pop/whoosh/riser/typing),
  ayrıca geniş bant gürültü kabarmaları için ayrı **whoosh** dedektörü; `panns_inference` kuruluysa AudioSet etiketleri.
- **ducking:** konuşma segmentlerinde müzik stem seviyesi öncesi/sırası, attack/release; mix − dialogue ile çapraz
  kontrol (ayrıştırma artefaktı şüphesi işaretlenir).

### Senkronizasyon

`sync` her kural için (kesme↔beat, giriş↔whoosh, metin↔söylenen kelime, giriş↔cümle başı, kesme↔duraklama …)
eşleşme oranını, işaretli Δ (ms) dağılımını ve **rastlantı tabanını** hesaplar; desen yalnızca isabet oranı
rastlantıyı anlamlı biçimde aşarsa (tek yönlü binom testi, p<0.05) işaretlenir.

## Çıktılar

```
reference-analysis/<ref_hash>/
  input/       reference.mp4, current-video-system.txt
  metadata/    video.json, frames.json (PTS), scenes.json, manifest.json, video_semantics.json
  visual/      frames/, evidence/ (kontak sayfaları, metin kırpıntıları), frame_metrics.json, keyframes.json,
               motion.json, objects.json, ocr.json, colors.json, segmentation.json, design_llm.json
  audio/       original.wav, dialogue.wav, music.wav, effects.wav, transcript.json, speech_style.json,
               speech_style_llm.json, loudness.json, beats.json, sfx.json, ducking.json, stems.json
  combined/    timeline.json, slices.json (0.5 s dilimler), synchronization.json, evidence.json, profile.json
  reports/     reference-analysis.md, system-comparison.md, improvement-tasks.json, gap-analysis.json, costs.json
```

Her ölçüm bir **provenance** bloğu taşır: `source_tool`, `method`, `confidence`, `observed_or_inferred`,
`frame_range`, `evidence_reference`. Örneğin bir elementin X konumu `observed`, easing ailesi `inferred`'dir.
`combined/evidence.json` tüm bu kayıtları tek listede toplar.

## Karşılaştırma nasıl çalışır?

1. **Deterministik boşluk analizi (her zaman):** ~30 yetenekten oluşan bir katalog (easing, spring/overshoot,
   stagger, kamera, parallax, geçişler, beat-senkron kurgu, animasyonlu metin, kelime-senkron metin, ducking,
   SFX, loudness, anlatım hızı, vurgu …). Her yetenek için referanstaki **ölçülmüş kanıt + zaman kodları** ile
   sistem açıklamasındaki anahtar kelime kanıtı (TR+EN, olumsuzluk algılamalı: "kullanılmıyor", "yok", "not
   supported") eşleştirilir; boşluklar öncelik skoruna göre sıralanır ve **ölçülebilir kabul kriterleri** üretilir
   (ör. "üretilen girişlerin medyan süresi 380–450 ms, easing fit = ease_out_cubic").
2. **LLM mimar incelemesi (anahtar varsa):** sistem açıklamasının tamamı, ölçülmüş profil, deterministik tablo ve
   yorum katmanı spesifikasyondaki karşılaştırma promptuyla modele verilir; rapor + JSON görev listesi döner.

Döngüyü kapatmak için: kendi sisteminizin ürettiği videoyu da aynı araçla analiz edin ve kabul kriterlerini
ölçülen değerlerle doğrulayın.

## GPU politikası

Öncelik: (1) CPU sunucusu → (2) managed API (GPU sağlayıcıda) → (3) kiralık GPU yalnızca başka yolla
ulaşılamayan, nihai raporu değiştirecek kadar daha iyi bir sonuç gerekiyorsa. SAM 2 yalnızca hareket
analizinin düşük güvenle izlediği elementler için ve yalnız ilgili klip gönderilerek çağrılır.

## Sınırlamalar

- Orijinal CSS/GSAP/Three.js kodu, kesin font dosyası ve gerçek easing fonksiyonu render edilmiş videodan geri
  elde edilemez; sistem görünür hareketi ölçer ve en yakın matematiksel karşılığı verir.
- Element takibi en iyi hareketli grafikte (sabit/yavaş arka plan) çalışır; hareketli kamera ve yoğun örtüşmede
  güven düşer ve element SAM 2 adayı olarak işaretlenir.
- Ayrıştırılmış stem'ler model tahminidir; katman sızıntısı olabilir. Ayrıştırma yoksa ses analizleri mix üzerinde
  düşük güvenle yapılır, ducking ölçülmez.
- OCR anahtar karelerde çalışır; metnin görünme anı, bağlı bir element animasyonu yoksa iki anahtar kare arasındadır.
- Harici API adapter'ları (Music AI, AssemblyAI, Deepgram, Replicate, fal, Twelve Labs, OpenAI, Gemini) sağlayıcıların
  yayımlanmış şemalarına göre yazıldı ancak bu depoda canlı anahtarla doğrulanmadı; ilk gerçek çalıştırmada
  `status` ve `manifest.json` çıktısını kontrol edin. Claude adapter'ı SDK'ya karşı sahte taşıma katmanıyla test edilir.
- Lisanslar: CoTracker (CC-BY-NC), openSMILE, bazı madmom modelleri ticari kullanıma uygun değildir ve bu projede
  kullanılmaz. Parselmouth GPLv3'tür — dağıtım şeklinize göre değerlendirin. Referans videoların kullanım hakları ve
  üçüncü taraf API'lerin veri saklama şartları ayrıca kontrol edilmelidir.

## Entegrasyon

- **BullMQ:** `integrations/bullmq/worker.mjs` — mevcut Node backend'i `reference-analysis` kuyruğuna iş ekler,
  worker CLI'ı çalıştırır, ilerlemeyi aşama bazında bildirir ve görev listesini döndürür.
- **PostgreSQL:** `reference-analyzer export --id ref_xxx --postgres "$DATABASE_URL"` (şema: `src/refanalyzer/schema.sql`).
- **R2/S3:** `storage.s3_bucket`, `storage.s3_endpoint` ayarlayıp `export --s3`.
- Yeni sağlayıcı eklemek: `src/refanalyzer/providers/` altına adapter yazıp `config.AUTO_ORDER` ve ilgili aşamaya bağlayın.

## Testler

```bash
pip install -e ".[full,dev]"
pytest -q
```

Testler bilinen doğruluk değerleri olan iki sentetik video üretir (ease-out-cubic kayma, doğrusal metin fade'i,
ease-out-back ölçek, 120 BPM beat, whoosh; kamera pan'ı, A→B hareket, siyaha fade, dissolve) ve tüm hattı
`--offline` çalıştırarak ölçümlerin doğruluğunu kontrol eder.

Ayar örneği: `config/refanalyzer.example.toml`. Örnek sistem açıklaması biçimi: `examples/current-video-system.example.txt`.
