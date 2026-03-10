"""
=============================================================================
HAFTA 4 CUMARTESİ — UYGULAMA 01
Hugging Face pipeline() & AutoTokenizer Derinlemesi
=============================================================================
Kapsam:
  - 5 farklı pipeline görevi: duygu analizi, NER, soru-cevap,
    sıfır-atış sınıflandırma, metin üretimi
  - AutoTokenizer: WordPiece alt-kelime tokenizasyonu
  - Offset mapping ile karakter-token hizalaması
  - Özel tokenlar: [CLS] [SEP] [PAD] [MASK] [UNK]
  - Single vs batched tokenizasyon hız karşılaştırması
  - Güven skoru analizi & karar eşiği görselleştirmesi
  - Tokenizasyon adımları: ham metin → token ID'leri
  - Kapsamlı görselleştirme (7 grafik)
  - HuggingFace yoksa tam simülasyon modu

Kurulum: pip install transformers torch numpy matplotlib scikit-learn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import time
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

print("=" * 65)
print("  HAFTA 4 CUMARTESİ — Uygulama 01")
print("  Hugging Face pipeline() & AutoTokenizer")
print(f"  transformers : {'✅ yüklü' if HF_AVAILABLE else '❌  pip install transformers torch'}")
print(f"  Mod          : {'🤖 Gerçek Model' if HF_AVAILABLE else '🎭 Simülasyon'}")
print("=" * 65)

PALETTE = {
    "navy":   "#0E4D78",
    "blue":   "#1565C0",
    "cyan":   "#0891B2",
    "teal":   "#0D9488",
    "amber":  "#D97706",
    "green":  "#059669",
    "red":    "#DC2626",
    "slate":  "#1E293B",
    "gray":   "#64748B",
    "purple": "#7C3AED",
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1 — 5 PIPELINE GÖREVİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 1: Hugging Face pipeline() — 5 NLP Görevi")
print("═" * 55)

# ── Görev tanımları (hem gerçek hem simülasyon) ───────────────
GOREV_VERILERI = {
    "duygu": {
        "model":    "distilbert-base-uncased-finetuned-sst-2-english",
        "task":     "sentiment-analysis",
        "ornekler": [
            "This movie was absolutely wonderful and touching!",
            "I hated this product, complete waste of money.",
            "The service was okay, nothing special.",
            "Best experience I've ever had, truly outstanding!",
            "Very disappointing and frustrating experience.",
            "It works fine, meets my expectations.",
        ],
        "sim_yanit": [
            [{"label": "POSITIVE", "score": 0.9998}],
            [{"label": "NEGATIVE", "score": 0.9996}],
            [{"label": "POSITIVE", "score": 0.5843}],
            [{"label": "POSITIVE", "score": 0.9999}],
            [{"label": "NEGATIVE", "score": 0.9991}],
            [{"label": "POSITIVE", "score": 0.7234}],
        ],
    },
    "ner": {
        "model":    "dslim/bert-base-NER",
        "task":     "ner",
        "ornekler": [
            "Elon Musk founded SpaceX in California in 2002.",
            "Apple Inc. was created by Steve Jobs in Cupertino.",
            "The Eiffel Tower is located in Paris, France.",
        ],
        "sim_yanit": [
            [{"word": "Elon Musk", "entity_group": "PER", "score": 0.9987, "start": 0,  "end": 9},
             {"word": "SpaceX",   "entity_group": "ORG", "score": 0.9976, "start": 18, "end": 24},
             {"word": "California","entity_group":"LOC", "score": 0.9954, "start": 28, "end": 38}],
            [{"word": "Apple Inc.","entity_group": "ORG","score": 0.9992,"start": 0,"end": 10},
             {"word": "Steve Jobs","entity_group":"PER","score": 0.9981,"start": 22,"end": 32},
             {"word": "Cupertino", "entity_group":"LOC","score": 0.9968,"start": 36,"end": 45}],
            [{"word": "Eiffel Tower","entity_group":"LOC","score": 0.9834,"start": 4,"end": 16},
             {"word": "Paris",    "entity_group":"LOC","score": 0.9993,"start": 32,"end": 37},
             {"word": "France",   "entity_group":"LOC","score": 0.9989,"start": 39,"end": 45}],
        ],
    },
    "qa": {
        "model":    "deepset/roberta-base-squad2",
        "task":     "question-answering",
        "ornekler": [
            {"question": "Who created GPT-3?",
             "context":  "GPT-3 was developed by OpenAI and released in 2020. It has 175 billion parameters."},
            {"question": "How many parameters does BERT-large have?",
             "context":  "BERT was introduced by Google in 2018. BERT-base has 110M parameters while BERT-large has 340M parameters."},
            {"question": "What is fine-tuning?",
             "context":  "Fine-tuning is the process of adapting a pre-trained model to a specific task by continuing training on task-specific data with a small learning rate."},
        ],
        "sim_yanit": [
            {"answer": "OpenAI", "score": 0.9923, "start": 25, "end": 31},
            {"answer": "340M parameters", "score": 0.9876, "start": 89, "end": 103},
            {"answer": "adapting a pre-trained model to a specific task", "score": 0.8734, "start": 30, "end": 77},
        ],
    },
    "zs": {
        "model":    "facebook/bart-large-mnli",
        "task":     "zero-shot-classification",
        "ornekler": [
            {"text": "The stock market crashed and millions lost their savings.",
             "labels": ["economics", "sports", "technology", "politics"]},
            {"text": "The new transformer architecture achieves state-of-the-art results on NLP benchmarks.",
             "labels": ["artificial intelligence", "cooking", "history", "music"]},
            {"text": "The team scored three goals in the final minutes to win the championship.",
             "labels": ["sports", "science", "finance", "travel"]},
        ],
        "sim_yanit": [
            {"sequence": "...", "labels": ["economics","politics","technology","sports"],
             "scores": [0.8923, 0.0734, 0.0241, 0.0102]},
            {"sequence": "...", "labels": ["artificial intelligence","history","music","cooking"],
             "scores": [0.9876, 0.0067, 0.0043, 0.0014]},
            {"sequence": "...", "labels": ["sports","finance","travel","science"],
             "scores": [0.9734, 0.0134, 0.0089, 0.0043]},
        ],
    },
    "uretim": {
        "model":    "gpt2",
        "task":     "text-generation",
        "ornekler": [
            "Artificial intelligence is transforming",
            "The future of machine learning",
            "Deep learning models can",
        ],
        "sim_yanit": [
            [{"generated_text": "Artificial intelligence is transforming the way we work, making tasks faster and more accurate than ever before."}],
            [{"generated_text": "The future of machine learning lies in more efficient architectures and better data pipelines that can scale."}],
            [{"generated_text": "Deep learning models can recognize patterns in data that humans would never be able to detect on their own."}],
        ],
    },
}

gorev_sonuclari = {}
gorev_latency  = {}

for gorev_adi, gorev in GOREV_VERILERI.items():
    print(f"\n  [{gorev_adi.upper()}]  {gorev['model'][:50]}")
    t0 = time.time()

    if HF_AVAILABLE:
        try:
            if gorev_adi == "ner":
                pipe = pipeline(gorev["task"], model=gorev["model"],
                                aggregation_strategy="simple")
                sonuclar = [pipe(ornek) for ornek in gorev["ornekler"]]
            elif gorev_adi in ("duygu", "uretim"):
                pipe     = pipeline(gorev["task"], model=gorev["model"])
                sonuclar = [pipe(ornek) for ornek in gorev["ornekler"]]
            elif gorev_adi == "qa":
                pipe     = pipeline(gorev["task"], model=gorev["model"])
                sonuclar = [pipe(**ornek) for ornek in gorev["ornekler"]]
            elif gorev_adi == "zs":
                pipe     = pipeline(gorev["task"], model=gorev["model"])
                sonuclar = [pipe(ornek["text"],
                                 candidate_labels=ornek["labels"])
                            for ornek in gorev["ornekler"]]
        except Exception as e:
            print(f"    ⚠️  Model yüklenemedi ({e}), simülasyon kullanılıyor.")
            sonuclar = gorev["sim_yanit"]
    else:
        time.sleep(0.04)
        sonuclar = gorev["sim_yanit"]

    gorev_latency[gorev_adi]  = time.time() - t0
    gorev_sonuclari[gorev_adi] = sonuclar

    # Özet çıktı
    if gorev_adi == "duygu":
        for ornek, s in zip(gorev["ornekler"], sonuclar):
            r = s[0] if isinstance(s, list) else s
            emoji = "✅" if r["label"] == "POSITIVE" else "❌"
            print(f"    {emoji} [{r['score']:.4f}] {ornek[:52]}")
    elif gorev_adi == "ner":
        for i, (ornek, s) in enumerate(zip(gorev["ornekler"], sonuclar)):
            varlıklar = ", ".join(f"{e['word']}({e['entity_group']})" for e in s)
            print(f"    {i+1}. {varlıklar}")
    elif gorev_adi == "qa":
        for ornek, s in zip(gorev["ornekler"], sonuclar):
            print(f"    ❓ {ornek['question'][:45]:48} → '{s['answer']}' ({s['score']:.3f})")
    elif gorev_adi == "zs":
        for ornek, s in zip(gorev["ornekler"], sonuclar):
            print(f"    🏷️  {ornek['text'][:50]:53} → {s['labels'][0]} ({s['scores'][0]:.3f})")
    elif gorev_adi == "uretim":
        for ornek, s in zip(gorev["ornekler"], sonuclar):
            gen = s[0]["generated_text"] if isinstance(s, list) else s
            print(f"    ✍️  {gen[:80]}")

    print(f"    ⏱  {gorev_latency[gorev_adi]:.2f}s")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2 — OTOTOKENİZER DERİNLEMESİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 2: AutoTokenizer — WordPiece & Offset Mapping")
print("═" * 55)

MODEL_ADI = "bert-base-uncased"

if HF_AVAILABLE:
    print(f"  Model yükleniyor: {MODEL_ADI}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ADI)
    print(f"  Vocab boyutu: {tok.vocab_size:,}")
else:
    tok = None
    print(f"  [SİMÜLASYON] {MODEL_ADI}")

TEST_CUMLELER = [
    "Transformers are revolutionizing natural language processing.",
    "The quick brown fox jumps over the lazy dog.",
    "BERT uses WordPiece tokenization with a vocabulary of 30,522 tokens.",
    "Fine-tuning pre-trained models saves computational resources significantly.",
    "Attention mechanisms allow models to focus on relevant parts of input.",
]

def simulate_wordpiece(text):
    """WordPiece tokenizasyonunu simüle et."""
    specials   = ["[CLS]", "[SEP]"]
    words      = text.lower().split()
    tokens     = ["[CLS]"]
    offsets    = [(0, 0)]

    pos = 0
    for word in words:
        # Baştaki boşluğu atla
        while pos < len(text) and text[pos] == " ":
            pos += 1
        start = pos
        # Uzun kelimeler için basit alt-kelime bölme simülasyonu
        if len(word) > 8 and np.random.random() < 0.5:
            mid = len(word) // 2
            tokens.append(word[:mid])
            offsets.append((start, start + mid))
            tokens.append("##" + word[mid:])
            offsets.append((start + mid, start + len(word)))
        else:
            tokens.append(word.rstrip(".,!?"))
            offsets.append((start, start + len(word.rstrip(".,!?"))))
        pos += len(word) + 1

    tokens.append("[SEP]")
    offsets.append((len(text), len(text)))
    return tokens, offsets

tokenizasyon_sonuclari = []
print("\n  Tokenizasyon adımları:")

for cumle in TEST_CUMLELER:
    if HF_AVAILABLE and tok:
        enc     = tok(cumle, return_offsets_mapping=True, return_tensors="pt")
        token_ids  = enc["input_ids"][0].tolist()
        tokens     = tok.convert_ids_to_tokens(token_ids)
        offsets    = enc["offset_mapping"][0].tolist()
        attn_mask  = enc["attention_mask"][0].tolist()
    else:
        tokens, offsets = simulate_wordpiece(cumle)
        token_ids  = list(range(101, 101 + len(tokens)))
        attn_mask  = [1] * len(tokens)

    n_tokens     = len(tokens)
    n_subword    = sum(1 for t in tokens if t.startswith("##"))
    n_special    = sum(1 for t in tokens if t in ("[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"))

    tokenizasyon_sonuclari.append({
        "cumle":     cumle,
        "tokens":    tokens,
        "token_ids": token_ids,
        "offsets":   offsets,
        "attn_mask": attn_mask,
        "n_tokens":  n_tokens,
        "n_subword": n_subword,
        "n_special": n_special,
    })

    preview_tokens = tokens[:10]
    print(f"\n  Girdi: '{cumle[:55]}...'")
    print(f"    Token sayısı: {n_tokens}  |  Alt-kelime: {n_subword}  |  Özel: {n_special}")
    print(f"    Tokenlar : {preview_tokens}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3 — BATCH vs SINGLE TOKENİZASYON HIZ KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 3: Batch vs Single Tokenizasyon Hız Karşılaştırması")
print("═" * 55)

BATCH_BOYUTLARI = [1, 8, 32, 128, 512]
ORNEKLER        = [TEST_CUMLELER[i % len(TEST_CUMLELER)] for i in range(512)]

hiz_kayitlari = {"single": [], "batched": []}

for batch_n in BATCH_BOYUTLARI:
    subset = ORNEKLER[:batch_n]

    # Single (tek tek)
    t0 = time.time()
    if HF_AVAILABLE and tok:
        for s in subset:
            tok(s, truncation=True, max_length=128)
    else:
        for s in subset:
            simulate_wordpiece(s)
        time.sleep(batch_n * 0.0002)
    t_single = (time.time() - t0) * 1000  # ms

    # Batched
    t0 = time.time()
    if HF_AVAILABLE and tok:
        tok(subset, truncation=True, max_length=128, padding=True)
    else:
        for s in subset:
            simulate_wordpiece(s)
        time.sleep(batch_n * 0.00008)  # Daha hızlı
    t_batched = (time.time() - t0) * 1000  # ms

    hiz_kayitlari["single"].append(t_single)
    hiz_kayitlari["batched"].append(t_batched)
    speedup = t_single / max(t_batched, 0.001)
    print(f"    n={batch_n:4d}: Single={t_single:7.1f}ms  Batched={t_batched:7.1f}ms  Hızlanma={speedup:.1f}×")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4 — GÜVENİ SKORU & EŞİK ANALİZİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 4: Güven Skoru Analizi & Eşik Seçimi")
print("═" * 55)

GENIS_ORNEKLER = [
    ("This is an outstanding product, I love it!", 1),
    ("Absolutely terrible, never buying again.",    0),
    ("It's okay, not great but not bad either.",    1),  # Belirsiz
    ("Wonderful! Five stars all the way!",          1),
    ("I'm so disappointed with this purchase.",     0),
    ("The product works as expected.",              1),  # Düşük duygu
    ("Incredible value for money!",                 1),
    ("Poor quality, broke after one week.",         0),
    ("Not sure if I like it or not.",               1),  # Belirsiz
    ("Highly recommend to everyone!",               1),
    ("Worst experience of my life.",                0),
    ("It's fine, does what it's supposed to do.",   1),
    ("Surprisingly good for the price!",            1),
    ("Complete garbage, avoid at all costs.",       0),
    ("Average product, nothing special.",           1),  # Nötr
    ("Best purchase I've made all year!",           1),
]

if HF_AVAILABLE:
    try:
        duygu_pipe = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")
        esik_sonuclari = []
        for metin, gercek_etiket in GENIS_ORNEKLER:
            r = duygu_pipe(metin)[0]
            tahmin_skor  = r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]
            esik_sonuclari.append({
                "metin":        metin,
                "gercek":       gercek_etiket,
                "pos_skor":     tahmin_skor,
                "tahmin":       1 if r["label"] == "POSITIVE" else 0,
                "dogru":        (1 if r["label"] == "POSITIVE" else 0) == gercek_etiket,
            })
    except:
        HF_AVAILABLE = False

if not HF_AVAILABLE:
    # Simülasyon
    np.random.seed(42)
    esik_sonuclari = []
    for metin, gercek_etiket in GENIS_ORNEKLER:
        base = 0.85 if gercek_etiket == 1 else 0.15
        skor = np.clip(base + np.random.randn() * 0.12, 0.01, 0.99)
        esik_sonuclari.append({
            "metin":    metin,
            "gercek":   gercek_etiket,
            "pos_skor": skor,
            "tahmin":   1 if skor > 0.5 else 0,
            "dogru":    (1 if skor > 0.5 else 0) == gercek_etiket,
        })

# Farklı eşikler için F1 hesapla
from sklearn.metrics import f1_score, precision_score, recall_score

esikler  = np.linspace(0.1, 0.95, 50)
gercekler = np.array([r["gercek"] for r in esik_sonuclari])
skorlar   = np.array([r["pos_skor"] for r in esik_sonuclari])

f1_scores       = []
precision_scores = []
recall_scores    = []
for esik in esikler:
    tahminler = (skorlar >= esik).astype(int)
    f1_scores.append(f1_score(gercekler, tahminler, zero_division=0))
    precision_scores.append(precision_score(gercekler, tahminler, zero_division=0))
    recall_scores.append(recall_score(gercekler, tahminler, zero_division=0))

en_iyi_esik = esikler[np.argmax(f1_scores)]
en_iyi_f1   = max(f1_scores)
print(f"\n  En iyi eşik: {en_iyi_esik:.3f}  (F1={en_iyi_f1:.4f})")
print(f"  Varsayılan eşik (0.5): F1={f1_scores[np.argmin(np.abs(esikler-0.5))]:.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5 — ÖZEL TOKEN ANALİZİ & VOCABULARy İstatistikleri
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 5: Kelime Dağılımı & Token Uzunluk İstatistikleri")
print("═" * 55)

tum_tokenlar = []
for res in tokenizasyon_sonuclari:
    tum_tokenlar.extend([t for t in res["tokens"]
                         if t not in ("[CLS]", "[SEP]", "[PAD]")])

token_sayilari   = Counter(tum_tokenlar)
subword_sayisi   = sum(1 for t in tum_tokenlar if t.startswith("##"))
normal_sayisi    = sum(1 for t in tum_tokenlar if not t.startswith("##") and t not in ("[CLS]","[SEP]","[PAD]"))
token_uzunluklari = [res["n_tokens"] for res in tokenizasyon_sonuclari]

print(f"  Toplam token (özel hariç)  : {len(tum_tokenlar)}")
print(f"  Normal token               : {normal_sayisi}")
print(f"  Alt-kelime (##) token      : {subword_sayisi}")
print(f"  Alt-kelime oranı           : {subword_sayisi/max(len(tum_tokenlar),1):.2%}")
print(f"  Ort. token/cümle           : {np.mean(token_uzunluklari):.1f}")
print(f"  En sık 5 token: {token_sayilari.most_common(5)}")

# ─────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n  Görselleştirmeler oluşturuluyor...")

fig = plt.figure(figsize=(22, 20))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)
fig.suptitle("Hugging Face pipeline() & AutoTokenizer — Kapsamlı Analiz",
             fontsize=15, fontweight="bold")

colors_list = list(PALETTE.values())

# ── a. Pipeline görev gecikmesi ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
gorev_labels = list(gorev_latency.keys())
latencies    = [gorev_latency[g] for g in gorev_labels]
bars1 = ax1.bar(range(len(gorev_labels)), latencies,
                color=[colors_list[i] for i in range(len(gorev_labels))],
                alpha=0.88)
for bar, v in zip(bars1, latencies):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{v:.2f}s", ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(range(len(gorev_labels)))
ax1.set_xticklabels(["Duygu", "NER", "QA", "Zero-Shot", "Üretim"],
                    rotation=20, fontsize=10)
ax1.set_title("Pipeline Görev Gecikmeleri", fontweight="bold")
ax1.set_ylabel("Gecikme (saniye)")
ax1.grid(axis="y", alpha=0.3)

# ── b. Duygu analizi güven skorları ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
duygu_ornekler = GOREV_VERILERI["duygu"]["ornekler"]
duygu_sonuclar = gorev_sonuclari["duygu"]
duygu_skorlar  = []
duygu_etiketler = []
for s in duygu_sonuclar:
    r = s[0] if isinstance(s, list) else s
    duygu_skorlar.append(r["score"])
    duygu_etiketler.append(r["label"])

renk_duygu = [PALETTE["green"] if e == "POSITIVE" else PALETTE["red"]
              for e in duygu_etiketler]
bars2 = ax2.barh(range(len(duygu_skorlar)), duygu_skorlar,
                 color=renk_duygu, alpha=0.85)
for bar, v in zip(bars2, duygu_skorlar):
    ax2.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=9)
ax2.axvline(0.5, color="black", ls="--", lw=1.5, label="Eşik=0.5")
ax2.set_yticks(range(len(duygu_ornekler)))
ax2.set_yticklabels([f"Ör.{i+1}: {o[:22]}..." for i, o in enumerate(duygu_ornekler)],
                    fontsize=8)
ax2.set_title("Duygu Analizi — Güven Skorları", fontweight="bold")
ax2.set_xlabel("Güven Skoru")
ax2.legend(fontsize=9)
patch_pos = mpatches.Patch(color=PALETTE["green"], label="POSITIVE")
patch_neg = mpatches.Patch(color=PALETTE["red"],   label="NEGATIVE")
ax2.legend(handles=[patch_pos, patch_neg], fontsize=9)
ax2.grid(axis="x", alpha=0.3)

# ── c. Sıfır-atış sınıflandırma bar ─────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
zs_ornek = GOREV_VERILERI["zs"]["ornekler"][0]
zs_sonuc  = gorev_sonuclari["zs"][0]
labels_zs = zs_sonuc.get("labels", zs_ornek["labels"])
scores_zs = zs_sonuc.get("scores", [0.7, 0.15, 0.1, 0.05])
sort_idx  = np.argsort(scores_zs)[::-1]
labels_s  = [labels_zs[i] for i in sort_idx]
scores_s  = [scores_zs[i] for i in sort_idx]

bars3 = ax3.bar(range(len(labels_s)), scores_s,
                color=[colors_list[i] for i in range(len(labels_s))],
                alpha=0.85)
for bar, v in zip(bars3, scores_s):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{v:.3f}", ha="center", fontsize=11)
ax3.set_xticks(range(len(labels_s)))
ax3.set_xticklabels([l[:12] for l in labels_s], rotation=15, fontsize=10)
ax3.set_title("Zero-Shot Sınıflandırma\n(Borsa çöküşü haberi)", fontweight="bold")
ax3.set_ylabel("Olasılık")
ax3.grid(axis="y", alpha=0.3)
ax3.set_ylim(0, 1.1)

# ── d. Batch vs Single tokenizasyon ──────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(BATCH_BOYUTLARI, hiz_kayitlari["single"],  "o-", lw=2.5,
         color=PALETTE["red"],  label="Single (tek tek)", markersize=9)
ax4.plot(BATCH_BOYUTLARI, hiz_kayitlari["batched"], "s-", lw=2.5,
         color=PALETTE["green"], label="Batched", markersize=9)
ax4.fill_between(BATCH_BOYUTLARI, hiz_kayitlari["single"],
                 hiz_kayitlari["batched"], alpha=0.15, color=PALETTE["green"],
                 label="Zaman tasarrufu")
ax4.set_xscale("log")
ax4.set_xlabel("Batch Boyutu (log ölçeği)")
ax4.set_ylabel("Süre (ms)")
ax4.set_title("Batch vs Single Tokenizasyon Hızı", fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.set_xticks(BATCH_BOYUTLARI)
ax4.set_xticklabels([str(b) for b in BATCH_BOYUTLARI])

# ── e. Eşik optimizasyonu — F1/Precision/Recall ──────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(esikler, f1_scores,        lw=2.5, color=PALETTE["blue"],
         label="F1 Skoru")
ax5.plot(esikler, precision_scores, lw=2.5, color=PALETTE["green"],
         label="Precision", ls="--")
ax5.plot(esikler, recall_scores,    lw=2.5, color=PALETTE["red"],
         label="Recall", ls="-.")
ax5.axvline(en_iyi_esik, color=PALETTE["amber"], ls="--", lw=2,
            label=f"Opt. Eşik={en_iyi_esik:.2f}")
ax5.axvline(0.5, color="gray", ls=":", lw=1.5, label="Default=0.50")
ax5.scatter([en_iyi_esik], [en_iyi_f1], s=120,
            color=PALETTE["amber"], zorder=6)
ax5.set_xlabel("Karar Eşiği")
ax5.set_ylabel("Skor")
ax5.set_title("Eşik Optimizasyonu — F1 / Precision / Recall", fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)
ax5.set_xlim(0.1, 0.95)

# ── f. Token uzunluk dağılımı ────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
tum_token_uzunluklar = [len(r["tokens"]) for r in tokenizasyon_sonuclari]
ax6.bar(range(len(tum_token_uzunluklar)), tum_token_uzunluklar,
        color=[colors_list[i % len(colors_list)] for i in range(len(tum_token_uzunluklar))],
        alpha=0.85)
ax6.axhline(np.mean(tum_token_uzunluklar), color=PALETTE["amber"],
            ls="--", lw=2, label=f"Ort={np.mean(tum_token_uzunluklar):.1f}")
ax6.set_xticks(range(len(TEST_CUMLELER)))
ax6.set_xticklabels([f"C{i+1}" for i in range(len(TEST_CUMLELER))])
ax6.set_title("Token Uzunluğu / Cümle", fontweight="bold")
ax6.set_ylabel("Token Sayısı")
ax6.legend(fontsize=10)
ax6.grid(axis="y", alpha=0.3)

# ── g. WordPiece tokenizasyon görsel ─────────────────────────
ax7 = fig.add_subplot(gs[2, :])
ax7.axis("off")
ax7.set_title("WordPiece Tokenizasyon Adımları — Örnek Görselleştirme",
              fontweight="bold", fontsize=12)

ornek_res = tokenizasyon_sonuclari[0]
tokens_gos = ornek_res["tokens"][:16]   # İlk 16 token
token_ids_gos = ornek_res["token_ids"][:16]

# Her token için renkli kutu
OZEL_RENKLER = {
    "[CLS]": PALETTE["navy"],
    "[SEP]": PALETTE["navy"],
    "[PAD]": PALETTE["gray"],
    "[MASK]": PALETTE["amber"],
    "[UNK]": PALETTE["red"],
}

n_tok = len(tokens_gos)
box_w = min(0.9 / n_tok, 0.055)
box_h = 0.28
start_x = (1.0 - n_tok * (box_w + 0.008)) / 2

for i, (tok_str, tid) in enumerate(zip(tokens_gos, token_ids_gos)):
    x = start_x + i * (box_w + 0.008)
    color = OZEL_RENKLER.get(tok_str, PALETTE["purple"] if tok_str.startswith("##")
                              else PALETTE["teal"])
    ax7.add_patch(mpatches.FancyBboxPatch(
        (x, 0.52), box_w, box_h,
        boxstyle="round,pad=0.01", facecolor=color,
        edgecolor="white", transform=ax7.transAxes, linewidth=1.5
    ))
    tok_disp = tok_str[:7] if len(tok_str) > 7 else tok_str
    ax7.text(x + box_w/2, 0.68, tok_disp, transform=ax7.transAxes,
             fontsize=max(6.5, 8 - n_tok * 0.15),
             color="white", ha="center", va="center", fontweight="bold")
    ax7.text(x + box_w/2, 0.54, str(tid), transform=ax7.transAxes,
             fontsize=max(6, 7 - n_tok * 0.1),
             color="#E2E8F0", ha="center", va="center")

ax7.text(0.5, 0.88,
         f"Girdi: \"{ornek_res['cumle'][:70]}...\"",
         transform=ax7.transAxes, fontsize=11, ha="center",
         color=PALETTE["slate"])
ax7.text(0.5, 0.38,
         f"Toplam: {ornek_res['n_tokens']} token  |  "
         f"Alt-kelime (##): {ornek_res['n_subword']}  |  "
         f"Özel tokenlar: {ornek_res['n_special']}",
         transform=ax7.transAxes, fontsize=11, ha="center",
         color=PALETTE["gray"])

# Renk açıklamaları
legend_items = [
    mpatches.Patch(color=PALETTE["navy"],   label="Özel ([CLS],[SEP])"),
    mpatches.Patch(color=PALETTE["teal"],   label="Normal token"),
    mpatches.Patch(color=PALETTE["purple"], label="Alt-kelime (##)"),
    mpatches.Patch(color=PALETTE["amber"],  label="[MASK]"),
    mpatches.Patch(color=PALETTE["gray"],   label="[PAD]"),
]
ax7.legend(handles=legend_items, loc="lower center", ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, 0.04))

# ── h. Attention mask görselleştirmesi ───────────────────────
ax8 = fig.add_subplot(gs[3, :2])
# Padding'li tokenizasyon örneği
PADDED_ORNEKLER = [
    "Short text.",
    "A medium length sentence for visualization.",
    "This is a much longer sentence that will require more tokens to properly encode the content.",
]
MAX_LEN = 30

padded_matrix = np.zeros((len(PADDED_ORNEKLER), MAX_LEN))
if HF_AVAILABLE and tok:
    enc_padded = tok(PADDED_ORNEKLER, padding="max_length",
                     max_length=MAX_LEN, truncation=True,
                     return_tensors="pt")
    padded_matrix = enc_padded["attention_mask"].numpy()
else:
    for i, ornek in enumerate(PADDED_ORNEKLER):
        tokens_sim, _ = simulate_wordpiece(ornek)
        n = min(len(tokens_sim), MAX_LEN)
        padded_matrix[i, :n] = 1

im8 = ax8.imshow(padded_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax8.set_yticks(range(len(PADDED_ORNEKLER)))
ax8.set_yticklabels([f"Örnek {i+1}: '{s[:25]}...'" for i, s in enumerate(PADDED_ORNEKLER)],
                    fontsize=10)
ax8.set_xlabel("Token Pozisyonu (0-29)")
ax8.set_title("Attention Mask — Padding Görselleştirmesi\n(Mavi=1 gerçek token, Beyaz=0 padding)",
              fontweight="bold")
ax8.set_xticks(range(0, MAX_LEN, 5))
plt.colorbar(im8, ax=ax8, fraction=0.03)

# ── i. NER varlık tipi dağılımı ─────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
ner_tipler = {}
for sonuc_listesi in gorev_sonuclari["ner"]:
    for ent in sonuc_listesi:
        tip = ent.get("entity_group", ent.get("entity", "OTHER"))
        ner_tipler[tip] = ner_tipler.get(tip, 0) + 1

if ner_tipler:
    ax9.pie(list(ner_tipler.values()), labels=list(ner_tipler.keys()),
            colors=[PALETTE["navy"], PALETTE["teal"], PALETTE["amber"],
                    PALETTE["green"], PALETTE["blue"]],
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
    ax9.set_title("NER — Varlık Tipi Dağılımı\n(3 örnek cümle)", fontweight="bold")
else:
    ax9.text(0.5, 0.5, "NER verisi yok", ha="center", va="center")
    ax9.axis("off")

plt.savefig("h4c_01_pipeline_tokenizer.png", dpi=150, bbox_inches="tight")
print("    ✅ h4c_01_pipeline_tokenizer.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Pipeline görevleri test edildi : {len(GOREV_VERILERI)}")
print(f"  Toplam örnek sayısı            : "
      f"{sum(len(v['ornekler']) for v in GOREV_VERILERI.values())}")
print(f"  Tokenize edilen cümle          : {len(TEST_CUMLELER)}")
print(f"  Ort. token/cümle               : {np.mean(token_uzunluklari):.1f}")
print(f"  Batch hızlanması (n=512)       : "
      f"{hiz_kayitlari['single'][-1]/max(hiz_kayitlari['batched'][-1],0.1):.1f}×")
print(f"  Eşik optimizasyonu             : "
      f"varsayılan={esikler[np.argmin(np.abs(esikler-0.5))]:.2f} → "
      f"optimal={en_iyi_esik:.3f}")
print("  ✅ UYGULAMA 01 TAMAMLANDI — h4c_01_pipeline_tokenizer.png")
print("=" * 65)
