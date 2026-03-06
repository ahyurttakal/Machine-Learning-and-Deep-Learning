"""
=============================================================================
UYGULAMA 04 — Çoklu NLP Pipeline: Beş Görev, Bir Kod
=============================================================================
Kapsam:
  1. Duygu Analizi — güven skoru + çok dilli
  2. NER — PER / ORG / LOC / MISC varlık tespiti + görselleştirme
  3. Zero-Shot Sınıflandırma — etiket listesi serbest, fine-tuning yok
  4. Soru-Cevap (Extractive QA) — bağlam + soru → metin içi cevap
  5. Metin Üretimi — GPT-2 beam search vs nucleus sampling karşılaştırması
  + Tüm görevlerin karşılaştırmalı görselleştirmesi

Kurulum: pip install transformers torch numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import time
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline, set_seed
    HF_AVAILABLE = True
    set_seed(42)
    print("✅ HuggingFace transformers yüklendi.")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace bulunamadı — simülasyon modu aktif.")
    print("   pip install transformers torch")

print("=" * 65)
print("  UYGULAMA 04 — Çoklu NLP Pipeline: 5 Görev")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# GÖREV 1: DUYGU ANALİZİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  GÖREV 1: Duygu Analizi (Sentiment Analysis)")
print("═" * 55)

SENTIMENT_TEXTS = [
    "This movie was absolutely fantastic! The acting was superb.",
    "Terrible film. Waste of time and money. Very disappointing.",
    "The plot was decent but the ending felt rushed and unsatisfying.",
    "An absolute masterpiece. One of the best films I've ever seen!",
    "Not bad, not great. Just an average movie experience overall.",
    "I loved every moment of it. Highly recommend to everyone!",
    "The special effects were amazing but the story was weak.",
    "Boring and predictable. I fell asleep halfway through the movie.",
]

if HF_AVAILABLE:
    print("  Model yükleniyor: distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True, max_length=512,
    )
    sentiment_results = sentiment_pipe(SENTIMENT_TEXTS)
else:
    # Simülasyon verileri
    sentiment_results = [
        {"label":"POSITIVE","score":0.9998}, {"label":"NEGATIVE","score":0.9997},
        {"label":"NEGATIVE","score":0.7234}, {"label":"POSITIVE","score":0.9999},
        {"label":"NEGATIVE","score":0.5621}, {"label":"POSITIVE","score":0.9995},
        {"label":"POSITIVE","score":0.8123}, {"label":"NEGATIVE","score":0.9876},
    ]

print("\n  Sonuçlar:")
for text, res in zip(SENTIMENT_TEXTS, sentiment_results):
    emoji = "😊" if res["label"] == "POSITIVE" else "😠"
    print(f"  {emoji} [{res['label']:8}  {res['score']:.3f}] {text[:60]}...")

# ─────────────────────────────────────────────────────────────
# GÖREV 2: NER
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  GÖREV 2: Varlık Tanıma (NER)")
print("═" * 55)

NER_TEXTS = [
    "Elon Musk, the CEO of Tesla and SpaceX, visited Berlin last week.",
    "The World Health Organization released a report on COVID-19 in Geneva.",
    "Barack Obama studied at Harvard Law School in Cambridge, Massachusetts.",
]

if HF_AVAILABLE:
    print("  Model yükleniyor: dslim/bert-base-NER")
    ner_pipe = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )
    ner_results = {text: ner_pipe(text) for text in NER_TEXTS}
else:
    ner_results = {
        NER_TEXTS[0]: [
            {"entity_group":"PER","word":"Elon Musk","score":0.9991,"start":0,"end":9},
            {"entity_group":"ORG","word":"Tesla","score":0.9987,"start":22,"end":27},
            {"entity_group":"ORG","word":"SpaceX","score":0.9979,"start":32,"end":38},
            {"entity_group":"LOC","word":"Berlin","score":0.9954,"start":48,"end":54},
        ],
        NER_TEXTS[1]: [
            {"entity_group":"ORG","word":"World Health Organization","score":0.9982,"start":4,"end":28},
            {"entity_group":"MISC","word":"COVID-19","score":0.9934,"start":53,"end":61},
            {"entity_group":"LOC","word":"Geneva","score":0.9971,"start":65,"end":71},
        ],
        NER_TEXTS[2]: [
            {"entity_group":"PER","word":"Barack Obama","score":0.9993,"start":0,"end":12},
            {"entity_group":"ORG","word":"Harvard Law School","score":0.9961,"start":24,"end":42},
            {"entity_group":"LOC","word":"Cambridge","score":0.9948,"start":46,"end":55},
            {"entity_group":"LOC","word":"Massachusetts","score":0.9936,"start":57,"end":70},
        ],
    }

for text, ents in ner_results.items():
    print(f"\n  Metin: '{text[:70]}...'")
    for ent in ents:
        tag_emoji = {"PER":"👤","ORG":"🏢","LOC":"📍","MISC":"🔖"}.get(ent["entity_group"], "?")
        print(f"    {tag_emoji} [{ent['entity_group']:4}] '{ent['word']}' (güven={ent['score']:.3f})")

# ─────────────────────────────────────────────────────────────
# GÖREV 3: ZERO-SHOT SINIFLANDIRMA
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  GÖREV 3: Zero-Shot Sınıflandırma")
print("═" * 55)
print("  (Fine-tuning yok! Etiket listesi serbest)")

ZS_SAMPLES = [
    {
        "text": "The new iPhone has an incredible camera and blazing fast processor.",
        "labels": ["technology", "sports", "politics", "food", "travel"],
    },
    {
        "text": "The team scored three goals in the final minutes of the championship.",
        "labels": ["football", "basketball", "tennis", "swimming", "athletics"],
    },
    {
        "text": "Parliament voted to approve the new climate change legislation.",
        "labels": ["politics", "environment", "economy", "health", "education"],
    },
    {
        "text": "The restaurant offers an amazing fusion of Italian and Japanese cuisine.",
        "labels": ["restaurant review", "travel guide", "recipe", "food criticism", "history"],
    },
]

if HF_AVAILABLE:
    print("  Model yükleniyor: facebook/bart-large-mnli")
    zs_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    zs_results = []
    for sample in ZS_SAMPLES:
        res = zs_pipe(sample["text"], candidate_labels=sample["labels"])
        zs_results.append(res)
else:
    zs_results = [
        {"labels":["technology","travel","food","politics","sports"],
         "scores":[0.9234,0.0412,0.0189,0.0098,0.0067], "sequence":ZS_SAMPLES[0]["text"]},
        {"labels":["football","athletics","basketball","swimming","tennis"],
         "scores":[0.8912,0.0542,0.0321,0.0134,0.0091], "sequence":ZS_SAMPLES[1]["text"]},
        {"labels":["politics","environment","economy","health","education"],
         "scores":[0.7123,0.1987,0.0543,0.0234,0.0113], "sequence":ZS_SAMPLES[2]["text"]},
        {"labels":["restaurant review","food criticism","recipe","travel guide","history"],
         "scores":[0.6543,0.1876,0.0921,0.0543,0.0117], "sequence":ZS_SAMPLES[3]["text"]},
    ]

for i, (sample, res) in enumerate(zip(ZS_SAMPLES, zs_results)):
    print(f"\n  [{i+1}] '{sample['text'][:65]}...'")
    for lbl, score in zip(res["labels"][:3], res["scores"][:3]):
        bar = "█" * int(score * 30)
        print(f"    {lbl:<20} {score:.3f} {bar}")

# ─────────────────────────────────────────────────────────────
# GÖREV 4: SORU-CEVAP (Extractive QA)
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  GÖREV 4: Soru-Cevap (Extractive QA)")
print("═" * 55)

QA_PAIRS = [
    {
        "context": (
            "The Transformer architecture was introduced in the paper 'Attention is All You Need' "
            "by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequences "
            "in parallel, unlike recurrent networks. BERT, released by Google in 2018, is an "
            "encoder-only Transformer pre-trained on masked language modeling."
        ),
        "questions": [
            "When was the Transformer architecture introduced?",
            "Who introduced the Transformer?",
            "What does BERT stand for?",
            "What is BERT pre-trained on?",
        ],
    },
    {
        "context": (
            "Python is a high-level, interpreted programming language created by Guido van Rossum "
            "and first released in 1991. It emphasizes code readability and simplicity. "
            "Python has become the most popular language for machine learning and data science, "
            "with libraries like TensorFlow, PyTorch, and scikit-learn."
        ),
        "questions": [
            "Who created Python?",
            "When was Python first released?",
            "What is Python popular for?",
        ],
    },
]

if HF_AVAILABLE:
    print("  Model yükleniyor: deepset/roberta-base-squad2")
    qa_pipe = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
    )
    qa_results = []
    for pair in QA_PAIRS:
        pair_results = []
        for q in pair["questions"]:
            res = qa_pipe(question=q, context=pair["context"])
            pair_results.append(res)
        qa_results.append(pair_results)
else:
    qa_results = [
        [
            {"answer":"2017","score":0.9234,"start":104,"end":108},
            {"answer":"Vaswani et al.","score":0.8876,"start":65,"end":79},
            {"answer":"encoder-only Transformer","score":0.7123,"start":221,"end":245},
            {"answer":"masked language modeling","score":0.8543,"start":278,"end":302},
        ],
        [
            {"answer":"Guido van Rossum","score":0.9512,"start":72,"end":88},
            {"answer":"1991","score":0.9823,"start":110,"end":114},
            {"answer":"machine learning and data science","score":0.8234,"start":195,"end":228},
        ],
    ]

for pair_idx, (pair, results) in enumerate(zip(QA_PAIRS, qa_results)):
    print(f"\n  Bağlam [{pair_idx+1}]: '{pair['context'][:80]}...'")
    for q, res in zip(pair["questions"], results):
        score_bar = "▓" * int(res["score"] * 20)
        print(f"    ❓ {q}")
        print(f"    ✅ '{res['answer']}'  (güven={res['score']:.3f}) {score_bar}")

# ─────────────────────────────────────────────────────────────
# GÖREV 5: METİN ÜRETİMİ (GPT-2)
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  GÖREV 5: Metin Üretimi (GPT-2)")
print("═" * 55)
print("  Beam Search vs Nucleus (Top-p) Sampling karşılaştırması")

PROMPTS = [
    "Artificial intelligence will transform the world by",
    "The future of natural language processing involves",
]

if HF_AVAILABLE:
    print("  Model yükleniyor: gpt2")
    gen_pipe = pipeline("text-generation", model="gpt2", truncation=True)
    gen_results = {}
    for prompt in PROMPTS:
        # Beam Search
        beam = gen_pipe(
            prompt,
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
        )
        # Nucleus Sampling (top-p)
        nucleus = gen_pipe(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.8,
            num_return_sequences=2,
        )
        gen_results[prompt] = {"beam": beam, "nucleus": nucleus}
else:
    gen_results = {
        PROMPTS[0]: {
            "beam": [{"generated_text": PROMPTS[0]+" enabling machines to understand and generate human language with unprecedented accuracy, automating tasks that previously required human intelligence across industries."}],
            "nucleus": [
                {"generated_text": PROMPTS[0]+" creating new possibilities for creativity, problem-solving, and collaboration between humans and intelligent systems in ways we never imagined before."},
                {"generated_text": PROMPTS[0]+" disrupting traditional industries while creating new ones, from autonomous vehicles to personalized medicine and climate change solutions."},
            ],
        },
        PROMPTS[1]: {
            "beam": [{"generated_text": PROMPTS[1]+" large-scale pre-trained models that can understand context across long documents and multiple languages simultaneously."}],
            "nucleus": [
                {"generated_text": PROMPTS[1]+" multimodal understanding where text, images, and audio are processed together, enabling richer human-computer interaction."},
                {"generated_text": PROMPTS[1]+" continual learning systems that can adapt to new domains without forgetting previous knowledge."},
            ],
        },
    }

for prompt, gens in gen_results.items():
    print(f"\n  Prompt: '{prompt}'")
    print(f"  📐 Beam Search:")
    print(f"     {gens['beam'][0]['generated_text']}")
    print(f"  🎲 Nucleus Sampling (örnek 1):")
    print(f"     {gens['nucleus'][0]['generated_text']}")
    if len(gens["nucleus"]) > 1:
        print(f"  🎲 Nucleus Sampling (örnek 2):")
        print(f"     {gens['nucleus'][1]['generated_text']}")

# ─────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n[6] Görselleştirmeler hazırlanıyor...")

ENT_COLORS = {"PER":"#DC2626","ORG":"#D97706","LOC":"#059669","MISC":"#7C3AED"}
TASK_COLORS = ["#DC2626","#D97706","#7C3AED","#059669","#0F766E"]

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)
fig.suptitle("Çoklu NLP Pipeline — 5 Görev Karşılaştırması", fontsize=15, fontweight="bold")

# ── a. Duygu analizi dağılımı ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
scores_pos = [r["score"] if r["label"]=="POSITIVE" else 1-r["score"] for r in sentiment_results]
x_s        = np.arange(len(SENTIMENT_TEXTS))
bar_colors = ["#059669" if r["label"]=="POSITIVE" else "#DC2626" for r in sentiment_results]
bars1 = ax1.bar(x_s, scores_pos, color=bar_colors, alpha=0.85, edgecolor="white")
for bar, r in zip(bars1, sentiment_results):
    lbl_txt = "+" if r["label"]=="POSITIVE" else "-"
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             lbl_txt, ha="center", fontsize=14, fontweight="bold",
             color="#059669" if r["label"]=="POSITIVE" else "#DC2626")
ax1.axhline(0.5, color="gray", ls="--", lw=1.5, label="Eşik=0.5")
ax1.set_xticks(x_s)
ax1.set_xticklabels([f"#{i+1}" for i in x_s])
ax1.set_ylim(0, 1.15); ax1.set_title("Duygu Analizi — Güven Skorları", fontweight="bold")
ax1.set_ylabel("Pozitif Olasılık"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)

# ── b. NER varlık dağılımı ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
all_ents  = [ent for ents in ner_results.values() for ent in ents]
ent_types = [e["entity_group"] for e in all_ents]
from collections import Counter
ent_counts = Counter(ent_types)
bars2 = ax2.bar(ent_counts.keys(), ent_counts.values(),
                color=[ENT_COLORS.get(k,"#6B7280") for k in ent_counts.keys()],
                alpha=0.85, edgecolor="white")
for bar, (k, v) in zip(bars2, ent_counts.items()):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             str(v), ha="center", fontsize=12, fontweight="bold")
ax2.set_title("NER — Varlık Tipi Dağılımı", fontweight="bold")
ax2.set_ylabel("Adet"); ax2.grid(axis="y", alpha=0.3)

# ── c. NER güven skoru ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 3])
ent_scores_by_type = {}
for ent in all_ents:
    t = ent["entity_group"]
    ent_scores_by_type.setdefault(t, []).append(ent["score"])
positions = list(ent_scores_by_type.keys())
data_bp   = [ent_scores_by_type[k] for k in positions]
bps = ax3.boxplot(data_bp, labels=positions, patch_artist=True)
for patch, pos in zip(bps["boxes"], positions):
    patch.set_facecolor(ENT_COLORS.get(pos, "#6B7280"))
    patch.set_alpha(0.75)
ax3.set_title("NER Güven Skoru Dağılımı", fontweight="bold")
ax3.set_ylabel("Güven Skoru"); ax3.grid(axis="y", alpha=0.3)

# ── d. Zero-Shot skor barları ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
for i, res in enumerate(zs_results):
    top_n = 3
    labels_top  = res["labels"][:top_n]
    scores_top  = res["scores"][:top_n]
    x_z = np.arange(top_n) + i * (top_n + 1)
    ax4.bar(x_z, scores_top, color=TASK_COLORS[i], alpha=0.8, edgecolor="white")
    for x, lbl in zip(x_z, labels_top):
        ax4.text(x, -0.05, lbl, ha="center", fontsize=7.5, rotation=35, va="top")
ax4.set_xticks([]); ax4.set_ylim(-0.2, 1.1)
ax4.set_title("Zero-Shot Sınıflandırma — Top-3 Skor (4 metin)", fontweight="bold")
ax4.set_ylabel("Güven Skoru"); ax4.grid(axis="y", alpha=0.3)
legend_patches = [mpatches.Patch(color=TASK_COLORS[i], label=f"Metin {i+1}")
                  for i in range(4)]
ax4.legend(handles=legend_patches, loc="upper right", fontsize=9)

# ── e. QA güven skoru ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2:])
all_qa_scores = []
all_qa_labels = []
for pair, results in zip(QA_PAIRS, qa_results):
    for q, res in zip(pair["questions"], results):
        all_qa_scores.append(res["score"])
        all_qa_labels.append(q[:35]+"...")
x_qa = np.arange(len(all_qa_scores))
bar_qa = ax5.barh(x_qa, all_qa_scores,
                   color=["#D97706" if s > 0.85 else "#059669" if s > 0.70 else "#DC2626"
                          for s in all_qa_scores], alpha=0.85)
for bar, v in zip(bar_qa, all_qa_scores):
    ax5.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=9)
ax5.set_yticks(x_qa)
ax5.set_yticklabels(all_qa_labels, fontsize=9)
ax5.set_xlim(0, 1.15)
ax5.set_title("Soru-Cevap — Güven Skoru (Extractive QA)", fontweight="bold")
ax5.axvline(0.8, color="gray", ls="--", lw=1.5, label="Güven eşiği=0.80")
ax5.legend(); ax5.grid(axis="x", alpha=0.3)

# ── f. Görev özeti tablo ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.axis("off")
ax6.set_title("5 NLP Görevi — Özet Karşılaştırma", fontweight="bold", pad=10)

tasks_summary = [
    ["Görev", "Pipeline Adı", "Önerilen Model", "Çıktı Tipi", "Kullanım Alanı"],
    ["Duygu Analizi", "sentiment-analysis", "DistilBERT-SST2", "Etiket + Skor", "Ürün yorumu, sosyal medya"],
    ["NER", "ner", "dslim/bert-base-NER", "Varlık Listesi", "Bilgi çıkarma, arama"],
    ["Zero-Shot Sınıf.", "zero-shot-classification", "facebook/bart-mnli", "Etiket + Skor", "Kategorisiz sınıflama"],
    ["Soru-Cevap", "question-answering", "deepset/roberta-squad2", "Cevap Metni", "Sözleşme analizi, destek"],
    ["Metin Üretimi", "text-generation", "GPT-2", "Üretilen Metin", "İçerik üretimi, chatbot"],
]
col_widths = [0.18, 0.22, 0.28, 0.18, 0.24]
col_starts = [0.0, 0.18, 0.40, 0.68, 0.82]
row_height  = 0.14

for row_i, row in enumerate(tasks_summary):
    for col_i, (cell, cx, cw) in enumerate(zip(row, col_starts, col_widths)):
        bg_color = TASK_COLORS[row_i-1] if row_i > 0 else "#431407"
        txt_color = "white"
        alpha = 0.85 if row_i > 0 else 1.0
        ax6.add_patch(mpatches.FancyBboxPatch(
            (cx+0.004, 0.95-row_i*row_height), cw-0.008, row_height-0.02,
            boxstyle="round,pad=0.01", facecolor=bg_color,
            edgecolor="white", linewidth=1.5,
            transform=ax6.transAxes, alpha=alpha
        ))
        ax6.text(cx+cw/2, 0.95-row_i*row_height+row_height/2-0.01, cell,
                 transform=ax6.transAxes,
                 fontsize=10.5 if row_i == 0 else 10,
                 ha="center", va="center", color=txt_color,
                 fontweight="bold" if row_i == 0 else "normal", wrap=True)

plt.savefig("04_nlp_pipeline.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 04_nlp_pipeline.png")
plt.close()

print("\n" + "═" * 55)
print("  5 GÖREV ÖZET:")
print(f"  1. Duygu Analizi  — {len(SENTIMENT_TEXTS)} metin işlendi")
pos_count = sum(1 for r in sentiment_results if r["label"]=="POSITIVE")
print(f"     Pozitif: {pos_count}/{len(SENTIMENT_TEXTS)} | Ortalama güven: {np.mean([r['score'] for r in sentiment_results]):.3f}")
print(f"  2. NER            — {sum(len(v) for v in ner_results.values())} varlık bulundu")
print(f"  3. Zero-Shot      — {len(ZS_SAMPLES)} metin, her biri {len(ZS_SAMPLES[0]['labels'])} aday etiketle")
total_qa = sum(len(p["questions"]) for p in QA_PAIRS)
avg_qa   = np.mean([r["score"] for pair_res in qa_results for r in pair_res])
print(f"  4. Soru-Cevap     — {total_qa} soru | Ortalama güven: {avg_qa:.3f}")
print(f"  5. Metin Üretimi  — Beam Search + Nucleus Sampling karşılaştırıldı")
print("\n" + "=" * 65)
print("  ✅ UYGULAMA 04 TAMAMLANDI — 04_nlp_pipeline.png")
print("=" * 65)
