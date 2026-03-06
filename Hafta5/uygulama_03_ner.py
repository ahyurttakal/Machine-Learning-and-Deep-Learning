"""
=============================================================================
UYGULAMA 03 — NER: Varlık Tanıma (Token Sınıflandırma)
=============================================================================
Kapsam:
  - BIO etiket şeması (B-PER, I-PER, B-ORG, B-LOC, O ...)
  - BERT token sınıflandırma başlığı (AutoModelForTokenClassification)
  - Offset mapping ile alt-kelime (subword) → karakter hizalama
  - Varlık düzeyinde F1 hesabı (seqeval kütüphanesi)
  - HuggingFace pipeline("ner") ile hızlı çıkarım
  - Renkli terminale varlık görselleştirme
  - Keras simülasyon modu (HF yoksa)

Kurulum: pip install transformers datasets evaluate seqeval torch
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
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    from datasets import load_dataset
    import evaluate
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

print("=" * 65)
print("  UYGULAMA 03 — NER: Varlık Tanıma")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# BIO ETİKET ŞEMASI AÇIKLAMASI
# ─────────────────────────────────────────────────────────────
print("\n[1] BIO etiket şeması:")

BIO_LABELS = {
    "O":     "Varlık dışı",
    "B-PER": "Kişi adı başlangıcı",
    "I-PER": "Kişi adı devamı",
    "B-ORG": "Kurum/Organizasyon başlangıcı",
    "I-ORG": "Kurum devamı",
    "B-LOC": "Konum/Yer başlangıcı",
    "I-LOC": "Konum devamı",
    "B-MISC":"Diğer varlık başlangıcı",
    "I-MISC":"Diğer varlık devamı",
}
for tag, desc in BIO_LABELS.items():
    print(f"    {tag:<8}: {desc}")

# Örnek cümle BIO etiketleme
example_tokens = ["Elon", "Musk", "Tesla", "'da", "CEO", "olarak", "görev", "yapıyor", "."]
example_labels = ["B-PER","I-PER","B-ORG","I-ORG","O","O","O","O","O"]
print("\n    Örnek cümle BIO etiketleme:")
for tok, lbl in zip(example_tokens, example_labels):
    print(f"      {tok:<12} → {lbl}")

# ─────────────────────────────────────────────────────────────
# OFFSET MAPPING AÇIKLAMASI
# ─────────────────────────────────────────────────────────────
print("\n[2] Offset mapping — alt-kelime hizalama:")

OFFSET_DEMO = True
if HF_AVAILABLE:
    tokenizer_demo = AutoTokenizer.from_pretrained("bert-base-cased")
    demo_text = "Elon Musk founded SpaceX in California."
    enc = tokenizer_demo(
        demo_text,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    tokens  = tokenizer_demo.convert_ids_to_tokens(enc["input_ids"][0])
    offsets = enc["offset_mapping"][0].tolist()

    print(f"    Metin: '{demo_text}'")
    print(f"    {'Token':<15} {'Offset':>12} {'Karakter'}")
    for tok, (s, e) in zip(tokens, offsets):
        chars = demo_text[s:e] if e > s else "<özel>"
        print(f"    {tok:<15} ({s:>3},{e:>3})   '{chars}'")
else:
    print("    (HuggingFace olmadan offset demo atlandı)")

# ─────────────────────────────────────────────────────────────
# HF MEVCUT → GERÇEK NER EĞİTİMİ
# ─────────────────────────────────────────────────────────────
if HF_AVAILABLE:
    MODEL_NAME = "bert-base-cased"     # NER için cased model önemli!
    BATCH_SIZE = 16
    EPOCHS     = 3

    print(f"\n[3] CoNLL-2003 NER veri seti yükleniyor...")
    raw_ds  = load_dataset("conll2003", trust_remote_code=True)
    label_names = raw_ds["train"].features["ner_tags"].feature.names
    id2label    = {i: l for i, l in enumerate(label_names)}
    label2id    = {l: i for i, l in enumerate(label_names)}
    NUM_LABELS  = len(label_names)

    print(f"    Etiket sayısı: {NUM_LABELS} → {label_names}")
    print(f"    Eğitim: {len(raw_ds['train']):,} | Val: {len(raw_ds['validation']):,} | Test: {len(raw_ds['test']):,}")

    tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align_labels(examples):
        """
        Tokenizasyon + BIO etiket hizalama.
        Alt-kelimeler (##...) için özel token = -100 (kayıp hesaplamada yoksayılır).
        """
        tokenized = tokenizer_ner(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,  # önceden token'lanmış metin
            max_length=128,
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)           # [CLS], [SEP], [PAD]
                elif wid != prev_word_id:
                    label_ids.append(labels[wid])    # Kelimenin ilk alt-kelimesi
                else:
                    label_ids.append(-100)           # Devam alt-kelimeleri yoksay
                prev_word_id = wid
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    print("    Tokenize & etiket hizalama...")
    tokenized_ds = raw_ds.map(tokenize_and_align_labels, batched=True,
                               remove_columns=raw_ds["train"].column_names)
    data_collator_ner = DataCollatorForTokenClassification(tokenizer_ner)

    # Seqeval metriği (varlık düzeyinde F1)
    seqeval_metric = evaluate.load("seqeval")

    def compute_ner_metrics(p):
        logits, labels = p
        predictions = np.argmax(logits, axis=-1)
        true_labels = [
            [label_names[l] for l in label if l != -100]
            for label in labels
        ]
        true_preds  = [
            [label_names[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        result = seqeval_metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": result["overall_precision"],
            "recall":    result["overall_recall"],
            "f1":        result["overall_f1"],
            "accuracy":  result["overall_accuracy"],
        }

    print(f"\n[4] NER modeli eğitiliyor ({MODEL_NAME})...")
    model_ner = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    trainable_ner = sum(p.numel() for p in model_ner.parameters() if p.requires_grad)
    print(f"    Toplam parametre: {trainable_ner:,}")

    args_ner = TrainingArguments(
        output_dir="./results_ner",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )
    trainer_ner = Trainer(
        model=model_ner, args=args_ner,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer_ner,
        data_collator=data_collator_ner,
        compute_metrics=compute_ner_metrics,
    )
    t0 = time.time()
    trainer_ner.train()
    t_ner = time.time() - t0
    print(f"    Eğitim süresi: {t_ner:.0f}s")

    # Test seti değerlendirmesi
    test_results = trainer_ner.evaluate(tokenized_ds["test"])
    print(f"\n    Test Seti Sonuçları:")
    print(f"    Precision : {test_results['eval_precision']:.4f}")
    print(f"    Recall    : {test_results['eval_recall']:.4f}")
    print(f"    F1        : {test_results['eval_f1']:.4f}")
    print(f"    Accuracy  : {test_results['eval_accuracy']:.4f}")

    # Pipeline ile çıkarım
    print("\n[5] NER pipeline ile örnek çıkarım:")
    ner_pipe = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner,
                        aggregation_strategy="simple")

    test_sentences = [
        "Angela Merkel served as Chancellor of Germany for 16 years.",
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "The United Nations headquarters is located in New York City.",
    ]
    ner_outputs = {}
    for sent in test_sentences:
        entities = ner_pipe(sent)
        ner_outputs[sent] = entities
        print(f"\n    Metin: '{sent}'")
        for ent in entities:
            print(f"      [{ent['entity_group']}] '{ent['word']}' "
                  f"(skor={ent['score']:.3f}, pozisyon={ent['start']}-{ent['end']})")

    # Sınıf bazında F1 hesapla
    all_preds_flat, all_labels_flat = [], []
    for batch in tokenized_ds["test"].select(range(min(500, len(tokenized_ds["test"])))):
        with torch.no_grad():
            inp = {k: torch.tensor([v]) for k, v in batch.items()
                   if k in ["input_ids", "attention_mask"]}
            out = model_ner(**inp)
        preds  = out.logits.argmax(-1)[0].tolist()
        labels = batch["labels"]
        for p, l in zip(preds, labels):
            if l != -100:
                all_preds_flat.append(id2label[p])
                all_labels_flat.append(id2label[l])

    per_class_result = seqeval_metric.compute(
        predictions=[all_preds_flat], references=[all_labels_flat]
    )
    # Her varlık tipi için F1
    entity_f1 = {k: v for k, v in per_class_result.items()
                 if isinstance(v, dict) and "f1" in v}

# ─────────────────────────────────────────────────────────────
# SEÇENEK B: Simülasyon (HF yok)
# ─────────────────────────────────────────────────────────────
else:
    print("\n[SİMÜLASYON MODU] HF yok — sentetik NER sonuçları ile görsel demo...")

    label_names = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]
    NUM_LABELS  = len(label_names)

    # Sentetik metrikler
    test_results = {
        "eval_precision": 0.8842,
        "eval_recall":    0.8917,
        "eval_f1":        0.8879,
        "eval_accuracy":  0.9713,
    }
    entity_f1 = {
        "PER":  {"f1": 0.9312, "precision": 0.9425, "recall": 0.9201},
        "ORG":  {"f1": 0.8654, "precision": 0.8721, "recall": 0.8589},
        "LOC":  {"f1": 0.9187, "precision": 0.9251, "recall": 0.9124},
        "MISC": {"f1": 0.7843, "precision": 0.7921, "recall": 0.7766},
    }
    ner_outputs = {
        "Angela Merkel served as Chancellor of Germany for 16 years.": [
            {"entity_group":"PER","word":"Angela Merkel","score":0.9987,"start":0,"end":13},
            {"entity_group":"LOC","word":"Germany","score":0.9912,"start":37,"end":44},
        ],
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.": [
            {"entity_group":"ORG","word":"Apple Inc.","score":0.9976,"start":0,"end":10},
            {"entity_group":"PER","word":"Steve Jobs","score":0.9991,"start":26,"end":36},
            {"entity_group":"LOC","word":"Cupertino","score":0.9854,"start":40,"end":49},
            {"entity_group":"LOC","word":"California","score":0.9943,"start":51,"end":61},
        ],
    }
    for sent, ents in ner_outputs.items():
        print(f"\n    Metin: '{sent}'")
        for ent in ents:
            print(f"      [{ent['entity_group']}] '{ent['word']}' (skor={ent['score']:.3f})")

# ─────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n[6] Görselleştirmeler hazırlanıyor...")

ENT_COLORS = {"PER":"#DC2626","ORG":"#D97706","LOC":"#059669","MISC":"#7C3AED","O":"#E5E7EB"}

fig = plt.figure(figsize=(22, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
fig.suptitle("NER — BERT Token Sınıflandırma (CoNLL-2003)", fontsize=15, fontweight="bold")

# ── a. Genel metrikler bar ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
metric_names_ner = ["Precision","Recall","F1","Accuracy"]
metric_vals_ner  = [test_results["eval_precision"], test_results["eval_recall"],
                    test_results["eval_f1"], test_results["eval_accuracy"]]
colors_m = ["#DC2626","#D97706","#7C3AED","#059669"]
bars1 = ax1.bar(metric_names_ner, metric_vals_ner, color=colors_m, alpha=0.85, width=0.55)
for bar, v in zip(bars1, metric_vals_ner):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax1.set_ylim(0.80, 1.02); ax1.set_title("Genel NER Metrikleri (Test)", fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# ── b. Varlık tipi F1 ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ent_names = list(entity_f1.keys())
ent_f1s   = [entity_f1[e]["f1"] for e in ent_names]
ent_precs = [entity_f1[e]["precision"] for e in ent_names]
ent_recs  = [entity_f1[e]["recall"]    for e in ent_names]
x_e = np.arange(len(ent_names)); w_e = 0.26
ax2.bar(x_e-w_e, ent_precs, w_e, label="Precision", color="#DC2626", alpha=0.8)
ax2.bar(x_e,     ent_recs,  w_e, label="Recall",    color="#D97706", alpha=0.8)
ax2.bar(x_e+w_e, ent_f1s,   w_e, label="F1",        color="#7C3AED", alpha=0.8)
ax2.set_xticks(x_e); ax2.set_xticklabels(ent_names, fontsize=12)
ax2.set_ylim(0.7, 1.02); ax2.set_title("Varlık Tipi Bazında Metrikler", fontweight="bold")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

# ── c. NER çıktısı görselleştirme ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")
ax3.set_title("NER Pipeline Örnek Çıktıları", fontweight="bold")

y_pos = 0.92
for sent, ents in list(ner_outputs.items())[:2]:
    words  = sent.split()
    x_pos  = 0.02
    ax3.text(x_pos, y_pos, "Metin:", transform=ax3.transAxes,
             fontsize=10, fontweight="bold", color="#431407")
    y_pos -= 0.06

    for ent in ents:
        color = ENT_COLORS.get(ent["entity_group"], "#E5E7EB")
        ax3.add_patch(mpatches.FancyBboxPatch(
            (x_pos, y_pos-0.04), min(len(ent["word"])*0.013, 0.55), 0.07,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor="white",
            transform=ax3.transAxes, alpha=0.85
        ))
        ax3.text(x_pos+0.01, y_pos-0.005, f"{ent['word']} [{ent['entity_group']}]",
                 transform=ax3.transAxes, fontsize=9, color="white", fontweight="bold")
        y_pos -= 0.1
    y_pos -= 0.06

# Renk açıklaması
legend_patches = [mpatches.Patch(color=c, label=k)
                  for k, c in ENT_COLORS.items() if k != "O"]
ax3.legend(handles=legend_patches, loc="lower left",
           bbox_to_anchor=(0.0, -0.02), ncol=2, fontsize=9)

# ── d. BIO şeması görsel ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.axis("off")
ax4.set_title("BIO Etiket Şeması — Token Hizalama Örneği", fontweight="bold")

bio_demo_tokens = ["[CLS]","Angela","Mer","##kel","worked","at","Apple","Inc",".",  "[SEP]"]
bio_demo_labels = ["O",     "B-PER","I-PER","[sub]","O",      "O",  "B-ORG","I-ORG","O","O"]
bio_demo_colors = ["#E5E7EB","#DC2626","#DC2626","#F9A8A8","#E5E7EB","#E5E7EB","#D97706","#D97706","#E5E7EB","#E5E7EB"]

for i, (tok, lbl, col) in enumerate(zip(bio_demo_tokens, bio_demo_labels, bio_demo_colors)):
    x = 0.04 + i * 0.094
    ax4.add_patch(mpatches.FancyBboxPatch(
        (x, 0.55), 0.085, 0.25, boxstyle="round,pad=0.01",
        facecolor=col, edgecolor="#9CA3AF",
        transform=ax4.transAxes, linewidth=1
    ))
    ax4.text(x+0.042, 0.68, tok, transform=ax4.transAxes,
             fontsize=9, ha="center", fontweight="bold",
             color="white" if col != "#E5E7EB" else "#374151")
    ax4.text(x+0.042, 0.45, lbl, transform=ax4.transAxes,
             fontsize=8.5, ha="center", color="#374151",
             fontweight="bold" if "sub" not in lbl else "normal",
             fontstyle="italic" if "sub" in lbl else "normal")

ax4.text(0.5, 0.18,
         "## ön eki → alt-kelime devamı (WordPiece).  "
         "[sub] etiket = -100 (kayıp hesaplamada yoksayılır).  "
         "Model yalnızca kelimenin ilk alt-kelimesini etiketler.",
         transform=ax4.transAxes, fontsize=10.5, ha="center", color="#431407",
         bbox=dict(boxstyle="round", facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.5))

# ── e. Güven skoru dağılımı ──────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
# Sentetik ya da gerçek skor dağılımı
all_scores = []
for ents in ner_outputs.values():
    all_scores.extend([e["score"] for e in ents])
if all_scores:
    ax5.hist(all_scores, bins=min(20, len(all_scores)*2),
             color="#DC2626", alpha=0.75, edgecolor="white")
    ax5.axvline(np.mean(all_scores), color="#D97706", ls="--", lw=2,
                label=f"Ort={np.mean(all_scores):.3f}")
    ax5.set_xlabel("NER Güven Skoru"); ax5.set_ylabel("Frekans")
    ax5.set_title("Varlık Güven Skoru Dağılımı", fontweight="bold")
    ax5.legend(); ax5.grid(axis="y", alpha=0.3)
else:
    ax5.text(0.5, 0.5, "Veri yok", ha="center", va="center",
             transform=ax5.transAxes, fontsize=14, color="gray")
    ax5.axis("off")

plt.savefig("03_ner_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 03_ner_analiz.png")
plt.close()

print("\n" + "=" * 65)
print(f"  Test F1       : {test_results['eval_f1']:.4f}")
print(f"  Test Accuracy : {test_results['eval_accuracy']:.4f}")
print("  ✅ UYGULAMA 03 TAMAMLANDI — 03_ner_analiz.png")
print("=" * 65)
