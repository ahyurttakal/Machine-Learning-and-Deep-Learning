"""
=============================================================================
UYGULAMA 02 — BERT Fine-Tuning: Hugging Face ile IMDB
=============================================================================
Kapsam:
  - DistilBERT/BERT AutoTokenizer + AutoModelForSequenceClassification
  - 3 fine-tuning stratejisi: Feature Extraction / Katman Dondurma / Tam FT
  - HuggingFace Trainer API ile eğitim
  - AdamW + Linear LR Warmup scheduler
  - Kapsamlı metrik analizi: Precision/Recall/F1/AUC + confusion matrix
  - Eşik optimizasyonu ve tahmin güven analizi

Kurulum: pip install transformers datasets evaluate scikit-learn torch numpy matplotlib
NOT: PyTorch tabanlıdır (transformers'ın varsayılan backend'i)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings("ignore")

# HuggingFace bağımlılıkları
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )
    from datasets import load_dataset, Dataset
    import evaluate
    HF_AVAILABLE = True
    print("✅ HuggingFace kütüphaneleri bulundu.")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace bulunamadı. TensorFlow/Keras simülasyonu ile devam ediliyor.")
    print("   pip install transformers datasets evaluate torch")

import sklearn.metrics as skm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_recall_curve,
)

print("=" * 65)
print("  UYGULAMA 02 — BERT Fine-Tuning (IMDB)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# SEÇENEK A: HuggingFace mevcut → gerçek BERT FT
# ─────────────────────────────────────────────────────────────
if HF_AVAILABLE:
    MODEL_NAME = "distilbert-base-uncased"   # daha hızlı demo; bert-base-uncased da olur
    MAX_LEN    = 256
    BATCH_SIZE = 16
    EPOCHS_FE  = 2    # Feature Extraction
    EPOCHS_FT  = 3    # Fine-Tuning

    print(f"\n[1] Model ve tokenizer yükleniyor: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"    Tokenizer: {type(tokenizer).__name__}  vocab={tokenizer.vocab_size:,}")

    # ── 1. Veri seti ──────────────────────────────────────────
    print("\n[2] IMDB veri seti yükleniyor (HuggingFace datasets)...")
    dataset = load_dataset("imdb")

    # Küçük subset (demo için hız odaklı)
    TRAIN_SIZE = 4000
    VAL_SIZE   = 1000
    TEST_SIZE  = 2000

    train_raw = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    val_raw   = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE))
    test_raw  = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=MAX_LEN,
        )

    print("    Tokenize ediliyor...")
    train_tok = train_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok   = val_raw.map(tokenize_fn,   batched=True, remove_columns=["text"])
    test_tok  = test_raw.map(tokenize_fn,  batched=True, remove_columns=["text"])
    train_tok = train_tok.rename_column("label", "labels")
    val_tok   = val_tok.rename_column("label",   "labels")
    test_tok  = test_tok.rename_column("label",  "labels")
    train_tok.set_format("torch"); val_tok.set_format("torch"); test_tok.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrik fonksiyonu
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1  = f1_score(labels, predictions)
        auc = roc_auc_score(labels, logits[:, 1])
        return {"accuracy": acc["accuracy"], "f1": f1, "auc": auc}

    all_results = {}

    # ── Strateji 1: Feature Extraction (sadece başlık) ────────
    print("\n[3a] Strateji 1: Feature Extraction (encoder donduruldu)...")
    model_fe = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Tüm parametreleri dondur
    for param in model_fe.base_model.parameters():
        param.requires_grad = False
    # Sınıflandırıcı başlığı aktif
    trainable_fe = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
    total_fe     = sum(p.numel() for p in model_fe.parameters())
    print(f"    Eğitilen param: {trainable_fe:,} / {total_fe:,} ({100*trainable_fe/total_fe:.1f}%)")

    args_fe = TrainingArguments(
        output_dir="./results_fe",
        num_train_epochs=EPOCHS_FE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-3,
        load_best_model_at_end=False,
        logging_steps=50,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )
    trainer_fe = Trainer(
        model=model_fe, args=args_fe,
        train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    trainer_fe.train()
    t_fe = time.time() - t0

    preds_fe = trainer_fe.predict(test_tok)
    logits_fe = preds_fe.predictions
    proba_fe  = torch.softmax(torch.tensor(logits_fe), dim=-1).numpy()[:, 1]
    y_pred_fe = (proba_fe >= 0.5).astype(int)
    y_true    = preds_fe.label_ids
    all_results["Feature Extraction"] = {
        "proba": proba_fe, "pred": y_pred_fe,
        "auc": roc_auc_score(y_true, proba_fe),
        "f1": f1_score(y_true, y_pred_fe),
        "acc": (y_pred_fe == y_true).mean(),
        "time": t_fe, "trainable": trainable_fe,
    }
    print(f"    AUC={all_results['Feature Extraction']['auc']:.4f}  "
          f"F1={all_results['Feature Extraction']['f1']:.4f}  t={t_fe:.0f}s")

    # ── Strateji 2: Kısmi Fine-Tuning (son 2 katman) ──────────
    print("\n[3b] Strateji 2: Kısmi Fine-Tuning (son 2 transformer katmanı)...")
    model_pft = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    for param in model_pft.base_model.parameters():
        param.requires_grad = False
    # DistilBERT'in son 2 transformer bloğunu aç
    for layer in model_pft.base_model.transformer.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable_pft = sum(p.numel() for p in model_pft.parameters() if p.requires_grad)
    print(f"    Eğitilen param: {trainable_pft:,} / {total_fe:,} ({100*trainable_pft/total_fe:.1f}%)")

    args_pft = TrainingArguments(
        output_dir="./results_pft",
        num_train_epochs=EPOCHS_FE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-4,
        load_best_model_at_end=False,
        logging_steps=50,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )
    trainer_pft = Trainer(
        model=model_pft, args=args_pft,
        train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    trainer_pft.train()
    t_pft = time.time() - t0

    preds_pft  = trainer_pft.predict(test_tok)
    logits_pft = preds_pft.predictions
    proba_pft  = torch.softmax(torch.tensor(logits_pft), dim=-1).numpy()[:, 1]
    y_pred_pft = (proba_pft >= 0.5).astype(int)
    all_results["Kısmi Fine-Tuning"] = {
        "proba": proba_pft, "pred": y_pred_pft,
        "auc": roc_auc_score(y_true, proba_pft),
        "f1": f1_score(y_true, y_pred_pft),
        "acc": (y_pred_pft == y_true).mean(),
        "time": t_pft, "trainable": trainable_pft,
    }
    print(f"    AUC={all_results['Kısmi Fine-Tuning']['auc']:.4f}  "
          f"F1={all_results['Kısmi Fine-Tuning']['f1']:.4f}  t={t_pft:.0f}s")

    # ── Strateji 3: Tam Fine-Tuning ───────────────────────────
    print("\n[3c] Strateji 3: Tam Fine-Tuning (tüm parametreler)...")
    model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    trainable_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"    Eğitilen param: {trainable_ft:,} ({100*trainable_ft/total_fe:.1f}%)")

    args_ft = TrainingArguments(
        output_dir="./results_ft",
        num_train_epochs=EPOCHS_FT,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,       # Tam FT için küçük LR kritik!
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        logging_steps=50,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )
    trainer_ft = Trainer(
        model=model_ft, args=args_ft,
        train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    trainer_ft.train()
    t_ft = time.time() - t0

    preds_ft   = trainer_ft.predict(test_tok)
    logits_ft  = preds_ft.predictions
    proba_ft   = torch.softmax(torch.tensor(logits_ft), dim=-1).numpy()[:, 1]
    y_pred_ft  = (proba_ft >= 0.5).astype(int)
    all_results["Tam Fine-Tuning"] = {
        "proba": proba_ft, "pred": y_pred_ft,
        "auc": roc_auc_score(y_true, proba_ft),
        "f1": f1_score(y_true, y_pred_ft),
        "acc": (y_pred_ft == y_true).mean(),
        "time": t_ft, "trainable": trainable_ft,
    }
    print(f"    AUC={all_results['Tam Fine-Tuning']['auc']:.4f}  "
          f"F1={all_results['Tam Fine-Tuning']['f1']:.4f}  t={t_ft:.0f}s")

# ─────────────────────────────────────────────────────────────
# SEÇENEK B: HuggingFace yok → Keras/TF simülasyonu
# ─────────────────────────────────────────────────────────────
else:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\n[SİMÜLASYON MODU] HuggingFace yok — Keras modelleriyle demo...")

    VOCAB_SIZE = 20000; MAX_LEN = 200; BATCH_SIZE = 128

    (X_tr_raw, y_tr_all), (X_te_raw, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    pad = lambda s: keras.preprocessing.sequence.pad_sequences(s, maxlen=MAX_LEN, padding="post")
    X_all = pad(X_tr_raw); X_test = pad(X_te_raw)
    X_val, X_train = X_all[:5000], X_all[5000:10000]
    y_val, y_train = y_tr_all[:5000], y_tr_all[5000:10000]
    y_true = y_test

    def make_sim_model(freeze_frac=0.0, name="sim"):
        inp = keras.Input(shape=(MAX_LEN,))
        x   = layers.Embedding(VOCAB_SIZE, 64, mask_zero=True)(inp)
        x   = layers.Bidirectional(layers.LSTM(64, dropout=0.2))(x)
        x   = layers.Dense(32, activation="gelu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m   = keras.Model(inp, out, name=name)
        # Simüle dondurma
        if freeze_frac > 0:
            for layer in m.layers[:int(len(m.layers)*freeze_frac)]:
                layer.trainable = False
        m.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
        return m

    all_results = {}
    for name, freeze in [("Feature Extraction", 0.9), ("Kısmi Fine-Tuning", 0.5), ("Tam Fine-Tuning", 0.0)]:
        print(f"    [{name}] eğitiliyor...")
        tf.random.set_seed(42)
        m = make_sim_model(freeze, name=name.replace(" ", "_"))
        m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=8, batch_size=BATCH_SIZE, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", patience=4,
                                                        restore_best_weights=True, mode="max")])
        proba = m.predict(X_test, verbose=0).flatten()
        y_pred = (proba >= 0.5).astype(int)
        all_results[name] = {
            "proba": proba, "pred": y_pred,
            "auc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, y_pred),
            "acc": (y_pred == y_test).mean(),
            "time": 0.0, "trainable": sum(np.prod(w.shape) for w in m.trainable_weights),
        }
        print(f"      AUC={all_results[name]['auc']:.4f}  F1={all_results[name]['f1']:.4f}")

# ─────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n[4] Görselleştirmeler hazırlanıyor...")

PALETTE = {
    "Feature Extraction": "#6B7280",
    "Kısmi Fine-Tuning":  "#D97706",
    "Tam Fine-Tuning":    "#DC2626",
}

fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.46, wspace=0.38)
fig.suptitle("BERT Fine-Tuning Stratejileri — IMDB (DistilBERT)", fontsize=15, fontweight="bold")

# ── a. AUC karşılaştırma ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
names_r = list(all_results.keys())
aucs_r  = [r["auc"] for r in all_results.values()]
bars1   = ax1.barh(names_r, aucs_r, color=[PALETTE[n] for n in names_r], alpha=0.85)
for bar, v in zip(bars1, aucs_r):
    ax1.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax1.set_xlim(0.75, 1.0); ax1.set_title("Test AUC", fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

# ── b. F1 karşılaştırma ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
f1s_r = [r["f1"] for r in all_results.values()]
bars2 = ax2.barh(names_r, f1s_r, color=[PALETTE[n] for n in names_r], alpha=0.85)
for bar, v in zip(bars2, f1s_r):
    ax2.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax2.set_xlim(0.75, 1.0); ax2.set_title("Test F1-Score", fontweight="bold")
ax2.grid(axis="x", alpha=0.3)

# ── c. ROC eğrisi ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for name, res in all_results.items():
    fpr, tpr, _ = roc_curve(y_true, res["proba"])
    ax3.plot(fpr, tpr, lw=2.5, color=PALETTE[name],
             label=f"{name}\n(AUC={res['auc']:.4f})")
ax3.plot([0,1],[0,1],"k--",lw=1); ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
ax3.set_title("ROC Eğrisi", fontweight="bold"); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# ── d. Tahmin dağılımı ───────────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
best_name = max(all_results, key=lambda k: all_results[k]["auc"])
proba_best = all_results[best_name]["proba"]
ax4.hist(proba_best[y_true==0], bins=50, alpha=0.65, color="#DC2626",
         label="Negatif", density=True)
ax4.hist(proba_best[y_true==1], bins=50, alpha=0.65, color="#059669",
         label="Pozitif", density=True)
ax4.axvline(0.5, color="k", ls="--", lw=1.5)
ax4.set_xlabel("Tahmin Olasılığı"); ax4.set_ylabel("Yoğunluk")
ax4.set_title(f"Güven Dağılımı ({best_name})", fontweight="bold")
ax4.legend(); ax4.grid(axis="y", alpha=0.3)

# ── e. Confusion Matrix (en iyi model) ───────────────────────
ax5 = fig.add_subplot(gs[1, 0])
import seaborn as sns
cm = confusion_matrix(y_true, all_results[best_name]["pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
            ax=ax5, annot_kws={"fontsize":14})
ax5.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
ax5.set_xlabel("Tahmin"); ax5.set_ylabel("Gerçek")

# ── f. Precision-Recall ──────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 1])
for name, res in all_results.items():
    prec, rec, _ = precision_recall_curve(y_true, res["proba"])
    ax6.plot(rec, prec, lw=2.5, color=PALETTE[name], label=name)
ax6.set_xlabel("Recall"); ax6.set_ylabel("Precision")
ax6.set_title("Precision-Recall Eğrisi", fontweight="bold")
ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

# ── g. Özet metin kutusu ─────────────────────────────────────
ax7 = fig.add_subplot(gs[1, 2:])
ax7.axis("off")
summary_lines = ["Strateji Karşılaştırması:\n"]
for name, res in all_results.items():
    summary_lines.append(
        f"{name}:\n"
        f"  Accuracy={res['acc']:.4f}  F1={res['f1']:.4f}  AUC={res['auc']:.4f}\n"
        f"  Eğitilebilir param: {res['trainable']:,}\n"
    )
summary_lines.append("\nÖneriler:\n"
    "  • Küçük veri + hız → Feature Extraction\n"
    "  • Orta veri + dengeli → Kısmi Fine-Tuning\n"
    "  • Büyük veri + GPU → Tam Fine-Tuning (LR=2e-5)\n"
    "  • Az GPU → LoRA (r=8-64, sadece Q/V matrisleri)")
ax7.text(0.02, 0.98, "\n".join(summary_lines), transform=ax7.transAxes,
         fontsize=11, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF7ED",
                   edgecolor="#DC2626", linewidth=1.5))
ax7.set_title("Sonuç Özeti", fontweight="bold")

plt.savefig("02_bert_finetuning.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 02_bert_finetuning.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print("  Çıktı: 02_bert_finetuning.png")
print("=" * 65)
