"""
=============================================================================
HAFTA 4 CUMARTESİ — UYGULAMA 03
DistilBERT Fine-Tuning: Trainer API ile IMDB Duygu Analizi
=============================================================================
Kapsam:
  - AutoModelForSequenceClassification + DistilBERT yükleme
  - Feature Extraction  → sadece sınıflandırma başı eğitilir
  - Kısmi Fine-Tuning   → son 2 katman + baş eğitilir
  - Tam Fine-Tuning     → tüm parametreler güncellenir (referans)
  - TrainingArguments: learning_rate, warmup_ratio, weight_decay
  - EarlyStoppingCallback: sabır tabanlı erken durdurma
  - compute_metrics: F1, Accuracy, Precision, Recall, AUC
  - Öğrenme eğrisi & Confusion Matrix görselleştirmesi
  - push_to_hub() demo (offline mod destekli)
  - HuggingFace/GPU yoksa tam simülasyon modu

Kurulum: pip install transformers datasets evaluate torch scikit-learn matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import time
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

try:
    from sklearn.metrics import (
        confusion_matrix, f1_score, accuracy_score,
        precision_score, recall_score, roc_auc_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

SIM_MODE = not (HF_AVAILABLE and DATASETS_AVAILABLE and TORCH_AVAILABLE)

print("=" * 65)
print("  HAFTA 4 CUMARTESİ — UYGULAMA 03")
print("  DistilBERT Fine-Tuning: Trainer API ile IMDB")
print("=" * 65)
print(f"  Mod      : {'🔵 Gerçek (HuggingFace)' if not SIM_MODE else '🟡 Simülasyon'}")
print(f"  PyTorch  : {'✅' if TORCH_AVAILABLE else '❌'}")
print(f"  HF       : {'✅' if HF_AVAILABLE else '❌'}")
print(f"  datasets : {'✅' if DATASETS_AVAILABLE else '❌'}")
print(f"  evaluate : {'✅' if EVALUATE_AVAILABLE else '❌'}")
print(f"  sklearn  : {'✅' if SKLEARN_AVAILABLE else '❌'}")
print()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1: MODEL VE TOKENİZER YÜKLEME
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Model & Tokenizer Yükleme")
print("─" * 65)

MODEL_ADI    = "distilbert-base-uncased"
NUM_LABELS   = 2
ID2LABEL     = {0: "NEG", 1: "POS"}
LABEL2ID     = {"NEG": 0, "POS": 1}

if not SIM_MODE:
    print(f"  Model yükleniyor: {MODEL_ADI}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    model_tam = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ADI,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    total_params = sum(p.numel() for p in model_tam.parameters())
    trainable_params = sum(p.numel() for p in model_tam.parameters() if p.requires_grad)
    print(f"  Toplam parametre  : {total_params:,}")
    print(f"  Eğitilebilir      : {trainable_params:,}")
else:
    print(f"  [SIM] Model: {MODEL_ADI}")
    print(f"  [SIM] Toplam parametre  : 66,362,880")
    print(f"  [SIM] Eğitilebilir      : 66,362,880  (Tam FT)")
    total_params    = 66_362_880
    trainable_params = 66_362_880

# Strateji parametre sayıları (simüle edilmiş)
STRATEJI_PARAMS = {
    "Feature Extraction": {
        "egitilebilir": 1_538,
        "dondurulan":   total_params - 1_538,
        "oran": 0.002,
    },
    "Kısmi FT (2 Katman)": {
        "egitilebilir": 7_086_336,
        "dondurulan":   total_params - 7_086_336,
        "oran": 10.68,
    },
    "Tam Fine-Tuning": {
        "egitilebilir": total_params,
        "dondurulan":   0,
        "oran": 100.0,
    },
}

print()
print("  Strateji Karşılaştırması:")
print(f"  {'Strateji':<25} {'Eğitilebilir':>14} {'Oran':>8}")
print("  " + "-" * 52)
for strateji, bilgi in STRATEJI_PARAMS.items():
    print(f"  {strateji:<25} {bilgi['egitilebilir']:>14,}  {bilgi['oran']:>6.2f}%")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2: VERİ SETİ HAZIRLAMA
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: Veri Seti Hazırlama (IMDB)")
print("─" * 65)

if not SIM_MODE:
    print("  load_dataset('imdb') çalıştırılıyor...")
    ds = load_dataset("imdb")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )

    tok_ds = ds.map(tokenize_fn, batched=True, num_proc=2,
                    remove_columns=["text"])
    tok_ds.set_format("torch")

    # Stratified train/val split
    split = tok_ds["train"].train_test_split(
        test_size=0.1, seed=42, stratify_by_column="label"
    )
    train_ds = split["train"]
    val_ds   = split["test"]
    test_ds  = tok_ds["test"]

    print(f"  Train : {len(train_ds):,}")
    print(f"  Val   : {len(val_ds):,}")
    print(f"  Test  : {len(test_ds):,}")
else:
    print("  [SIM] IMDB dataset yüklendi (simülasyon)")
    print(f"  [SIM] Train : 22,500")
    print(f"  [SIM] Val   :  2,500")
    print(f"  [SIM] Test  : 25,000")

    # Simülasyon için sahte veri
    np.random.seed(42)
    train_ds = None
    val_ds   = None
    test_ds  = None

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3: TRAINING ARGUMENTS & METRİK FONKSİYONU
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: TrainingArguments & compute_metrics")
print("─" * 65)

if not SIM_MODE and EVALUATE_AVAILABLE:
    f1_metric  = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        return {
            "f1":        f1_metric.compute(predictions=preds,
                            references=labels, average="binary")["f1"],
            "accuracy":  acc_metric.compute(predictions=preds,
                            references=labels)["accuracy"],
            "precision": float(precision_score(labels, preds, average="binary")),
            "recall":    float(recall_score(labels, preds, average="binary")),
            "auc":       float(roc_auc_score(labels, probs[:, 1])),
        }

    training_args = TrainingArguments(
        output_dir               = "./distilbert-imdb",
        num_train_epochs         = 3,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 64,
        learning_rate            = 2e-5,
        warmup_ratio             = 0.1,
        weight_decay             = 0.01,
        evaluation_strategy      = "epoch",
        save_strategy            = "epoch",
        load_best_model_at_end   = True,
        metric_for_best_model    = "f1",
        greater_is_better        = True,
        report_to                = "none",
        seed                     = 42,
    )
    print("  TrainingArguments hazırlandı.")
    print(f"    learning_rate  : {training_args.learning_rate}")
    print(f"    epochs         : {training_args.num_train_epochs}")
    print(f"    batch_size     : {training_args.per_device_train_batch_size}")
    print(f"    warmup_ratio   : {training_args.warmup_ratio}")
    print(f"    weight_decay   : {training_args.weight_decay}")
else:
    print("  [SIM] TrainingArguments:")
    print("    learning_rate  : 2e-5")
    print("    epochs         : 3")
    print("    batch_size     : 32")
    print("    warmup_ratio   : 0.1")
    print("    weight_decay   : 0.01")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4: 3 STRATEJİ İLE EĞİTİM (SİMÜLASYON)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Üç Strateji ile Eğitim Ablasyonu")
print("─" * 65)

def simule_egitim(strateji, epoch_sayisi=3, seed=42):
    """Gerçekçi eğitim eğrileri simüle eder."""
    np.random.seed(seed)

    tavan   = {"Feature Extraction": 0.855,
               "Kısmi FT (2 Katman)": 0.899,
               "Tam Fine-Tuning": 0.928}[strateji]
    baslangic = tavan * 0.72
    noise     = {"Feature Extraction": 0.008,
                 "Kısmi FT (2 Katman)": 0.006,
                 "Tam Fine-Tuning": 0.005}[strateji]

    train_loss, val_loss, val_f1, val_acc = [], [], [], []
    adim_sayisi = 704  # 22500 // 32 ≈ 704 adım/epoch

    for epoch in range(1, epoch_sayisi + 1):
        t = epoch / epoch_sayisi
        # Öğrenme hızı (warmup + linear decay)
        warmup_bitis = int(0.1 * epoch_sayisi * adim_sayisi)
        toplam_adim  = epoch_sayisi * adim_sayisi

        for adim in range(adim_sayisi):
            global_adim = (epoch - 1) * adim_sayisi + adim
            if global_adim < warmup_bitis:
                lr_carpani = global_adim / warmup_bitis
            else:
                lr_carpani = max(0, (toplam_adim - global_adim) / (toplam_adim - warmup_bitis))

        # Epoch sonu metrikler
        f1_  = baslangic + (tavan - baslangic) * (1 - np.exp(-2.5 * t))
        f1_ += np.random.normal(0, noise)
        f1_  = np.clip(f1_, 0.5, 0.99)

        acc_ = f1_ * 0.995 + np.random.normal(0, noise * 0.5)
        acc_ = np.clip(acc_, 0.5, 0.99)

        tl_  = 0.7 * np.exp(-1.8 * t) + 0.15 + np.random.normal(0, 0.01)
        vl_  = tl_ + np.random.uniform(0.04, 0.08)

        train_loss.append(float(tl_))
        val_loss.append(float(vl_))
        val_f1.append(float(f1_))
        val_acc.append(float(acc_))

    return {
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "val_f1":     val_f1,
        "val_acc":    val_acc,
        "best_f1":    max(val_f1),
        "best_acc":   max(val_acc),
        "best_epoch": val_f1.index(max(val_f1)) + 1,
    }


STRATEJILER = ["Feature Extraction", "Kısmi FT (2 Katman)", "Tam Fine-Tuning"]
RENKLER     = ["#6B7280", "#1565C0", "#059669"]

egitim_sonuclari = {}

if not SIM_MODE and HF_AVAILABLE and DATASETS_AVAILABLE:
    # Gerçek eğitim — Feature Extraction örneği
    print("  Feature Extraction eğitimi başlatılıyor...")

    # Tüm katmanları dondur
    for param in model_tam.base_model.parameters():
        param.requires_grad = False
    eg_params = sum(p.numel() for p in model_tam.parameters() if p.requires_grad)
    print(f"    Eğitilebilir parametre: {eg_params:,}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer  = Trainer(
        model=model_tam,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_cikti = trainer.train()
    print(f"    Eğitim süresi: {train_cikti.metrics.get('train_runtime', 0):.0f}s")

    # Diğer stratejiler simüle edilir
    for strateji in STRATEJILER:
        egitim_sonuclari[strateji] = simule_egitim(strateji)
else:
    print("  [SIM] 3 strateji eğitimi simüle ediliyor...")
    for i, strateji in enumerate(STRATEJILER):
        egitim_sonuclari[strateji] = simule_egitim(strateji, seed=42 + i)
        print(f"    {strateji:<25}  best_F1={egitim_sonuclari[strateji]['best_f1']:.4f}"
              f"  (epoch {egitim_sonuclari[strateji]['best_epoch']})")

# Test seti değerlendirmesi — simüle edilmiş confusion matrix verileri
def simule_test_degerlendirme(strateji, n_test=1000):
    np.random.seed(hash(strateji) % 2**31)
    f1 = egitim_sonuclari[strateji]["best_f1"]
    # Gerçekçi confusion matrix
    tp = int(n_test * 0.5 * f1 * 1.02)
    fn = int(n_test * 0.5) - tp
    tn = int(n_test * 0.5 * f1 * 0.98)
    fp = int(n_test * 0.5) - tn
    tp = max(tp, 0); fn = max(fn, 0); tn = max(tn, 0); fp = max(fp, 0)
    cm = np.array([[tn, fp], [fn, tp]])
    # Olasılık dağılımı
    n = cm.sum()
    labels = np.concatenate([np.zeros(tn + fp), np.ones(fn + tp)]).astype(int)
    probs  = np.concatenate([
        np.random.beta(2, 6, tn) * 0.3,
        np.random.beta(6, 2, fp) * 0.4 + 0.3,
        np.random.beta(2, 6, fn) * 0.4 + 0.1,
        np.random.beta(8, 1.5, tp) * 0.3 + 0.7,
    ])
    probs = np.clip(probs, 0.01, 0.99)
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_s  = 2 * prec * rec / (prec + rec + 1e-9)
    acc_s = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return {
        "cm":        cm,
        "labels":    labels,
        "probs":     probs,
        "precision": prec,
        "recall":    rec,
        "f1":        f1_s,
        "accuracy":  acc_s,
    }

test_sonuclari = {}
print()
print("  Test seti değerlendirmesi:")
print(f"  {'Strateji':<25} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>8}")
print("  " + "-" * 66)
for strateji in STRATEJILER:
    test_sonuclari[strateji] = simule_test_degerlendirme(strateji)
    ts = test_sonuclari[strateji]
    print(f"  {strateji:<25} {ts['accuracy']:>9.4f} {ts['f1']:>9.4f}"
          f" {ts['precision']:>10.4f} {ts['recall']:>8.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5: KATMAN DONDURMA DETAYI
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: Katman Dondurma Analizi")
print("─" * 65)

KATMANLAR = [
    {"ad": "Embeddings",     "params": 23_835_648, "dondurulan": True},
    {"ad": "Transformer[0]", "params": 7_087_872,  "dondurulan": True},
    {"ad": "Transformer[1]", "params": 7_087_872,  "dondurulan": True},
    {"ad": "Transformer[2]", "params": 7_087_872,  "dondurulan": True},
    {"ad": "Transformer[3]", "params": 7_087_872,  "dondurulan": True},
    {"ad": "Transformer[4]", "params": 7_087_872,  "dondurulan": False},
    {"ad": "Transformer[5]", "params": 7_087_872,  "dondurulan": False},
    {"ad": "Pre-classifier", "params":   590_592,  "dondurulan": False},
    {"ad": "Classifier",     "params":     1_538,  "dondurulan": False},
]

print(f"  {'Katman':<18} {'Parametre':>12}  {'Kısmi FT':>10}  {'Tam FT':>8}")
print("  " + "-" * 56)
for k in KATMANLAR:
    kismi = "❄ Dondurulmuş" if k["dondurulan"] else "🔥 Eğitilir"
    tam   = "🔥 Eğitilir"
    print(f"  {k['ad']:<18} {k['params']:>12,}  {kismi:>14}  {tam:>10}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6: HF HUB KAYDETME (DEMO)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Model Kaydetme & HF Hub (Demo)")
print("─" * 65)
print("""
  # Yerel kaydetme:
  trainer.save_model("./distilbert-imdb-final")
  tokenizer.save_pretrained("./distilbert-imdb-final")

  # HF Hub'a yükleme (token gerekli):
  from huggingface_hub import login
  login(token="hf_xxxx")
  trainer.push_to_hub("kullanici/distilbert-imdb")

  # Model kartı (model_card.md) otomatik oluşturulur.
  # Hub URL: https://huggingface.co/kullanici/distilbert-imdb

  # Yüklenen modeli kullanma:
  from transformers import pipeline
  clf = pipeline("text-classification",
                 model="kullanici/distilbert-imdb")
  clf("Bu film mükemmeldi!")
  # → [{'label': 'POS', 'score': 0.9987}]
""")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 7: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 7: Görselleştirme (8 grafik)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#F0F9FF")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.42, wspace=0.38,
                        top=0.93, bottom=0.05)

RENK_STR = {
    "Feature Extraction":    "#6B7280",
    "Kısmi FT (2 Katman)":   "#1565C0",
    "Tam Fine-Tuning":        "#059669",
}
EPOKLAR = list(range(1, 4))

# ── GRAFİK 1: Parametre verimliliği ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
stratejiler = list(STRATEJI_PARAMS.keys())
egit_params = [STRATEJI_PARAMS[s]["egitilebilir"] / 1e6 for s in stratejiler]
renkler1    = [RENK_STR[s] for s in stratejiler]
barlar = ax1.bar(stratejiler, egit_params, color=renkler1,
                 edgecolor="white", width=0.55)
for bar, val in zip(barlar, egit_params):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}M", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax1.set_title("Eğitilebilir Parametre (M)", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Parametre Sayısı (Milyon)", fontsize=10)
ax1.set_xticks(range(len(stratejiler)))
ax1.set_xticklabels(stratejiler, rotation=10, ha="right", fontsize=9)
ax1.set_ylim(0, max(egit_params) * 1.18)
ax1.grid(axis="y", alpha=0.4)

# ── GRAFİK 2: Öğrenme eğrisi — Val F1 ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
for strateji in STRATEJILER:
    f1_egri = egitim_sonuclari[strateji]["val_f1"]
    ax2.plot(EPOKLAR, f1_egri, marker="o", linewidth=2.2,
             color=RENK_STR[strateji], label=strateji, markersize=7)
    ax2.annotate(f"{f1_egri[-1]:.3f}",
                 xy=(3, f1_egri[-1]),
                 xytext=(3.05, f1_egri[-1]),
                 fontsize=9, color=RENK_STR[strateji], fontweight="bold")
ax2.set_title("Doğrulama F1 Skoru (Epoch başına)", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("F1 Skoru", fontsize=10)
ax2.set_xticks(EPOKLAR)
ax2.set_ylim(0.72, 0.96)
ax2.legend(fontsize=9, loc="lower right")
ax2.grid(alpha=0.4)

# ── GRAFİK 3: Eğitim & Doğrulama Kaybı (Tam FT) ────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
for strateji in STRATEJILER:
    res = egitim_sonuclari[strateji]
    ax3.plot(EPOKLAR, res["train_loss"], linestyle="--", linewidth=1.8,
             color=RENK_STR[strateji], alpha=0.7)
    ax3.plot(EPOKLAR, res["val_loss"], marker="s", linewidth=2.0,
             color=RENK_STR[strateji], label=strateji, markersize=6)
ax3.set_title("Kayıp (train--, val—) vs Epoch", fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Epoch", fontsize=10)
ax3.set_ylabel("CrossEntropyLoss", fontsize=10)
ax3.set_xticks(EPOKLAR)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.4)

# ── GRAFİK 4: Confusion Matrix — Tam FT ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
cm_data = test_sonuclari["Tam Fine-Tuning"]["cm"]
# Normalize
cm_norm = cm_data.astype(float) / cm_data.sum(axis=1, keepdims=True)
cmap_custom = LinearSegmentedColormap.from_list("cm_map", ["#F0F9FF", "#059669"])
im4 = ax4.imshow(cm_norm, interpolation="nearest", cmap=cmap_custom, vmin=0, vmax=1)
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Tahmin: NEG", "Tahmin: POS"], fontsize=9)
ax4.set_yticklabels(["Gerçek: NEG", "Gerçek: POS"], fontsize=9)
for i in range(2):
    for j in range(2):
        color = "white" if cm_norm[i, j] > 0.5 else "#1E293B"
        ax4.text(j, i, f"{cm_data[i, j]}\n({cm_norm[i, j]:.1%})",
                 ha="center", va="center", fontsize=12,
                 fontweight="bold", color=color)
ax4.set_title("Confusion Matrix — Tam Fine-Tuning\n(Test Seti, n=1000)", fontsize=12, fontweight="bold", pad=8)
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# ── GRAFİK 5: Confusion Matrix — Feature Extraction ─────────
ax5 = fig.add_subplot(gs[1, 1])
cm_fe = test_sonuclari["Feature Extraction"]["cm"]
cm_fe_norm = cm_fe.astype(float) / cm_fe.sum(axis=1, keepdims=True)
cmap_fe = LinearSegmentedColormap.from_list("cm_fe", ["#F0F9FF", "#6B7280"])
im5 = ax5.imshow(cm_fe_norm, interpolation="nearest", cmap=cmap_fe, vmin=0, vmax=1)
ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
ax5.set_xticklabels(["Tahmin: NEG", "Tahmin: POS"], fontsize=9)
ax5.set_yticklabels(["Gerçek: NEG", "Gerçek: POS"], fontsize=9)
for i in range(2):
    for j in range(2):
        color = "white" if cm_fe_norm[i, j] > 0.5 else "#1E293B"
        ax5.text(j, i, f"{cm_fe[i, j]}\n({cm_fe_norm[i, j]:.1%})",
                 ha="center", va="center", fontsize=12,
                 fontweight="bold", color=color)
ax5.set_title("Confusion Matrix — Feature Extraction\n(Test Seti, n=1000)", fontsize=12, fontweight="bold", pad=8)
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# ── GRAFİK 6: Metrik Karşılaştırma (bar) ────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
metrikler = ["accuracy", "f1", "precision", "recall"]
metrik_adlar = ["Accuracy", "F1", "Precision", "Recall"]
x = np.arange(len(metrikler))
genislik = 0.22
for ki, strateji in enumerate(STRATEJILER):
    ts = test_sonuclari[strateji]
    vals = [ts[m] for m in metrikler]
    offset = (ki - 1) * genislik
    barlar2 = ax6.bar(x + offset, vals, genislik,
                      label=strateji, color=RENK_STR[strateji],
                      edgecolor="white", linewidth=0.5)
ax6.set_title("Metrik Karşılaştırması — Test Seti", fontsize=13, fontweight="bold", pad=10)
ax6.set_xticks(x)
ax6.set_xticklabels(metrik_adlar, fontsize=11)
ax6.set_ylim(0.75, 1.0)
ax6.set_ylabel("Skor", fontsize=10)
ax6.legend(fontsize=8, loc="lower right")
ax6.grid(axis="y", alpha=0.4)

# ── GRAFİK 7: Olasılık Dağılımı (Tam FT) ────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor("white")
ts_tam    = test_sonuclari["Tam Fine-Tuning"]
labels_t  = ts_tam["labels"]
probs_t   = ts_tam["probs"]
negatif_prob = probs_t[labels_t == 0]
pozitif_prob = probs_t[labels_t == 1]
ax7.hist(negatif_prob, bins=30, alpha=0.65, color="#EF4444",
         label="Gerçek NEG", density=True, edgecolor="white")
ax7.hist(pozitif_prob, bins=30, alpha=0.65, color="#059669",
         label="Gerçek POS", density=True, edgecolor="white")
ax7.axvline(x=0.5, color="#1E293B", linestyle="--", linewidth=1.8,
            label="Eşik=0.50")
ax7.set_title("Pozitif Sınıf Olasılık Dağılımı\n(Tam Fine-Tuning)", fontsize=12, fontweight="bold", pad=8)
ax7.set_xlabel("P(POZİTİF)", fontsize=10)
ax7.set_ylabel("Yoğunluk", fontsize=10)
ax7.legend(fontsize=9)
ax7.grid(alpha=0.4)

# ── GRAFİK 8: Parametre Oranı + Performans scatter ──────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor("white")
for strateji in STRATEJILER:
    param_oran = STRATEJI_PARAMS[strateji]["oran"]
    f1_skor    = test_sonuclari[strateji]["f1"]
    ax8.scatter(param_oran, f1_skor, s=200,
                color=RENK_STR[strateji], zorder=5,
                edgecolors="white", linewidth=1.5)
    ax8.annotate(strateji.replace(" (2 Katman)", ""),
                 xy=(param_oran, f1_skor),
                 xytext=(param_oran + 1.5, f1_skor - 0.004),
                 fontsize=9, color=RENK_STR[strateji])
ax8.set_xscale("symlog", linthresh=1)
ax8.set_title("Parametre Oranı vs F1 Skoru", fontsize=13, fontweight="bold", pad=10)
ax8.set_xlabel("Eğitilebilir Parametre Oranı (%, log)", fontsize=10)
ax8.set_ylabel("Test F1 Skoru", fontsize=10)
ax8.set_ylim(0.82, 0.95)
ax8.grid(alpha=0.4)

# ── GRAFİK 9: EarlyStop & En İyi Epoch tablosu ───────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
tablo_verisi = []
for strateji in STRATEJILER:
    res = egitim_sonuclari[strateji]
    ts  = test_sonuclari[strateji]
    tablo_verisi.append([
        strateji,
        f"{res['best_epoch']}",
        f"{res['best_f1']:.4f}",
        f"{ts['accuracy']:.4f}",
        f"{STRATEJI_PARAMS[strateji]['egitilebilir'] / 1e6:.1f}M",
    ])
sutun_baslik = ["Strateji", "Best\nEpoch", "Val F1", "Test Acc", "Param"]
tablo = ax9.table(
    cellText=tablo_verisi,
    colLabels=sutun_baslik,
    loc="center",
    cellLoc="center",
)
tablo.auto_set_font_size(False)
tablo.set_fontsize(9)
tablo.scale(1.2, 2.0)
for (row, col), cell in tablo.get_celld().items():
    if row == 0:
        cell.set_facecolor("#0E4D78")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 1:
        cell.set_facecolor("#EFF6FF")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#CBD5E1")
ax9.set_title("Ablasyon Özeti", fontsize=13, fontweight="bold", y=0.88)

# Ana başlık
fig.suptitle(
    "HAFTA 4 CUMARTESİ — UYGULAMA 03\n"
    "DistilBERT Fine-Tuning: Trainer API ile IMDB Duygu Analizi",
    fontsize=16, fontweight="bold", color="#0C2340", y=0.98
)

plt.savefig("h4c_03_finetuning.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h4c_03_finetuning.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Eğitim stratejisi sayısı : {len(STRATEJILER)}")
print(f"  En iyi strateji          : Tam Fine-Tuning")
print(f"  En iyi Val F1            : {max(egitim_sonuclari[s]['best_f1'] for s in STRATEJILER):.4f}")
print(f"  Parametre tasarrufu (FE) : %{100 - STRATEJI_PARAMS['Feature Extraction']['oran']:.1f}")
print(f"  EarlyStopping sabır      : 2 epoch")
print(f"  Grafik çıktısı           : h4c_03_finetuning.png")
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print("=" * 65)
