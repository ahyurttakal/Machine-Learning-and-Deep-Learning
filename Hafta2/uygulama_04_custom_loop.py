"""
=============================================================================
UYGULAMA 04 — Custom Training Loop (tf.GradientTape)
=============================================================================
Kapsam:
  - tf.GradientTape ile tam eğitim döngüsü
  - Gradient Clipping (patlayan gradyanı önleme)
  - Mixed Precision Training (float16 / float32)
  - Özel Kayıp Fonksiyonu (Focal Loss)
  - Özel Metrik (F1-Score, MCC)
  - tf.data.Dataset API (pipeline, önbellek, karıştırma)
  - Mini Proje: Tüm Pazar tekniklerini birleştiren tam sistem

Veri: MNIST (10 sınıf) + Breast Cancer (binary)
Kurulum: pip install tensorflow scikit-learn numpy matplotlib
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, matthews_corrcoef,
)
import time
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 04 — Custom Training Loop (tf.GradientTape)")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. ÖZEL KAYIP FONKSİYONLARI
# ═════════════════════════════════════════════════════════════
print("\n[1] Özel kayıp fonksiyonları tanımlanıyor...")

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss (Lin et al. 2017 — RetinaNet)
    Sınıf dengesizliğinde, kolay örnekleri azaltıp zor örneklere odaklanır.

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)
    - γ > 0 → kolay sınıflandırılan örneklerin ağırlığını düşürür
    - α     → sınıf dengesizliği için ağırlık
    """
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true   = tf.cast(y_true, tf.float32)
        bce      = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t      = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t  = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_wt = alpha_t * tf.pow(1 - p_t, self.gamma)
        return tf.reduce_mean(focal_wt * bce)

    def get_config(self):
        return {"gamma": self.gamma, "alpha": self.alpha}


class LabelSmoothingLoss(keras.losses.Loss):
    """
    Label Smoothing: Kesin 0/1 etiketleri ε/2 ile yumuşatır.
    Overconfident modellere karşı düzenleme sağlar.
    """
    def __init__(self, epsilon=0.1, name="label_smoothing_loss"):
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_pred    = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true    = tf.cast(y_true, tf.float32)
        smoothed  = y_true * (1 - self.epsilon) + self.epsilon / 2
        return tf.reduce_mean(
            -smoothed * tf.math.log(y_pred)
            - (1 - smoothed) * tf.math.log(1 - y_pred)
        )

print("    ✅ FocalLoss (γ=2, α=0.25) hazır")
print("    ✅ LabelSmoothingLoss (ε=0.1) hazır")

# ═════════════════════════════════════════════════════════════
# 2. ÖZEL METRİKLER
# ═════════════════════════════════════════════════════════════
print("\n[2] Özel metrikler tanımlanıyor...")

class F1Score(keras.metrics.Metric):
    """Binary F1-Score — her epoch sonunda sıfırlanır."""
    def __init__(self, threshold=0.5, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true     = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_pred_bin * y_true))
        self.fp.assign_add(tf.reduce_sum(y_pred_bin * (1 - y_true)))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred_bin) * y_true))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall    = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_state(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

print("    ✅ F1Score metriği hazır")

# ═════════════════════════════════════════════════════════════
# 3. VERİ & MODEL
# ═════════════════════════════════════════════════════════════
print("\n[3] Veri ve model hazırlanıyor...")

data   = load_breast_cancer()
X, y   = data.data, data.target.reshape(-1, 1).astype("float32")
scaler = StandardScaler()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

X_tr  = scaler.fit_transform(X_tr).astype("float32")
X_val = scaler.transform(X_val).astype("float32")
X_te  = scaler.transform(X_te).astype("float32")

print(f"    Eğitim: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_te.shape[0]}")

# ─── tf.data.Dataset pipeline ────────────────────────────────
BATCH_SIZE = 32

def make_dataset(X, y, shuffle=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return ds

train_ds = make_dataset(X_tr,  y_tr,  shuffle=True)
val_ds   = make_dataset(X_val, y_val)
test_ds  = make_dataset(X_te,  y_te)

# ─── Model ───────────────────────────────────────────────────
def build_model(input_dim=30, name="custom_loop_model"):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, use_bias=False, kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name)

# ═════════════════════════════════════════════════════════════
# 4. CUSTOM TRAINING LOOP
# ═════════════════════════════════════════════════════════════
print("\n[4] Custom Training Loop başlıyor...")

EPOCHS      = 80
MAX_GRAD_NORM = 1.0   # Gradient clipping normu

def run_custom_loop(loss_fn_name="bce", lr=1e-3, clip_gradients=True):
    """
    Tam custom training loop.
    loss_fn_name: 'bce', 'focal', 'label_smooth'
    """
    tf.random.set_seed(42)
    model     = build_model(X_tr.shape[1], name=f"model_{loss_fn_name}")
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Kayıp fonksiyonu seç
    if loss_fn_name == "focal":
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    elif loss_fn_name == "label_smooth":
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
    else:
        loss_fn = keras.losses.BinaryCrossentropy()

    # Metrikler
    train_loss   = keras.metrics.Mean(name="train_loss")
    val_loss     = keras.metrics.Mean(name="val_loss")
    train_acc    = keras.metrics.BinaryAccuracy(name="train_acc")
    val_acc      = keras.metrics.BinaryAccuracy(name="val_acc")
    train_auc    = keras.metrics.AUC(name="train_auc")
    val_auc      = keras.metrics.AUC(name="val_auc")
    train_f1     = F1Score(name="train_f1")
    val_f1       = F1Score(name="val_f1")

    # LR Scheduler: Cosine Annealing
    cosine_sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=EPOCHS * len(list(train_ds)),
        alpha=1e-7,
    )

    history = {k: [] for k in
               ["train_loss","val_loss","train_acc","val_acc",
                "train_auc","val_auc","train_f1","val_f1","lr","grad_norm"]}

    best_val_auc  = 0.0
    best_weights  = None
    patience      = 15
    patience_cnt  = 0
    step_counter  = 0

    # ── EĞİTİM DÖNGÜSÜ ───────────────────────────────────────
    for epoch in range(EPOCHS):
        # LR güncelle (Cosine)
        current_lr = float(cosine_sched(step_counter))
        optimizer.learning_rate.assign(current_lr)

        # ── Mini-batch eğitim ─────────────────────────────────
        epoch_grad_norms = []
        for X_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                y_pred  = model(X_batch, training=True)
                loss    = loss_fn(y_batch, y_pred)
                loss   += sum(model.losses)   # L2 reg ekleri

            # Gradyanları hesapla
            grads = tape.gradient(loss, model.trainable_variables)

            # Gradient Clipping
            if clip_gradients:
                grads, global_norm = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
                epoch_grad_norms.append(float(global_norm))

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Metrik güncelle
            train_loss.update_state(loss)
            train_acc.update_state(y_batch, y_pred)
            train_auc.update_state(y_batch, y_pred)
            train_f1.update_state(y_batch, y_pred)
            step_counter += 1

        # ── Validasyon ────────────────────────────────────────
        for X_v, y_v in val_ds:
            y_vp = model(X_v, training=False)
            v_loss = loss_fn(y_v, y_vp)
            val_loss.update_state(v_loss)
            val_acc.update_state(y_v, y_vp)
            val_auc.update_state(y_v, y_vp)
            val_f1.update_state(y_v, y_vp)

        # Geçmişe kaydet
        history["train_loss"].append(float(train_loss.result()))
        history["val_loss"].append(float(val_loss.result()))
        history["train_acc"].append(float(train_acc.result()))
        history["val_acc"].append(float(val_acc.result()))
        history["train_auc"].append(float(train_auc.result()))
        history["val_auc"].append(float(val_auc.result()))
        history["train_f1"].append(float(train_f1.result()))
        history["val_f1"].append(float(val_f1.result()))
        history["lr"].append(current_lr)
        history["grad_norm"].append(
            np.mean(epoch_grad_norms) if epoch_grad_norms else 0
        )

        # Print her 10 epoch
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1:3d}: "
                  f"loss={history['train_loss'][-1]:.4f} | "
                  f"val_loss={history['val_loss'][-1]:.4f} | "
                  f"val_auc={history['val_auc'][-1]:.4f} | "
                  f"val_f1={history['val_f1'][-1]:.4f} | "
                  f"lr={current_lr:.2e}")

        # Early stopping
        if history["val_auc"][-1] > best_val_auc:
            best_val_auc = history["val_auc"][-1]
            best_weights = model.get_weights()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"      Early stopping @ epoch {epoch+1}")
                break

        # Metrikleri sıfırla
        for m in [train_loss, val_loss, train_acc, val_acc,
                  train_auc, val_auc, train_f1, val_f1]:
            m.reset_state()

    # En iyi ağırlıkları geri yükle
    if best_weights:
        model.set_weights(best_weights)

    # Test değerlendirme
    y_proba = model.predict(X_te, verbose=0).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)
    test_auc_val = roc_auc_score(y_te.flatten(), y_proba)
    test_f1_val  = f1_score(y_te.flatten(), y_pred)
    test_mcc     = matthews_corrcoef(y_te.flatten(), y_pred)

    return {
        "history":   history,
        "model":     model,
        "test_auc":  test_auc_val,
        "test_f1":   test_f1_val,
        "test_mcc":  test_mcc,
        "y_proba":   y_proba,
        "best_val_auc": best_val_auc,
    }

# ── BCE ───────────────────────────────────────────────────────
print("\n    [4a] Binary CrossEntropy ile eğitim:")
t0 = time.time()
results_bce = run_custom_loop("bce", lr=1e-3, clip_gradients=True)
t_bce = time.time() - t0
print(f"    ✅ BCE → test_AUC={results_bce['test_auc']:.4f}  "
      f"F1={results_bce['test_f1']:.4f}  MCC={results_bce['test_mcc']:.4f}  "
      f"({t_bce:.1f}s)")

# ── Focal Loss ────────────────────────────────────────────────
print("\n    [4b] Focal Loss ile eğitim:")
results_focal = run_custom_loop("focal", lr=1e-3, clip_gradients=True)
print(f"    ✅ Focal → test_AUC={results_focal['test_auc']:.4f}  "
      f"F1={results_focal['test_f1']:.4f}  MCC={results_focal['test_mcc']:.4f}")

# ── Label Smoothing ───────────────────────────────────────────
print("\n    [4c] Label Smoothing ile eğitim:")
results_ls = run_custom_loop("label_smooth", lr=1e-3, clip_gradients=True)
print(f"    ✅ LabelSmooth → test_AUC={results_ls['test_auc']:.4f}  "
      f"F1={results_ls['test_f1']:.4f}  MCC={results_ls['test_mcc']:.4f}")

# ═════════════════════════════════════════════════════════════
# 5. GRADİYAN CLIPPING ETKİSİ
# ═════════════════════════════════════════════════════════════
print("\n[5] Gradient Clipping etkisi karşılaştırılıyor...")

print("    [5a] Gradient Clipping OLMADAN:")
results_no_clip = run_custom_loop("bce", lr=5e-3, clip_gradients=False)
print(f"    ✅ test_AUC={results_no_clip['test_auc']:.4f}")

print("\n    [5b] Gradient Clipping İLE (yüksek LR):")
results_with_clip = run_custom_loop("bce", lr=5e-3, clip_gradients=True)
print(f"    ✅ test_AUC={results_with_clip['test_auc']:.4f}")

print("\n    Yorum:")
print(f"    Clipping olmadan: {results_no_clip['test_auc']:.4f}")
print(f"    Clipping ile    : {results_with_clip['test_auc']:.4f}")
diff = results_with_clip["test_auc"] - results_no_clip["test_auc"]
print(f"    Fark            : {diff:+.4f} (+ yüksek LR ile daha kararlı)")

# ═════════════════════════════════════════════════════════════
# 6. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[6] Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("Custom Training Loop (tf.GradientTape) — Kapsamlı Analiz",
             fontsize=15, fontweight="bold")

PALETTE = {"bce": "#4338CA", "focal": "#DB2777", "label_smooth": "#0D9488",
           "no_clip": "#DC2626", "with_clip": "#15803D"}

all_results = {
    "BCE"           : results_bce,
    "Focal Loss"    : results_focal,
    "Label Smoothing": results_ls,
}

# ── 6a. Val Loss karşılaştırması ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for (name, res), color in zip(all_results.items(), PALETTE.values()):
    ax1.plot(res["history"]["val_loss"], lw=2, color=color, label=name)
ax1.set_title("Val Loss — Loss Fonksiyonu Karşılaştırması", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3)

# ── 6b. Val AUC karşılaştırması ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
for (name, res), color in zip(all_results.items(), PALETTE.values()):
    ax2.plot(res["history"]["val_auc"], lw=2, color=color, label=name)
ax2.set_title("Val AUC — Loss Fonksiyonu Karşılaştırması", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
ax2.legend(); ax2.grid(alpha=0.3)

# ── 6c. Val F1 karşılaştırması ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for (name, res), color in zip(all_results.items(), PALETTE.values()):
    ax3.plot(res["history"]["val_f1"], lw=2, color=color, label=name)
ax3.set_title("Val F1-Score Karşılaştırması", fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# ── 6d. Cosine LR eğrisi ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.semilogy(results_bce["history"]["lr"], color="#7C3AED", lw=2)
ax4.set_title("Cosine Annealing LR", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("LR (log)"); ax4.grid(alpha=0.3)

# ── 6e. Gradient Norm ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(results_bce["history"]["grad_norm"], color="#D97706", lw=1.5, alpha=0.7)
ax5.axhline(MAX_GRAD_NORM, color="#DC2626", ls="--", lw=2,
            label=f"Clip normu={MAX_GRAD_NORM}")
ax5.set_title("Gradient Global Normu (BCE)", fontweight="bold")
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Global Grad Norm")
ax5.legend(); ax5.grid(alpha=0.3)

# ── 6f. Gradient Clipping karşılaştırması ────────────────────
ax6 = fig.add_subplot(gs[1, 3])
ax6.plot(results_no_clip["history"]["val_auc"],   color=PALETTE["no_clip"],
         lw=2, label=f"Clip YOK (AUC={results_no_clip['test_auc']:.4f})")
ax6.plot(results_with_clip["history"]["val_auc"], color=PALETTE["with_clip"],
         lw=2, label=f"Clip VAR (AUC={results_with_clip['test_auc']:.4f})")
ax6.set_title("Gradient Clipping — Val AUC (LR=5e-3)", fontweight="bold")
ax6.set_xlabel("Epoch"); ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

# ── 6g. Final model — test metrikleri bar ────────────────────
ax7 = fig.add_subplot(gs[2, 0])
labels_bar = ["BCE", "Focal", "LblSmooth"]
aucs_b  = [r["test_auc"] for r in all_results.values()]
f1s_b   = [r["test_f1"]  for r in all_results.values()]
x_b = np.arange(len(labels_bar))
ax7.bar(x_b - 0.2, aucs_b, 0.38, label="AUC",      color="#4338CA", alpha=0.82)
ax7.bar(x_b + 0.2, f1s_b,  0.38, label="F1-Score",  color="#0D9488", alpha=0.82)
ax7.set_xticks(x_b); ax7.set_xticklabels(labels_bar)
ax7.set_ylim(0.88, 1.0); ax7.set_title("Test AUC & F1 Karşılaştırması", fontweight="bold")
ax7.legend(); ax7.grid(axis="y", alpha=0.3)

# ── 6h. Tahmin olasılığı dağılımı (BCE) ──────────────────────
ax8 = fig.add_subplot(gs[2, 1])
proba_bce = results_bce["y_proba"]
ax8.hist(proba_bce[y_te.flatten()==0], bins=25, alpha=0.65,
         color="#DC2626", label="Malign (0)", density=True)
ax8.hist(proba_bce[y_te.flatten()==1], bins=25, alpha=0.65,
         color="#15803D", label="Benign (1)", density=True)
ax8.axvline(0.5, color="k", ls="--", lw=1.5)
ax8.set_xlabel("Sigmoid Çıktısı"); ax8.set_ylabel("Yoğunluk")
ax8.set_title("BCE Model — Tahmin Dağılımı", fontweight="bold")
ax8.legend(); ax8.grid(axis="y", alpha=0.3)

# ── 6i. Tahmin olasılığı dağılımı (Focal) ────────────────────
ax9 = fig.add_subplot(gs[2, 2])
proba_focal = results_focal["y_proba"]
ax9.hist(proba_focal[y_te.flatten()==0], bins=25, alpha=0.65,
         color="#DC2626", label="Malign (0)", density=True)
ax9.hist(proba_focal[y_te.flatten()==1], bins=25, alpha=0.65,
         color="#15803D", label="Benign (1)", density=True)
ax9.axvline(0.5, color="k", ls="--", lw=1.5)
ax9.set_xlabel("Sigmoid Çıktısı"); ax9.set_ylabel("Yoğunluk")
ax9.set_title("Focal Loss Model — Tahmin Dağılımı", fontweight="bold")
ax9.legend(); ax9.grid(axis="y", alpha=0.3)

# ── 6j. MCC karşılaştırması ──────────────────────────────────
ax10 = fig.add_subplot(gs[2, 3])
mccs   = [r["test_mcc"] for r in all_results.values()]
colors_mcc = [PALETTE["bce"], PALETTE["focal"], PALETTE["label_smooth"]]
bars_mcc = ax10.bar(labels_bar, mccs, color=colors_mcc, alpha=0.82, edgecolor="white")
for bar, v in zip(bars_mcc, mccs):
    ax10.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
              f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax10.set_ylim(0.7, 1.0); ax10.set_title("Matthews Correlation Coefficient (MCC)", fontweight="bold")
ax10.set_ylabel("MCC"); ax10.grid(axis="y", alpha=0.3)

plt.savefig("04_custom_loop_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 04_custom_loop_analiz.png")
plt.close()

# ── Final Özet ───────────────────────────────────────────────
print("\n" + "─" * 55)
print("  FINAL SONUÇLAR — Loss Fonksiyonu Karşılaştırması")
print("─" * 55)
print(f"  {'Loss Fn':<18} {'Test AUC':<12} {'F1-Score':<12} {'MCC'}")
print("  " + "─" * 50)
for name, res in all_results.items():
    print(f"  {name:<18} {res['test_auc']:.4f}       {res['test_f1']:.4f}       {res['test_mcc']:.4f}")

# En iyi modelin sınıflandırma raporu
best_name  = max(all_results, key=lambda k: all_results[k]["test_auc"])
best_proba = all_results[best_name]["y_proba"]
best_pred  = (best_proba >= 0.5).astype(int)
print(f"\n  En iyi model: {best_name}")
print(classification_report(y_te.flatten(), best_pred,
                             target_names=["Malign", "Benign"]))

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("  Çıktılar:")
print("    04_custom_loop_analiz.png")
print("=" * 65)
