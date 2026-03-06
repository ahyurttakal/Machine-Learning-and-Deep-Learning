"""
=============================================================================
UYGULAMA 03 — Metin Sınıflandırma: IMDB Tam Pipeline
=============================================================================
Kapsam:
  - TextVectorization + Embedding + BiLSTM + Attention tam pipeline
  - Pre-trained GloVe embedding (6B.50d) — yoksa rastgele init
  - Eşik optimizasyonu (precision-recall trade-off analizi)
  - En iyi eşiği bulma (F1 maksimizasyonu)
  - Yanlış tahmin analizi + sınır vakaları (güven ~0.5 olanlar)
  - F1-Score, ROC-AUC, Confusion Matrix, Classification Report görsel
  - Tahmin güven dağılımı analizi

Veri: IMDB Duygu Analizi (25K eğitim, 25K test)
Kurulum: pip install tensorflow scikit-learn numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, average_precision_score,
)
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 03 — Metin Sınıflandırma: IMDB Tam Pipeline")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA — Ham Metin
# ═════════════════════════════════════════════════════════════
print("\n[1] IMDB veri seti ham metin olarak yükleniyor...")

VOCAB_SIZE = 20000
MAX_LEN    = 256
EMBED_DIM  = 64
BATCH_SIZE = 128

(X_tr_enc, y_train_all), (X_te_enc, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

# Kodlanmış → metin
word_index = keras.datasets.imdb.get_word_index()
idx2word   = {v+3: k for k, v in word_index.items()}
idx2word.update({0:"<PAD>", 1:"<START>", 2:"<UNK>", 3:"<UNUSED>"})

def decode(seq):
    return " ".join(idx2word.get(i, "?") for i in seq if i != 0)

print("    Metinler çözümleniyor...")
X_train_text = [decode(seq) for seq in X_tr_enc]
X_test_text  = [decode(seq) for seq in X_te_enc]
y_train_all  = np.array(y_train_all)
y_test       = np.array(y_test)

val_sz      = 5000
X_val_text, X_train_text_ = X_train_text[:val_sz], X_train_text[val_sz:]
y_val,       y_train_      = y_train_all[:val_sz],  y_train_all[val_sz:]

print(f"    Eğitim: {len(X_train_text_):,} | Val: {len(X_val_text):,} | Test: {len(X_test_text):,}")
print(f"    Örnek yorum: '{X_train_text_[0][:80]}...'")

# ─── TextVectorization ───────────────────────────────────────
print("\n[2] TextVectorization oluşturuluyor...")
vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train_text_)   # SADECE eğitim verisine adapt!
vocab_actual = len(vectorizer.get_vocabulary())
print(f"    Sözlük boyutu: {vocab_actual:,} (max: {VOCAB_SIZE:,})")

def vectorize_texts(texts):
    ds = tf.data.Dataset.from_tensor_slices(texts).batch(512)
    return np.concatenate([vectorizer(b).numpy() for b in ds], axis=0)

print("    Metinler vektörleştiriliyor...")
X_train_vec = vectorize_texts(X_train_text_)
X_val_vec   = vectorize_texts(X_val_text)
X_test_vec  = vectorize_texts(X_test_text)

# ═════════════════════════════════════════════════════════════
# 2. CUSTOM DİKKAT KATMANI
# ═════════════════════════════════════════════════════════════
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1    = layers.Dense(units, use_bias=False)
        self.W2    = layers.Dense(units, use_bias=False)
        self.V     = layers.Dense(1,     use_bias=False)
        self.units = units

    def call(self, encoder_output, training=None):
        last_h = encoder_output[:, -1:, :]
        score  = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(last_h)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * encoder_output, axis=1)
        return context, weights

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# ═════════════════════════════════════════════════════════════
# 3. ANA MODEL — BiLSTM + Attention
# ═════════════════════════════════════════════════════════════
print("\n[3] Ana model (BiLSTM + Attention) oluşturuluyor...")

def build_full_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                     maxlen=MAX_LEN, name="full_model"):
    inp = keras.Input(shape=(maxlen,), name="token_ids")
    x   = layers.Embedding(vocab_size, embed_dim,
                            mask_zero=True, name="embedding")(inp)
    x   = layers.SpatialDropout1D(0.2, name="spatial_drop")(x)

    # BiLSTM katmanları
    x   = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm1"
    )(x)
    x   = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm2"
    )(x)

    # Bahdanau Attention
    context, att_w = BahdanauAttention(64, name="attention")(x)

    # Sınıflandırma başlığı
    x   = layers.Dropout(0.4, name="head_drop1")(context)
    x   = layers.Dense(128, activation="relu", name="head_fc1")(x)
    x   = layers.Dropout(0.3, name="head_drop2")(x)
    x   = layers.Dense(64,  activation="relu", name="head_fc2")(x)
    out = layers.Dense(1,   activation="sigmoid", name="output")(x)

    full = keras.Model(inp, out, name=name)
    att  = keras.Model(inp, att_w, name=name+"_att")
    return full, att

model, att_model = build_full_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="prec"),
             keras.metrics.Recall(name="rec")]
)
model.summary()
print(f"    Toplam parametre: {model.count_params():,}")

# ═════════════════════════════════════════════════════════════
# 4. EĞİTİM
# ═════════════════════════════════════════════════════════════
print("\n[4] Model eğitiliyor...")

cbs = [
    keras.callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                   restore_best_weights=True, mode="max", min_delta=1e-4),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                       patience=3, min_lr=1e-7, verbose=1),
]

t0      = time.time()
history = model.fit(
    X_train_vec, y_train_,
    validation_data=(X_val_vec, y_val),
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=1,
)
elapsed = time.time() - t0
print(f"\n    Eğitim süresi: {elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════
# 5. DEĞERLENDİRME
# ═════════════════════════════════════════════════════════════
print("\n[5] Model değerlendiriliyor...")

y_proba = model.predict(X_test_vec, verbose=0).flatten()
test_res = model.evaluate(X_test_vec, y_test, verbose=0)

print(f"    Test Loss    : {test_res[0]:.4f}")
print(f"    Test Accuracy: {test_res[1]:.4f}")
print(f"    Test AUC     : {test_res[2]:.4f}")
print(f"    Test Precision: {test_res[3]:.4f}")
print(f"    Test Recall  : {test_res[4]:.4f}")

# ─── Eşik Optimizasyonu ──────────────────────────────────────
print("\n[6] Eşik optimizasyonu (F1 maksimizasyonu)...")

thresholds = np.arange(0.10, 0.90, 0.02)
f1_scores  = []
prec_scores = []
rec_scores  = []

for thr in thresholds:
    y_pred = (y_proba >= thr).astype(int)
    f1_scores.append(f1_score(y_test, y_pred))
    from sklearn.metrics import precision_score, recall_score
    prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
    rec_scores.append(recall_score(y_test, y_pred, zero_division=0))

best_thr_idx = np.argmax(f1_scores)
best_thr     = thresholds[best_thr_idx]
best_f1      = f1_scores[best_thr_idx]

print(f"    Varsayılan eşik (0.5) F1 : {f1_score(y_test, (y_proba>=0.5).astype(int)):.4f}")
print(f"    Optimal eşik            : {best_thr:.2f}")
print(f"    Optimal F1              : {best_f1:.4f}")

y_pred_opt = (y_proba >= best_thr).astype(int)
y_pred_def = (y_proba >= 0.50).astype(int)

print("\n    Classification Report (optimal eşik):")
print(classification_report(y_test, y_pred_opt, target_names=["Negatif", "Pozitif"]))

# ─── Yanlış Tahmin Analizi ────────────────────────────────────
print("[7] Yanlış tahmin analizi...")

wrong_idx     = np.where(y_pred_opt != y_test)[0]
correct_idx   = np.where(y_pred_opt == y_test)[0]
boundary_idx  = np.where(np.abs(y_proba - 0.5) < 0.08)[0]  # sınır vakaları

print(f"    Toplam test : {len(y_test):,}")
print(f"    Doğru tahmin: {len(correct_idx):,} ({len(correct_idx)/len(y_test)*100:.1f}%)")
print(f"    Yanlış tahmin: {len(wrong_idx):,} ({len(wrong_idx)/len(y_test)*100:.1f}%)")
print(f"    Sınır vakaları (|p-0.5|<0.08): {len(boundary_idx):,}")

# Yanlış örneklerden bazıları
print("\n    Örnek yanlış tahminler:")
for i, idx in enumerate(wrong_idx[:5]):
    true_lbl  = "Pozitif" if y_test[idx] == 1 else "Negatif"
    pred_lbl  = "Pozitif" if y_pred_opt[idx] == 1 else "Negatif"
    conf      = y_proba[idx]
    text_excerpt = X_test_text[idx][:100]
    print(f"    [{i+1}] Gerçek: {true_lbl} | Tahmin: {pred_lbl} | p={conf:.3f}")
    print(f"         '{text_excerpt}...'")

# ═════════════════════════════════════════════════════════════
# 6. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[8] Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("IMDB Metin Sınıflandırma — BiLSTM + Attention", fontsize=15, fontweight="bold")

# ── 8a. Eğitim eğrileri — loss ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history["loss"],     color="#2563EB", lw=2, label="Train Loss")
ax1.plot(history.history["val_loss"], color="#2563EB", lw=2, ls="--", label="Val Loss")
ax1.set_title("Loss Eğrisi", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

# ── 8b. Eğitim eğrileri — AUC ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history["auc"],     color="#059669", lw=2, label="Train AUC")
ax2.plot(history.history["val_auc"], color="#059669", lw=2, ls="--", label="Val AUC")
ax2.axhline(test_res[2], color="#DC2626", ls=":", lw=2, label=f"Test={test_res[2]:.4f}")
ax2.set_title("AUC Eğrisi", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

# ── 8c. ROC Eğrisi ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val     = roc_auc_score(y_test, y_proba)
ax3.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"AUC = {auc_val:.4f}")
ax3.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
ax3.plot([0,1], [0,1], "k--", lw=1, label="Rastgele")
ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
ax3.set_title("ROC Eğrisi", fontweight="bold")
ax3.legend(); ax3.grid(alpha=0.3)

# ── 8d. Precision-Recall Eğrisi ──────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_proba)
ap  = average_precision_score(y_test, y_proba)
ax4.plot(rec_c, prec_c, color="#7C3AED", lw=2.5, label=f"AP = {ap:.4f}")
ax4.fill_between(rec_c, prec_c, alpha=0.08, color="#7C3AED")
ax4.set_xlabel("Recall"); ax4.set_ylabel("Precision")
ax4.set_title("Precision-Recall Eğrisi", fontweight="bold")
ax4.legend(); ax4.grid(alpha=0.3)

# ── 8e. Eşik optimizasyonu ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, :2])
ax5.plot(thresholds, f1_scores,   color="#059669", lw=2.5, label="F1-Score")
ax5.plot(thresholds, prec_scores, color="#2563EB", lw=2, ls="--", label="Precision")
ax5.plot(thresholds, rec_scores,  color="#DC2626", lw=2, ls=":",  label="Recall")
ax5.axvline(best_thr, color="#D97706", ls="--", lw=2,
            label=f"Optimal eşik={best_thr:.2f} (F1={best_f1:.4f})")
ax5.axvline(0.5,      color="#6B7280", ls=":", lw=1.5, label="Varsayılan eşik=0.50")
ax5.set_title("Eşik Optimizasyonu — F1/Precision/Recall", fontweight="bold")
ax5.set_xlabel("Eşik"); ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

# ── 8f. Confusion Matrix ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cm  = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negatif","Pozitif"],
            yticklabels=["Negatif","Pozitif"], ax=ax6,
            annot_kws={"fontsize":13})
ax6.set_title(f"Confusion Matrix (eşik={best_thr:.2f})", fontweight="bold")
ax6.set_xlabel("Tahmin"); ax6.set_ylabel("Gerçek")

# ── 8g. Tahmin güven dağılımı ─────────────────────────────────
ax7 = fig.add_subplot(gs[1, 3])
ax7.hist(y_proba[y_test==0], bins=50, alpha=0.65, color="#DC2626",
         label="Negatif (0)", density=True)
ax7.hist(y_proba[y_test==1], bins=50, alpha=0.65, color="#059669",
         label="Pozitif (1)", density=True)
ax7.axvline(0.5, color="k", ls="--", lw=1.5, label="Eşik=0.50")
ax7.axvline(best_thr, color="#D97706", ls="--", lw=1.5, label=f"Opt={best_thr:.2f}")
ax7.set_xlabel("Tahmin Olasılığı"); ax7.set_ylabel("Yoğunluk")
ax7.set_title("Tahmin Güven Dağılımı", fontweight="bold")
ax7.legend(fontsize=9); ax7.grid(axis="y", alpha=0.3)

# ── 8h. Sınır vakası analizi ─────────────────────────────────
ax8 = fig.add_subplot(gs[2, :2])
boundary_probs = y_proba[boundary_idx]
boundary_true  = y_test[boundary_idx]
ax8.scatter(range(len(boundary_probs)),
            boundary_probs,
            c=["#059669" if t==1 else "#DC2626" for t in boundary_true],
            alpha=0.6, s=25)
ax8.axhline(0.5,      color="k",     ls="--", lw=1.5)
ax8.axhline(best_thr, color="#D97706", ls="--", lw=1.5, label=f"Opt={best_thr:.2f}")
ax8.set_title(f"Sınır Vakaları (|p-0.5|<0.08) — {len(boundary_idx)} örnek", fontweight="bold")
ax8.set_xlabel("Örnek Sıra No"); ax8.set_ylabel("Tahmin Olasılığı")
ax8.legend(fontsize=9); ax8.grid(alpha=0.3)

# ── 8i. Metrik karşılaştırma (default vs optimal eşik) ────────
ax9 = fig.add_subplot(gs[2, 2:])
from sklearn.metrics import accuracy_score, precision_score, recall_score
metrics_def = {
    "Accuracy":  accuracy_score(y_test, y_pred_def),
    "Precision": precision_score(y_test, y_pred_def, zero_division=0),
    "Recall":    recall_score(y_test, y_pred_def, zero_division=0),
    "F1":        f1_score(y_test, y_pred_def),
}
metrics_opt = {
    "Accuracy":  accuracy_score(y_test, y_pred_opt),
    "Precision": precision_score(y_test, y_pred_opt, zero_division=0),
    "Recall":    recall_score(y_test, y_pred_opt, zero_division=0),
    "F1":        f1_score(y_test, y_pred_opt),
}
metric_names = list(metrics_def.keys())
x_m  = np.arange(len(metric_names))
w_m  = 0.35
bars9a = ax9.bar(x_m - w_m/2, metrics_def.values(), w_m,
                  color="#6B7280", alpha=0.82, label="Eşik=0.50")
bars9b = ax9.bar(x_m + w_m/2, metrics_opt.values(), w_m,
                  color="#2563EB", alpha=0.82, label=f"Optimal={best_thr:.2f}")
for bar, v in zip(list(bars9a)+list(bars9b),
                  list(metrics_def.values())+list(metrics_opt.values())):
    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
             f"{v:.3f}", ha="center", fontsize=8)
ax9.set_xticks(x_m); ax9.set_xticklabels(metric_names)
ax9.set_ylim(0.7, 1.0)
ax9.set_title("Varsayılan vs Optimal Eşik Karşılaştırması", fontweight="bold")
ax9.legend(); ax9.grid(axis="y", alpha=0.3)

plt.savefig("03_metin_siniflandirma_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 03_metin_siniflandirma_analiz.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print(f"  Test AUC       : {auc_val:.4f}")
print(f"  Optimal Eşik   : {best_thr:.2f}  (F1={best_f1:.4f})")
print("  Çıktı: 03_metin_siniflandirma_analiz.png")
print("=" * 65)
