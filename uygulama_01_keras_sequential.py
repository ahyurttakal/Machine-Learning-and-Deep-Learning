"""
=============================================================================
UYGULAMA 01 — Keras Sequential API: Derinlemesine
=============================================================================
Kapsam:
  - Dense + BatchNormalization + Dropout tam mimari
  - 6 farklı aktivasyon fonksiyonunu karşılaştırma
  - He / Glorot / LeCun initializer deneyi
  - L1, L2, ElasticNet regularization karşılaştırması
  - model.summary(), param sayma, katman erişimi
  - Training history görselleştirme (loss & metrik eğrileri)
  - Overfitting tespiti ve teşhisi

Veri: Breast Cancer Wisconsin (sklearn)
Kurulum: pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

# ── TensorFlow deterministik mod ────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 65)
print("  UYGULAMA 01 — Keras Sequential API: Derinlemesine")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[1] Veri yükleniyor: Breast Cancer Wisconsin...")
data = load_breast_cancer()
X, y = data.data, data.target

print(f"    Örnekler: {X.shape[0]:,}  |  Özellikler: {X.shape[1]}")
print(f"    Sınıflar: {np.bincount(y)}  (0=malign, 1=benign)")
print(f"    Sınıf oranı: {np.mean(y):.1%} benign")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"    Eğitim: {X_train_s.shape[0]} | Val: {X_val_s.shape[0]} | Test: {X_test_s.shape[0]}")

INPUT_DIM = X_train_s.shape[1]

# ═════════════════════════════════════════════════════════════
# 2. TEMEL MODEL FONKSİYONU
# ═════════════════════════════════════════════════════════════
def build_model(activation="relu", initializer="he_normal",
                reg=None, dropout_rate=0.3, batch_norm=True,
                name="model"):
    """
    Esnek model oluşturucu.
    - activation  : katman aktivasyon fonksiyonu
    - initializer : kernel init stratejisi
    - reg         : regularizer nesnesi ya da None
    - dropout_rate: Dropout oranı (0 → kapalı)
    - batch_norm  : BatchNormalization eklensin mi?
    """
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(INPUT_DIM,)))

    for units, dr in [(256, dropout_rate), (128, dropout_rate * 0.8), (64, dropout_rate * 0.5)]:
        model.add(layers.Dense(
            units,
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            use_bias=not batch_norm,  # BN ile bias gereksiz
        ))
        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dr > 0:
            model.add(layers.Dropout(dr))

    model.add(layers.Dense(1, activation="sigmoid",
                           kernel_initializer="glorot_uniform"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model

# ─── Temel model özeti ────────────────────────────────────────
print("\n[2] Temel model özeti:")
base_model = build_model(name="temel_model")
base_model.summary()

total_params = base_model.count_params()
print(f"\n    Toplam parametre: {total_params:,}")
print(f"    Eğitilebilir   : {sum(tf.size(v).numpy() for v in base_model.trainable_variables):,}")

# ═════════════════════════════════════════════════════════════
# 3. AKTİVASYON FONKSİYONLARI KARŞILAŞTIRMASI
# ═════════════════════════════════════════════════════════════
print("\n[3] Aktivasyon fonksiyonları karşılaştırılıyor...")
print("    (Her aktivasyon için 50 epoch eğitim)")

EPOCHS_QUICK = 50
callbacks_quick = [
    keras.callbacks.EarlyStopping(monitor="val_auc", patience=10,
                                   restore_best_weights=True, mode="max"),
]

activation_configs = {
    "relu":      {"init": "he_normal"},
    "elu":       {"init": "he_normal"},
    "selu":      {"init": "lecun_normal"},
    "tanh":      {"init": "glorot_uniform"},
    "gelu":      {"init": "he_normal"},
    "swish":     {"init": "he_normal"},
}

activation_results = {}

for act, cfg in activation_configs.items():
    tf.random.set_seed(42)
    m = build_model(activation=act, initializer=cfg["init"],
                    name=f"act_{act}")
    h = m.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS_QUICK,
        batch_size=32,
        callbacks=callbacks_quick,
        verbose=0,
    )
    test_res = m.evaluate(X_test_s, y_test, verbose=0)
    val_auc  = max(h.history["val_auc"])
    test_auc = test_res[2]
    activation_results[act] = {
        "history": h.history,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "epochs_run": len(h.history["loss"]),
    }
    print(f"    {act:<10}: val_AUC={val_auc:.4f} | test_AUC={test_auc:.4f} | {len(h.history['loss'])} epoch")

# ═════════════════════════════════════════════════════════════
# 4. REGULARİZASYON KARŞILAŞTIRMASI
# ═════════════════════════════════════════════════════════════
print("\n[4] Regularization stratejileri karşılaştırılıyor...")

reg_configs = {
    "Yok":        None,
    "L2 (1e-4)":  regularizers.l2(1e-4),
    "L2 (1e-3)":  regularizers.l2(1e-3),
    "L1 (1e-4)":  regularizers.l1(1e-4),
    "L1+L2":      regularizers.l1_l2(l1=1e-5, l2=1e-4),
}

reg_results = {}
for reg_name, reg_obj in reg_configs.items():
    tf.random.set_seed(42)
    m = build_model(reg=reg_obj, name=f"reg_{reg_name.replace(' ','_')}")
    h = m.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks_quick,
        verbose=0,
    )
    test_res = m.evaluate(X_test_s, y_test, verbose=0)
    # Overfit göstergesi: train-val auc farkı
    best_epoch = np.argmax(h.history["val_auc"])
    overfit_gap = h.history["auc"][best_epoch] - h.history["val_auc"][best_epoch]
    reg_results[reg_name] = {
        "test_auc":    test_res[2],
        "val_auc":     max(h.history["val_auc"]),
        "overfit_gap": overfit_gap,
        "history":     h.history,
    }
    print(f"    {reg_name:<15}: test_AUC={test_res[2]:.4f}  overfit_gap={overfit_gap:.4f}")

# ═════════════════════════════════════════════════════════════
# 5. EN İYİ MODELİ DETAYLI EĞİT
# ═════════════════════════════════════════════════════════════
print("\n[5] En iyi model detaylı eğitiliyor...")

best_act = max(activation_results, key=lambda k: activation_results[k]["val_auc"])
print(f"    En iyi aktivasyon: {best_act}")

tf.random.set_seed(42)
final_model = build_model(
    activation=best_act,
    initializer="he_normal",
    reg=regularizers.l2(1e-4),
    dropout_rate=0.35,
    batch_norm=True,
    name="final_model",
)

callbacks_full = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=20,
        restore_best_weights=True, mode="max", min_delta=1e-4,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=8,
        min_lr=1e-7, verbose=1,
    ),
]

history = final_model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks_full,
    verbose=1,
)

# Test değerlendirmesi
test_loss, test_acc, test_auc = final_model.evaluate(X_test_s, y_test, verbose=0)
y_proba = final_model.predict(X_test_s, verbose=0).flatten()
y_pred  = (y_proba >= 0.5).astype(int)

print(f"\n    ── Test Sonuçları ──")
print(f"    Loss    : {test_loss:.4f}")
print(f"    Accuracy: {test_acc:.4f}")
print(f"    AUC     : {test_auc:.4f}")
print(f"\n    Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Malign", "Benign"]))

# ═════════════════════════════════════════════════════════════
# 6. KATMAN ERİŞİMİ & ARA ÇIKTILAR
# ═════════════════════════════════════════════════════════════
print("\n[6] Katman erişimi ve ara çıktılar...")

print("    Tüm katmanlar:")
for i, layer in enumerate(final_model.layers):
    trainable = getattr(layer, "trainable_weights", [])
    p_count = sum(tf.size(w).numpy() for w in trainable)
    print(f"      [{i:2d}] {layer.name:<30} çıktı={str(layer.output_shape):<22} params={p_count:,}")

# Ara çıktı modeli (ilk Dense bloğunun çıktısı)
intermediate_model = keras.Model(
    inputs=final_model.input,
    outputs=final_model.layers[3].output,  # İlk Activation çıktısı
)
intermediate_output = intermediate_model.predict(X_test_s[:5], verbose=0)
print(f"\n    İlk activation çıktısı (5 örnek, ilk 8 nöron):")
print(f"    {intermediate_output[:, :8].round(3)}")

# ═════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[7] Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle("Keras Sequential — Kapsamlı Analiz", fontsize=16, fontweight="bold")

# ── 7a. Aktivasyon karşılaştırma ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
acts = list(activation_results.keys())
val_aucs  = [activation_results[a]["val_auc"]  for a in acts]
test_aucs = [activation_results[a]["test_auc"] for a in acts]
x_pos = np.arange(len(acts))
bars1 = ax1.bar(x_pos - 0.18, val_aucs,  0.35, label="Val AUC",  color="#4338CA", alpha=0.85)
bars2 = ax1.bar(x_pos + 0.18, test_aucs, 0.35, label="Test AUC", color="#7C3AED", alpha=0.85)
ax1.set_xticks(x_pos); ax1.set_xticklabels(acts, fontsize=11)
ax1.set_ylim(0.93, 1.0); ax1.set_title("Aktivasyon Fonksiyonu Karşılaştırması", fontweight="bold")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)
for bar in [*bars1, *bars2]:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0008,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

# ── 7b. Regularization karşılaştırma ─────────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
reg_names  = list(reg_results.keys())
tauc_vals  = [reg_results[r]["test_auc"] for r in reg_names]
gap_vals   = [reg_results[r]["overfit_gap"] for r in reg_names]
x_pos2 = np.arange(len(reg_names))
ax2b = ax2.twinx()
bars = ax2.bar(x_pos2, tauc_vals, color="#0D9488", alpha=0.75, label="Test AUC")
ax2b.plot(x_pos2, gap_vals, "o-", color="#DC2626", linewidth=2, markersize=7, label="Overfit Gap")
ax2.set_xticks(x_pos2); ax2.set_xticklabels(reg_names, fontsize=9, rotation=15)
ax2.set_ylabel("Test AUC", color="#0D9488"); ax2b.set_ylabel("Overfit Gap", color="#DC2626")
ax2.set_title("Regularization Karşılaştırması", fontweight="bold")
ax2.set_ylim(0.90, 1.0); ax2.grid(axis="y", alpha=0.3)

# ── 7c. Eğitim eğrileri — Loss ────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(history.history["loss"],     color="#4338CA", lw=2, label="Train Loss")
ax3.plot(history.history["val_loss"], color="#7C3AED", lw=2, ls="--", label="Val Loss")
ax3.set_title("Final Model — Loss Eğrisi", fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Binary Crossentropy")
ax3.legend(); ax3.grid(alpha=0.3)

# ── 7d. Eğitim eğrileri — AUC ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2:])
ax4.plot(history.history["auc"],     color="#0D9488", lw=2, label="Train AUC")
ax4.plot(history.history["val_auc"], color="#DB2777", lw=2, ls="--", label="Val AUC")
ax4.set_title("Final Model — AUC Eğrisi", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("AUC")
ax4.legend(); ax4.grid(alpha=0.3)

# ── 7e. ROC Eğrisi ───────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)
ax5.plot(fpr, tpr, color="#4338CA", lw=2.5, label=f"ROC AUC = {auc_val:.4f}")
ax5.fill_between(fpr, tpr, alpha=0.08, color="#4338CA")
ax5.plot([0,1],[0,1],"k--",lw=1,label="Rastgele")
ax5.set_xlabel("False Positive Rate"); ax5.set_ylabel("True Positive Rate")
ax5.set_title("ROC Eğrisi", fontweight="bold")
ax5.legend(); ax5.grid(alpha=0.3)

# ── 7f. Tahmin dağılımı ──────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2:])
ax6.hist(y_proba[y_test==0], bins=30, alpha=0.65, color="#DC2626", label="Malign (0)", density=True)
ax6.hist(y_proba[y_test==1], bins=30, alpha=0.65, color="#15803D", label="Benign (1)", density=True)
ax6.axvline(0.5, color="k", ls="--", lw=1.5, label="Eşik=0.5")
ax6.set_xlabel("Sigmoid Çıktısı (Tahmin Olasılığı)")
ax6.set_ylabel("Yoğunluk"); ax6.set_title("Tahmin Olasılığı Dağılımı", fontweight="bold")
ax6.legend(); ax6.grid(axis="y", alpha=0.3)

plt.savefig("01_keras_sequential_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 01_keras_sequential_analiz.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print(f"  Final Test AUC: {test_auc:.4f}")
print("  Çıktı: 01_keras_sequential_analiz.png")
print("=" * 65)
