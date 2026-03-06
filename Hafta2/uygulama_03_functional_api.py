"""
=============================================================================
UYGULAMA 03 — Keras Functional API: Çoklu Giriş & Çıkış
=============================================================================
Kapsam:
  - Çoklu girdi mimarisi (sayısal + kategorik dal)
  - Concatenate ve Add katmanları
  - Skip Connection (Residual Block) elle inşa
  - Çoklu çıkış + loss ağırlıklandırması
  - Auxiliary loss ile eğitim
  - Model görselleştirme (keras.utils.plot_model)
  - Dallar arası bilgi akışını izleme

Veri: Breast Cancer (sayısal) + elle oluşturulan kategorik dal
Kurulum: pip install tensorflow scikit-learn numpy matplotlib
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 03 — Keras Functional API: Çoklu Giriş & Çıkış")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[1] Veri hazırlanıyor...")
data    = load_breast_cancer()
X_raw   = data.data
y_main  = data.target     # Binary: 0=malign, 1=benign

# Kategorik dal için: özelliklerin yüksek/orta/düşük grupları
# (Gerçek senaryoda farklı veri kaynakları olabilir)
np.random.seed(42)
severity = np.where(X_raw[:, 0] > np.percentile(X_raw[:, 0], 66), 2,
           np.where(X_raw[:, 0] > np.percentile(X_raw[:, 0], 33), 1, 0))
# Onehot encode
cat_features = np.eye(3)[severity]                    # (n, 3)

# Ek auxiliary hedef: 3-sınıflı risk grubu
y_aux = severity                                       # 3 sınıf

X_num  = X_raw
scaler = StandardScaler()

X_n_tr, X_n_te, X_c_tr, X_c_te, y_m_tr, y_m_te, y_a_tr, y_a_te = train_test_split(
    X_num, cat_features, y_main, y_aux,
    test_size=0.2, stratify=y_main, random_state=42
)
X_n_tr, X_n_val, X_c_tr, X_c_val, y_m_tr, y_m_val, y_a_tr, y_a_val = train_test_split(
    X_n_tr, X_c_tr, y_m_tr, y_a_tr,
    test_size=0.2, stratify=y_m_tr, random_state=42
)

X_n_tr  = scaler.fit_transform(X_n_tr)
X_n_val = scaler.transform(X_n_val)
X_n_te  = scaler.transform(X_n_te)

print(f"    Sayısal girdi: {X_n_tr.shape}")
print(f"    Kategorik girdi: {X_c_tr.shape}")
print(f"    Ana hedef (binary): {np.bincount(y_m_tr)}")
print(f"    Yardımcı hedef (3-sınıf): {np.bincount(y_a_tr)}")

# ═════════════════════════════════════════════════════════════
# 2. RESIDUAL BLOCK FONKSİYONU
# ═════════════════════════════════════════════════════════════
def residual_block(x, units, dropout_rate=0.2, name_prefix="res"):
    """
    Skip Connection (Artık Bağlantı) bloğu.
    Giriş boyutu çıkış boyutuna eşit değilse projection katmanı eklenir.
    """
    shortcut = x
    # Boyut uyumsuzluğu varsa projeksiyon
    if x.shape[-1] != units:
        shortcut = layers.Dense(units, use_bias=False,
                                 name=f"{name_prefix}_proj")(x)

    # Ana yol
    x = layers.Dense(units, use_bias=False,
                      name=f"{name_prefix}_fc1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act1")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    x = layers.Dense(units, use_bias=False,
                      name=f"{name_prefix}_fc2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

    # Skip connection ekle
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_act2")(x)
    return x

# ═════════════════════════════════════════════════════════════
# 3. MİMARİ 1: Basit Çoklu Girdi
# ═════════════════════════════════════════════════════════════
print("\n[2] Mimari 1: Basit Çoklu Girdi modeli oluşturuluyor...")

def build_multi_input_model(num_dim=30, cat_dim=3):
    """
    İki ayrı girdi → ayrı işleme → birleştir → ortak çıkış
    """
    # Girdi katmanları
    inp_num = keras.Input(shape=(num_dim,), name="sayisal_giris")
    inp_cat = keras.Input(shape=(cat_dim,), name="kategorik_giris")

    # Sayısal dal
    x1 = layers.Dense(128, use_bias=False, name="num_fc1")(inp_num)
    x1 = layers.BatchNormalization(name="num_bn1")(x1)
    x1 = layers.Activation("relu", name="num_act1")(x1)
    x1 = layers.Dropout(0.3, name="num_drop1")(x1)
    x1 = layers.Dense(64, use_bias=False, name="num_fc2")(x1)
    x1 = layers.BatchNormalization(name="num_bn2")(x1)
    x1 = layers.Activation("relu", name="num_act2")(x1)

    # Kategorik dal
    x2 = layers.Dense(32, activation="relu", name="cat_fc1")(inp_cat)
    x2 = layers.Dense(16, activation="relu", name="cat_fc2")(x2)

    # Birleştirme (Concatenate)
    merged = layers.Concatenate(name="birlestir")([x1, x2])
    x = layers.Dense(64, activation="relu", name="merge_fc")(merged)
    x = layers.Dropout(0.25, name="merge_drop")(x)

    # Çıkış
    output = layers.Dense(1, activation="sigmoid", name="ana_cikis")(x)

    model = keras.Model(
        inputs=[inp_num, inp_cat],
        outputs=output,
        name="coklu_giris_modeli"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model

model_mi = build_multi_input_model(X_n_tr.shape[1], X_c_tr.shape[1])
model_mi.summary()

# ═════════════════════════════════════════════════════════════
# 4. MİMARİ 2: Residual + Çoklu Çıkış
# ═════════════════════════════════════════════════════════════
print("\n[3] Mimari 2: Residual + Çoklu Çıkış modeli oluşturuluyor...")

def build_residual_multi_output(num_dim=30, cat_dim=3):
    """
    Residual bloklar + iki çıkış:
    1. Ana çıkış: binary (malign/benign)
    2. Yardımcı çıkış: 3-sınıflı risk grubu
    """
    inp_num = keras.Input(shape=(num_dim,), name="sayisal_giris")
    inp_cat = keras.Input(shape=(cat_dim,), name="kategorik_giris")

    # Sayısal dal — Residual bloklar
    x1 = layers.Dense(128, use_bias=False, name="stem_num")(inp_num)
    x1 = layers.BatchNormalization(name="stem_bn")(x1)
    x1 = layers.Activation("relu", name="stem_act")(x1)
    x1 = residual_block(x1, 128, dropout_rate=0.3, name_prefix="res1")
    x1 = residual_block(x1, 64,  dropout_rate=0.2, name_prefix="res2")

    # Kategorik dal — basit
    x2 = layers.Dense(32, activation="relu", name="cat_embed")(inp_cat)

    # Birleştir
    merged = layers.Concatenate(name="concat")([x1, x2])

    # Paylaşılan gövde
    shared = layers.Dense(64, use_bias=False, name="shared_fc")(merged)
    shared = layers.BatchNormalization(name="shared_bn")(shared)
    shared = layers.Activation("relu", name="shared_act")(shared)
    shared = layers.Dropout(0.2, name="shared_drop")(shared)

    # ── Ana Çıkış (binary) ───────────────────────────────────
    main_head = layers.Dense(32, activation="relu", name="main_head_fc")(shared)
    main_out  = layers.Dense(1, activation="sigmoid", name="ana_cikis")(main_head)

    # ── Yardımcı Çıkış (3-sınıf) ─────────────────────────────
    aux_head = layers.Dense(16, activation="relu", name="aux_head_fc")(shared)
    aux_out  = layers.Dense(3, activation="softmax", name="yardimci_cikis")(aux_head)

    model = keras.Model(
        inputs=[inp_num, inp_cat],
        outputs=[main_out, aux_out],
        name="residual_coklu_cikis"
    )

    # Çoklu loss: ana çıkış ağırlık=1.0, yardımcı=0.3
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "ana_cikis":       "binary_crossentropy",
            "yardimci_cikis":  "sparse_categorical_crossentropy",
        },
        loss_weights={
            "ana_cikis":      1.0,
            "yardimci_cikis": 0.3,   # Auxiliary loss düşük ağırlıklı
        },
        metrics={
            "ana_cikis":      [keras.metrics.AUC(name="auc")],
            "yardimci_cikis": ["accuracy"],
        },
    )
    return model

model_res = build_residual_multi_output(X_n_tr.shape[1], X_c_tr.shape[1])
model_res.summary()

# ═════════════════════════════════════════════════════════════
# 5. EĞİTİM
# ═════════════════════════════════════════════════════════════
print("\n[4] Modeller eğitiliyor...")

cb_common = [
    keras.callbacks.EarlyStopping(
        monitor="val_ana_cikis_auc", patience=15,
        restore_best_weights=True, mode="max",
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=7, min_lr=1e-7,
    ),
]

# ── Model 1: Basit çoklu girdi ────────────────────────────────
print("    Mimari 1 eğitiliyor...")
cb_mi = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=15,
        restore_best_weights=True, mode="max",
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=7, min_lr=1e-7,
    ),
]
hist_mi = model_mi.fit(
    [X_n_tr, X_c_tr], y_m_tr,
    validation_data=([X_n_val, X_c_val], y_m_val),
    epochs=150, batch_size=32, callbacks=cb_mi, verbose=0,
)

test_mi = model_mi.evaluate([X_n_te, X_c_te], y_m_te, verbose=0)
print(f"    Mimari 1 Test AUC: {test_mi[2]:.4f}")

# ── Model 2: Residual çoklu çıkış ────────────────────────────
print("    Mimari 2 (Residual) eğitiliyor...")
hist_res = model_res.fit(
    [X_n_tr, X_c_tr],
    {"ana_cikis": y_m_tr, "yardimci_cikis": y_a_tr},
    validation_data=(
        [X_n_val, X_c_val],
        {"ana_cikis": y_m_val, "yardimci_cikis": y_a_val},
    ),
    epochs=150, batch_size=32, callbacks=cb_common, verbose=0,
)

test_res_eval = model_res.evaluate(
    [X_n_te, X_c_te],
    {"ana_cikis": y_m_te, "yardimci_cikis": y_a_te},
    verbose=0,
)
main_pred = model_res.predict([X_n_te, X_c_te], verbose=0)[0].flatten()
main_auc  = roc_auc_score(y_m_te, main_pred)
print(f"    Mimari 2 Ana Çıkış AUC: {main_auc:.4f}")

# ═════════════════════════════════════════════════════════════
# 6. SKIP CONNECTION ETKİSİ GÖSTERİMİ
# ═════════════════════════════════════════════════════════════
print("\n[5] Skip connection analizi...")

# Residual bloğun giriş ve çıkışını karşılaştır
intermediate = keras.Model(
    inputs=model_res.input,
    outputs={
        "res1_giris": model_res.get_layer("stem_act").output,
        "res1_cikis": model_res.get_layer("res1_act2").output,
        "res2_cikis": model_res.get_layer("res2_act2").output,
    },
)
mid_out = intermediate.predict([X_n_te[:50], X_c_te[:50]], verbose=0)

print(f"    Residual Blok 1 Giriş  — L2 normu ort: "
      f"{np.mean(np.linalg.norm(mid_out['res1_giris'], axis=1)):.4f}")
print(f"    Residual Blok 1 Çıkış  — L2 normu ort: "
      f"{np.mean(np.linalg.norm(mid_out['res1_cikis'], axis=1)):.4f}")
print(f"    Residual Blok 2 Çıkış  — L2 normu ort: "
      f"{np.mean(np.linalg.norm(mid_out['res2_cikis'], axis=1)):.4f}")
print("    ↑ Skip connection sayesinde bilgi akışı korunur")

# ═════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[6] Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Functional API — Çoklu Giriş/Çıkış & Residual Bloklar",
             fontsize=15, fontweight="bold")

# ── 6a. Mimari 1 — Loss eğrisi ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(hist_mi.history["loss"],     color="#4338CA", lw=2, label="Train")
ax1.plot(hist_mi.history["val_loss"], color="#DB2777", lw=2, ls="--", label="Val")
ax1.set_title("Çoklu Girdi — Loss", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

# ── 6b. Mimari 1 — AUC eğrisi ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(hist_mi.history["auc"],     color="#0D9488", lw=2, label="Train AUC")
ax2.plot(hist_mi.history["val_auc"], color="#D97706", lw=2, ls="--", label="Val AUC")
ax2.axhline(test_mi[2], color="#15803D", ls=":", lw=2, label=f"Test={test_mi[2]:.4f}")
ax2.set_title("Çoklu Girdi — AUC", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# ── 6c. Mimari 2 — Ana + Aux Loss ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(hist_res.history.get("ana_cikis_loss", []),
         color="#4338CA", lw=2, label="Ana Train")
ax3.plot(hist_res.history.get("val_ana_cikis_loss", []),
         color="#4338CA", lw=2, ls="--", label="Ana Val")
ax3.plot(hist_res.history.get("yardimci_cikis_loss", []),
         color="#DB2777", lw=1.5, label="Aux Train", alpha=0.7)
ax3.plot(hist_res.history.get("val_yardimci_cikis_loss", []),
         color="#DB2777", lw=1.5, ls="--", label="Aux Val", alpha=0.7)
ax3.set_title("Residual Çoklu Çıkış — Loss", fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# ── 6d. Mimari 2 — AUC ───────────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
key_auc     = "ana_cikis_auc"
key_val_auc = "val_ana_cikis_auc"
if key_auc in hist_res.history:
    ax4.plot(hist_res.history[key_auc],     color="#7C3AED", lw=2, label="Train AUC")
    ax4.plot(hist_res.history[key_val_auc], color="#D97706", lw=2, ls="--", label="Val AUC")
ax4.axhline(main_auc, color="#15803D", ls=":", lw=2, label=f"Test={main_auc:.4f}")
ax4.set_title("Residual Model — Ana Çıkış AUC", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# ── 6e. Mimari karşılaştırma bar ─────────────────────────────
ax5 = fig.add_subplot(gs[1, 0])
mimariler = ["Çoklu\nGirdi", "Residual+\nAux Loss"]
aucs_comp  = [test_mi[2], main_auc]
colors_bar = ["#4338CA", "#7C3AED"]
bars = ax5.bar(mimariler, aucs_comp, color=colors_bar, alpha=0.82, edgecolor="white")
for bar, v in zip(bars, aucs_comp):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
ax5.set_ylim(0.9, 1.0); ax5.set_title("Mimari AUC Karşılaştırması", fontweight="bold")
ax5.grid(axis="y", alpha=0.3)

# ── 6f. Skip connection aktivasyon analizi ────────────────────
ax6 = fig.add_subplot(gs[1, 1])
labels_norm = ["Res1 Giriş", "Res1 Çıkış", "Res2 Çıkış"]
norms_val = [
    np.mean(np.linalg.norm(mid_out["res1_giris"], axis=1)),
    np.mean(np.linalg.norm(mid_out["res1_cikis"], axis=1)),
    np.mean(np.linalg.norm(mid_out["res2_cikis"], axis=1)),
]
ax6.bar(labels_norm, norms_val, color=["#0D9488","#4338CA","#7C3AED"], alpha=0.82)
ax6.set_title("Skip Connection — Aktivasyon Norm", fontweight="bold")
ax6.set_ylabel("L2 Normu (ort)"); ax6.grid(axis="y", alpha=0.3)

# ── 6g. Model 2 tahmin dağılımı ───────────────────────────────
ax7 = fig.add_subplot(gs[1, 2])
ax7.hist(main_pred[y_m_te==0], bins=25, alpha=0.65,
         color="#DC2626", label="Malign (0)", density=True)
ax7.hist(main_pred[y_m_te==1], bins=25, alpha=0.65,
         color="#15803D", label="Benign (1)", density=True)
ax7.axvline(0.5, color="k", ls="--", lw=1.5, label="Eşik=0.5")
ax7.set_xlabel("Sigmoid Çıktısı"); ax7.set_ylabel("Yoğunluk")
ax7.set_title("Residual Model — Tahmin Dağılımı", fontweight="bold")
ax7.legend(); ax7.grid(axis="y", alpha=0.3)

# ── 6h. Yardımcı çıkış accuracy ──────────────────────────────
ax8 = fig.add_subplot(gs[1, 3])
aux_key     = "yardimci_cikis_accuracy"
val_aux_key = "val_yardimci_cikis_accuracy"
if aux_key in hist_res.history:
    ax8.plot(hist_res.history[aux_key],     color="#DB2777", lw=2, label="Train")
    ax8.plot(hist_res.history[val_aux_key], color="#D97706", lw=2, ls="--", label="Val")
ax8.set_title("Yardımcı Çıkış (3-sınıf) Accuracy", fontweight="bold")
ax8.set_xlabel("Epoch"); ax8.legend(); ax8.grid(alpha=0.3)

plt.savefig("03_functional_api_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 03_functional_api_analiz.png")
plt.close()

# ─── Final rapor ─────────────────────────────────────────────
print("\n    ── Final Rapor ──")
y_pred_main = (main_pred >= 0.5).astype(int)
print(classification_report(y_m_te, y_pred_main,
                             target_names=["Malign", "Benign"]))

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print(f"  Mimari 1 Test AUC: {test_mi[2]:.4f}")
print(f"  Mimari 2 Test AUC: {main_auc:.4f}")
print("  Çıktı: 03_functional_api_analiz.png")
print("=" * 65)
