"""
=============================================================================
UYGULAMA 01 — CNN Temelleri & Mimari Tasarımı
=============================================================================
Kapsam:
  - Conv2D + BatchNorm + Activation + MaxPool + GAP tam pipeline
  - Filtre sayısı ablasyon çalışması (16 / 32 / 64 / 128)
  - Kernel boyutu karşılaştırması (3×3 vs 5×5 vs 7×7)
  - Stride=2 Conv vs MaxPool karşılaştırması
  - DepthwiseSeparableConv vs standart Conv2D
  - Feature map görselleştirme (conv1 filtrelerinin çıktıları)
  - Parametre sayısı & FLOPs analizi

Veri: CIFAR-10 (10 sınıf, 32×32 renkli)
Kurulum: pip install tensorflow numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import time
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

print("=" * 65)
print("  UYGULAMA 01 — CNN Temelleri & Mimari Tasarımı")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[1] CIFAR-10 yükleniyor...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()

# Normalize [0,1]
X_train = X_train_raw.astype("float32") / 255.0
X_test  = X_test_raw.astype("float32")  / 255.0
y_train = y_train_raw.flatten()
y_test  = y_test_raw.flatten()

# Eğitim / validasyon ayrımı
val_size  = 5000
X_val, X_train_ = X_train[:val_size], X_train[val_size:]
y_val, y_train_ = y_train[:val_size], y_train[val_size:]

print(f"    Eğitim : {X_train_.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"    Şekil  : {X_train_.shape[1:]}  |  Sınıf sayısı: {len(np.unique(y_train_))}")

# Global ortalamlama & standart sapma (per-channel)
mean = X_train_.mean(axis=(0,1,2), keepdims=True)
std  = X_train_.std(axis=(0,1,2), keepdims=True) + 1e-7
X_train_n = (X_train_ - mean) / std
X_val_n   = (X_val   - mean) / std
X_test_n  = (X_test  - mean) / std

# ═════════════════════════════════════════════════════════════
# 2. CONV BLOĞU & MODEL OLUŞTURUCULAR
# ═════════════════════════════════════════════════════════════
def conv_block(x, filters, kernel=3, strides=1, padding="same", name="cb"):
    """Conv → BatchNorm → ReLU bloğu."""
    x = layers.Conv2D(
        filters, kernel, strides=strides, padding=padding,
        use_bias=False, kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x

def depthwise_sep_block(x, filters, name="ds"):
    """DepthwiseConv → Conv1×1 → BN → ReLU."""
    x = layers.DepthwiseConv2D(
        3, padding="same", use_bias=False,
        depthwise_initializer="he_normal", name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(
        filters, 1, use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x

def build_cnn(
    filters=(32, 64, 128),
    kernel=3,
    use_stride=False,     # True → stride=2 Conv, False → MaxPool
    use_dsconv=False,     # True → DepthwiseSep kullan
    use_gap=True,         # True → GlobalAvgPool, False → Flatten
    dropout_rate=0.3,
    name="cnn",
):
    """Esnek CNN oluşturucu."""
    inp = keras.Input(shape=(32,32,3))
    x   = inp

    for i, f in enumerate(filters):
        if use_dsconv and i > 0:   # İlk blok standart
            x = depthwise_sep_block(x, f, name=f"ds{i+1}")
        else:
            x = conv_block(x, f, kernel=kernel, name=f"cb{i+1}")

        # Downsample: stride=2 Conv veya MaxPool
        if use_stride:
            x = conv_block(x, f, kernel=3, strides=2, name=f"down{i+1}")
        else:
            x = layers.MaxPooling2D(2, name=f"pool{i+1}")(x)

        x = layers.Dropout(dropout_rate * (i+1) / len(filters), name=f"drop{i+1}")(x)

    # Classifier
    if use_gap:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="fc_drop")(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ═════════════════════════════════════════════════════════════
# 3. TEMEL MODEL — ÖZET
# ═════════════════════════════════════════════════════════════
print("\n[2] Temel CNN modeli:")
base_model = build_cnn(name="base_cnn")
base_model.summary()
print(f"    Toplam parametre: {base_model.count_params():,}")

# ═════════════════════════════════════════════════════════════
# 4. EĞİTİM YARDIMCISI
# ═════════════════════════════════════════════════════════════
EPOCHS_QUICK = 20
BATCH_SIZE   = 128

def quick_train(model, X_tr, y_tr, X_v, y_v, epochs=EPOCHS_QUICK, verbose=0):
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=5, min_lr=1e-7,
        ),
    ]
    t0 = time.time()
    h  = model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
                   epochs=epochs, batch_size=BATCH_SIZE,
                   callbacks=cbs, verbose=verbose)
    elapsed = time.time() - t0
    test_acc = model.evaluate(X_test_n, y_test, verbose=0)[1]
    best_val = max(h.history["val_accuracy"])
    return h.history, test_acc, best_val, elapsed

# ═════════════════════════════════════════════════════════════
# 5. FİLTRE SAYISI ABLASYONU
# ═════════════════════════════════════════════════════════════
print("\n[3] Filtre sayısı ablasyonu (4 konfigürasyon)...")

filter_configs = {
    "Küçük (16-32-64)":    (16,  32,  64),
    "Orta  (32-64-128)":   (32,  64, 128),
    "Büyük (64-128-256)":  (64, 128, 256),
    "Geniş (128-256-512)": (128,256, 512),
}
filter_results = {}
for name, f_cfg in filter_configs.items():
    tf.random.set_seed(42)
    m = build_cnn(filters=f_cfg, name=name.split()[0])
    hist, t_acc, v_acc, elapsed = quick_train(m, X_train_n, y_train_, X_val_n, y_val)
    params = m.count_params()
    filter_results[name] = {
        "test_acc": t_acc, "val_acc": v_acc,
        "params": params, "history": hist, "time": elapsed,
    }
    print(f"    {name}: test_acc={t_acc:.4f}  params={params:,}  t={elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════
# 6. KERNEL BOYUTU KARŞILAŞTIRMASI
# ═════════════════════════════════════════════════════════════
print("\n[4] Kernel boyutu karşılaştırması (3×3 / 5×5 / 7×7)...")

kernel_results = {}
for k in [3, 5, 7]:
    tf.random.set_seed(42)
    m = build_cnn(filters=(32,64,128), kernel=k, name=f"k{k}")
    hist, t_acc, v_acc, elapsed = quick_train(m, X_train_n, y_train_, X_val_n, y_val)
    kernel_results[f"{k}×{k}"] = {
        "test_acc": t_acc, "val_acc": v_acc,
        "params": m.count_params(), "history": hist, "time": elapsed,
    }
    print(f"    kernel {k}×{k}: test_acc={t_acc:.4f}  params={m.count_params():,}  t={elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════
# 7. STRIDE vs MAXPOOL / GAP vs FLATTEN / DSConv vs CONV2D
# ═════════════════════════════════════════════════════════════
print("\n[5] Mimari seçenekleri karşılaştırması...")

arch_configs = {
    "Standart (MaxPool+GAP)":  dict(use_stride=False, use_dsconv=False, use_gap=True),
    "Stride Conv":             dict(use_stride=True,  use_dsconv=False, use_gap=True),
    "DSConv":                  dict(use_stride=False, use_dsconv=True,  use_gap=True),
    "GAP yerine Flatten":      dict(use_stride=False, use_dsconv=False, use_gap=False),
}
arch_results = {}
for name, cfg in arch_configs.items():
    tf.random.set_seed(42)
    m = build_cnn(filters=(32,64,128), name=name.split()[0], **cfg)
    hist, t_acc, v_acc, elapsed = quick_train(m, X_train_n, y_train_, X_val_n, y_val)
    arch_results[name] = {
        "test_acc": t_acc, "val_acc": v_acc,
        "params": m.count_params(), "history": hist, "time": elapsed,
    }
    print(f"    {name:<30}: test_acc={t_acc:.4f}  params={m.count_params():,}")

# ═════════════════════════════════════════════════════════════
# 8. FEATURE MAP GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[6] Feature map görselleştirme için en iyi model eğitiliyor...")

tf.random.set_seed(42)
best_model = build_cnn(filters=(64,128,256), name="best_cnn_viz")
_, _, _, _ = quick_train(best_model, X_train_n, y_train_, X_val_n, y_val,
                          epochs=25, verbose=1)

# İlk Conv katmanının çıktısını alacak ara model
first_conv_name = "cb1_relu"
feat_model = keras.Model(
    inputs=best_model.input,
    outputs=best_model.get_layer(first_conv_name).output,
)

# 8 farklı sınıftan birer örnek seç
sample_imgs, sample_labels = [], []
for cls in range(8):
    idx = np.where(y_test == cls)[0][0]
    sample_imgs.append(X_test_n[idx])
    sample_labels.append(cls)
sample_imgs = np.array(sample_imgs)

feature_maps = feat_model.predict(sample_imgs, verbose=0)
print(f"    Feature map şekli: {feature_maps.shape}  (8 örnek × filtre sayısı)")

# ═════════════════════════════════════════════════════════════
# 9. PARAMETRE VE FLOPs ANALİZİ
# ═════════════════════════════════════════════════════════════
print("\n[7] Parametre analizi:")
print(f"    {'Mimari':<32} {'Parametre':>12} {'Test Acc':>10}")
print("    " + "─" * 55)
for name, r in {**filter_results, **arch_results}.items():
    print(f"    {name:<32} {r['params']:>12,} {r['test_acc']:>10.4f}")

# ═════════════════════════════════════════════════════════════
# 10. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[8] Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("CNN Temelleri — CIFAR-10 Ablasyon Çalışması", fontsize=15, fontweight="bold")

# ── 10a. Filtre sayısı — test acc ────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for (name, r), color in zip(filter_results.items(),
                             ["#6B7280","#0F766E","#059669","#1D4ED8"]):
    ax1.plot(r["history"]["val_accuracy"], lw=2, color=color, label=f"{name} ({r['test_acc']:.4f})")
ax1.set_title("Filtre Sayısı Ablasyonu — Val Accuracy", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Accuracy")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# ── 10b. Parametre vs doğruluk scatter ───────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
all_res_combined = {**filter_results, **kernel_results, **arch_results}
param_vals = [r["params"]/1e6 for r in all_res_combined.values()]
acc_vals   = [r["test_acc"]   for r in all_res_combined.values()]
colors_scatter = (["#6B7280","#0F766E","#059669","#1D4ED8"] +
                  ["#7C3AED","#D97706","#DC2626"] +
                  ["#0F766E","#1D4ED8","#D97706","#DC2626"])
for name, (pv, av, col) in zip(
    all_res_combined.keys(),
    [(p,a,c) for p,a,c in zip(param_vals, acc_vals, colors_scatter)]
):
    ax2.scatter(pv, av, s=90, color=col, zorder=3)
    ax2.annotate(name.split("(")[0][:12].strip(),
                 (pv, av), textcoords="offset points",
                 xytext=(5,3), fontsize=8)
ax2.set_xlabel("Parametre (Milyon)"); ax2.set_ylabel("Test Accuracy")
ax2.set_title("Parametre vs Test Accuracy", fontweight="bold")
ax2.grid(alpha=0.3)

# ── 10c. Kernel boyutu karşılaştırması ───────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for (name, r), color in zip(kernel_results.items(), ["#0F766E","#059669","#0891B2"]):
    ax3.plot(r["history"]["val_accuracy"], lw=2, color=color, label=name)
ax3.set_title("Kernel Boyutu Karşılaştırması", fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.legend(); ax3.grid(alpha=0.3)

# ── 10d. Mimari seçenekleri bar ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
arch_names = [n.split("(")[0][:14].strip() for n in arch_results.keys()]
arch_accs  = [r["test_acc"] for r in arch_results.values()]
arch_params = [r["params"]/1e6 for r in arch_results.values()]
x_pos = np.arange(len(arch_names))
bars = ax4.bar(x_pos, arch_accs, color=["#0F766E","#059669","#0891B2","#D97706"], alpha=0.82)
for bar, v in zip(bars, arch_accs):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
ax4.set_xticks(x_pos); ax4.set_xticklabels(arch_names, fontsize=8, rotation=10)
ax4.set_ylim(0.6, 0.85); ax4.set_title("Mimari Seçenekleri — Test Acc", fontweight="bold")
ax4.grid(axis="y", alpha=0.3)

# ── 10e. Feature maps — örnek görüntü ────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
sample_orig = (sample_imgs[0] * std[0] + mean[0]).clip(0,1)
ax5.imshow(sample_orig)
ax5.set_title(f"Girdi: {CIFAR_CLASSES[sample_labels[0]]}", fontweight="bold")
ax5.axis("off")

# ── 10f. Feature map (ilk 12 kanal) ──────────────────────────
ax6 = fig.add_subplot(gs[1, 3])
n_show = 9
cols_fm = 3
rows_fm = n_show // cols_fm
fig2, axes2 = plt.subplots(rows_fm, cols_fm, figsize=(8,6))
fig2.suptitle(f"Feature Maps — {CIFAR_CLASSES[sample_labels[0]]} (Conv1 çıktısı)", fontweight="bold")
for idx in range(n_show):
    ax_t = axes2[idx//cols_fm, idx%cols_fm]
    fmap = feature_maps[0, :, :, idx]
    ax_t.imshow(fmap, cmap="viridis")
    ax_t.axis("off"); ax_t.set_title(f"Kanal {idx+1}", fontsize=8)
plt.tight_layout()
plt.savefig("01b_feature_maps.png", dpi=130, bbox_inches="tight")
plt.close(fig2)
print("    ✅ Kaydedildi: 01b_feature_maps.png")

ax6.axis("off")
ax6.text(0.5, 0.5, "Feature Maps\nayrı kaydedildi:\n01b_feature_maps.png",
         ha="center", va="center", fontsize=12, transform=ax6.transAxes)

# ── 10g. Eğitim süresi karşılaştırması ───────────────────────
ax7 = fig.add_subplot(gs[2, :2])
all_names  = list(filter_results.keys()) + list(arch_results.keys())
all_times  = [r["time"] for r in list(filter_results.values()) + list(arch_results.values())]
all_colors = (["#6B7280","#0F766E","#059669","#1D4ED8"] +
              ["#0F766E","#1D4ED8","#D97706","#DC2626"])
x_all = np.arange(len(all_names))
ax7.bar(x_all, all_times, color=all_colors, alpha=0.8, edgecolor="white")
ax7.set_xticks(x_all)
ax7.set_xticklabels([n.split("(")[0][:16].strip() for n in all_names],
                    rotation=30, ha="right", fontsize=9)
ax7.set_ylabel("Eğitim Süresi (sn)"); ax7.set_title("Eğitim Süresi Karşılaştırması", fontweight="bold")
ax7.grid(axis="y", alpha=0.3)

# ── 10h. Ornek görüntüler ─────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2:])
ax8.axis("off")
n_show_img = 8
for idx in range(n_show_img):
    ax_s = fig.add_axes([0.505+idx*0.061, 0.03, 0.056, 0.12])
    orig = (sample_imgs[idx] * std[0] + mean[0]).clip(0,1)
    ax_s.imshow(orig)
    ax_s.set_title(CIFAR_CLASSES[sample_labels[idx]], fontsize=7)
    ax_s.axis("off")

plt.savefig("01_cnn_temelleri_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 01_cnn_temelleri_analiz.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print("  Çıktılar: 01_cnn_temelleri_analiz.png  |  01b_feature_maps.png")
print("=" * 65)
