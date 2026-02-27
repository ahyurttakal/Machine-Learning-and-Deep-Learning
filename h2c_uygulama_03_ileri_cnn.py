"""
=============================================================================
UYGULAMA 03 — İleri CNN: ResNet Blokları & Channel Attention
=============================================================================
Kapsam:
  - Basic ResBlock (ResNet-18 / 34 stili)
  - Bottleneck ResBlock (ResNet-50+ stili)
  - PreActResNet v2 (BN→ReLU→Conv sırası)
  - DepthwiseSeparableConv bloğu
  - SE Block (Squeeze-and-Excitation / Channel Attention)
  - Tüm blokları birleştiren tam model
  - Parametre başına accuracy karşılaştırması
  - Aktivasyon istatistikleri analizi (dying ReLU tespiti)

Veri: CIFAR-10
Kurulum: pip install tensorflow numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 03 — İleri CNN: ResNet Blokları & Channel Attention")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[1] CIFAR-10 yükleniyor...")
(X_raw, y_raw), (X_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()
X_raw  = X_raw.astype("float32") / 255.0
X_test = X_test_raw.astype("float32") / 255.0
y_raw  = y_raw.flatten(); y_test = y_test_raw.flatten()
val_size = 5000
X_val, X_train = X_raw[:val_size], X_raw[val_size:]
y_val, y_train = y_raw[:val_size], y_raw[val_size:]
mean = X_train.mean(axis=(0,1,2), keepdims=True)
std  = X_train.std(axis=(0,1,2),  keepdims=True) + 1e-7
X_train_n = (X_train - mean) / std
X_val_n   = (X_val   - mean) / std
X_test_n  = (X_test  - mean) / std

BATCH_SIZE = 128
EPOCHS     = 30

# Augmentation
aug_layer = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="aug")

def make_ds(X, y, aug=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X, tf.float32), tf.constant(y, tf.int32)))
    if shuffle: ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if aug:
        ds = ds.map(lambda x,y: (aug_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train_n, y_train, aug=True)
val_ds   = make_ds(X_val_n,   y_val,   shuffle=False)
test_ds  = make_ds(X_test_n,  y_test,  shuffle=False)

# ═════════════════════════════════════════════════════════════
# 2. BLOK TANIMLARI
# ═════════════════════════════════════════════════════════════
print("\n[2] Blok tanımları yazılıyor...")

# ─── Basic ResBlock ──────────────────────────────────────────
def basic_resblock(x, filters, strides=1, name="basic"):
    """
    ResNet v1 Basic Block:
    Conv(stride) → BN → ReLU → Conv → BN → Add → ReLU
    """
    shortcut = x

    # Boyut uyumsuzluğu: projeksiyon shortcut
    if strides != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=strides, use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    # Ana yol
    x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x

# ─── Bottleneck ResBlock ─────────────────────────────────────
def bottleneck_resblock(x, filters, strides=1, expansion=4, name="bottle"):
    """
    ResNet-50+ Bottleneck Block:
    Conv 1×1 (compress) → Conv 3×3 → Conv 1×1 (expand) → Add → ReLU
    """
    inner_filters = filters // expansion
    shortcut      = x

    if strides != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=strides, use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    # 1×1 Compress
    x = layers.Conv2D(inner_filters, 1, use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    # 3×3 Spatial
    x = layers.Conv2D(inner_filters, 3, strides=strides, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4), name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    # 1×1 Expand
    x = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_c3")(x)
    x = layers.BatchNormalization(name=f"{name}_bn3")(x)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu3")(x)
    return x

# ─── PreActResBlock v2 ────────────────────────────────────────
def preact_resblock(x, filters, strides=1, name="preact"):
    """
    PreActResNet v2 (He et al. 2016):
    BN → ReLU → Conv → BN → ReLU → Conv → Add (ReLU YOK son)
    Gradyan akışı daha temiz; shortcut üzerinde normalizasyon yok.
    """
    shortcut = x

    if strides != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=strides, use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_proj")(shortcut)

    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4), name=f"{name}_conv1")(x)

    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4), name=f"{name}_conv2")(x)

    # Doğrudan add — son ReLU yok (v2'nin farkı)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return x

# ─── SE Block (Squeeze-and-Excitation) ───────────────────────
def se_block(x, ratio=16, name="se"):
    """
    Squeeze-and-Excitation (Channel Attention):
    Her kanalın globl istatistiğini kullanarak kanalları yeniden ağırlıklandırır.
    Hu et al. 2018 — ImageNet kazananı.
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Dense(channels // ratio, activation="relu",
                      use_bias=False, name=f"{name}_fc1")(se)
    se = layers.Dense(channels, activation="sigmoid",
                      use_bias=False, name=f"{name}_fc2")(se)
    se = layers.Reshape((1,1,channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])

# ─── DS Conv Bloğu ────────────────────────────────────────────
def ds_block(x, filters, strides=1, name="ds"):
    """Depthwise Separable Conv bloğu."""
    if strides > 1:
        x = layers.DepthwiseConv2D(3, strides=strides, padding="same", use_bias=False,
                                    name=f"{name}_dw")(x)
    else:
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                                    name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x

print("    ✅ BasicResBlock, BottleneckResBlock, PreActResBlock, SE Block, DSBlock hazır")

# ═════════════════════════════════════════════════════════════
# 3. MODELLERİ OLUŞTUR
# ═════════════════════════════════════════════════════════════
print("\n[3] Modeller oluşturuluyor...")

def build_basic_resnet(name="basic_resnet"):
    """Basic ResBlock kullanan küçük ResNet."""
    inp = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    x = basic_resblock(x, 64,  strides=1, name="b1_1")
    x = basic_resblock(x, 64,  strides=1, name="b1_2")
    x = basic_resblock(x, 128, strides=2, name="b2_1")
    x = basic_resblock(x, 128, strides=1, name="b2_2")
    x = basic_resblock(x, 256, strides=2, name="b3_1")
    x = basic_resblock(x, 256, strides=1, name="b3_2")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name=name)

def build_bottleneck_resnet(name="bottle_resnet"):
    """Bottleneck ResBlock kullanan model."""
    inp = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    x = bottleneck_resblock(x, 128, strides=1, name="b1_1")
    x = bottleneck_resblock(x, 128, strides=1, name="b1_2")
    x = bottleneck_resblock(x, 256, strides=2, name="b2_1")
    x = bottleneck_resblock(x, 256, strides=1, name="b2_2")
    x = bottleneck_resblock(x, 512, strides=2, name="b3_1")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name=name)

def build_preact_resnet(name="preact_resnet"):
    """PreAct ResNet v2."""
    inp = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name="stem")(inp)
    x = preact_resblock(x, 64,  strides=1, name="p1_1")
    x = preact_resblock(x, 64,  strides=1, name="p1_2")
    x = preact_resblock(x, 128, strides=2, name="p2_1")
    x = preact_resblock(x, 128, strides=1, name="p2_2")
    x = preact_resblock(x, 256, strides=2, name="p3_1")
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Activation("relu", name="final_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name=name)

def build_se_resnet(name="se_resnet"):
    """SE Block'lu ResNet."""
    inp = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name="stem")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = basic_resblock(x, 64, name="b1_1")
    x = se_block(x, ratio=8, name="se1")
    x = basic_resblock(x, 128, strides=2, name="b2_1")
    x = se_block(x, ratio=8, name="se2")
    x = basic_resblock(x, 256, strides=2, name="b3_1")
    x = se_block(x, ratio=16, name="se3")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name=name)

def build_ds_net(name="ds_net"):
    """DepthwiseSeparable tabanlı hafif model."""
    inp = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(32, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name="stem")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    for f, s in [(64,1),(64,1),(128,2),(128,1),(256,2),(256,1)]:
        x = ds_block(x, f, strides=s, name=f"ds_f{f}_s{s}_{np.random.randint(999)}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name=name)

# ═════════════════════════════════════════════════════════════
# 4. EĞİTİM
# ═════════════════════════════════════════════════════════════
print("\n[4] Modeller eğitiliyor...")

def train_model(model, name=""):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=5, min_lr=1e-7,
        ),
    ]
    h = model.fit(train_ds, validation_data=val_ds,
                  epochs=EPOCHS, callbacks=cbs, verbose=0)
    ta = model.evaluate(test_ds, verbose=0)[1]
    params = model.count_params()
    best_val = max(h.history["val_accuracy"])
    print(f"    {name:<25}: test_acc={ta:.4f}  val_acc={best_val:.4f}  params={params:,}")
    return h.history, ta, params

model_defs = [
    ("Basic ResNet",      build_basic_resnet),
    ("Bottleneck ResNet", build_bottleneck_resnet),
    ("PreAct ResNet v2",  build_preact_resnet),
    ("SE-ResNet",         build_se_resnet),
    ("DS-Net (hafif)",    build_ds_net),
]

results = {}
for mname, mfunc in model_defs:
    tf.random.set_seed(42)
    hist, ta, params = train_model(mfunc(), mname)
    results[mname] = {"history": hist, "test_acc": ta, "params": params}

# ═════════════════════════════════════════════════════════════
# 5. AKTİVASYON İSTATİSTİKLERİ (Dying ReLU tespiti)
# ═════════════════════════════════════════════════════════════
print("\n[5] Aktivasyon istatistikleri analizi...")

# En iyi modeli yeniden al
tf.random.set_seed(42)
analysis_model = build_basic_resnet("analysis_basic")
analysis_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
analysis_model.fit(train_ds, validation_data=val_ds,
                   epochs=15, verbose=0)

# Tüm ReLU çıktılarını kaydet
relu_layers = [l for l in analysis_model.layers if "relu" in l.name]
activation_model = keras.Model(
    inputs=analysis_model.input,
    outputs=[l.output for l in relu_layers],
)
sample_batch = X_train_n[:512]
all_acts = activation_model.predict(sample_batch, verbose=0)
if not isinstance(all_acts, list): all_acts = [all_acts]

print("    Katman başına sıfır aktivasyon yüzdesi (Dying ReLU):")
dead_stats = {}
for layer, act in zip(relu_layers, all_acts):
    pct_dead = np.mean(act == 0) * 100
    dead_stats[layer.name] = pct_dead
    status = "⚠️  Dying!" if pct_dead > 80 else "✅"
    print(f"    {layer.name:<35}: {pct_dead:5.1f}% sıfır  {status}")

# ═════════════════════════════════════════════════════════════
# 6. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[6] Görselleştirmeler hazırlanıyor...")

palette = {
    "Basic ResNet":      "#0F766E",
    "Bottleneck ResNet": "#059669",
    "PreAct ResNet v2":  "#0891B2",
    "SE-ResNet":         "#7C3AED",
    "DS-Net (hafif)":    "#D97706",
}

fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("İleri CNN — ResNet Blokları & Channel Attention", fontsize=15, fontweight="bold")

# ── 6a. Val accuracy ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for name, res in results.items():
    ax1.plot(res["history"]["val_accuracy"], lw=2,
             color=palette[name], label=f"{name} ({res['test_acc']:.4f})")
ax1.set_title("Val Accuracy — Tüm Modeller", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Accuracy")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# ── 6b. Parametre vs Accuracy ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
for name, res in results.items():
    ax2.scatter(res["params"]/1e6, res["test_acc"],
                s=120, color=palette[name], zorder=3, label=name)
    ax2.annotate(name.split("(")[0][:12].strip(),
                 (res["params"]/1e6, res["test_acc"]),
                 textcoords="offset points", xytext=(5,4), fontsize=9)
ax2.set_xlabel("Parametre (M)"); ax2.set_ylabel("Test Accuracy")
ax2.set_title("Parametre Verimliliği", fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# ── 6c. Test accuracy bar ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
names_b = list(results.keys())
accs_b  = [r["test_acc"] for r in results.values()]
clrs_b  = [palette[n] for n in names_b]
bars = ax3.barh([n.split("(")[0].strip() for n in names_b],
                accs_b, color=clrs_b, alpha=0.82)
for bar, v in zip(bars, accs_b):
    ax3.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax3.set_xlim(0.60, 0.90)
ax3.set_title("Test Accuracy Karşılaştırması", fontweight="bold")
ax3.grid(axis="x", alpha=0.3)

# ── 6d. Dying ReLU bar ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
layer_names_short = [n[:20] for n in dead_stats.keys()]
dead_pcts = list(dead_stats.values())
bar_colors = ["#DC2626" if p > 80 else "#0F766E" for p in dead_pcts]
ax4.barh(layer_names_short, dead_pcts, color=bar_colors, alpha=0.82)
ax4.axvline(80, color="#DC2626", ls="--", lw=1.5, label="Dying ReLU eşiği")
ax4.set_title("Katman Başına Sıfır Akt. Oranı\n(Dying ReLU Analizi)", fontweight="bold")
ax4.set_xlabel("% Sıfır Aktivasyon"); ax4.legend(); ax4.grid(axis="x", alpha=0.3)

# ── 6e. Parametre ve verimlilik radar ────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
acc_per_M = {n: r["test_acc"]/(r["params"]/1e6) for n, r in results.items()}
ax5.bar([n.split("(")[0][:12].strip() for n in acc_per_M.keys()],
        acc_per_M.values(),
        color=[palette[n] for n in acc_per_M.keys()], alpha=0.82)
ax5.set_title("Parametre Başına Accuracy\n(Verimlilik)", fontweight="bold")
ax5.set_ylabel("Accuracy / Milyon Param")
ax5.tick_params(axis="x", rotation=20); ax5.grid(axis="y", alpha=0.3)

# ── 6f. Val loss karşılaştırma ────────────────────────────────
ax6 = fig.add_subplot(gs[1, 3])
for name, res in results.items():
    ax6.plot(res["history"]["val_loss"], lw=2,
             color=palette[name], label=name.split("(")[0].strip())
ax6.set_title("Val Loss Karşılaştırması", fontweight="bold")
ax6.set_xlabel("Epoch"); ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

plt.savefig("03_ileri_cnn_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 03_ileri_cnn_analiz.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 03 TAMAMLANDI")
best_m = max(results, key=lambda k: results[k]["test_acc"])
print(f"  En iyi model: {best_m} (test_acc={results[best_m]['test_acc']:.4f})")
print("  Çıktı: 03_ileri_cnn_analiz.png")
print("=" * 65)
