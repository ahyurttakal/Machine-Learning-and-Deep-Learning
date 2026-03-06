"""
=============================================================================
UYGULAMA 02 — Data Augmentation Stratejileri
=============================================================================
Kapsam:
  - tf.data.Dataset pipeline: shuffle, cache, prefetch, AUTOTUNE
  - Keras Preprocessing Layers (GPU içi augmentation)
  - CutMix implementasyonu (bölge yapıştırma + label mixing)
  - MixUp implementasyonu (piksel + label interpolasyonu)
  - Augmentation şiddeti ablasyon çalışması
  - Augmentation'lı vs augmentationsız val_accuracy karşılaştırması
  - Augmentasyonlu görüntüleri görselleştirme

Veri: CIFAR-10
Kurulum: pip install tensorflow numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

print("=" * 65)
print("  UYGULAMA 02 — Data Augmentation Stratejileri")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[1] CIFAR-10 yükleniyor...")
(X_raw, y_raw), (X_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()
X_raw    = X_raw.astype("float32") / 255.0
X_test   = X_test_raw.astype("float32") / 255.0
y_raw    = y_raw.flatten()
y_test   = y_test_raw.flatten()

val_size  = 5000
X_val,   X_train = X_raw[:val_size],  X_raw[val_size:]
y_val,   y_train = y_raw[:val_size],  y_raw[val_size:]

mean = X_train.mean(axis=(0,1,2), keepdims=True)
std  = X_train.std(axis=(0,1,2), keepdims=True)  + 1e-7

X_train_n = (X_train - mean) / std
X_val_n   = (X_val   - mean) / std
X_test_n  = (X_test  - mean) / std

print(f"    Eğitim: {X_train_n.shape[0]:,} | Val: {X_val_n.shape[0]:,}")

# ═════════════════════════════════════════════════════════════
# 2. AUGMENTATİON MODÜLLERI
# ═════════════════════════════════════════════════════════════
print("\n[2] Augmentation modülleri tanımlanıyor...")

def build_augmenter(
    flip=True, rotation=0.1, zoom=0.15,
    brightness=0.2, contrast=0.2, translation=0.1,
):
    """Parametre kontrollü Keras augmenter."""
    aug_layers = []
    if flip:
        aug_layers.append(layers.RandomFlip("horizontal"))
    if rotation > 0:
        aug_layers.append(layers.RandomRotation(rotation))
    if zoom > 0:
        aug_layers.append(layers.RandomZoom(zoom))
    if brightness > 0:
        aug_layers.append(layers.RandomBrightness(brightness))
    if contrast > 0:
        aug_layers.append(layers.RandomContrast(contrast))
    if translation > 0:
        aug_layers.append(layers.RandomTranslation(translation, translation))
    return keras.Sequential(aug_layers, name="augmenter")

# ─── CutMix ──────────────────────────────────────────────────
def cutmix(images, labels, alpha=1.0):
    """
    CutMix: Bir görüntünün dikdörtgen bölgesini diğerinden doldur.
    Etiketler de lambda oranında karışır.
    """
    batch_size = tf.shape(images)[0]
    H = tf.shape(images)[1]
    W = tf.shape(images)[2]

    lam = tf.cast(np.random.beta(alpha, alpha), tf.float32)

    # Rastgele bölge boyutu
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(cut_ratio * tf.cast(H, tf.float32), tf.int32)
    cut_w = tf.cast(cut_ratio * tf.cast(W, tf.float32), tf.int32)

    # Merkez koordinatı
    cx = tf.random.uniform([], 0, W, dtype=tf.int32)
    cy = tf.random.uniform([], 0, H, dtype=tf.int32)

    x1 = tf.maximum(cx - cut_w // 2, 0)
    y1 = tf.maximum(cy - cut_h // 2, 0)
    x2 = tf.minimum(cx + cut_w // 2, W)
    y2 = tf.minimum(cy + cut_h // 2, H)

    # Shuffle ile ikinci batch oluştur
    indices = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)

    # Maske: y1:y2, x1:x2 bölgesi images2'den gelir
    mask = tf.zeros_like(images)
    ones_patch = tf.ones([y2-y1, x2-x1, 3])
    mask_padded = tf.pad(ones_patch,
                         [[y1, H-y2], [x1, W-x2], [0,0]])
    mixed = images * (1 - mask_padded) + images2 * mask_padded

    # Lambda güncelle (gerçek alan oranı)
    lam = 1.0 - tf.cast((y2-y1)*(x2-x1), tf.float32) / tf.cast(H*W, tf.float32)

    # One-hot mix
    labels_oh  = tf.one_hot(labels,  10)
    labels2_oh = tf.one_hot(labels2, 10)
    mixed_labels = lam * labels_oh + (1.0 - lam) * labels2_oh
    return mixed, mixed_labels

def mixup(images, labels, alpha=0.4):
    """
    MixUp: İki görüntüyü piksel düzeyinde karıştır.
    Etiketler de aynı oranda karışır.
    """
    batch_size = tf.shape(images)[0]
    lam = tf.cast(np.random.beta(alpha, alpha), tf.float32)

    indices = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)

    mixed       = lam * images + (1 - lam) * images2
    labels_oh   = tf.one_hot(labels,  10)
    labels2_oh  = tf.one_hot(labels2, 10)
    mixed_labels = lam * labels_oh + (1-lam) * labels2_oh
    return mixed, mixed_labels

print("    ✅ CutMix, MixUp hazır")

# ═════════════════════════════════════════════════════════════
# 3. tf.data PİPELİNE
# ═════════════════════════════════════════════════════════════
BATCH_SIZE = 128

def make_pipeline(X, y, augmenter=None, augtype=None, shuffle=True):
    """
    tf.data pipeline: shuffle → augment → batch → prefetch.
    augtype: None / 'cutmix' / 'mixup'
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X, dtype=tf.float32), tf.constant(y, dtype=tf.int32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)

    if augmenter is not None:
        @tf.function
        def apply_aug(imgs, lbls):
            imgs = augmenter(imgs, training=True)
            return imgs, lbls
        ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)

    if augtype == "cutmix":
        @tf.function
        def apply_cutmix(imgs, lbls):
            return cutmix(imgs, lbls, alpha=1.0)
        ds = ds.map(apply_cutmix, num_parallel_calls=tf.data.AUTOTUNE)

    elif augtype == "mixup":
        @tf.function
        def apply_mixup(imgs, lbls):
            return mixup(imgs, lbls, alpha=0.4)
        ds = ds.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE).cache()

# ═════════════════════════════════════════════════════════════
# 4. TEMEL CNN MODELİ (Basit)
# ═════════════════════════════════════════════════════════════
def build_simple_cnn(name="aug_cnn"):
    inp = keras.Input(shape=(32,32,3))
    x = inp
    for f in [64, 128, 256]:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inp, out, name=name)
    return model

def compile_model(model, lr=1e-3, use_cce=True):
    loss = "sparse_categorical_crossentropy" if use_cce else "categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,
        metrics=["accuracy"],
    )
    return model

# ═════════════════════════════════════════════════════════════
# 5. AUGMENTATİON STRATEJİLERİ KARŞILAŞTIRMASI
# ═════════════════════════════════════════════════════════════
print("\n[3] Augmentation stratejileri karşılaştırılıyor (her biri 25 epoch)...")

EPOCHS = 25

def train_pipeline(train_ds, val_ds, model, epochs=EPOCHS):
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
                  epochs=epochs, callbacks=cbs, verbose=0)
    test_acc = model.evaluate(
        tf.data.Dataset.from_tensor_slices((X_test_n, y_test)).batch(256).prefetch(2),
        verbose=0
    )[1]
    return h.history, test_acc

val_ds_base = (tf.data.Dataset.from_tensor_slices(
    (tf.constant(X_val_n,dtype=tf.float32), tf.constant(y_val,dtype=tf.int32)))
    .batch(256).prefetch(tf.data.AUTOTUNE))

strategies = {}

# ─── 5a. Augmentation yok ────────────────────────────────────
print("    [a] Augmentation yok...")
tf.random.set_seed(42)
ds_none  = make_pipeline(X_train_n, y_train, augmenter=None, augtype=None)
m        = compile_model(build_simple_cnn("no_aug"))
hist, ta = train_pipeline(ds_none, val_ds_base, m)
strategies["Yok"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

# ─── 5b. Basit augmentation ──────────────────────────────────
print("    [b] Basit augmentation (flip+rotation)...")
tf.random.set_seed(42)
aug_basit = build_augmenter(flip=True, rotation=0.1, zoom=0, brightness=0, contrast=0, translation=0)
ds_basit  = make_pipeline(X_train_n, y_train, augmenter=aug_basit)
m         = compile_model(build_simple_cnn("basic_aug"))
hist, ta  = train_pipeline(ds_basit, val_ds_base, m)
strategies["Basit (flip+rot)"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

# ─── 5c. Orta augmentation ───────────────────────────────────
print("    [c] Orta augmentation...")
tf.random.set_seed(42)
aug_orta = build_augmenter(flip=True, rotation=0.1, zoom=0.15, brightness=0.2, contrast=0.15, translation=0.1)
ds_orta  = make_pipeline(X_train_n, y_train, augmenter=aug_orta)
m        = compile_model(build_simple_cnn("mid_aug"))
hist, ta = train_pipeline(ds_orta, val_ds_base, m)
strategies["Orta"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

# ─── 5d. Agresif augmentation ────────────────────────────────
print("    [d] Agresif augmentation...")
tf.random.set_seed(42)
aug_agresif = build_augmenter(flip=True, rotation=0.25, zoom=0.3, brightness=0.3, contrast=0.3, translation=0.2)
ds_agresif  = make_pipeline(X_train_n, y_train, augmenter=aug_agresif)
m           = compile_model(build_simple_cnn("hard_aug"))
hist, ta    = train_pipeline(ds_agresif, val_ds_base, m)
strategies["Agresif"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

# ─── 5e. CutMix ──────────────────────────────────────────────
print("    [e] CutMix...")
tf.random.set_seed(42)
aug_base  = build_augmenter(flip=True, rotation=0.1, zoom=0, brightness=0.1, contrast=0.1, translation=0)
ds_cutmix = make_pipeline(X_train_n, y_train, augmenter=aug_base, augtype="cutmix")
m         = compile_model(build_simple_cnn("cutmix"), use_cce=False)
hist, ta  = train_pipeline(ds_cutmix, val_ds_base, m)
strategies["CutMix"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

# ─── 5f. MixUp ───────────────────────────────────────────────
print("    [f] MixUp...")
tf.random.set_seed(42)
ds_mixup  = make_pipeline(X_train_n, y_train, augmenter=aug_base, augtype="mixup")
m         = compile_model(build_simple_cnn("mixup"), use_cce=False)
hist, ta  = train_pipeline(ds_mixup, val_ds_base, m)
strategies["MixUp"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

# ─── 5g. Orta + CutMix ───────────────────────────────────────
print("    [g] Orta + CutMix (en iyi kombinasyon)...")
tf.random.set_seed(42)
ds_combo  = make_pipeline(X_train_n, y_train, augmenter=aug_orta, augtype="cutmix")
m         = compile_model(build_simple_cnn("combo"), use_cce=False)
hist, ta  = train_pipeline(ds_combo, val_ds_base, m)
strategies["Orta+CutMix"] = {"history": hist, "test_acc": ta}
print(f"        test_acc={ta:.4f}")

print("\n    ─── Özet ───")
for name, r in strategies.items():
    print(f"    {name:<20}: test_acc={r['test_acc']:.4f}")

# ═════════════════════════════════════════════════════════════
# 6. AUGMENTATİON GÖRSELLEŞTİRMESİ
# ═════════════════════════════════════════════════════════════
print("\n[4] Augmentation görselleştirmesi...")

# Orijinal görüntü + 7 augmented versiyon
aug_orta_viz = build_augmenter(flip=True, rotation=0.15, zoom=0.2, brightness=0.25, contrast=0.2, translation=0.12)
sample_idx  = np.where(y_train == 3)[0][0]  # bir kedi
orig_img    = X_train_n[sample_idx:sample_idx+1]

fig_aug, axes_aug = plt.subplots(2, 8, figsize=(20, 5))
fig_aug.suptitle(f"Data Augmentation — Orta Şiddet ({CIFAR_CLASSES[3]})",
                 fontsize=13, fontweight="bold")

# Geri normalize
def denorm(img):
    return (img * std[0] + mean[0]).clip(0,1)

axes_aug[0,0].imshow(denorm(orig_img[0])); axes_aug[0,0].set_title("Orijinal", fontsize=9); axes_aug[0,0].axis("off")
for col in range(1, 8):
    aug_img = aug_orta_viz(orig_img, training=True).numpy()[0]
    axes_aug[0,col].imshow(denorm(aug_img))
    axes_aug[0,col].set_title(f"Aug #{col}", fontsize=9)
    axes_aug[0,col].axis("off")

# CutMix vizualizasyonu
dog_idx  = np.where(y_train == 5)[0][0]
cat_imgs = X_train_n[sample_idx:sample_idx+4]
dog_imgs = X_train_n[dog_idx:dog_idx+4]
cat_lbs  = y_train[sample_idx:sample_idx+4]
dog_lbs  = y_train[dog_idx:dog_idx+4]

# Kagıt görüntüleri birleştir
combined_imgs = np.concatenate([cat_imgs, dog_imgs], axis=0)
combined_lbs  = np.concatenate([cat_lbs, dog_lbs], axis=0)
mixed_imgs_cm, mixed_lbs_cm = cutmix(
    tf.constant(combined_imgs, tf.float32),
    tf.constant(combined_lbs, tf.int32), alpha=1.0
)

for col in range(8):
    if col < len(mixed_imgs_cm):
        axes_aug[1,col].imshow(denorm(mixed_imgs_cm[col].numpy()))
        top_cls = CIFAR_CLASSES[np.argmax(mixed_lbs_cm[col].numpy())]
        axes_aug[1,col].set_title(f"CutMix→{top_cls}", fontsize=8)
    axes_aug[1,col].axis("off")

plt.tight_layout()
plt.savefig("02b_augmentation_ornekleri.png", dpi=130, bbox_inches="tight")
print("    ✅ Kaydedildi: 02b_augmentation_ornekleri.png")
plt.close()

# ═════════════════════════════════════════════════════════════
# 7. KARŞILAŞTIRMA GÖRSELLEŞTİRMESİ
# ═════════════════════════════════════════════════════════════
print("\n[5] Karşılaştırma görselleştirmeleri...")

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Data Augmentation Stratejileri — Karşılaştırmalı Analiz",
             fontsize=15, fontweight="bold")

palette = {
    "Yok":          "#6B7280",
    "Basit (flip+rot)": "#0F766E",
    "Orta":         "#059669",
    "Agresif":      "#DC2626",
    "CutMix":       "#7C3AED",
    "MixUp":        "#D97706",
    "Orta+CutMix":  "#1D4ED8",
}

# ── 7a. Val accuracy eğrileri ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for name, res in strategies.items():
    ax1.plot(res["history"]["val_accuracy"], lw=2,
             color=palette[name], label=f"{name} (test={res['test_acc']:.4f})")
ax1.set_title("Val Accuracy — Tüm Stratejiler", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Accuracy")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# ── 7b. Test accuracy bar ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
names_bar = list(strategies.keys())
accs_bar  = [r["test_acc"] for r in strategies.values()]
clrs_bar  = [palette[n] for n in names_bar]
bars = ax2.barh(names_bar, accs_bar, color=clrs_bar, alpha=0.82, edgecolor="white")
for bar, v in zip(bars, accs_bar):
    ax2.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax2.set_xlim(0.55, 0.88); ax2.set_title("Test Accuracy Karşılaştırması", fontweight="bold")
ax2.grid(axis="x", alpha=0.3)

# ── 7c. Overfitting: train-val gap ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for name, res in strategies.items():
    tr_acc  = res["history"].get("accuracy", [0])
    val_acc = res["history"].get("val_accuracy", [0])
    n_ep    = min(len(tr_acc), len(val_acc))
    gap     = [t - v for t,v in zip(tr_acc[:n_ep], val_acc[:n_ep])]
    ax3.plot(gap, lw=1.5, color=palette[name], label=name)
ax3.axhline(0, color="k", ls="--", lw=1)
ax3.set_title("Train-Val Accuracy Farkı (Overfit Göstergesi)", fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Train - Val Acc")
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

# ── 7d. Val loss karşılaştırma ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for name, res in strategies.items():
    ax4.plot(res["history"]["val_loss"], lw=2,
             color=palette[name], label=name)
ax4.set_title("Val Loss Karşılaştırması", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# ── 7e. Augmentation örnekleri küçük önizleme ─────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
# Orta augmentor ile 9 örnek
aug_show = build_augmenter(flip=True, rotation=0.15, zoom=0.2, brightness=0.2, contrast=0.2, translation=0.1)
sample_batch = X_train_n[:9]
for idx in range(9):
    row_s, col_s = idx//3, idx%3
    ax_inner = fig.add_axes([0.68+col_s*0.105, 0.04+row_s*0.105, 0.10, 0.095])
    aug_img = aug_show(sample_batch[idx:idx+1], training=True).numpy()[0]
    ax_inner.imshow(denorm(aug_img))
    ax_inner.axis("off")
ax5.set_title("Augmented Örnekler (3×3)", fontweight="bold", y=1.02)

plt.savefig("02_augmentation_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 02_augmentation_analiz.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 02 TAMAMLANDI")
best_strategy = max(strategies, key=lambda k: strategies[k]["test_acc"])
print(f"  En iyi strateji: {best_strategy} (test_acc={strategies[best_strategy]['test_acc']:.4f})")
print("  Çıktılar: 02_augmentation_analiz.png  |  02b_augmentation_ornekleri.png")
print("=" * 65)
