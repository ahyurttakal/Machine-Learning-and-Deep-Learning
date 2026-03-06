"""
=============================================================================
UYGULAMA 04 — Transfer Learning & Grad-CAM
=============================================================================
Kapsam:
  - EfficientNetB0 ile 3 farklı strateji:
      1) Feature Extraction (tüm base dondurulmuş)
      2) Partial Fine-Tuning (son 30 katman açık)
      3) Full Fine-Tuning (tüm ağırlıklar güncellendi)
  - 2 aşamalı eğitim (head → fine-tune)
  - Catastrophic forgetting demosu
  - Grad-CAM ısı haritası görselleştirme
  - Doğru / yanlış tahmin analizi
  - Sınıf başına doğruluk raporu

Veri: CIFAR-10 (96×96'ya büyütülmüş — EfficientNet giriş boyutu)
Kurulum: pip install tensorflow numpy matplotlib opencv-python
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]
IMG_SIZE  = 64    # EfficientNetB0 min: 32 (önerilen: 224, ama hız için 64)
BATCH_SIZE = 64

print("=" * 65)
print("  UYGULAMA 04 — Transfer Learning & Grad-CAM")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA — Resize
# ═════════════════════════════════════════════════════════════
print("\n[1] CIFAR-10 yükleniyor ve yeniden boyutlandırılıyor...")
(X_raw, y_raw), (X_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()

def resize_dataset(X, size=IMG_SIZE):
    """CIFAR-10 32×32 → size×size."""
    X_f = X.astype("float32") / 255.0
    return tf.image.resize(X_f, [size, size]).numpy()

print(f"    Yeniden boyutlandırılıyor: 32×32 → {IMG_SIZE}×{IMG_SIZE}...")
X_res     = resize_dataset(X_raw)
X_test_r  = resize_dataset(X_test_raw)
y_raw_f   = y_raw.flatten()
y_test_f  = y_test_raw.flatten()

val_size   = 5000
X_val,   X_train = X_res[:val_size],   X_res[val_size:]
y_val_f, y_train = y_raw_f[:val_size], y_raw_f[val_size:]

print(f"    Eğitim: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Boyut: {IMG_SIZE}×{IMG_SIZE}×3")

# EfficientNet kendi ön işlemeye sahip; [0,255] aralığı bekler
X_train_u = (X_train * 255).astype("float32")
X_val_u   = (X_val   * 255).astype("float32")
X_test_u  = (X_test_r* 255).astype("float32")

# ─── Augmentation (transfer learning için agresif olabilir) ──
aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="aug")

def make_ds(X, y, augment=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X, tf.float32), tf.constant(y, tf.int32)))
    if shuffle: ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(lambda x,y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train_u, y_train, augment=True)
val_ds   = make_ds(X_val_u,   y_val_f, shuffle=False)
test_ds  = make_ds(X_test_u,  y_test_f, shuffle=False)

# ═════════════════════════════════════════════════════════════
# 2. HEAD OLUŞTURUCU
# ═════════════════════════════════════════════════════════════
def build_head(base, n_classes=10, dropout=0.3, name="tl_model"):
    """Base model üzerine sınıflandırma başlığı ekle."""
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # training=False → BN ve Dropout inference modunda
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.BatchNormalization(name="head_bn")(x)
    x   = layers.Dropout(dropout, name="head_drop1")(x)
    x   = layers.Dense(256, activation="relu", name="head_fc1")(x)
    x   = layers.Dropout(dropout * 0.7, name="head_drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name=name)

# ═════════════════════════════════════════════════════════════
# 3. STRATEJİ 1 — FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════
print("\n[2] Strateji 1: Feature Extraction (Base Dondurulmuş)...")

base_fe = keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_fe.trainable = False
model_fe = build_head(base_fe, name="fe_model")
model_fe.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print(f"    Eğitilebilir param: {sum(tf.size(v).numpy() for v in model_fe.trainable_variables):,}")

cbs_fe = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                   restore_best_weights=True, mode="max"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=4, min_lr=1e-7),
]
print("    ⏳ Aşama 1 eğitimi (head only, 25 epoch max)...")
hist_fe = model_fe.fit(train_ds, validation_data=val_ds,
                        epochs=25, callbacks=cbs_fe, verbose=1)
acc_fe = model_fe.evaluate(test_ds, verbose=0)[1]
print(f"    ✅ Feature Extraction test_acc={acc_fe:.4f}")

# ═════════════════════════════════════════════════════════════
# 4. STRATEJİ 2 — PARTIAL FINE-TUNING (İki Aşama)
# ═════════════════════════════════════════════════════════════
print("\n[3] Strateji 2: Partial Fine-Tuning (Son 30 katman)...")

base_pf = keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_pf.trainable = False
model_pf = build_head(base_pf, name="pf_model")
model_pf.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"],
)

# Aşama 1: head only
print("    ⏳ Aşama 1: Head eğitimi...")
cbs_ph1 = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                          restore_best_weights=True, mode="max"),]
hist_pf1 = model_pf.fit(train_ds, validation_data=val_ds,
                          epochs=20, callbacks=cbs_ph1, verbose=0)

# Aşama 2: Son N katmanı aç
print("    ⏳ Aşama 2: Fine-tuning (son 30 katman)...")
base_pf.trainable = True
# Alt katmanları tekrar dondur
for layer in base_pf.layers[:-30]:
    layer.trainable = False
# BatchNorm katmanları HER ZAMAN dondurulur
for layer in base_pf.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Çok küçük LR ile yeniden derle
model_pf.compile(
    optimizer=keras.optimizers.Adam(
        keras.optimizers.schedules.CosineDecay(1e-5, decay_steps=20*len(list(train_ds)), alpha=1e-7)
    ),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"],
)
print(f"    Eğitilebilir param (fine-tune): {sum(tf.size(v).numpy() for v in model_pf.trainable_variables):,}")

cbs_pf2 = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                   restore_best_weights=True, mode="max"),
]
hist_pf2 = model_pf.fit(train_ds, validation_data=val_ds,
                          epochs=20, callbacks=cbs_pf2, verbose=1)
acc_pf = model_pf.evaluate(test_ds, verbose=0)[1]
print(f"    ✅ Partial Fine-Tuning test_acc={acc_pf:.4f}")

# ═════════════════════════════════════════════════════════════
# 5. STRATEJİ 3 — FULL FINE-TUNING (Catastrophic Forgetting Demo)
# ═════════════════════════════════════════════════════════════
print("\n[4] Strateji 3: Full Fine-Tuning (Tüm ağ açık)...")

base_ff = keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_ff.trainable = False
model_ff = build_head(base_ff, name="ff_model")
model_ff.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"],
)
print("    ⏳ Aşama 1: Head eğitimi (10 epoch)...")
model_ff.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
acc_ff_asamal = model_ff.evaluate(test_ds, verbose=0)[1]
print(f"    Head sonrası acc: {acc_ff_asamal:.4f}")

# Tüm ağı aç (BN hariç)
print("    ⏳ Aşama 2: Full fine-tuning (çok küçük LR)...")
base_ff.trainable = True
for layer in base_ff.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
model_ff.compile(
    optimizer=keras.optimizers.Adam(
        keras.optimizers.schedules.CosineDecay(2e-5, decay_steps=15*len(list(train_ds)))
    ),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"],
)
cbs_ff = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                         restore_best_weights=True, mode="max")]
hist_ff2 = model_ff.fit(train_ds, validation_data=val_ds,
                         epochs=15, callbacks=cbs_ff, verbose=1)
acc_ff = model_ff.evaluate(test_ds, verbose=0)[1]
print(f"    ✅ Full Fine-Tuning test_acc={acc_ff:.4f}")

# ═════════════════════════════════════════════════════════════
# 6. GRAD-CAM
# ═════════════════════════════════════════════════════════════
print("\n[5] Grad-CAM hesaplanıyor...")

def compute_gradcam(model, img_array, last_conv_layer_name):
    """
    Grad-CAM ısı haritası hesapla.
    img_array: (1, H, W, 3) float32 [0-255]
    """
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        pred_idx     = tf.argmax(preds[0])
        class_channel = preds[:, pred_idx]

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out    = conv_outputs[0]
    heatmap     = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0.0)
    mx = tf.math.reduce_max(heatmap)
    if mx > 0: heatmap = heatmap / mx
    return heatmap.numpy(), int(pred_idx.numpy()), float(tf.reduce_max(preds[0]).numpy())

def overlay_heatmap(original_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Isı haritasını orijinal görüntüyle çakıştır."""
    h, w = original_img.shape[:2]
    heatmap_r = cv2.resize(heatmap, (w, h))
    heatmap_u = np.uint8(255 * heatmap_r)
    colored   = cv2.applyColorMap(heatmap_u, colormap)
    colored   = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    orig_u    = np.uint8(original_img * 255) if original_img.max() <= 1.0 else np.uint8(original_img)
    superimposed = cv2.addWeighted(orig_u, 1 - alpha, colored, alpha, 0)
    return superimposed.astype(np.float32) / 255.0

# EfficientNetB0'ın son conv katmanı
last_conv = "top_conv"   # EfficientNetB0'da bu isim

# Test setinden doğru ve yanlış tahminler bul
y_pred_all = model_pf.predict(test_ds, verbose=0)
y_pred_cls = np.argmax(y_pred_all, axis=1)
correct_idx   = np.where(y_pred_cls == y_test_f)[0]
incorrect_idx = np.where(y_pred_cls != y_test_f)[0]

# 4 doğru + 4 yanlış tahmin
n_show = 4
correct_samples   = correct_idx[:n_show]
incorrect_samples = incorrect_idx[:n_show]

print(f"    {len(correct_idx):,} doğru  |  {len(incorrect_idx):,} yanlış tahmin")

gradcam_results = {"correct": [], "incorrect": []}
for idx in correct_samples:
    img = X_test_u[idx:idx+1]
    hm, pred, conf = compute_gradcam(model_pf, img, last_conv)
    gradcam_results["correct"].append({
        "img": X_test_r[idx], "hm": hm,
        "true": y_test_f[idx], "pred": pred, "conf": conf,
    })

for idx in incorrect_samples:
    img = X_test_u[idx:idx+1]
    hm, pred, conf = compute_gradcam(model_pf, img, last_conv)
    gradcam_results["incorrect"].append({
        "img": X_test_r[idx], "hm": hm,
        "true": y_test_f[idx], "pred": pred, "conf": conf,
    })

print("    ✅ Grad-CAM hesaplandı")

# ═════════════════════════════════════════════════════════════
# 7. SINIF BAZINDA DOĞRULUK ANALİZİ
# ═════════════════════════════════════════════════════════════
print("\n[6] Sınıf bazında doğruluk analizi...")
class_acc = {}
for cls in range(10):
    mask = y_test_f == cls
    if mask.sum() > 0:
        cls_preds = y_pred_cls[mask]
        class_acc[CIFAR_CLASSES[cls]] = np.mean(cls_preds == cls)
        print(f"    {CIFAR_CLASSES[cls]:<12}: {class_acc[CIFAR_CLASSES[cls]]:.4f}")

# ═════════════════════════════════════════════════════════════
# 8. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[7] Görselleştirmeler hazırlanıyor...")

# ── 8a. Strateji karşılaştırma ────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle("Transfer Learning Stratejileri Karşılaştırması", fontsize=14, fontweight="bold")

# Val acc - FE
axes1[0].plot(hist_fe.history["accuracy"],     color="#0F766E", lw=2, label="Train")
axes1[0].plot(hist_fe.history["val_accuracy"], color="#0F766E", lw=2, ls="--", label="Val")
axes1[0].axhline(acc_fe, color="#DC2626", ls=":", lw=2, label=f"Test={acc_fe:.4f}")
axes1[0].set_title("Strateji 1: Feature Extraction", fontweight="bold")
axes1[0].set_xlabel("Epoch"); axes1[0].legend(); axes1[0].grid(alpha=0.3)

# Val acc - PF
epochs_phase2_start = len(hist_pf1.history["val_accuracy"])
all_val_pf = (hist_pf1.history["val_accuracy"] +
              hist_pf2.history["val_accuracy"])
all_tr_pf  = (hist_pf1.history["accuracy"] +
              hist_pf2.history["accuracy"])
axes1[1].plot(all_tr_pf,  color="#059669", lw=2, label="Train")
axes1[1].plot(all_val_pf, color="#059669", lw=2, ls="--", label="Val")
axes1[1].axvline(epochs_phase2_start, color="#D97706", ls="--", lw=2, label="Fine-tune başladı")
axes1[1].axhline(acc_pf, color="#DC2626", ls=":", lw=2, label=f"Test={acc_pf:.4f}")
axes1[1].set_title("Strateji 2: 2-Aşamalı Fine-Tuning", fontweight="bold")
axes1[1].set_xlabel("Epoch"); axes1[1].legend(fontsize=8); axes1[1].grid(alpha=0.3)

# Bar karşılaştırma
strat_names = ["Feature\nExtraction", "Partial\nFine-Tuning", "Full\nFine-Tuning"]
strat_accs  = [acc_fe, acc_pf, acc_ff]
strat_clrs  = ["#0F766E", "#059669", "#7C3AED"]
bars = axes1[2].bar(strat_names, strat_accs, color=strat_clrs, alpha=0.82, edgecolor="white", width=0.5)
for bar, v in zip(bars, strat_accs):
    axes1[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                  f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
axes1[2].set_ylim(0.5, 1.0); axes1[2].set_title("Test Accuracy Karşılaştırması", fontweight="bold")
axes1[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("04_transfer_learning_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 04_transfer_learning_analiz.png")
plt.close()

# ── 8b. Grad-CAM görselleştirme ───────────────────────────────
fig2, axes2 = plt.subplots(4, 6, figsize=(20, 14))
fig2.suptitle("Grad-CAM — Modelin Odaklandığı Bölgeler (Partial Fine-Tuning)", fontsize=14, fontweight="bold")

for row, (category, row_idx) in enumerate([("correct",0),("correct",1),("incorrect",2),("incorrect",3)]):
    cat_key  = "correct" if row_idx < 2 else "incorrect"
    data_idx = row_idx if row_idx < 2 else row_idx - 2
    if data_idx >= len(gradcam_results[cat_key]):
        continue
    item = gradcam_results[cat_key][data_idx]

    orig_img   = item["img"]
    hm         = item["hm"]
    true_label = CIFAR_CLASSES[int(item["true"])]
    pred_label = CIFAR_CLASSES[item["pred"]]
    conf       = item["conf"]
    correct    = int(item["true"]) == item["pred"]

    overlay    = overlay_heatmap(orig_img, hm, alpha=0.5)

    # Sütun 1: Orijinal
    axes2[row,0].imshow(orig_img.clip(0,1))
    axes2[row,0].set_title(f"Gerçek: {true_label}", fontsize=10, fontweight="bold")
    axes2[row,0].axis("off")

    # Sütun 2: Isı haritası
    axes2[row,1].imshow(hm, cmap="jet")
    axes2[row,1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes2[row,1].axis("off")

    # Sütun 3: Üst üste çakıştırma
    color = "#059669" if correct else "#DC2626"
    axes2[row,2].imshow(overlay)
    status = "✅ Doğru" if correct else "❌ Yanlış"
    axes2[row,2].set_title(f"Tahmin: {pred_label} ({conf:.2f})\n{status}", fontsize=9)
    axes2[row,2].axis("off")

    # Sütunlar 3-5: Diğer kategorilerden örnekler
    other_correct = [i for i in (correct_samples if cat_key=="correct" else incorrect_samples)
                     if i != (correct_samples if cat_key=="correct" else incorrect_samples)[data_idx]]
    for cidx in range(3):
        axes2[row,3+cidx].axis("off")

# Kalan eksenleri kapat
for r in range(4):
    for c in range(6):
        if axes2[r,c].has_data() == False:
            axes2[r,c].axis("off")

plt.tight_layout()
plt.savefig("04b_gradcam_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 04b_gradcam_analiz.png")
plt.close()

# ── 8c. Sınıf bazında doğruluk ────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
cls_names = list(class_acc.keys())
cls_accs  = list(class_acc.values())
bar_c     = ["#059669" if a >= np.mean(cls_accs) else "#DC2626" for a in cls_accs]
bars_c    = ax3.bar(cls_names, cls_accs, color=bar_c, alpha=0.82, edgecolor="white")
ax3.axhline(np.mean(cls_accs), color="#D97706", ls="--", lw=2, label=f"Ortalama={np.mean(cls_accs):.4f}")
ax3.axhline(acc_pf, color="#0F766E", ls=":", lw=2, label=f"Genel={acc_pf:.4f}")
for bar, v in zip(bars_c, cls_accs):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{v:.3f}", ha="center", fontsize=9)
ax3.set_ylim(0.4, 1.05); ax3.set_title("Sınıf Bazında Test Doğruluğu (Partial Fine-Tuning)", fontweight="bold")
ax3.legend(); ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("04c_sinif_dogrulugu.png", dpi=130, bbox_inches="tight")
print("    ✅ Kaydedildi: 04c_sinif_dogrulugu.png")
plt.close()

# ─── Özet ────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("  SONUÇLAR — Transfer Learning Karşılaştırması")
print("─" * 55)
print(f"  Feature Extraction   : test_acc = {acc_fe:.4f}")
print(f"  Partial Fine-Tuning  : test_acc = {acc_pf:.4f}  ← Önerilen")
print(f"  Full Fine-Tuning     : test_acc = {acc_ff:.4f}")
print(f"\n  En düşük sınıf acc: {min(class_acc, key=class_acc.get)} = {min(class_acc.values()):.4f}")
print(f"  En yüksek sınıf acc: {max(class_acc, key=class_acc.get)} = {max(class_acc.values()):.4f}")

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("  Çıktılar:")
print("    04_transfer_learning_analiz.png")
print("    04b_gradcam_analiz.png")
print("    04c_sinif_dogrulugu.png")
print("=" * 65)
