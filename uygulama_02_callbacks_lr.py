"""
=============================================================================
UYGULAMA 02 — Callbacks & Learning Rate Scheduling
=============================================================================
Kapsam:
  - EarlyStopping (restore_best_weights, min_delta)
  - ModelCheckpoint (.keras formatı)
  - ReduceLROnPlateau ince ayar
  - Custom Callback: LRLogger + EpochPlotter + GradientMonitor
  - Cosine Annealing Decay
  - Warmup + Cosine Decay (Transformer stili)
  - Cyclical Learning Rate
  - LR Finder (ısıtma deneyi)
  - Tüm LR stratejilerini karşılaştırma

Veri: MNIST (el yazısı rakamlar — çok sınıflı problem)
Kurulum: pip install tensorflow numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 02 — Callbacks & Learning Rate Scheduling")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA — MNIST
# ═════════════════════════════════════════════════════════════
print("\n[1] MNIST yükleniyor...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

# Flatten + normalize
X_train_full = X_train_raw.reshape(-1, 784).astype("float32") / 255.0
X_test       = X_test_raw.reshape(-1, 784).astype("float32") / 255.0
y_train_full = y_train_raw
y_test       = y_test_raw

# Küçük subset (hızlı deney için)
TRAIN_SIZE = 10000
VAL_SIZE   = 2000
idx = np.random.permutation(len(X_train_full))
X_train = X_train_full[idx[:TRAIN_SIZE]]
y_train = y_train_full[idx[:TRAIN_SIZE]]
X_val   = X_train_full[idx[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]]
y_val   = y_train_full[idx[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]]

print(f"    Eğitim: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"    Sınıflar: {np.unique(y_train)}")

# ═════════════════════════════════════════════════════════════
# 2. TEMEL MODEL
# ═════════════════════════════════════════════════════════════
def build_mnist_model(lr=1e-3, name="mnist_model"):
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(256, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.25),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ], name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

print("\n[2] Model özeti:")
build_mnist_model().summary()

# ═════════════════════════════════════════════════════════════
# 3. CUSTOM CALLBACKS
# ═════════════════════════════════════════════════════════════
print("\n[3] Custom Callback'ler tanımlanıyor...")

class LRLogger(keras.callbacks.Callback):
    """Her epoch sonunda LR'yi ve metrikleri kaydeder."""
    def on_train_begin(self, logs=None):
        self.lr_history     = []
        self.loss_history   = []
        self.val_loss_history = []
        self.acc_history    = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = float(self.model.optimizer.learning_rate)
        self.lr_history.append(current_lr)
        self.loss_history.append(logs.get("loss", 0))
        self.val_loss_history.append(logs.get("val_loss", 0))
        self.acc_history.append(logs.get("accuracy", 0))
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1:3d}: lr={current_lr:.2e}  "
                  f"loss={logs.get('loss',0):.4f}  "
                  f"val_loss={logs.get('val_loss',0):.4f}  "
                  f"acc={logs.get('accuracy',0):.4f}")

class GradientMonitor(keras.callbacks.Callback):
    """
    Her epoch'ta gradient normunu hesaplar.
    Vanishing / Exploding gradient tespiti için.
    """
    def __init__(self, X_sample, y_sample, freq=5):
        super().__init__()
        self.X_sample  = tf.constant(X_sample[:128], dtype=tf.float32)
        self.y_sample  = tf.constant(y_sample[:128])
        self.freq      = freq
        self.grad_norms = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        with tf.GradientTape() as tape:
            pred = self.model(self.X_sample, training=True)
            loss = self.model.compiled_loss(self.y_sample, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        norms = [tf.norm(g).numpy() for g in grads if g is not None]
        mean_norm = np.mean(norms)
        self.grad_norms.append(mean_norm)
        if mean_norm < 1e-4:
            print(f"      ⚠️  Epoch {epoch+1}: Düşük gradyan normu! ({mean_norm:.2e}) — Vanishing gradient şüphesi")

class WarmupCosineSchedule(keras.callbacks.Callback):
    """
    Warmup aşaması + Cosine Annealing.
    İlk warmup_epochs epoch boyunca LR'yi 0 → max'a çıkar,
    sonra cosine ile düşür.
    """
    def __init__(self, max_lr=1e-3, min_lr=1e-7,
                 warmup_epochs=5, total_epochs=60):
        super().__init__()
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_log = []

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        self.model.optimizer.learning_rate.assign(lr)
        self.lr_log.append(lr)

class CyclicalLR(keras.callbacks.Callback):
    """
    Cyclical Learning Rate (triangular policy).
    LR'yi base_lr → max_lr arasında periyodik döngüye sokar.
    """
    def __init__(self, base_lr=1e-5, max_lr=1e-2, step_size=200):
        super().__init__()
        self.base_lr  = base_lr
        self.max_lr   = max_lr
        self.step_size = step_size
        self.iteration = 0
        self.lr_log    = []

    def on_train_batch_begin(self, batch, logs=None):
        cycle    = math.floor(1 + self.iteration / (2 * self.step_size))
        x        = abs(self.iteration / self.step_size - 2 * cycle + 1)
        lr       = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        self.model.optimizer.learning_rate.assign(lr)
        self.iteration += 1
        self.lr_log.append(lr)

print("    ✅ LRLogger, GradientMonitor, WarmupCosineSchedule, CyclicalLR hazır")

# ═════════════════════════════════════════════════════════════
# 4. EarlyStopping & ModelCheckpoint — DETAYLI DEMO
# ═════════════════════════════════════════════════════════════
print("\n[4] EarlyStopping + ModelCheckpoint demosu...")

tf.random.set_seed(42)
model_es = build_mnist_model(name="model_earlystop")
lr_logger = LRLogger()
grad_mon  = GradientMonitor(X_train, y_train, freq=5)

callbacks_demo = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        mode="max",
        min_delta=0.001,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_mnist_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=6,
        min_lr=1e-7,
        min_delta=1e-4,
        verbose=1,
    ),
    lr_logger,
    grad_mon,
]

print("    ⏳ Eğitim başlıyor (max 100 epoch, EarlyStopping ile kesilebilir)...")
history_es = model_es.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks_demo,
    verbose=0,
)

test_acc_es = model_es.evaluate(X_test, y_test, verbose=0)[1]
print(f"\n    EarlyStopping ile {len(history_es.history['loss'])} epoch çalıştı")
print(f"    Test Accuracy: {test_acc_es:.4f}")
print(f"    Min LR ulaşıldı: {min(lr_logger.lr_history):.2e}")

# ═════════════════════════════════════════════════════════════
# 5. LR STRATEJİLERİ KARŞILAŞTIRMASI
# ═════════════════════════════════════════════════════════════
print("\n[5] LR stratejileri karşılaştırılıyor...")

COMP_EPOCHS = 40

lr_strategies = {}

# ── Sabit LR ─────────────────────────────────────────────────
tf.random.set_seed(42)
m = build_mnist_model(lr=1e-3, name="sabit_lr")
h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=COMP_EPOCHS, batch_size=128, verbose=0)
lr_strategies["Sabit LR (1e-3)"] = h.history

# ── ReduceLROnPlateau ─────────────────────────────────────────
tf.random.set_seed(42)
m = build_mnist_model(lr=1e-3, name="reduce_lr")
rlr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.4, patience=5, min_lr=1e-7)
h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=COMP_EPOCHS, batch_size=128, callbacks=[rlr], verbose=0)
lr_strategies["ReduceLROnPlateau"] = h.history

# ── Cosine Annealing ──────────────────────────────────────────
tf.random.set_seed(42)
m = build_mnist_model(lr=1e-3, name="cosine")
cosine_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=COMP_EPOCHS * len(X_train) // 128,
    alpha=1e-7,
)
m.optimizer.learning_rate = cosine_schedule
h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=COMP_EPOCHS, batch_size=128, verbose=0)
lr_strategies["Cosine Annealing"] = h.history

# ── Warmup + Cosine ───────────────────────────────────────────
tf.random.set_seed(42)
m = build_mnist_model(lr=1e-7, name="warmup_cosine")
warmup_cb = WarmupCosineSchedule(
    max_lr=1e-3, min_lr=1e-7,
    warmup_epochs=5, total_epochs=COMP_EPOCHS
)
h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=COMP_EPOCHS, batch_size=128, callbacks=[warmup_cb], verbose=0)
lr_strategies["Warmup + Cosine"] = h.history
lr_strategies["__warmup_lr_log"] = warmup_cb.lr_log

print("    ✅ 4 LR stratejisi tamamlandı:")
for name, hist in lr_strategies.items():
    if name.startswith("__"):
        continue
    print(f"      {name:<25}: final val_acc={hist['val_accuracy'][-1]:.4f}  "
          f"peak={max(hist['val_accuracy']):.4f}")

# ═════════════════════════════════════════════════════════════
# 6. LR FINDER
# ═════════════════════════════════════════════════════════════
print("\n[6] LR Finder çalıştırılıyor...")

class LRFinder(keras.callbacks.Callback):
    """
    LR'yi log-lineer artırarak optimal aralığı bul.
    Loss düşmeye başladığı yer → min LR
    Loss artmadan önceki yer  → max LR
    """
    def __init__(self, min_lr=1e-7, max_lr=1.0, n_steps=200):
        super().__init__()
        self.min_lr  = min_lr
        self.max_lr  = max_lr
        self.n_steps = n_steps
        self.step    = 0
        self.lrs     = []
        self.losses  = []

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.step / self.n_steps)
        self.model.optimizer.learning_rate.assign(lr)

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate))
        self.losses.append(logs.get("loss", 0))
        self.step += 1
        if self.step >= self.n_steps:
            self.model.stop_training = True

tf.random.set_seed(42)
m_finder = build_mnist_model(lr=1e-7, name="lr_finder_model")
lr_finder = LRFinder(min_lr=1e-7, max_lr=1.0, n_steps=200)
m_finder.fit(X_train[:2000], y_train[:2000],
             epochs=5, batch_size=32, callbacks=[lr_finder], verbose=0)

print(f"    ✅ LR Finder tamamlandı: {len(lr_finder.lrs)} adım")

# ═════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[7] Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle("Callbacks & Learning Rate Scheduling — Kapsamlı Analiz",
             fontsize=15, fontweight="bold")

# ── 7a. EarlyStopping — Loss ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history_es.history["loss"],     color="#4338CA", lw=2, label="Train")
ax1.plot(history_es.history["val_loss"], color="#DB2777", lw=2, ls="--", label="Val")
best_ep = np.argmin(history_es.history["val_loss"])
ax1.axvline(best_ep, color="#D97706", ls=":", lw=2, label=f"Best ep={best_ep+1}")
ax1.set_title("EarlyStopping — Loss", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

# ── 7b. LR geçmişi (ReduceLROnPlateau etkisi) ────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(lr_logger.lr_history, color="#0D9488", lw=2)
ax2.set_title("LR Geçmişi (ReduceLROnPlateau)", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("LR (log ölçek)"); ax2.grid(alpha=0.3)

# ── 7c. Gradient normu ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
if grad_mon.grad_norms:
    x_gn = [(i + 1) * 5 for i in range(len(grad_mon.grad_norms))]
    ax3.plot(x_gn, grad_mon.grad_norms, "o-", color="#7C3AED", lw=2, markersize=5)
    ax3.axhline(1e-4, color="#DC2626", ls="--", lw=1.5, label="Vanishing eşiği")
    ax3.set_title("Gradient Normu (GradientMonitor)", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Ort. Gradient Normu"); ax3.legend()
ax3.grid(alpha=0.3)

# ── 7d. LR stratejileri — val_accuracy ───────────────────────
ax4 = fig.add_subplot(gs[1, :2])
colors = ["#4338CA", "#0D9488", "#DB2777", "#D97706"]
for (name, hist), color in zip(
    [(k, v) for k, v in lr_strategies.items() if not k.startswith("__")], colors
):
    ax4.plot(hist["val_accuracy"], lw=2, color=color, label=name)
ax4.set_title("LR Stratejileri — Val Accuracy Karşılaştırması", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Val Accuracy")
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# ── 7e. Warmup + Cosine LR eğrisi ────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
warmup_log = lr_strategies.get("__warmup_lr_log", [])
ax5.plot(warmup_log, color="#DB2777", lw=2)
ax5.axvline(5, color="#D97706", ls="--", lw=1.5, label="Warmup bitti")
ax5.set_title("Warmup + Cosine LR Eğrisi", fontweight="bold")
ax5.set_xlabel("Epoch"); ax5.set_ylabel("LR")
ax5.legend(); ax5.grid(alpha=0.3)

# ── 7f. LR Finder ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
smoothed = pd.Series(lr_finder.losses).rolling(5, center=True).mean()
import pandas as pd
smoothed = pd.Series(lr_finder.losses).rolling(5, center=True).mean()
ax6.semilogx(lr_finder.lrs, lr_finder.losses, color="#9CA3AF", lw=1, alpha=0.5, label="Ham")
ax6.semilogx(lr_finder.lrs, smoothed, color="#4338CA", lw=2.5, label="Düzeltilmiş")
# En hızlı düşüşü bul
try:
    grad_lr = np.gradient(smoothed.fillna(method="bfill"))
    best_lr_idx = np.argmin(grad_lr[10:-10]) + 10
    ax6.axvline(lr_finder.lrs[best_lr_idx], color="#DC2626", ls="--", lw=2,
                label=f"Önerilen LR ≈ {lr_finder.lrs[best_lr_idx]:.2e}")
except Exception:
    pass
ax6.set_title("LR Finder — Optimal LR Aralığı", fontweight="bold")
ax6.set_xlabel("LR (log ölçek)"); ax6.set_ylabel("Loss")
ax6.legend(); ax6.grid(alpha=0.3); ax6.set_xlim(1e-7, 1.0)

# ── 7g. Final model — val acc ────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(history_es.history["accuracy"],     color="#4338CA", lw=2, label="Train Acc")
ax7.plot(history_es.history["val_accuracy"], color="#DB2777", lw=2, ls="--", label="Val Acc")
ax7.axhline(test_acc_es, color="#15803D", ls=":", lw=2, label=f"Test Acc={test_acc_es:.4f}")
ax7.set_title("EarlyStopping Model — Accuracy", fontweight="bold")
ax7.set_xlabel("Epoch"); ax7.legend(fontsize=9); ax7.grid(alpha=0.3)

plt.savefig("02_callbacks_lr_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 02_callbacks_lr_analiz.png")
plt.close()

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print(f"  Best Model Test Accuracy: {test_acc_es:.4f}")
print("  Çıktılar: 02_callbacks_lr_analiz.png | best_mnist_model.keras")
print("=" * 65)
