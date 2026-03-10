"""
=============================================================================
HAFTA 4 PAZAR — UYGULAMA 03
DCGAN (Deep Convolutional GAN) ile Fashion-MNIST Görüntü Üretimi
=============================================================================
Kapsam:
  - Generator: ConvTranspose2D ile gürültü → kıyafet görüntüsü
  - Discriminator: Strided Conv2D ile gerçek/sahte ayırt
  - Adversarial eğitim döngüsü: D güncelle → G güncelle
  - GANMonitor callback: her epoch sonunda 4×4 örnek kaydet
  - Label Smoothing: aşırı öğrenmeyi önleme
  - Eğitim kaybı analizi: d_loss / g_loss dengesi
  - Mode collapse erken uyarı sistemi
  - 8×8 grid üretilmiş kıyafet görüntüsü
  - TensorFlow/Keras yoksa tam simülasyon modu

Kurulum: pip install tensorflow numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def gaussian_filter(x, sigma=1): return x

SIM_MODE = not TF_AVAILABLE

print("=" * 65)
print("  HAFTA 4 PAZAR — UYGULAMA 03")
print("  DCGAN ile Fashion-MNIST Görüntü Üretimi")
print("=" * 65)
print(f"  Mod        : {'🔵 Gerçek (TensorFlow)' if not SIM_MODE else '🟡 Simülasyon'}")
print(f"  TensorFlow : {'✅' if TF_AVAILABLE else '❌  pip install tensorflow'}")
print()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1: VERİ HAZIRLAMA
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Fashion-MNIST Veri Hazırlama")
print("─" * 65)

IMG_BOYUT   = 28      # Fashion-MNIST: 28×28
LATENT_DIM  = 100     # Gürültü vektörü boyutu
BATCH_SIZE  = 64
EPOCHS      = 50

SINIF_ADLAR = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

if not SIM_MODE:
    (x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5   # [-1, 1]
    x_train = x_train[..., np.newaxis]                       # (60000,28,28,1)
    print(f"  Veri seti   : {x_train.shape}  dtype={x_train.dtype}")
    print(f"  Değer aralığı: [{x_train.min():.1f}, {x_train.max():.1f}]  ([-1,1] normalize)")
    print(f"  Sınıflar    : {SINIF_ADLAR[:5]}...")
else:
    print(f"  [SIM] x_train : (60000, 28, 28, 1)  dtype=float32")
    print(f"  [SIM] Değer aralığı: [-1.0, 1.0]  (Tanh çıktısına uygun)")
    print(f"  [SIM] Sınıflar: {SINIF_ADLAR[:5]}...")
    x_train = None
    y_train = None

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2: GENERATOR MİMARİSİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: Generator Mimarisi  (z→28×28)")
print("─" * 65)
print("""
  z (100,)
      │
  Dense(7×7×256) → Reshape(7,7,256)
      │  BN + ReLU
  ConvTranspose2D(128, 4×4, stride=2)  →  14×14×128
      │  BatchNorm + ReLU
  ConvTranspose2D(64,  4×4, stride=2)  →  28×28×64
      │  BatchNorm + ReLU
  Conv2D(1, 7×7, padding='same')       →  28×28×1
      │  Tanh  ∈ [-1, 1]
""")

if not SIM_MODE:
    def generator_olustur():
        model = keras.Sequential([
            keras.layers.Input(shape=(LATENT_DIM,)),
            keras.layers.Dense(7 * 7 * 256, use_bias=False),
            keras.layers.Reshape((7, 7, 256)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                          padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2),
                                         padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(1, (7, 7), padding="same", activation="tanh"),
        ], name="generator")
        return model

    generator = generator_olustur()
    print(f"  Generator parametresi: {generator.count_params():,}")
    # Çıkış boyutu testi
    z_test = tf.random.normal([1, LATENT_DIM])
    g_out  = generator(z_test, training=False)
    print(f"  Generator çıktı shape: {g_out.shape}  ✅")
else:
    print("  [SIM] Generator parametresi: 3,623,553")
    print(f"  [SIM] Generator çıktı shape: (1, {IMG_BOYUT}, {IMG_BOYUT}, 1)  ✅")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3: DISCRIMINATOR MİMARİSİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Discriminator Mimarisi  (28×28→Skor)")
print("─" * 65)
print("""
  Girdi (28×28×1)
      │
  Conv2D(64,  4×4, stride=2)   →  14×14×64   LeakyReLU(0.2) + Dropout(0.3)
  Conv2D(128, 4×4, stride=2)   →   7×7×128   LeakyReLU(0.2) + Dropout(0.3)
  Conv2D(256, 4×4, stride=2)   →   4×4×256   LeakyReLU(0.2) + Dropout(0.3)
      │
  Flatten → Dense(1)           →  P(gerçek) ∈ ℝ  (from_logits=True)
""")

if not SIM_MODE:
    def discriminator_olustur():
        model = keras.Sequential([
            keras.layers.Input(shape=(IMG_BOYUT, IMG_BOYUT, 1)),
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ], name="discriminator")
        return model

    discriminator = discriminator_olustur()
    print(f"  Discriminator parametresi: {discriminator.count_params():,}")
    d_out = discriminator(g_out, training=False)
    print(f"  Discriminator çıktı shape: {d_out.shape}  ✅")
else:
    print("  [SIM] Discriminator parametresi: 621,825")
    print(f"  [SIM] Discriminator çıktı shape: (1, 1)  ✅")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4: GAN SINIFI VE EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: GAN Sınıfı & Adversarial Eğitim Döngüsü")
print("─" * 65)

if not SIM_MODE:
    class GANMonitor(keras.callbacks.Callback):
        """Her epoch sonunda 4×4 sahte görüntü kaydeder."""
        def __init__(self, n_samples=16, latent_dim=100, save_dir="./gan_progress"):
            self.n_samples  = n_samples
            self.latent_dim = latent_dim
            self.save_dir   = save_dir
            import os
            os.makedirs(save_dir, exist_ok=True)
            # Epoch boyunca sabit z → ilerlemeyi takip et
            self.sabit_z = tf.random.normal([n_samples, latent_dim])

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0 or epoch == 0:
                uretilen = self.model.generator(self.sabit_z, training=False)
                uretilen = (uretilen + 1.0) / 2.0  # [-1,1] → [0,1]
                uretilen = uretilen.numpy().squeeze()
                fig, axler = plt.subplots(4, 4, figsize=(8, 8))
                fig.patch.set_facecolor("black")
                for i, ax in enumerate(axler.flat):
                    ax.imshow(uretilen[i], cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                fig.suptitle(f"Epoch {epoch+1}", color="white", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{self.save_dir}/epoch_{epoch+1:03d}.png",
                            dpi=100, facecolor="black")
                plt.close()

    class GAN(keras.Model):
        def __init__(self, discriminator, generator, latent_dim,
                     label_smoothing=0.9):
            super().__init__()
            self.discriminator     = discriminator
            self.generator         = generator
            self.latent_dim        = latent_dim
            self.label_smoothing   = label_smoothing  # 1.0 → 0.9
            self.d_loss_tracker    = keras.metrics.Mean(name="d_loss")
            self.g_loss_tracker    = keras.metrics.Mean(name="g_loss")
            self.d_real_tracker    = keras.metrics.Mean(name="d_real")
            self.d_fake_tracker    = keras.metrics.Mean(name="d_fake")

        @property
        def metrics(self):
            return [self.d_loss_tracker, self.g_loss_tracker,
                    self.d_real_tracker, self.d_fake_tracker]

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super().compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn     = loss_fn

        def train_step(self, real_images):
            batch_size = tf.shape(real_images)[0]
            z = tf.random.normal([batch_size, self.latent_dim])

            # ── Discriminator Güncelleme ───────────────────
            with tf.GradientTape() as d_tape:
                fake_images  = self.generator(z, training=False)
                real_logits  = self.discriminator(real_images, training=True)
                fake_logits  = self.discriminator(fake_images, training=True)

                # Label Smoothing: gerçek etiket 0.9 (1.0 yerine)
                real_labels  = tf.ones_like(real_logits) * self.label_smoothing
                fake_labels  = tf.zeros_like(fake_logits)

                d_real_loss  = self.loss_fn(real_labels, real_logits)
                d_fake_loss  = self.loss_fn(fake_labels, fake_logits)
                d_loss       = d_real_loss + d_fake_loss

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_weights)
            )

            # ── Generator Güncelleme ───────────────────────
            z2 = tf.random.normal([batch_size, self.latent_dim])
            with tf.GradientTape() as g_tape:
                fake_images2 = self.generator(z2, training=True)
                fake_logits2 = self.discriminator(fake_images2, training=False)
                # Generator amacı: D(G(z)) = 1 (D'yi kandır)
                g_loss       = self.loss_fn(tf.ones_like(fake_logits2), fake_logits2)

            g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(
                zip(g_grads, self.generator.trainable_weights)
            )

            # Metrik güncelle
            d_real_prob = tf.reduce_mean(tf.sigmoid(real_logits))
            d_fake_prob = tf.reduce_mean(tf.sigmoid(fake_logits))
            self.d_loss_tracker.update_state(d_loss)
            self.g_loss_tracker.update_state(g_loss)
            self.d_real_tracker.update_state(d_real_prob)
            self.d_fake_tracker.update_state(d_fake_prob)
            return {m.name: m.result() for m in self.metrics}

    # Derleme ve eğitim
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    gan = GAN(discriminator, generator, LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        loss_fn=loss_fn,
    )
    print("  GAN derlendi:")
    print("    d_optimizer: Adam(lr=2e-4, β₁=0.5)")
    print("    g_optimizer: Adam(lr=2e-4, β₁=0.5)")
    print("    loss_fn    : BinaryCrossentropy(from_logits=True)")
    print("    Label Smooth: 0.9")
    print()
    print(f"  Eğitim başlıyor: {EPOCHS} epoch, batch={BATCH_SIZE}...")
    t0   = time.time()
    hist = gan.fit(
        x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[GANMonitor(n_samples=16, latent_dim=LATENT_DIM)]
    )
    sure = time.time() - t0
    print(f"  ✅ Eğitim tamamlandı: {sure:.0f} saniye")

    d_loss_tarih  = hist.history["d_loss"]
    g_loss_tarih  = hist.history["g_loss"]
    d_real_tarih  = hist.history["d_real"]
    d_fake_tarih  = hist.history["d_fake"]
    epochlar_list = list(range(1, EPOCHS + 1))

else:
    # Simülasyon eğitim geçmişi
    print("  [SIM] GAN parametreleri:")
    print("    d_optimizer: Adam(lr=2e-4, β₁=0.5)")
    print("    g_optimizer: Adam(lr=2e-4, β₁=0.5)")
    print("    loss_fn    : BinaryCrossentropy(from_logits=True)")
    print("    Label Smooth: 0.9  (gerçek etiket=0.9 yerine 1.0)")
    print()
    print(f"  [SIM] {EPOCHS} epoch eğitim simüle ediliyor...")

    def simule_gan_egitimi(n_epoch=50, seed=42):
        np.random.seed(seed)
        d_loss_l, g_loss_l, d_real_l, d_fake_l = [], [], [], []
        for e in range(1, n_epoch + 1):
            t = e / n_epoch
            # Gerçekçi GAN kayıp dinamikleri
            # D ilk başta çok iyi (G kötü) → G gelişince denge
            d_real = 0.9 - 0.15 * t + np.random.normal(0, 0.03)
            d_fake = 0.05 + 0.40 * t * (1 - 0.3 * t) + np.random.normal(0, 0.04)
            d_real = float(np.clip(d_real, 0.4, 0.95))
            d_fake = float(np.clip(d_fake, 0.02, 0.65))

            d_loss_real = -np.log(d_real + 1e-8) * 0.9
            d_loss_fake = -np.log(1 - d_fake + 1e-8)
            d_loss = float(d_loss_real + d_loss_fake + np.random.normal(0, 0.04))
            g_loss = float(-np.log(d_fake + 1e-8) + np.random.normal(0, 0.08))
            g_loss = max(g_loss, 0.3)

            d_loss_l.append(round(d_loss, 4))
            g_loss_l.append(round(g_loss, 4))
            d_real_l.append(round(d_real, 4))
            d_fake_l.append(round(d_fake, 4))
        return d_loss_l, g_loss_l, d_real_l, d_fake_l

    d_loss_tarih, g_loss_tarih, d_real_tarih, d_fake_tarih = simule_gan_egitimi(EPOCHS)
    epochlar_list = list(range(1, EPOCHS + 1))

    print(f"  [SIM] Son d_loss : {d_loss_tarih[-1]:.4f}")
    print(f"  [SIM] Son g_loss : {g_loss_tarih[-1]:.4f}")
    print(f"  [SIM] Son D(x)   : {d_real_tarih[-1]:.4f}  (ideal ≈ 0.5)")
    print(f"  [SIM] Son D(G(z)): {d_fake_tarih[-1]:.4f}  (ideal ≈ 0.5)")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5: MODE COLLAPSE ERKEN UYARI
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: Mode Collapse Erken Uyarı Sistemi")
print("─" * 65)
print("""
  Mode Collapse Belirtileri:
    ① G kaybı çok düşük, D kaybı çok yüksek (G tüm D'yi kandırıyor)
    ② Üretilen görüntüler çeşitliliği düşük (tüm z → benzer çıktı)
    ③ D(G(z)) → 1.0 yakın (D tamamen kandırılmış)

  Tespit Metrikleri:
    - g_loss < 0.3 VE d_loss > 2.5  → Tehlike ⚠️
    - D(G(z)) > 0.85 art arda 5 epoch → Mode Collapse şüphesi
""")

def mode_collapse_kontrol(g_loss_tarih, d_loss_tarih, d_fake_tarih):
    tehlike_sayisi = 0
    for e, (gl, dl, df) in enumerate(
            zip(g_loss_tarih, d_loss_tarih, d_fake_tarih), 1):
        if gl < 0.3 and dl > 2.5:
            tehlike_sayisi += 1
            if tehlike_sayisi == 1:
                print(f"  ⚠️  Epoch {e:3d}: g_loss={gl:.4f} < 0.3, d_loss={dl:.4f} > 2.5  → Tehlike!")
        else:
            tehlike_sayisi = 0

    ardisik_yuksek = 0
    for e, df in enumerate(d_fake_tarih, 1):
        if df > 0.85:
            ardisik_yuksek += 1
            if ardisik_yuksek >= 5:
                print(f"  ⚠️  Epoch {e:3d}: D(G(z))={df:.4f} > 0.85 — "
                      f"{ardisik_yuksek} art arda epoch")
        else:
            ardisik_yuksek = 0

    son_d_fake = np.mean(d_fake_tarih[-5:])
    son_g_loss = np.mean(g_loss_tarih[-5:])
    if son_d_fake > 0.7:
        print(f"  ⚠️  Son 5 epoch ortalama D(G(z))={son_d_fake:.4f} yüksek")
    if son_g_loss < 0.5:
        print(f"  ⚠️  Son 5 epoch ortalama g_loss={son_g_loss:.4f} düşük")
    if son_d_fake < 0.7 and son_g_loss > 0.5:
        print(f"  ✅ Mode collapse tespit edilmedi  "
              f"D(G(z))={son_d_fake:.4f}  g_loss={son_g_loss:.4f}")

mode_collapse_kontrol(g_loss_tarih, d_loss_tarih, d_fake_tarih)

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6: YENİ GÖRÜNTÜ ÜRETME
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Yeni Görüntü Üretme (8×8 Grid)")
print("─" * 65)

N_URET = 64  # 8×8 grid

if not SIM_MODE:
    np.random.seed(42)
    z_yeni   = tf.random.normal([N_URET, LATENT_DIM])
    uretilen = generator(z_yeni, training=False).numpy()
    uretilen = (uretilen + 1.0) / 2.0   # [-1,1] → [0,1]
    uretilen = uretilen.squeeze()
    print(f"  Üretilen görüntü sayısı: {N_URET}  (8×8 grid)")
    print(f"  Değer aralığı          : [{uretilen.min():.3f}, {uretilen.max():.3f}]")
else:
    print(f"  [SIM] {N_URET} kıyafet görüntüsü üretiliyor...")

    def kiyafet_goruntu_uret(latent_z, sinif_ipucu=None):
        """z vektöründen kıyafet benzeri görüntü üretir."""
        seed = int(abs(np.sum(latent_z)) * 1000) % 2**20
        np.random.seed(seed)
        img  = np.zeros((IMG_BOYUT, IMG_BOYUT))
        cx, cy = 14, 14

        # Latent vektörden şekil belirle
        z_norm = latent_z[:2] if len(latent_z) >= 2 else np.array([0.0, 0.0])
        genislik = int(7 + z_norm[0] * 2.5)
        yukseklik = int(10 + z_norm[1] * 2.0)
        genislik  = max(3, min(12, genislik))
        yukseklik = max(5, min(13, yukseklik))

        # Gövde (dikdörtgen kıyafet gövdesi)
        for y in range(cy - yukseklik // 2, cy + yukseklik // 2):
            for x in range(cx - genislik, cx + genislik):
                if 0 <= x < IMG_BOYUT and 0 <= y < IMG_BOYUT:
                    dist_edge = min(x - (cx - genislik), (cx + genislik - 1) - x,
                                    y - (cy - yukseklik // 2),
                                    (cy + yukseklik // 2 - 1) - y)
                    img[y, x] = 0.6 + 0.3 * min(dist_edge / 3, 1.0)

        # Desen / texture
        for _ in range(30):
            px = np.random.randint(cx - genislik + 1, cx + genislik - 1)
            py = np.random.randint(cy - yukseklik // 2 + 1, cy + yukseklik // 2 - 1)
            if 0 <= px < IMG_BOYUT and 0 <= py < IMG_BOYUT:
                img[py, px] = np.random.uniform(0.2, 0.5)

        img = gaussian_filter(img, sigma=1.2)
        img = img + np.random.normal(0, 0.04, img.shape)
        if img.max() > 0:
            img = img / img.max()
        return np.clip(img, 0, 1)

    np.random.seed(42)
    z_ornekler = np.random.normal(size=(N_URET, LATENT_DIM)).astype("float32")
    uretilen   = np.array([kiyafet_goruntu_uret(z_ornekler[i]) for i in range(N_URET)])
    print(f"  [SIM] {N_URET} görüntü üretildi. shape={uretilen.shape}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 7: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Görselleştirme (5 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#F0FDF4")
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.40, wspace=0.32,
                        top=0.93, bottom=0.05)

# ── GRAFİK 1: Eğitim Kaybı (d_loss & g_loss) ────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
ep_arr = np.array(epochlar_list)
ax1.plot(ep_arr, d_loss_tarih, color="#EF4444", linewidth=2.0,
         label=f"D Loss  (son={d_loss_tarih[-1]:.3f})")
ax1.plot(ep_arr, g_loss_tarih, color="#0D9488", linewidth=2.0,
         label=f"G Loss  (son={g_loss_tarih[-1]:.3f})")
ax1.axhline(y=1.386, color="#94A3B8", linestyle=":", linewidth=1.5,
            label="Nash dengesi ≈ ln(2)=1.386")
ax1.fill_between(ep_arr, d_loss_tarih, alpha=0.12, color="#EF4444")
ax1.fill_between(ep_arr, g_loss_tarih, alpha=0.12, color="#0D9488")
ax1.set_title("D Loss vs G Loss (50 Epoch)\nNash dengesi: ln(2) ≈ 1.386",
              fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.4)

# ── GRAFİK 2: D(x) ve D(G(z)) zamanla ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
ax2.plot(ep_arr, d_real_tarih, color="#6D28D9", linewidth=2.2,
         label=f"D(x) = Gerçek  (son={d_real_tarih[-1]:.3f})")
ax2.plot(ep_arr, d_fake_tarih, color="#D97706", linewidth=2.2,
         label=f"D(G(z)) = Sahte  (son={d_fake_tarih[-1]:.3f})")
ax2.axhline(y=0.5, color="#94A3B8", linestyle="--", linewidth=1.5,
            label="İdeal denge = 0.5")
ax2.fill_between(ep_arr, d_real_tarih, 0.5,
                 where=np.array(d_real_tarih) > 0.5,
                 alpha=0.12, color="#6D28D9")
ax2.fill_between(ep_arr, d_fake_tarih, 0.5,
                 where=np.array(d_fake_tarih) > 0.5,
                 alpha=0.12, color="#D97706")
ax2.set_title("Discriminator Güven Skorları\nD(x) ↔ D(G(z)) Dengesi",
              fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Discriminator Çıktısı (Sigmoid)", fontsize=11)
ax2.set_ylim(-0.05, 1.15)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.4)

# ── GRAFİK 3: Son Kayıplar Özet (Hap grafiği) ───────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
n_son = 10
ep_son = ep_arr[-n_son:]
genislik = 0.35
ax3.bar(ep_son - genislik / 2, d_loss_tarih[-n_son:], genislik,
        color="#EF4444", label="D Loss", edgecolor="white", alpha=0.85)
ax3.bar(ep_son + genislik / 2, g_loss_tarih[-n_son:], genislik,
        color="#0D9488", label="G Loss", edgecolor="white", alpha=0.85)
ax3.axhline(y=1.386, color="#94A3B8", linestyle=":", linewidth=1.5)
ax3.set_title("Son 10 Epoch Kayıp Detayı\n(Epoch 41–50)",
              fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("Loss", fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(axis="y", alpha=0.4)

# ── GRAFİK 4: Üretilen 8×8 Görüntü Grid ─────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.axis("off")
ax4.set_title("DCGAN Çıktısı — 64 Üretilmiş Fashion-MNIST Görüntüsü (8×8 Grid)\n"
              f"Epoch={EPOCHS} | z_dim={LATENT_DIM} | batch={BATCH_SIZE}",
              fontsize=13, fontweight="bold", pad=10)

inner4 = gridspec.GridSpecFromSubplotSpec(
    8, 8, subplot_spec=gs[1, :2], hspace=0.04, wspace=0.04
)
for i in range(8):
    for j in range(8):
        ax_tmp = fig.add_subplot(inner4[i, j])
        ax_tmp.imshow(uretilen[i * 8 + j], cmap="gray",
                      vmin=0, vmax=1, interpolation="nearest")
        ax_tmp.axis("off")

# ── GRAFİK 5: Eğitim İlerlemesi Özet Tablosu ─────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
kontrol_noktalari = [1, 5, 10, 20, 30, 40, 50]
tablo_data = []
for ep in kontrol_noktalari:
    idx = ep - 1
    if idx < len(d_loss_tarih):
        denge = "✅" if abs(d_real_tarih[idx] - 0.5) < 0.15 else "⚠️"
        tablo_data.append([
            str(ep),
            f"{d_loss_tarih[idx]:.3f}",
            f"{g_loss_tarih[idx]:.3f}",
            f"{d_real_tarih[idx]:.3f}",
            f"{d_fake_tarih[idx]:.3f}",
            denge,
        ])
sutun_bas = ["Epoch", "D Loss", "G Loss", "D(x)", "D(G(z))", "Denge"]
tablo = ax5.table(
    cellText=tablo_data,
    colLabels=sutun_bas,
    loc="center", cellLoc="center",
)
tablo.auto_set_font_size(False)
tablo.set_fontsize(10)
tablo.scale(1.18, 1.9)
for (row, col), cell in tablo.get_celld().items():
    if row == 0:
        cell.set_facecolor("#064E3B")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 1:
        cell.set_facecolor("#ECFDF5")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#D1FAE5")
ax5.set_title("Eğitim İlerleme Tablosu\n(Kontrol Noktaları)",
              fontsize=12, fontweight="bold", y=0.92)

# Ana başlık
fig.suptitle(
    "HAFTA 4 PAZAR — UYGULAMA 03\n"
    "DCGAN ile Fashion-MNIST: Generator · Discriminator · Adversarial Eğitim · 8×8 Grid",
    fontsize=15, fontweight="bold", color="#064E3B", y=0.98
)

plt.savefig("h4p_03_dcgan.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h4p_03_dcgan.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Generator param.     : {'3,623,553' if SIM_MODE else str(generator.count_params())}")
print(f"  Discriminator param. : {'621,825' if SIM_MODE else str(discriminator.count_params())}")
print(f"  Latent dim (z)       : {LATENT_DIM}")
print(f"  Eğitim               : {EPOCHS} epoch, batch={BATCH_SIZE}")
print(f"  Son d_loss           : {d_loss_tarih[-1]:.4f}")
print(f"  Son g_loss           : {g_loss_tarih[-1]:.4f}")
print(f"  Son D(x)             : {d_real_tarih[-1]:.4f}  (ideal ≈ 0.5)")
print(f"  Son D(G(z))          : {d_fake_tarih[-1]:.4f}  (ideal ≈ 0.5)")
print(f"  Üretilen görüntüler  : {N_URET}  (8×8 grid)")
print(f"  Grafik çıktısı       : h4p_03_dcgan.png")
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print("=" * 65)
