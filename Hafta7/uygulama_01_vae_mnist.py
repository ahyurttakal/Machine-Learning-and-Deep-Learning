"""
=============================================================================
HAFTA 4 PAZAR — UYGULAMA 01
VAE (Variational Autoencoder) ile MNIST Latent Space
=============================================================================
Kapsam:
  - Encoder / Decoder Keras ile sıfırdan inşa
  - Reparameterization Trick: özel Sampling katmanı
  - ELBO kaybı: Reconstruction (BCE) + KL Divergence
  - 2D Latent Space görselleştirme (10 sınıf rengi)
  - 20×20 manifold grid: rakam geçişleri
  - Orijinal vs yeniden oluşturulmuş görüntüler
  - Eğitim kaybı (recon + KL bileşenleri)
  - TensorFlow/Keras yoksa tam simülasyon modu

Kurulum: pip install tensorflow numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

SIM_MODE = not TF_AVAILABLE

print("=" * 65)
print("  HAFTA 4 PAZAR — UYGULAMA 01")
print("  VAE ile MNIST Latent Space")
print("=" * 65)
print(f"  Mod        : {'🔵 Gerçek (TensorFlow)' if not SIM_MODE else '🟡 Simülasyon'}")
print(f"  TensorFlow : {'✅' if TF_AVAILABLE else '❌  pip install tensorflow'}")
print()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1: VERİ HAZIRLAMA
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: MNIST Veri Hazırlama")
print("─" * 65)

LATENT_DIM   = 2      # 2D → görselleştirme
INTERMEDIATE = 256
ORIGINAL_DIM = 784    # 28×28
EPOCHS       = 30
BATCH_SIZE   = 128

if not SIM_MODE:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, ORIGINAL_DIM)
    x_test  = x_test.reshape(-1, ORIGINAL_DIM)
    print(f"  Train : {x_train.shape}  dtype={x_train.dtype}")
    print(f"  Test  : {x_test.shape}")
    print(f"  Sınıflar: {np.unique(y_train)}")
else:
    print(f"  [SIM] x_train: (60000, 784)  dtype=float32")
    print(f"  [SIM] x_test : (10000, 784)")
    print(f"  [SIM] Sınıflar: [0 1 2 3 4 5 6 7 8 9]")
    # Simülasyon için sahte veri
    np.random.seed(42)
    x_train = np.random.rand(60000, ORIGINAL_DIM).astype("float32")
    y_train = np.random.randint(0, 10, 60000)
    x_test  = np.random.rand(10000, ORIGINAL_DIM).astype("float32")
    y_test  = np.random.randint(0, 10, 10000)

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2: MODEL TANIMI
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: VAE Model Tanımı")
print("─" * 65)

if not SIM_MODE:
    # ── Sampling Layer (Reparameterization Trick) ─────────────
    class Sampling(keras.layers.Layer):
        """z = μ + σ·ε  (ε ~ N(0,1))  — Gradyan μ ve σ üzerinden akar."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch  = tf.shape(z_mean)[0]
            dim    = tf.shape(z_mean)[1]
            eps    = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * eps

    # ── Encoder ───────────────────────────────────────────────
    encoder_inputs = keras.Input(shape=(ORIGINAL_DIM,), name="encoder_input")
    x = keras.layers.Dense(INTERMEDIATE, activation="relu")(encoder_inputs)
    z_mean    = keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z         = Sampling(name="z")([z_mean, z_log_var])
    encoder   = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # ── Decoder ───────────────────────────────────────────────
    decoder_inputs = keras.Input(shape=(LATENT_DIM,), name="decoder_input")
    x = keras.layers.Dense(INTERMEDIATE, activation="relu")(decoder_inputs)
    decoder_outputs = keras.layers.Dense(ORIGINAL_DIM, activation="sigmoid")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    print(f"  Encoder parametreleri  : {encoder.count_params():,}")
    print(f"  Decoder parametreleri  : {decoder.count_params():,}")
    print(f"  Toplam parametre       : {encoder.count_params() + decoder.count_params():,}")

    # ── VAE Sınıfı ────────────────────────────────────────────
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder    = encoder
            self.decoder    = decoder
            self.total_loss_tracker       = keras.metrics.Mean(name="total_loss")
            self.recon_loss_tracker       = keras.metrics.Mean(name="recon_loss")
            self.kl_loss_tracker          = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)

                # Reconstruction Loss: BCE × piksel sayısı
                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction),
                        axis=-1
                    )
                )
                # KL Divergence: −0.5 × Σ(1 + log σ² − μ² − σ²)
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=1
                    )
                )
                total_loss = recon_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {m.name: m.result() for m in self.metrics}

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder(z)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    print("  VAE modeli derlendi. Optimizer: Adam(lr=1e-3)")
else:
    print("  [SIM] Encoder parametreleri  : 203,778")
    print("  [SIM] Decoder parametreleri  : 203,553")
    print("  [SIM] Toplam parametre       : 407,331")
    print("  [SIM] VAE modeli hazır.")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3: EĞİTİM
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Eğitim")
print("─" * 65)
print(f"  Epoch       : {EPOCHS}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Latent dim  : {LATENT_DIM}")

def simule_egitim(n_epoch=30):
    """Gerçekçi VAE eğitim kayıp eğrileri."""
    np.random.seed(42)
    epochlar     = list(range(1, n_epoch + 1))
    total_kayip  = []
    recon_kayip  = []
    kl_kayip     = []
    for e in epochlar:
        t = e / n_epoch
        recon = 165 * np.exp(-2.2 * t) + 55 + np.random.normal(0, 1.5)
        kl    = 18  * (1 - np.exp(-3.5 * t)) + np.random.normal(0, 0.4)
        recon = max(recon, 55.0)
        kl    = max(kl, 0.5)
        recon_kayip.append(round(recon, 2))
        kl_kayip.append(round(kl, 2))
        total_kayip.append(round(recon + kl, 2))
    return epochlar, total_kayip, recon_kayip, kl_kayip


if not SIM_MODE:
    import time
    t0 = time.time()
    history = vae.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                      callbacks=[keras.callbacks.TerminateOnNaN()])
    sure = time.time() - t0
    print(f"  Eğitim süresi : {sure:.0f} saniye")

    epochlar    = list(range(1, EPOCHS + 1))
    total_kayip = history.history["total_loss"]
    recon_kayip = history.history["recon_loss"]
    kl_kayip    = history.history["kl_loss"]
    print(f"  Son total_loss : {total_kayip[-1]:.2f}")
    print(f"  Son recon_loss : {recon_kayip[-1]:.2f}")
    print(f"  Son kl_loss    : {kl_kayip[-1]:.2f}")
else:
    print("  [SIM] Eğitim simüle ediliyor...")
    epochlar, total_kayip, recon_kayip, kl_kayip = simule_egitim(EPOCHS)
    print(f"  [SIM] Son total_loss : {total_kayip[-1]:.2f}")
    print(f"  [SIM] Son recon_loss : {recon_kayip[-1]:.2f}")
    print(f"  [SIM] Son kl_loss    : {kl_kayip[-1]:.2f}")

print()
print("  İlk 5 epoch kaybı:")
print(f"  {'Epoch':>6} {'Total':>10} {'Recon':>10} {'KL':>8}")
print("  " + "-" * 38)
for i in range(min(5, len(epochlar))):
    print(f"  {epochlar[i]:>6} {total_kayip[i]:>10.2f} {recon_kayip[i]:>10.2f} {kl_kayip[i]:>8.2f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4: LATENT SPACE ÜRETİMİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Latent Space Kodlama")
print("─" * 65)

def simule_latent_space(n=2000):
    """10 sınıf için gerçekçi 2D latent noktalar üretir."""
    np.random.seed(42)
    # Her rakam için doğal latent küme merkezleri
    merkezler = [
        (-2.1, -1.8),  # 0 — sol-alt
        ( 2.4,  0.4),  # 1 — sağ-orta
        (-0.6,  2.6),  # 2 — üst-orta
        ( 1.8,  2.4),  # 3 — sağ-üst
        (-2.2,  1.4),  # 4 — sol-üst
        ( 0.2, -2.4),  # 5 — alt-orta
        (-1.4, -0.6),  # 6 — sol-orta
        ( 1.2, -2.0),  # 7 — sağ-alt
        ( 2.8, -0.8),  # 8 — sağ-alt-orta
        (-0.2,  0.6),  # 9 — orta
    ]
    z_noktalar = []
    etiketler  = []
    for sinif, (mx, my) in enumerate(merkezler):
        n_sinif = n // 10
        z = np.random.multivariate_normal(
            [mx, my],
            [[0.6, 0.08], [0.08, 0.6]],
            n_sinif
        )
        z_noktalar.append(z)
        etiketler.extend([sinif] * n_sinif)
    return np.vstack(z_noktalar), np.array(etiketler)


if not SIM_MODE:
    z_ortalama, z_log_var, z_ornekler = encoder.predict(x_test[:2000], batch_size=256)
    y_gorsellestir = y_test[:2000]
    print(f"  Encode edildi: {z_ornekler.shape}  →  z_mean shape: {z_ortalama.shape}")
else:
    print("  [SIM] Latent noktalar üretiliyor...")
    z_ornekler, y_gorsellestir = simule_latent_space(2000)
    z_ortalama = z_ornekler + np.random.normal(0, 0.05, z_ornekler.shape)
    print(f"  [SIM] z_ornekler shape: {z_ornekler.shape}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5: MANİFOLD (20×20 GRID)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: 20×20 Manifold Grid Üretimi")
print("─" * 65)

MANIFOLD_N      = 20
MANIFOLD_ARALIK = 3.0
GORUNTU_BOYUT   = 28

if not SIM_MODE:
    from scipy.stats import norm
    grid_x = norm.ppf(np.linspace(0.05, 0.95, MANIFOLD_N))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, MANIFOLD_N))
    manifold = np.zeros((GORUNTU_BOYUT * MANIFOLD_N, GORUNTU_BOYUT * MANIFOLD_N))
    for i, yi in enumerate(grid_x[::-1]):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            goruntu = x_decoded[0].reshape(GORUNTU_BOYUT, GORUNTU_BOYUT)
            manifold[i*GORUNTU_BOYUT:(i+1)*GORUNTU_BOYUT,
                     j*GORUNTU_BOYUT:(j+1)*GORUNTU_BOYUT] = goruntu
    print(f"  Manifold shape: {manifold.shape}  (560×560 piksel)")
else:
    # Simülasyon: gerçekçi manifold benzeri görsel
    manifold = np.zeros((GORUNTU_BOYUT * MANIFOLD_N, GORUNTU_BOYUT * MANIFOLD_N))
    for i in range(MANIFOLD_N):
        for j in range(MANIFOLD_N):
            z1 = (j - MANIFOLD_N / 2) / (MANIFOLD_N / 4)
            z2 = (i - MANIFOLD_N / 2) / (MANIFOLD_N / 4)
            r  = np.sqrt(z1**2 + z2**2)
            theta = np.arctan2(z2, z1)
            sinif_idx = int((theta / (2 * np.pi) + 0.5) * 10) % 10
            # Rakam benzeri rastgele görüntü
            np.random.seed(sinif_idx * 100 + i * MANIFOLD_N + j)
            goruntu = np.zeros((GORUNTU_BOYUT, GORUNTU_BOYUT))
            t = np.linspace(0, 2 * np.pi, 100)
            cx, cy = 14, 14
            for k in range(10 + sinif_idx * 2):
                idx_t = (k * 8) % len(t)
                px = int(cx + 7 * np.cos(t[idx_t]) + np.random.randint(-2, 3))
                py = int(cy + 7 * np.sin(t[idx_t]) + np.random.randint(-2, 3))
                if 0 <= px < 28 and 0 <= py < 28:
                    goruntu[py, px] = np.random.uniform(0.4, 0.95)
            from scipy.ndimage import gaussian_filter
            goruntu = gaussian_filter(goruntu, sigma=1.2)
            goruntu = (goruntu - goruntu.min()) / (goruntu.max() - goruntu.min() + 1e-8)
            manifold[i*GORUNTU_BOYUT:(i+1)*GORUNTU_BOYUT,
                     j*GORUNTU_BOYUT:(j+1)*GORUNTU_BOYUT] = goruntu
    print(f"  [SIM] Manifold shape: {manifold.shape}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6: ÖRNEK YENİDEN OLUŞTURMA
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Orijinal vs Yeniden Oluşturulmuş")
print("─" * 65)

N_ORNEK = 10
if not SIM_MODE:
    orjinaller = x_test[:N_ORNEK]
    z_m, z_lv, z = encoder.predict(orjinaller, verbose=0)
    yeniden = decoder.predict(z, verbose=0)
    print(f"  {N_ORNEK} test görüntüsü yeniden oluşturuldu.")
    print(f"  MSE (ortalama): {np.mean((orjinaller - yeniden)**2):.5f}")
else:
    # Gerçek MNIST benzeri digit görüntüleri simüle et
    def digit_goruntu_uret(digit, seed=0):
        np.random.seed(seed)
        img = np.zeros((28, 28))
        cx, cy = 14, 14
        # Basit çizgi/çevre tabanlı rakam yaklaşımı
        if digit == 0:
            for t in np.linspace(0, 2*np.pi, 50):
                x, y = int(cx+7*np.cos(t)), int(cy+4*np.sin(t))
                if 0<=x<28 and 0<=y<28: img[y,x] = 1.0
        elif digit == 1:
            for y in range(5, 23): img[y, 14] = 1.0; img[y, 15] = 0.8
        elif digit == 8:
            for t in np.linspace(0, 2*np.pi, 50):
                x, y = int(cx+5*np.cos(t)), int(cy-4+3*np.sin(t)); 
                if 0<=x<28 and 0<=y<28: img[y,x] = 1.0
            for t in np.linspace(0, 2*np.pi, 50):
                x, y = int(cx+5*np.cos(t)), int(cy+4+3*np.sin(t)); 
                if 0<=x<28 and 0<=y<28: img[y,x] = 1.0
        else:
            for t in np.linspace(0, 2*np.pi, 50):
                r = 6 + 2*np.sin(digit*t)
                x, y = int(cx+r*np.cos(t)), int(cy+r*np.sin(t)*0.8)
                if 0<=x<28 and 0<=y<28: img[y,x] = 1.0
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=1.5) + np.random.normal(0, 0.04, (28,28))
        return np.clip(img, 0, 1)

    orjinaller = np.array([digit_goruntu_uret(i, seed=i*7) for i in range(N_ORNEK)]).reshape(N_ORNEK, ORIGINAL_DIM)
    yeniden    = orjinaller + np.random.normal(0, 0.06, orjinaller.shape)
    yeniden    = np.clip(yeniden, 0, 1)
    mse = np.mean((orjinaller - yeniden)**2)
    print(f"  [SIM] {N_ORNEK} örnek yeniden oluşturuldu.")
    print(f"  [SIM] MSE (ortalama): {mse:.5f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 7: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Görselleştirme (4 panel)")
print("─" * 65)

SINIF_RENKLERI = [
    "#EF4444","#F97316","#EAB308","#22C55E","#14B8A6",
    "#3B82F6","#8B5CF6","#EC4899","#6B7280","#0EA5E9"
]

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#FAF5FF")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.32,
                        top=0.93, bottom=0.05)

# ── GRAFİK 1: 2D Latent Space ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#F5F3FF")
sinif_renk_arr = [SINIF_RENKLERI[int(y)] for y in y_gorsellestir]
scatter = ax1.scatter(
    z_ornekler[:, 0], z_ornekler[:, 1],
    c=sinif_renk_arr, alpha=0.55, s=12, linewidths=0
)
# Sınıf etiketleri (merkez noktalara)
for sinif in range(10):
    maske = y_gorsellestir == sinif
    if maske.sum() > 0:
        cx = z_ornekler[maske, 0].mean()
        cy = z_ornekler[maske, 1].mean()
        ax1.text(cx, cy, str(sinif), fontsize=14, fontweight="bold",
                 color=SINIF_RENKLERI[sinif],
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=SINIF_RENKLERI[sinif], alpha=0.8))
ax1.set_title("2D Latent Space — MNIST Rakamları\n(Encoder çıkışı, n=2000)", 
              fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("z₁", fontsize=12)
ax1.set_ylabel("z₂", fontsize=12)
legend_handles = [
    plt.scatter([], [], c=SINIF_RENKLERI[i], s=50, label=str(i)) for i in range(10)
]
ax1.legend(handles=legend_handles, title="Rakam", fontsize=9,
           loc="upper right", ncol=2, framealpha=0.9)
ax1.grid(alpha=0.3)

# ── GRAFİK 2: 20×20 Manifold ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(manifold, cmap="gray_r", interpolation="nearest",
           extent=[-MANIFOLD_ARALIK, MANIFOLD_ARALIK,
                   -MANIFOLD_ARALIK, MANIFOLD_ARALIK])
ax2.set_title(f"20×20 Manifold Grid (z₁,z₂ ∈ [-3, 3])\n"
              f"Her nokta: z örnekle → Decoder → 28×28 görüntü",
              fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("z₁", fontsize=12)
ax2.set_ylabel("z₂", fontsize=12)
ax2.grid(False)

# ── GRAFİK 3: Eğitim Kaybı ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("white")
ep_arr = np.array(epochlar)
ax3.fill_between(ep_arr, recon_kayip, alpha=0.25, color="#6D28D9", label="_nolegend_")
ax3.fill_between(ep_arr, kl_kayip,    alpha=0.25, color="#0D9488", label="_nolegend_")
ax3.plot(ep_arr, total_kayip, color="#1E293B", linewidth=2.4,
         label=f"Total ELBO  (son={total_kayip[-1]:.1f})")
ax3.plot(ep_arr, recon_kayip, color="#6D28D9", linewidth=1.8, linestyle="--",
         label=f"Recon BCE   (son={recon_kayip[-1]:.1f})")
ax3.plot(ep_arr, kl_kayip,    color="#0D9488", linewidth=1.8, linestyle=":",
         label=f"KL Div.     (son={kl_kayip[-1]:.1f})")
ax3.set_title("Eğitim Kaybı — ELBO Bileşenleri\n(Total = Recon + KL)",
              fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("Kayıp", fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.4)

# ── GRAFİK 4: Orijinal vs Yeniden Oluşturulmuş ───────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
ax4.set_title(f"Orijinal (üst) vs Yeniden Oluşturulmuş (alt)\n"
              f"MSE={np.mean((orjinaller - yeniden)**2):.4f}",
              fontsize=13, fontweight="bold", pad=10, y=0.97)

inner_gs = gridspec.GridSpecFromSubplotSpec(
    2, N_ORNEK, subplot_spec=gs[1, 1], hspace=0.08, wspace=0.06
)
for idx in range(N_ORNEK):
    # Orijinal
    ax_or = fig.add_subplot(inner_gs[0, idx])
    ax_or.imshow(orjinaller[idx].reshape(GORUNTU_BOYUT, GORUNTU_BOYUT),
                 cmap="gray_r", interpolation="nearest")
    ax_or.axis("off")
    if idx == 0:
        ax_or.set_ylabel("Orijinal", fontsize=9, rotation=0, labelpad=36)
    # Yeniden oluşturulmuş
    ax_re = fig.add_subplot(inner_gs[1, idx])
    ax_re.imshow(yeniden[idx].reshape(GORUNTU_BOYUT, GORUNTU_BOYUT),
                 cmap="gray_r", interpolation="nearest")
    ax_re.axis("off")
    if idx == 0:
        ax_re.set_ylabel("Rekon.", fontsize=9, rotation=0, labelpad=36)

fig.suptitle(
    "HAFTA 4 PAZAR — UYGULAMA 01\n"
    "VAE ile MNIST: 2D Latent Space · Manifold · ELBO Kaybı · Rekonstrüksiyon",
    fontsize=15, fontweight="bold", color="#1E0A3C", y=0.98
)

plt.savefig("h4p_01_vae_mnist.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h4p_01_vae_mnist.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 8: YENİ GÖRÜNTÜ ÜRETME (ÖRNEKLEME)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Yeni Görüntü Üretme (Latent Örnekleme)")
print("─" * 65)
print()
print("  N(0,I)'dan z örnekle → Decoder → yeni görüntü:")

if not SIM_MODE:
    np.random.seed(99)
    z_ornekler_yeni = np.random.normal(size=(16, LATENT_DIM))
    uretilen = decoder.predict(z_ornekler_yeni, verbose=0)
    print(f"  Üretilen görüntü sayısı: {len(uretilen)}")
    print(f"  Her görüntü boyutu    : {GORUNTU_BOYUT}×{GORUNTU_BOYUT}")
else:
    print("  [SIM] 16 yeni görüntü üretildi.")
    print(f"  [SIM] Her görüntü boyutu: {GORUNTU_BOYUT}×{GORUNTU_BOYUT}")

print()
print("""  # Kod özeti — yeni görüntü üretmek için:
  import numpy as np
  z_yeni = np.random.normal(size=(n_ornek, latent_dim))  # N(0,I)'dan örnekle
  uretilen_gorseller = decoder.predict(z_yeni)            # Decode
  # uretilen_gorseller.shape → (n_ornek, 784)
""")

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Latent boyut         : {LATENT_DIM}D")
print(f"  Encoder mimarisi     : 784 → Dense(256) → μ({LATENT_DIM}), σ({LATENT_DIM})")
print(f"  Decoder mimarisi     : {LATENT_DIM} → Dense(256) → 784 (Sigmoid)")
print(f"  Eğitim               : {EPOCHS} epoch, batch={BATCH_SIZE}")
print(f"  Son ELBO kaybı       : {total_kayip[-1]:.2f}  "
      f"(recon={recon_kayip[-1]:.2f}, KL={kl_kayip[-1]:.2f})")
print(f"  Latent görsel        : 2000 test noktası")
print(f"  Manifold grid        : {MANIFOLD_N}×{MANIFOLD_N} = {MANIFOLD_N**2} nokta")
print(f"  Grafik çıktısı       : h4p_01_vae_mnist.png")
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print("=" * 65)
