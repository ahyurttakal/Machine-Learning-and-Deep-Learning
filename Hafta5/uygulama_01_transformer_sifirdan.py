"""
=============================================================================
UYGULAMA 01 — Transformer Sıfırdan: Attention, MHA, Encoder
=============================================================================
Kapsam:
  - Positional Encoding (sinüzoidal) — NumPy + görselleştirme
  - Scaled Dot-Product Attention — NumPy ile adım adım
  - Multi-Head Attention — tf.keras.layers.Layer olarak
  - Encoder Block: MHA + FFN + Add & LayerNorm
  - Tam Transformer Encoder modeli — IMDB sınıflandırma
  - LSTM vs Transformer karşılaştırması (parametre, hız, doğruluk)

Kurulum: pip install tensorflow numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 01 — Transformer Sıfırdan")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. POZİSYON KODLAMASI (Positional Encoding)
# ─────────────────────────────────────────────────────────────
print("\n[1] Positional Encoding hesaplanıyor...")

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Vaswani et al. (2017) sinüzoidal PE.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pos = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
    i   = np.arange(d_model)[np.newaxis, :]          # (1, d_model)
    div = np.power(10000.0, (2 * (i // 2)) / d_model)
    pe  = pos / div                                   # (max_len, d_model)
    pe[:, 0::2] = np.sin(pe[:, 0::2])               # çift indeksler
    pe[:, 1::2] = np.cos(pe[:, 1::2])               # tek indeksler
    return pe.astype(np.float32)

MAX_LEN  = 200
D_MODEL  = 128
pe_matrix = sinusoidal_positional_encoding(MAX_LEN, D_MODEL)
print(f"    PE matrisi şekli: {pe_matrix.shape}  (max_len={MAX_LEN}, d_model={D_MODEL})")
print(f"    PE değer aralığı : [{pe_matrix.min():.3f}, {pe_matrix.max():.3f}]")

# ─────────────────────────────────────────────────────────────
# 2. SCALED DOT-PRODUCT ATTENTION (NumPy)
# ─────────────────────────────────────────────────────────────
print("\n[2] Scaled Dot-Product Attention (NumPy ile adım adım)...")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)
    V: (batch, heads, seq_v, d_v)
    """
    d_k    = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)   # (B, H, seq_q, seq_k)

    # Opsiyonel mask (causal/padding)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax (satır bazında normalize)
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    weights = softmax(scores)      # (B, H, seq_q, seq_k)
    output  = weights @ V          # (B, H, seq_q, d_v)
    return output, weights

# Demo: 2 token, 4 boyutlu Q/K/V
B, H, T, dk = 1, 1, 5, 8
Q_demo = np.random.randn(B, H, T, dk).astype(np.float32)
K_demo = np.random.randn(B, H, T, dk).astype(np.float32)
V_demo = np.random.randn(B, H, T, dk).astype(np.float32)

output_demo, weights_demo = scaled_dot_product_attention(Q_demo, K_demo, V_demo)
print(f"    Girdi Q şekli     : {Q_demo.shape}")
print(f"    Attention weights : {weights_demo.shape}  (satır toplamı=1: {weights_demo[0,0,0].sum():.4f})")
print(f"    Çıktı şekli       : {output_demo.shape}")
print(f"    Örnek ağırlıklar  : {weights_demo[0,0,0].round(4)}")

# Causal mask demo
causal_mask = np.tril(np.ones((T, T)))[np.newaxis, np.newaxis, :, :]
_, causal_weights = scaled_dot_product_attention(Q_demo, K_demo, V_demo, causal_mask)
print(f"    Causal ağırlıklar (satır 0): {causal_weights[0,0,0].round(4)}  (gelecek=0)")

# ─────────────────────────────────────────────────────────────
# 3. MULTI-HEAD ATTENTION (Keras Layer)
# ─────────────────────────────────────────────────────────────
print("\n[3] Multi-Head Attention katmanı tanımlanıyor...")

class MultiHeadSelfAttention(keras.layers.Layer):
    """
    Sıfırdan Multi-Head (Self-)Attention.
    d_model = n_heads × d_k
    """
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model, n_heads'e tam bölünmeli"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.d_model = d_model

        # Q, K, V projeksiyon matrisleri
        self.W_q = layers.Dense(d_model, use_bias=False)
        self.W_k = layers.Dense(d_model, use_bias=False)
        self.W_v = layers.Dense(d_model, use_bias=False)
        # Çıkış projeksiyonu
        self.W_o = layers.Dense(d_model, use_bias=False)

    def split_heads(self, x, batch_size):
        """(B, T, d_model) → (B, n_heads, T, d_k)"""
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None, training=None):
        B = tf.shape(query)[0]

        Q = self.split_heads(self.W_q(query), B)  # (B, H, T, d_k)
        K = self.split_heads(self.W_k(key),   B)
        V = self.split_heads(self.W_v(value), B)

        # Scaled dot-product attention
        d_k     = tf.cast(self.d_k, tf.float32)
        scores  = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        if mask is not None:
            scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        weights = tf.nn.softmax(scores, axis=-1)   # (B, H, T, T)

        context = tf.matmul(weights, V)            # (B, H, T, d_k)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (B, -1, self.d_model))
        return self.W_o(context), weights

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "n_heads": self.n_heads}

print("    ✅ MultiHeadSelfAttention hazır")

# ─────────────────────────────────────────────────────────────
# 4. ENCODER BLOCK
# ─────────────────────────────────────────────────────────────
print("\n[4] Encoder Block tanımlanıyor...")

class EncoderBlock(keras.layers.Layer):
    """
    Tek Transformer Encoder bloğu:
    x → MHA → Add+LN → FFN → Add+LN
    """
    def __init__(self, d_model, n_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha  = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn  = keras.Sequential([
            layers.Dense(dff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, mask=None, training=None):
        # Multi-Head Self-Attention + Residual
        attn_out, _ = self.mha(x, x, x, mask=mask, training=training)
        x = self.ln1(x + self.drop(attn_out, training=training))
        # Feed-Forward + Residual
        ffn_out = self.ffn(x, training=training)
        x = self.ln2(x + self.drop(ffn_out, training=training))
        return x

print("    ✅ EncoderBlock hazır")

# ─────────────────────────────────────────────────────────────
# 5. TAM TRANSFORMER ENCODER MODELİ (IMDB)
# ─────────────────────────────────────────────────────────────
print("\n[5] IMDB veri seti yükleniyor...")

VOCAB_SIZE  = 20000
MAX_LEN_SEQ = 200
EMBED_DIM   = 64
N_HEADS     = 4
DFF         = 256
N_LAYERS    = 2
DROPOUT     = 0.1
BATCH_SIZE  = 128

(X_tr_raw, y_tr_all), (X_te_raw, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
pad = lambda s, ml: keras.preprocessing.sequence.pad_sequences(
    s, maxlen=ml, padding="post", truncating="post")
X_all  = pad(X_tr_raw, MAX_LEN_SEQ)
X_test = pad(X_te_raw, MAX_LEN_SEQ)
X_val, X_train = X_all[:5000], X_all[5000:]
y_val, y_train = y_tr_all[:5000], y_tr_all[5000:]
print(f"    Eğitim: {len(X_train):,} | Val: {len(X_val):,}")

# Transformer sınıflandırıcı
def build_transformer_classifier(vocab=VOCAB_SIZE, max_len=MAX_LEN_SEQ,
                                  embed_dim=EMBED_DIM, n_heads=N_HEADS,
                                  dff=DFF, n_layers=N_LAYERS, dropout=DROPOUT):
    inp = keras.Input(shape=(max_len,))

    # Embedding + Positional Encoding
    x   = layers.Embedding(vocab, embed_dim, mask_zero=False)(inp)
    pe  = sinusoidal_positional_encoding(max_len, embed_dim)
    pe_tensor = tf.constant(pe[np.newaxis, :, :])       # (1, max_len, embed_dim)
    x   = x + pe_tensor

    x   = layers.Dropout(dropout)(x)

    # N Encoder Bloğu
    for i in range(n_layers):
        x = EncoderBlock(embed_dim, n_heads, dff, dropout,
                         name=f"encoder_{i+1}")(x)

    # Global Average Pooling → sınıflandırma
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="gelu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="TransformerClassifier")

# LSTM baseline modeli
def build_lstm_baseline(vocab=VOCAB_SIZE, max_len=MAX_LEN_SEQ, embed_dim=EMBED_DIM):
    inp = keras.Input(shape=(max_len,))
    x   = layers.Embedding(vocab, embed_dim, mask_zero=True)(inp)
    x   = layers.Bidirectional(layers.LSTM(128, dropout=0.2))(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="BiLSTM_Baseline")

print("\n[6] Modeller oluşturuluyor ve eğitiliyor...")

results = {}

def train_model(model, name, epochs=12):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                       restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3),
    ]
    t0 = time.time()
    h  = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=epochs, batch_size=BATCH_SIZE, callbacks=cbs, verbose=1)
    elapsed  = time.time() - t0
    test_res = model.evaluate(X_test, y_test, verbose=0)
    results[name] = {
        "history": h.history, "test_acc": test_res[1],
        "test_auc": test_res[2], "params": model.count_params(), "time": elapsed
    }
    print(f"\n  [{name}] test_acc={test_res[1]:.4f}  auc={test_res[2]:.4f}  "
          f"params={model.count_params():,}  t={elapsed:.0f}s")

tf.random.set_seed(42)
train_model(build_lstm_baseline(), "BiLSTM Baseline")

tf.random.set_seed(42)
train_model(build_transformer_classifier(n_layers=2), "Transformer (2L)")

tf.random.set_seed(42)
train_model(build_transformer_classifier(n_layers=4), "Transformer (4L)")

# ─────────────────────────────────────────────────────────────
# 6. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n[7] Görselleştirmeler hazırlanıyor...")

PALETTE = {"BiLSTM Baseline": "#DC2626", "Transformer (2L)": "#D97706",
           "Transformer (4L)": "#7C3AED"}

fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("Transformer Sıfırdan — IMDB Sınıflandırma", fontsize=15, fontweight="bold")

# ── a. Positional Encoding ısı haritası ─────────────────────
ax1 = fig.add_subplot(gs[0, :2])
im  = ax1.imshow(pe_matrix[:60, :64].T, aspect="auto", cmap="RdBu",
                  vmin=-1, vmax=1, origin="lower")
plt.colorbar(im, ax=ax1)
ax1.set_xlabel("Pozisyon"); ax1.set_ylabel("Boyut indeksi")
ax1.set_title("Sinüzoidal Positional Encoding (ilk 60 pozisyon)", fontweight="bold")

# ── b. PE boyut örüntüleri ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
for dim, color in zip([0, 2, 6, 14], ["#DC2626", "#D97706", "#7C3AED", "#059669"]):
    ax2.plot(pe_matrix[:60, dim], lw=2, color=color, label=f"dim={dim}")
ax2.set_xlabel("Pozisyon"); ax2.set_ylabel("PE değeri")
ax2.set_title("PE Boyut Örüntüleri — Farklı Frekanslar", fontweight="bold")
ax2.legend(); ax2.grid(alpha=0.3)

# ── c. Attention ağırlıkları ısı haritası ───────────────────
ax3 = fig.add_subplot(gs[1, 0])
sns_data = weights_demo[0, 0]   # (T, T)
ax3.imshow(sns_data, cmap="YlOrRd", aspect="auto")
ax3.set_title("Attention Weights (demo, T=5)", fontweight="bold")
ax3.set_xlabel("Key pozisyon"); ax3.set_ylabel("Query pozisyon")
for i in range(T):
    for j in range(T):
        ax3.text(j, i, f"{sns_data[i,j]:.2f}", ha="center", va="center", fontsize=8)

# ── d. Causal mask ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(causal_weights[0, 0], cmap="YlOrRd", aspect="auto")
ax4.set_title("Causal Attention Weights", fontweight="bold")
ax4.set_xlabel("Key pozisyon"); ax4.set_ylabel("Query pozisyon")
for i in range(T):
    for j in range(T):
        v = causal_weights[0, 0, i, j]
        ax4.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

# ── e. Val AUC karşılaştırma ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2:])
for name, res in results.items():
    ax5.plot(res["history"]["val_auc"], lw=2.5, color=PALETTE[name],
             label=f"{name} (test={res['test_auc']:.4f})")
ax5.set_title("Val AUC — LSTM vs Transformer", fontweight="bold")
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Val AUC")
ax5.legend(); ax5.grid(alpha=0.3)

# ── f. Test AUC bar ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
names_s = list(results.keys())
aucs_s  = [r["test_auc"] for r in results.values()]
bars6   = ax6.bar(range(len(names_s)), aucs_s,
                   color=[PALETTE[n] for n in names_s], alpha=0.85)
for bar, v in zip(bars6, aucs_s):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax6.set_xticks(range(len(names_s)))
ax6.set_xticklabels([n[:10] for n in names_s], rotation=15, fontsize=9)
ax6.set_ylim(0.88, 0.98); ax6.set_title("Test AUC Karşılaştırması", fontweight="bold")
ax6.grid(axis="y", alpha=0.3)

# ── g. Parametre karşılaştırması ─────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
params_s = [r["params"]/1e6 for r in results.values()]
times_s  = [r["time"] for r in results.values()]
ax7b = ax7.twinx()
bars7a = ax7.bar(np.arange(len(names_s))-0.2, params_s, 0.38,
                  color=[PALETTE[n] for n in names_s], alpha=0.85, label="Param (M)")
bars7b = ax7b.bar(np.arange(len(names_s))+0.2, times_s, 0.38,
                   color=[PALETTE[n] for n in names_s], alpha=0.45, label="Süre (s)")
ax7.set_xticks(range(len(names_s)))
ax7.set_xticklabels([n[:10] for n in names_s], rotation=15, fontsize=9)
ax7.set_ylabel("Parametre (M)"); ax7b.set_ylabel("Süre (sn)")
ax7.set_title("Parametre & Eğitim Süresi", fontweight="bold")
ax7.grid(axis="y", alpha=0.3)

# ── h. Val Loss ──────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2:])
for name, res in results.items():
    ax8.plot(res["history"]["val_loss"], lw=2.5, color=PALETTE[name], label=name)
    ax8.plot(res["history"]["loss"], lw=1.5, color=PALETTE[name], ls="--", alpha=0.5)
ax8.set_title("Train (---) vs Val Loss (—)", fontweight="bold")
ax8.set_xlabel("Epoch"); ax8.legend(); ax8.grid(alpha=0.3)

plt.savefig("01_transformer_sifirdan.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 01_transformer_sifirdan.png")
plt.close()

print("\n" + "─" * 55)
print(f"  {'Model':<22} {'Test Acc':>10} {'AUC':>8} {'Param':>10}")
print("  " + "─" * 52)
for name, r in results.items():
    print(f"  {name:<22} {r['test_acc']:>10.4f} {r['test_auc']:>8.4f} {r['params']:>10,}")
print("\n" + "=" * 65)
print("  ✅ UYGULAMA 01 TAMAMLANDI — 01_transformer_sifirdan.png")
print("=" * 65)
