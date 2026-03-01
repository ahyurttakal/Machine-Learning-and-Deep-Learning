"""
=============================================================================
UYGULAMA 02 — BiLSTM, Yığılmış LSTM & Dikkat Mekanizması
=============================================================================
Kapsam:
  - BiLSTM vs tek yönlü LSTM karşılaştırması
  - 2 / 3 / 4 katmanlı Stacked LSTM ablasyonu
  - Custom Bahdanau Attention katmanı (tf.keras.layers.Layer)
  - Keras MultiHeadAttention ile metin sınıflandırma
  - Dikkat ağırlıklarını görselleştirme (hangi kelimeye baktı?)
  - Tüm modellerin karşılaştırmalı analizi

Veri: IMDB Duygu Analizi
Kurulum: pip install tensorflow numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
print("  UYGULAMA 02 — BiLSTM, Yığılmış LSTM & Dikkat Mekanizması")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[1] IMDB yükleniyor...")

VOCAB_SIZE = 20000
MAX_LEN    = 200
EMBED_DIM  = 64
BATCH_SIZE = 128
EPOCHS     = 10

(X_tr_raw, y_train_all), (X_te_raw, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

pad = lambda seqs, ml: keras.preprocessing.sequence.pad_sequences(
    seqs, maxlen=ml, padding="post", truncating="post"
)
X_all   = pad(X_tr_raw, MAX_LEN)
X_test  = pad(X_te_raw, MAX_LEN)

val_sz    = 5000
X_val,  X_train = X_all[:val_sz],        X_all[val_sz:]
y_val,  y_train = y_train_all[:val_sz],  y_train_all[val_sz:]
print(f"    Eğitim: {len(X_train):,} | Val: {len(X_val):,} | maxlen: {MAX_LEN}")

# Hızlı eğitim yardımcısı
def fit_model(model, epochs=EPOCHS, verbose=0):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                       restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                           patience=3, min_lr=1e-7),
    ]
    t0 = time.time()
    h  = model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=epochs, batch_size=BATCH_SIZE,
                   callbacks=cbs, verbose=verbose)
    elapsed  = time.time() - t0
    test_res = model.evaluate(X_test, y_test, verbose=0)
    return h.history, test_res, elapsed

# ═════════════════════════════════════════════════════════════
# 2. ÖZEL BAHDANAU DİKKAT KATMANI
# ═════════════════════════════════════════════════════════════
print("\n[2] Custom Bahdanau Attention katmanı tanımlanıyor...")

class BahdanauAttention(keras.layers.Layer):
    """
    Additive (Bahdanau) Dikkat Mekanizması.
    Tüm LSTM adımlarının çıktılarını ağırlıklı olarak birleştirir.
    Hangi zaman adımına ne kadar dikkat edildiğini öğrenir.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1    = layers.Dense(units, use_bias=False)
        self.W2    = layers.Dense(units, use_bias=False)
        self.V     = layers.Dense(1,     use_bias=False)
        self.units = units

    def call(self, encoder_output, training=None):
        """
        encoder_output: (batch, timesteps, features) — return_sequences=True
        Çıkış: context vektörü (batch, features), dikkat ağırlıkları (batch, timesteps, 1)
        """
        # Son adımı sorgu olarak kullan (basit versiyon)
        last_hidden = encoder_output[:, -1:, :]   # (B, 1, features)

        # Skor hesapla: W1(encoder_out) + W2(last_hidden)
        score = self.V(
            tf.nn.tanh(
                self.W1(encoder_output) +      # (B, T, units)
                self.W2(last_hidden)           # (B, 1, units) → broadcast
            )
        )  # (B, T, 1)

        # Softmax ile normalize
        attention_weights = tf.nn.softmax(score, axis=1)   # (B, T, 1)

        # Ağırlıklı topla → context vektörü
        context = attention_weights * encoder_output        # (B, T, features)
        context = tf.reduce_sum(context, axis=1)            # (B, features)

        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

print("    ✅ BahdanauAttention hazır")

# ═════════════════════════════════════════════════════════════
# 3. MODEL TANIMLAMALARI
# ═════════════════════════════════════════════════════════════
print("\n[3] Modeller tanımlanıyor...")

def build_unidirectional_lstm(units=128, name="uni_lstm"):
    """Tek yönlü LSTM — baseline."""
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inp)
    x   = layers.LSTM(units, dropout=0.2, recurrent_dropout=0.1)(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)

def build_bilstm(units=64, merge_mode="concat", name="bilstm"):
    """Bidirectional LSTM."""
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inp)
    x   = layers.Bidirectional(
        layers.LSTM(units, dropout=0.2, recurrent_dropout=0.1),
        merge_mode=merge_mode,
    )(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)

def build_stacked_lstm(n_layers=3, units=64, name="stacked"):
    """Yığılmış LSTM — n katman."""
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inp)
    for i in range(n_layers):
        rs  = (i < n_layers - 1)   # son katman hariç return_sequences=True
        drp = max(0.1, 0.3 - i * 0.05)
        x   = layers.LSTM(units, return_sequences=rs,
                          dropout=drp, recurrent_dropout=0.1,
                          name=f"lstm_{i+1}")(x)
        if rs:
            x = layers.Dropout(0.25)(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)

def build_lstm_with_attention(units=64, att_units=32, name="lstm_att"):
    """LSTM + Custom Bahdanau Attention."""
    inp     = keras.Input(shape=(MAX_LEN,))
    x       = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inp)
    seq_out = layers.LSTM(units, return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.1)(x)
    context, att_weights = BahdanauAttention(att_units, name="bahdanau_att")(seq_out)
    x   = layers.Dropout(0.3)(context)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    # Dikkat ağırlıklarını da çıkar
    full_model  = keras.Model(inp, out, name=name)
    att_model   = keras.Model(inp, att_weights, name=name+"_att")
    return full_model, att_model

def build_bilstm_with_attention(units=64, name="bilstm_att"):
    """BiLSTM + Bahdanau Attention."""
    inp     = keras.Input(shape=(MAX_LEN,))
    x       = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inp)
    seq_out = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        merge_mode="concat",
    )(x)
    context, att_weights = BahdanauAttention(units, name="att")(seq_out)
    x   = layers.Dropout(0.3)(context)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    full_model = keras.Model(inp, out, name=name)
    att_model  = keras.Model(inp, att_weights, name=name+"_att")
    return full_model, att_model

def build_multihead_attention(units=64, n_heads=4, name="mha"):
    """MultiHeadAttention + LSTM."""
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    # LSTM sequence çıktısı üzerine self-attention
    seq = layers.LSTM(units, return_sequences=True, dropout=0.2)(x)
    att_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=units // n_heads,
        dropout=0.1, name="mha_layer"
    )(seq, seq)                              # self-attention
    att_out = layers.LayerNormalization()(att_out + seq)   # residual
    x   = layers.GlobalAveragePooling1D()(att_out)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)

# ═════════════════════════════════════════════════════════════
# 4. EĞİTİM
# ═════════════════════════════════════════════════════════════
print("\n[4] Modeller eğitiliyor...")

all_results = {}

# ─── Tek yönlü LSTM (baseline) ───────────────────────────────
print("    [a] Tek yönlü LSTM (baseline)...")
tf.random.set_seed(42)
m = build_unidirectional_lstm(name="uni_lstm")
hist, tr, elapsed = fit_model(m)
all_results["Uni-LSTM"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                            "params":m.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── BiLSTM (concat) ─────────────────────────────────────────
print("    [b] BiLSTM (concat)...")
tf.random.set_seed(42)
m = build_bilstm(units=64, merge_mode="concat", name="bilstm_concat")
hist, tr, elapsed = fit_model(m)
all_results["BiLSTM-concat"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                                 "params":m.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── BiLSTM (sum) ────────────────────────────────────────────
print("    [c] BiLSTM (sum)...")
tf.random.set_seed(42)
m = build_bilstm(units=64, merge_mode="sum", name="bilstm_sum")
hist, tr, elapsed = fit_model(m)
all_results["BiLSTM-sum"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                              "params":m.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── Stacked LSTM 2 katman ────────────────────────────────────
print("    [d] Stacked LSTM 2 katman...")
tf.random.set_seed(42)
m = build_stacked_lstm(n_layers=2, units=64, name="stacked_2")
hist, tr, elapsed = fit_model(m)
all_results["Stacked-2"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                             "params":m.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── Stacked LSTM 3 katman ────────────────────────────────────
print("    [e] Stacked LSTM 3 katman...")
tf.random.set_seed(42)
m = build_stacked_lstm(n_layers=3, units=64, name="stacked_3")
hist, tr, elapsed = fit_model(m)
all_results["Stacked-3"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                             "params":m.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── LSTM + Attention ────────────────────────────────────────
print("    [f] LSTM + Bahdanau Attention...")
tf.random.set_seed(42)
m_att, m_att_weights = build_lstm_with_attention(name="lstm_att")
hist, tr, elapsed = fit_model(m_att)
all_results["LSTM+Attention"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                                  "params":m_att.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── BiLSTM + Attention ──────────────────────────────────────
print("    [g] BiLSTM + Bahdanau Attention...")
tf.random.set_seed(42)
m_biatt, m_biatt_weights = build_bilstm_with_attention(name="bilstm_att")
hist, tr, elapsed = fit_model(m_biatt)
all_results["BiLSTM+Att"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                              "params":m_biatt.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ─── MultiHeadAttention ──────────────────────────────────────
print("    [h] MultiHeadAttention...")
tf.random.set_seed(42)
m = build_multihead_attention(units=64, n_heads=4, name="mha")
hist, tr, elapsed = fit_model(m)
all_results["MultiHead-Att"] = {"history":hist, "test_acc":tr[1], "test_auc":tr[2],
                                 "params":m.count_params(), "time":elapsed}
print(f"        test_acc={tr[1]:.4f}  test_auc={tr[2]:.4f}  t={elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════
# 5. DİKKAT AĞIRLIKLARINI GÖRSELLEŞTİR
# ═════════════════════════════════════════════════════════════
print("\n[5] Dikkat ağırlıkları görselleştiriliyor...")

# IMDB kelime indeks haritası
word_index  = keras.datasets.imdb.get_word_index()
reverse_map = {v+3: k for k, v in word_index.items()}
reverse_map.update({0:"<PAD>", 1:"<START>", 2:"<UNK>", 3:"<UNUSED>"})

def decode_review(encoded_seq):
    return [reverse_map.get(idx, "?") for idx in encoded_seq if idx != 0]

# Örnek incele: bir pozitif, bir negatif yorum
pos_idx = np.where(y_test == 1)[0][0]
neg_idx = np.where(y_test == 0)[0][0]

def get_attention_weights(att_model, seq, maxlen=MAX_LEN):
    seq_padded = keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=maxlen, padding="post", truncating="post"
    )
    w = att_model.predict(seq_padded, verbose=0)
    return w[0, :, 0]   # (timesteps,)

# ─── Pozitif örnek ───────────────────────────────────────────
pos_raw    = X_te_raw[pos_idx]
pos_words  = decode_review(pos_raw[:MAX_LEN])
pos_weights= get_attention_weights(m_att_weights,
                                   keras.preprocessing.sequence.pad_sequences(
                                       [pos_raw], maxlen=MAX_LEN)[0])

# ─── Negatif örnek ───────────────────────────────────────────
neg_raw    = X_te_raw[neg_idx]
neg_words  = decode_review(neg_raw[:MAX_LEN])
neg_weights= get_attention_weights(m_att_weights,
                                   keras.preprocessing.sequence.pad_sequences(
                                       [neg_raw], maxlen=MAX_LEN)[0])

print(f"    Pozitif örnek dikkat max kelimeler:")
top_pos = sorted(enumerate(zip(pos_words[:50], pos_weights[:50])),
                 key=lambda x: x[1][1], reverse=True)[:8]
for rank, (i, (word, w)) in enumerate(top_pos):
    print(f"      [{rank+1}] '{word}' @ pozisyon {i}: ağırlık={w:.4f}")

# ═════════════════════════════════════════════════════════════
# 6. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[6] Görselleştirmeler hazırlanıyor...")

PALETTE = {
    "Uni-LSTM":      "#6B7280",
    "BiLSTM-concat": "#2563EB",
    "BiLSTM-sum":    "#3B82F6",
    "Stacked-2":     "#7C3AED",
    "Stacked-3":     "#9333EA",
    "LSTM+Attention":"#059669",
    "BiLSTM+Att":    "#0F766E",
    "MultiHead-Att": "#DC2626",
}

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("BiLSTM, Yığılmış LSTM & Dikkat Mekanizması — IMDB", fontsize=15, fontweight="bold")

# ── 6a. Val Accuracy tüm modeller ────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for name, res in all_results.items():
    ax1.plot(res["history"]["val_accuracy"], lw=1.8,
             color=PALETTE[name], label=f"{name} ({res['test_acc']:.4f})")
ax1.set_title("Val Accuracy — Tüm Mimariler", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Accuracy")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# ── 6b. Test AUC barları ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
names_sorted = sorted(all_results.keys(), key=lambda k: all_results[k]["test_auc"], reverse=True)
aucs_sorted  = [all_results[n]["test_auc"] for n in names_sorted]
bars2 = ax2.barh(names_sorted, aucs_sorted,
                  color=[PALETTE[n] for n in names_sorted], alpha=0.85)
for bar, v in zip(bars2, aucs_sorted):
    ax2.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax2.set_xlim(0.84, 0.97); ax2.set_title("Test AUC Sıralaması", fontweight="bold")
ax2.grid(axis="x", alpha=0.3)

# ── 6c. BiLSTM merge_mode karşılaştırması ────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for name in ["Uni-LSTM", "BiLSTM-concat", "BiLSTM-sum"]:
    ax3.plot(all_results[name]["history"]["val_auc"],
             lw=2, color=PALETTE[name], label=name)
ax3.set_title("Tek Yönlü vs BiLSTM", fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Val AUC")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# ── 6d. Stacked LSTM karşılaştırması ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for name in ["Uni-LSTM", "Stacked-2", "Stacked-3"]:
    ax4.plot(all_results[name]["history"]["val_auc"],
             lw=2, color=PALETTE[name], label=name)
ax4.set_title("Yığılmış LSTM Katman Sayısı", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Val AUC")
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# ── 6e. Attention ısı haritası — Pozitif yorum ───────────────
ax5 = fig.add_subplot(gs[1, 2:])
n_show_words = 30
words_show   = pos_words[:n_show_words]
weights_show = pos_weights[:n_show_words]
# Barplot yatay
y_pos = np.arange(n_show_words)
colors_att = plt.cm.RdYlGn(weights_show / (weights_show.max() + 1e-8))
ax5.barh(y_pos, weights_show, color=colors_att, edgecolor="none")
ax5.set_yticks(y_pos)
ax5.set_yticklabels([f"{w} [{i}]" for i, w in enumerate(words_show)], fontsize=7)
ax5.invert_yaxis()
ax5.set_title("Dikkat Ağırlıkları — Pozitif Yorum (LSTM+Att)", fontweight="bold")
ax5.set_xlabel("Dikkat Ağırlığı")
ax5.grid(axis="x", alpha=0.3)

# ── 6f. Parametre & Süre karşılaştırması ─────────────────────
ax6 = fig.add_subplot(gs[2, :2])
names_all  = list(all_results.keys())
params_all = [r["params"]/1e6 for r in all_results.values()]
times_all  = [r["time"] for r in all_results.values()]
x_pos_all  = np.arange(len(names_all))
width      = 0.38
ax6b = ax6.twinx()
bars6a = ax6.bar(x_pos_all - width/2, params_all, width,
                  color=[PALETTE[n] for n in names_all], alpha=0.75, label="Parametre (M)")
bars6b = ax6b.bar(x_pos_all + width/2, times_all, width,
                   color=[PALETTE[n] for n in names_all], alpha=0.45, label="Süre (sn)")
ax6.set_xticks(x_pos_all)
ax6.set_xticklabels([n[:12] for n in names_all], rotation=25, ha="right", fontsize=9)
ax6.set_ylabel("Parametre (M)"); ax6b.set_ylabel("Süre (sn)")
ax6.set_title("Parametre Sayısı & Eğitim Süresi", fontweight="bold")
ax6.grid(axis="y", alpha=0.3)

# ── 6g. Negatif yorum dikkat ─────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2:])
n_show_neg   = 30
words_neg    = neg_words[:n_show_neg]
weights_neg  = neg_weights[:n_show_neg]
y_neg        = np.arange(n_show_neg)
colors_neg   = plt.cm.RdYlGn(weights_neg / (weights_neg.max() + 1e-8))
ax7.barh(y_neg, weights_neg, color=colors_neg, edgecolor="none")
ax7.set_yticks(y_neg)
ax7.set_yticklabels([f"{w} [{i}]" for i, w in enumerate(words_neg)], fontsize=7)
ax7.invert_yaxis()
ax7.set_title("Dikkat Ağırlıkları — Negatif Yorum (LSTM+Att)", fontweight="bold")
ax7.set_xlabel("Dikkat Ağırlığı")
ax7.grid(axis="x", alpha=0.3)

plt.savefig("02_bilstm_attention_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 02_bilstm_attention_analiz.png")
plt.close()

# ─── Özet ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print(f"  {'Model':<20} {'Test Acc':>10} {'Test AUC':>10} {'Parametre':>12}")
print("  " + "─" * 55)
for name, r in sorted(all_results.items(), key=lambda x: x[1]["test_auc"], reverse=True):
    print(f"  {name:<20} {r['test_acc']:>10.4f} {r['test_auc']:>10.4f} {r['params']:>12,}")

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print("  Çıktı: 02_bilstm_attention_analiz.png")
print("=" * 65)
