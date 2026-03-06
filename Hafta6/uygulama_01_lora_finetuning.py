"""
=============================================================================
UYGULAMA 01 — LoRA Fine-Tuning: Matematik, PEFT ve SFT Simülasyonu
=============================================================================
Kapsam:
  - LoRA düşük-rank matris ayrışımının matematiksel gösterimi
  - r, alpha hiperparametrelerinin parametre sayısına etkisi
  - Keras ile LoRADense katmanı sıfırdan implementasyonu
  - Standard vs LoRA r=4 vs LoRA r=16 karşılaştırması (IMDB)
  - Instruction tuning (Alpaca) formatı açıklaması
  - PEFT API konfigürasyon kılavuzu

Kurulum: pip install tensorflow numpy matplotlib scikit-learn
=============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42); np.random.seed(42)
plt.rcParams.update({"font.family":"sans-serif",
                     "axes.spines.top":False,"axes.spines.right":False})

print("="*65)
print("  UYGULAMA 01 — LoRA Fine-Tuning")
print(f"  TensorFlow: {tf.__version__}")
print("="*65)

# ═══════════════════════════════════════════════════════════
# 1. LoRA MATEMATİĞİ
# ═══════════════════════════════════════════════════════════
print("\n[1] LoRA parametre analizi: W' = W₀ + (α/r)·B·A")

def lora_stats(d, k, r, alpha=None):
    if alpha is None: alpha = r
    full  = d * k
    lora  = d * r + r * k
    scale = alpha / r
    red   = 1 - lora / full
    return {"full":full,"lora":lora,"scale":scale,"reduction":red*100}

D = 4096   # LLaMA-7B d_model
print(f"\n  d=k={D} (LLaMA-7B benzeri)")
print(f"  {'r':>5} {'LoRA Param':>12} {'Full FT':>12} {'Tasarruf':>10} {'α/r Scale':>11}")
print("  "+"─"*55)
rank_data = {}
for r in [4,8,16,32,64,128]:
    st = lora_stats(D,D,r,alpha=r*2)
    rank_data[r] = st
    print(f"  {r:>5} {st['lora']:>12,} {st['full']:>12,} {st['reduction']:>9.2f}% {st['scale']:>11.2f}")

# LLaMA-7B toplam
r16 = lora_stats(D,D,16,32)
total_lora = r16["lora"] * 2 * 32   # Q+V, 32 katman
print(f"\n  LLaMA-7B toplam LoRA (r=16, Q+V, 32 katman): {total_lora:,}")
print(f"  → Tüm modelin %{100*total_lora/7e9:.4f}'i eğitiliyor")

# ═══════════════════════════════════════════════════════════
# 2. KERAS LoRA KATMANI
# ═══════════════════════════════════════════════════════════
print("\n[2] LoRADense katmanı tanımlanıyor...")

class LoRADense(keras.layers.Layer):
    """
    y = x·W₀ᵀ + (α/r)·x·Aᵀ·Bᵀ
    W₀ dondurulmuş, A (Gaussian) ve B (sıfır) eğitilebilir.
    """
    def __init__(self, units, rank=8, alpha=16.0, lora_dropout=0.0, **kw):
        super().__init__(**kw)
        self.units=units; self.rank=rank
        self.alpha=alpha; self.scale=alpha/rank
        self.lora_dropout=lora_dropout

    def build(self, input_shape):
        d = input_shape[-1]
        self.W0 = self.add_weight("W0",(d,self.units),
            initializer="glorot_uniform", trainable=False)  # DONDURULDU
        self.A  = self.add_weight("A",(d,self.rank),
            initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True)
        self.B  = self.add_weight("B",(self.rank,self.units),
            initializer="zeros", trainable=True)
        if self.lora_dropout > 0:
            self._drop = layers.Dropout(self.lora_dropout)
        super().build(input_shape)

    def call(self, x, training=None):
        base = x @ self.W0
        lx   = self._drop(x,training=training) if self.lora_dropout>0 else x
        return base + self.scale * ((lx @ self.A) @ self.B)

    def get_config(self):
        return {**super().get_config(),"units":self.units,"rank":self.rank,
                "alpha":self.alpha,"lora_dropout":self.lora_dropout}

# Birim testi
tl = LoRADense(256,rank=8,alpha=16.0)
_  = tl(tf.zeros((1,128)))
tp  = sum(w.numpy().size for w in tl.trainable_weights)
tot = sum(w.numpy().size for w in tl.weights)
print(f"  LoRADense(128→256, r=8): eğitilebilir={tp:,}/{tot:,} ({100*tp/tot:.1f}%)")

# ═══════════════════════════════════════════════════════════
# 3. IMDB KARŞILAŞTIRMA DENEYİ
# ═══════════════════════════════════════════════════════════
print("\n[3] IMDB üzerinde Standard vs LoRA karşılaştırması...")

VS=10000; ML=200; BS=256; ED=64
(Xtr,ytr),(Xte,yte)=keras.datasets.imdb.load_data(num_words=VS)
pad=lambda s: keras.preprocessing.sequence.pad_sequences(s,maxlen=ML,padding="post")
Xtr,Xte = pad(Xtr[:15000]),pad(Xte)
Xval,yval = pad(keras.datasets.imdb.load_data(num_words=VS)[0][0][15000:20000]),ytr[15000:20000]
ytr = ytr[:15000]

def make_std(name="Std"):
    inp=keras.Input(shape=(ML,))
    x=layers.Embedding(VS,ED,mask_zero=True)(inp)
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(256,activation="gelu")(x)
    x=layers.Dropout(0.3)(x)
    x=layers.Dense(128,activation="gelu")(x)
    out=layers.Dense(1,activation="sigmoid")(x)
    return keras.Model(inp,out,name=name)

def make_lora(r=8,a=16.0,name="LoRA"):
    inp=keras.Input(shape=(ML,))
    x=layers.Embedding(VS,ED,mask_zero=True)(inp)
    x=layers.GlobalAveragePooling1D()(x)
    x=LoRADense(256,rank=r,alpha=a,lora_dropout=0.1)(x)
    x=layers.Activation("gelu")(x)
    x=layers.Dropout(0.3)(x)
    x=LoRADense(128,rank=r,alpha=a)(x)
    x=layers.Activation("gelu")(x)
    out=layers.Dense(1,activation="sigmoid")(x)
    return keras.Model(inp,out,name=name)

def train(m,name,epochs=10):
    m.compile(keras.optimizers.Adam(2e-3),"binary_crossentropy",
              ["accuracy",keras.metrics.AUC(name="auc")])
    cbs=[keras.callbacks.EarlyStopping("val_auc",patience=4,
                                        restore_best_weights=True,mode="max")]
    t0=time.time()
    h=m.fit(Xtr,ytr,validation_data=(Xval,yval),epochs=epochs,
             batch_size=BS,callbacks=cbs,verbose=1)
    el=time.time()-t0
    res=m.evaluate(Xte,yte,verbose=0)
    tp=sum(tf.size(w).numpy() for w in m.trainable_weights)
    tot=m.count_params()
    print(f"  [{name}] acc={res[1]:.4f} auc={res[2]:.4f} "
          f"trainable={tp:,}/{tot:,} ({100*tp/tot:.1f}%) t={el:.0f}s")
    return h.history,res,tot,tp,el

results={}
for nm,fn in [("Standard",lambda:make_std()),
              ("LoRA r=4", lambda:make_lora(4,8.0,"LoRA_r4")),
              ("LoRA r=16",lambda:make_lora(16,32.0,"LoRA_r16"))]:
    print(f"\n  [{nm}] eğitiliyor...")
    tf.random.set_seed(42)
    h,res,tot,tp,el=train(fn(),nm)
    results[nm]={"hist":h,"res":res,"total":tot,"trainable":tp,"time":el}

# ═══════════════════════════════════════════════════════════
# 4. GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════
print("\n[4] Görselleştirmeler hazırlanıyor...")
PAL={"Standard":"#DC2626","LoRA r=4":"#0F766E","LoRA r=16":"#059669"}

fig=plt.figure(figsize=(22,16))
gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.5,wspace=0.38)
fig.suptitle("LoRA Fine-Tuning — Parametre Verimliliği & Performans",fontsize=15,fontweight="bold")

# a. Rank vs param
ax1=fig.add_subplot(gs[0,:2])
ranks=list(rank_data.keys())
lparams=[rank_data[r]["lora"] for r in ranks]
ax1.bar(range(len(ranks)),lparams,color="#0F766E",alpha=0.8,label="LoRA param")
ax1.axhline(rank_data[ranks[0]]["full"],color="#DC2626",ls="--",lw=2.5,
            label=f"Full FT: {rank_data[ranks[0]]['full']:,}")
for i,(r,v) in enumerate(zip(ranks,lparams)):
    ax1.text(i,v+200,f"{v:,}",ha="center",fontsize=8.5,fontweight="bold")
ax1.set_xticks(range(len(ranks)))
ax1.set_xticklabels([f"r={r}" for r in ranks])
ax1.set_title("Rank r vs Parametre Sayısı (d=k=4096)",fontweight="bold")
ax1.set_ylabel("Parametre"); ax1.legend(); ax1.grid(axis="y",alpha=0.3)

# b. Tasarruf
ax2=fig.add_subplot(gs[0,2])
reds=[rank_data[r]["reduction"] for r in ranks]
ax2.bar(range(len(ranks)),reds,color="#059669",alpha=0.85)
for i,v in enumerate(reds):
    ax2.text(i,v-2.5,f"{v:.1f}%",ha="center",fontsize=9,color="white",fontweight="bold")
ax2.set_xticks(range(len(ranks)))
ax2.set_xticklabels([f"r={r}" for r in ranks])
ax2.set_ylim(70,101); ax2.set_title("Parametre Tasarruf %",fontweight="bold")
ax2.grid(axis="y",alpha=0.3)

# c. Val AUC
ax3=fig.add_subplot(gs[0,3])
for nm,res in results.items():
    ax3.plot(res["hist"]["val_auc"],lw=2.5,color=PAL[nm],
             label=f"{nm} ({res['res'][2]:.4f})")
ax3.set_title("Val AUC Karşılaştırması",fontweight="bold")
ax3.set_xlabel("Epoch"); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# d. Trainable param bar
ax4=fig.add_subplot(gs[1,0])
names_r=list(results.keys())
ax4.bar(names_r,[results[n]["total"] for n in names_r],
        color=[PAL[n] for n in names_r],alpha=0.35,label="Toplam")
ax4.bar(names_r,[results[n]["trainable"] for n in names_r],
        color=[PAL[n] for n in names_r],alpha=0.9,label="Eğitilebilir")
ax4.set_title("Eğitilebilir vs Toplam",fontweight="bold")
ax4.set_ylabel("Parametre"); ax4.legend(fontsize=9); ax4.grid(axis="y",alpha=0.3)
ax4.tick_params(axis="x",labelsize=9)

# e. Test AUC
ax5=fig.add_subplot(gs[1,1])
aucs=[results[n]["res"][2] for n in names_r]
bars5=ax5.bar(names_r,aucs,color=[PAL[n] for n in names_r],alpha=0.85)
for bar,v in zip(bars5,aucs):
    ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
             f"{v:.4f}",ha="center",fontsize=11,fontweight="bold")
ax5.set_ylim(0.88,0.99); ax5.set_title("Test AUC",fontweight="bold")
ax5.grid(axis="y",alpha=0.3); ax5.tick_params(axis="x",labelsize=9)

# f. Süre
ax6=fig.add_subplot(gs[1,2])
times=[results[n]["time"] for n in names_r]
bars6=ax6.bar(names_r,times,color=[PAL[n] for n in names_r],alpha=0.85)
for bar,v in zip(bars6,times):
    ax6.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
             f"{v:.0f}s",ha="center",fontsize=11,fontweight="bold")
ax6.set_title("Eğitim Süresi",fontweight="bold"); ax6.grid(axis="y",alpha=0.3)
ax6.tick_params(axis="x",labelsize=9)

# g. Loss eğrisi
ax7=fig.add_subplot(gs[1,3])
for nm,res in results.items():
    ax7.plot(res["hist"]["val_loss"],lw=2.5,color=PAL[nm],label=nm)
    ax7.plot(res["hist"]["loss"],lw=1.5,color=PAL[nm],ls="--",alpha=0.5)
ax7.set_title("Train (---) vs Val Loss",fontweight="bold")
ax7.set_xlabel("Epoch"); ax7.legend(fontsize=9); ax7.grid(alpha=0.3)

# h. ΔW = A·B ısı haritası
ax8=fig.add_subplot(gs[2,:2])
A_d=np.random.randn(64,8)*0.02
B_d=np.zeros((8,64))
for _ in range(200):
    A_d-=np.random.randn(*A_d.shape)*0.001
    B_d-=np.random.randn(*B_d.shape)*0.001
dW=A_d@B_d
im=ax8.imshow(dW,cmap="RdYlGn",aspect="auto")
plt.colorbar(im,ax=ax8)
ax8.set_title("ΔW = A·B Isı Haritası (64×64, r=8)",fontweight="bold")
ax8.set_xlabel("k boyutu"); ax8.set_ylabel("d boyutu")

# i. PEFT kılavuzu
ax9=fig.add_subplot(gs[2,2:])
ax9.axis("off")
code="""PEFT LoRA Konfigürasyon Kılavuzu:

from peft import LoraConfig, get_peft_model, PeftModel

config = LoraConfig(
    r              = 16,        # rank: 4,8,16,32,64
    lora_alpha     = 32,        # scale = alpha/r = 2.0
    target_modules = [
        'q_proj','v_proj',      # dikkat: query+value
        'k_proj','o_proj',      # opsiyonel: key+output
        'gate_proj','up_proj',  # FFN (LLaMA/Mistral)
        'down_proj',
    ],
    lora_dropout   = 0.05,
    bias           = 'none',    # 'none'|'all'|'lora_only'
    task_type      = 'CAUSAL_LM',
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Kaydet / Yükle:
model.save_pretrained('./lora_adapter')
model = PeftModel.from_pretrained(base, './lora_adapter')

# Birleştir (inference):
merged = model.merge_and_unload()  # W₀ + B·A"""
ax9.text(0.02,0.98,code,transform=ax9.transAxes,fontsize=9.5,va="top",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4",facecolor="#F0FDF4",
                   edgecolor="#0F766E",linewidth=2))
ax9.set_title("PEFT Konfigürasyon Kılavuzu",fontweight="bold")

plt.savefig("01_lora_finetuning.png",dpi=150,bbox_inches="tight")
print("  ✅ 01_lora_finetuning.png kaydedildi")
plt.close()

print("\n"+"─"*55)
for nm,r in results.items():
    pct=100*r["trainable"]/r["total"]
    print(f"  {nm:<15} AUC={r['res'][2]:.4f}  "
          f"trainable={r['trainable']:,}/{r['total']:,} ({pct:.1f}%)")
print("\n"+"="*65)
print("  ✅ UYGULAMA 01 TAMAMLANDI — 01_lora_finetuning.png")
print("="*65)
