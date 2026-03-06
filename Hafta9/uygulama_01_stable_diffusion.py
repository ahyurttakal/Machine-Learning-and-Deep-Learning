"""
=============================================================================
HAFTA 5 CUMARTESİ — UYGULAMA 01
Stable Diffusion Pipeline — Text-to-Image & img2img
=============================================================================
Kapsam:
  - StableDiffusionPipeline: prompt → 512×512 görüntü
  - Prompt mühendisliği: pozitif/negatif prompt, stil örnekleri
  - guidance_scale ablasyonu: 1 → 15 arasında kalite farkı
  - num_inference_steps ablasyonu: 10 / 25 / 50 adım karşılaştırması
  - Scheduler karşılaştırması: DDIM vs DPMSolver++ vs Euler
  - img2img: başlangıç görüntüsünden yeni görüntü üretimi
  - Latent uzay interpolasyonu: iki prompt arası geçiş
  - GPU yoksa CPU modu + simülasyon destekli
  - Kapsamlı görselleştirme (8 panel)

Kurulum:
  pip install diffusers transformers accelerate pillow torch
  (GPU için): pip install xformers
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE  = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE  = False

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        DPMSolverMultistepScheduler,
        DDIMScheduler,
        EulerAncestralDiscreteScheduler,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

SIM_MODE = not (TORCH_AVAILABLE and DIFFUSERS_AVAILABLE)

print("=" * 65)
print("  HAFTA 5 CUMARTESİ — UYGULAMA 01")
print("  Stable Diffusion Pipeline")
print("=" * 65)
print(f"  Mod          : {'🔵 Gerçek' if not SIM_MODE else '🟡 Simülasyon'}")
print(f"  PyTorch      : {'✅' if TORCH_AVAILABLE else '❌  pip install torch'}")
print(f"  Diffusers    : {'✅' if DIFFUSERS_AVAILABLE else '❌  pip install diffusers'}")
print(f"  CUDA/GPU     : {'✅  ' + torch.cuda.get_device_name(0) if CUDA_AVAILABLE else '❌  CPU modunda çalışır'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: MODEL YÜKLEME
# ─────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Model Yükleme")
print("─" * 65)

MODEL_ID   = "stabilityai/stable-diffusion-2-1-base"
GORUNTU_W  = 512
GORUNTU_H  = 512
CIHAZ      = "cuda" if CUDA_AVAILABLE else "cpu"
DTYPE      = torch.float16 if CUDA_AVAILABLE else torch.float32 if TORCH_AVAILABLE else None

if not SIM_MODE:
    print(f"  Model      : {MODEL_ID}")
    print(f"  Cihaz      : {CIHAZ}  dtype={DTYPE}")
    print(f"  İndiriliyor (ilk çalıştırmada ~5GB)...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(CIHAZ)

    # Bellek optimizasyonu
    pipe.enable_attention_slicing()
    if CUDA_AVAILABLE:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  ✅ xformers bellek optimizasyonu aktif")
        except Exception:
            pass

    # Varsayılan scheduler: DPM-Solver++ (25 adımda kaliteli)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    print(f"  ✅ Model yüklendi  ({CIHAZ.upper()})")
    print(f"  U-Net parametreleri: ~860M")
else:
    print(f"  [SIM] Model: {MODEL_ID}")
    print(f"  [SIM] Gerçek diffusers yüklü olmadığından simülasyon modunda çalışıyor.")
    print(f"  [SIM] Görüntüler sentetik olarak üretilecek.")

def goruntu_uret_sim(prompt, seed=42, width=512, height=512,
                     steps=25, guidance=7.5):
    """Gerçekçi olmayan ama görselleştirme için yeterli sahte görüntü üretir."""
    np.random.seed(seed + int(guidance * 10) + steps)
    # Prompt'taki kelimeleri renk/doku ipucuna dönüştür
    tone   = 0.4 + 0.4 * np.sin(seed * 0.7)
    warm   = "sunset" in prompt or "fire" in prompt or "warm" in prompt
    cool   = "ocean" in prompt or "space" in prompt or "blue" in prompt
    dark   = "night" in prompt or "dark" in prompt or "shadow" in prompt

    img = np.random.rand(height, width, 3).astype(np.float32)
    # Kalite simülasyonu: daha fazla adım → daha az gürültü
    gurultu_guc = max(0.05, 0.5 - steps * 0.008)
    img = img * gurultu_guc

    # Gradient arka plan (prompt tonuna göre)
    y_grad = np.linspace(0, 1, height).reshape(-1, 1, 1)
    x_grad = np.linspace(0, 1, width).reshape(1, -1, 1)
    if warm:
        bg = np.ones((height, width, 3))
        bg[:,:,0] = 0.5 + 0.4 * y_grad.squeeze()
        bg[:,:,1] = 0.2 + 0.3 * x_grad.squeeze()
        bg[:,:,2] = 0.1
    elif cool:
        bg = np.ones((height, width, 3))
        bg[:,:,0] = 0.05 + 0.1  * y_grad.squeeze()
        bg[:,:,1] = 0.15 + 0.25 * x_grad.squeeze()
        bg[:,:,2] = 0.4 + 0.4 * (1 - y_grad.squeeze())
    elif dark:
        bg = np.ones((height, width, 3)) * 0.08
        bg[:,:,0] += 0.1 * x_grad.squeeze()
    else:
        bg = np.ones((height, width, 3))
        bg[:,:,0] = tone * y_grad.squeeze() + 0.2
        bg[:,:,1] = (1 - tone) * x_grad.squeeze() + 0.1
        bg[:,:,2] = 0.3 + 0.2 * (y_grad.squeeze() * x_grad.squeeze())

    # Guidance scale etkisi: yüksek guidance → daha doygun
    doygunluk = min(guidance / 10.0, 1.5)
    img = np.clip(bg * doygunluk + img, 0, 1)
    return (img * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: TEMEL TEXT-TO-IMAGE
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: Text-to-Image — Temel Kullanım")
print("─" * 65)

PROMTLAR = [
    {
        "baslik": "Fütüristik Şehir",
        "prompt": ("a futuristic cyberpunk city at night, neon lights, "
                   "ultra-detailed, 4K, cinematic, rainy streets, reflections"),
        "negatif": "blurry, low quality, watermark, ugly, distorted",
        "seed": 42,
    },
    {
        "baslik": "Sakin Orman",
        "prompt": ("a serene enchanted forest at golden hour, "
                   "mystical light rays, detailed foliage, photorealistic, 8K"),
        "negatif": "blurry, dark, overexposed, people, buildings",
        "seed": 77,
    },
    {
        "baslik": "Uzay Sahnesi",
        "prompt": ("a breathtaking nebula in deep space, vibrant colors, "
                   "stars, galactic dust, Hubble telescope style, extremely detailed"),
        "negatif": "blurry, artifacts, low resolution, cartoon",
        "seed": 123,
    },
    {
        "baslik": "Yağlıboya Portre",
        "prompt": ("portrait of a wise old philosopher, dramatic lighting, "
                   "oil painting style, renaissance, detailed brushstrokes, museum quality"),
        "negatif": "photo, realistic, modern, nsfw, blurry",
        "seed": 256,
    },
]

print(f"  {'Başlık':<22} {'Steps':>6} {'CFG':>6} {'Seed':>6}")
print("  " + "-" * 45)

uretilen_goruntular = []
for p in PROMTLAR:
    if not SIM_MODE:
        generator = torch.Generator(CIHAZ).manual_seed(p["seed"])
        with torch.autocast(CIHAZ) if CUDA_AVAILABLE else torch.no_grad():
            sonuc = pipe(
                prompt=p["prompt"],
                negative_prompt=p["negatif"],
                num_inference_steps=25,
                guidance_scale=7.5,
                height=GORUNTU_H,
                width=GORUNTU_W,
                generator=generator,
            )
        img_arr = np.array(sonuc.images[0])
    else:
        img_arr = goruntu_uret_sim(p["prompt"], p["seed"])
    uretilen_goruntular.append(img_arr)
    print(f"  {p['baslik']:<22} {25:>6} {7.5:>6} {p['seed']:>6}  ✅")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: GUIDANCE SCALE ABLASYONU
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: guidance_scale Ablasyonu (CFG Ağırlığı)")
print("─" * 65)
print("""
  CFG (Classifier-Free Guidance):
    ε_final = ε_uncon + w × (ε_cond − ε_uncon)

    w=1  : Tamamen koşulsuz (prompt etkisiz)
    w=7.5: Dengeli (önerilen başlangıç değeri)
    w=15 : Çok katı → artifaktlar, aşırı doygunluk
""")

CFG_DEGERLERI = [1.0, 3.5, 7.5, 12.0, 15.0]
cfg_goruntuleri = []
SABIT_PROMPT = (
    "a majestic mountain landscape at sunset, warm colors, "
    "snow peaks, golden light, photorealistic"
)

print(f"  {'CFG (w)':>8} {'Yorum'}")
print("  " + "-" * 45)
for cfg in CFG_DEGERLERI:
    if not SIM_MODE:
        generator = torch.Generator(CIHAZ).manual_seed(42)
        with torch.autocast(CIHAZ) if CUDA_AVAILABLE else torch.no_grad():
            sonuc = pipe(
                prompt=SABIT_PROMPT,
                num_inference_steps=25,
                guidance_scale=cfg,
                height=GORUNTU_H, width=GORUNTU_W,
                generator=generator,
            )
        img_arr = np.array(sonuc.images[0])
    else:
        img_arr = goruntu_uret_sim(SABIT_PROMPT + " warm sunset", 42,
                                   guidance=cfg)
    cfg_goruntuleri.append(img_arr)
    yorumlar = {
        1.0:  "Prompt etkisiz, dağınık",
        3.5:  "Zayıf prompt uyumu",
        7.5:  "İdeal denge ✅",
        12.0: "Güçlü, biraz sert",
        15.0: "Aşırı → artifakt ⚠️",
    }
    print(f"  {cfg:>8.1f}  {yorumlar[cfg]}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: ADIM SAYISI ABLASYONU
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Inference Steps Ablasyonu")
print("─" * 65)

ADIM_SAYILARI = [5, 10, 20, 50]
adim_goruntuleri = []

print(f"  {'Adım':>6} {'Süre (tahmini)':>18} {'Yorum'}")
print("  " + "-" * 55)
for adim in ADIM_SAYILARI:
    if not SIM_MODE:
        import time
        generator = torch.Generator(CIHAZ).manual_seed(42)
        t0 = time.time()
        with torch.autocast(CIHAZ) if CUDA_AVAILABLE else torch.no_grad():
            sonuc = pipe(
                prompt=SABIT_PROMPT,
                num_inference_steps=adim,
                guidance_scale=7.5,
                height=GORUNTU_H, width=GORUNTU_W,
                generator=generator,
            )
        sure = time.time() - t0
        img_arr = np.array(sonuc.images[0])
        sure_str = f"{sure:.1f}s"
    else:
        img_arr = goruntu_uret_sim(SABIT_PROMPT, 42, steps=adim)
        cpu_sure = {"CUDA": adim * 0.5, "CPU": adim * 8}.get(CIHAZ.upper(), adim * 8)
        sure_str = f"~{int(cpu_sure)}s (tahmini)"
    adim_goruntuleri.append(img_arr)
    yorumlar = {
        5:  "Çok kaba, sadece genel şekil",
        10: "Kabul edilebilir başlangıç",
        20: "İyi kalite ✅",
        50: "Yüksek kalite, yavaş",
    }
    print(f"  {adim:>6} {sure_str:>18}  {yorumlar[adim]}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 5: SCHEDULER KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: Scheduler (Gürültü Programı) Karşılaştırması")
print("─" * 65)
print("""
  Scheduler     Adım    Deterministik   Yorum
  ──────────────────────────────────────────────────────────
  DDIM           50     Evet ✅         Stabil, hızlı
  DPM-Solver++   25     Kısmen ✅       En hızlı+kaliteli ✅
  Euler-A        30     Hayır (stoch.)  Yaratıcı, çeşitli
""")

SCHEDULERLAR = {
    "DDIM (50 adım)":      {"adim": 50, "turu": "ddim"},
    "DPM-Solver++ (25)":   {"adim": 25, "turu": "dpm"},
    "Euler Ancestral (30)":{"adim": 30, "turu": "euler"},
}

sch_goruntuleri = []
for sch_adi, sch_bilgi in SCHEDULERLAR.items():
    if not SIM_MODE:
        if sch_bilgi["turu"] == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif sch_bilgi["turu"] == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        generator = torch.Generator(CIHAZ).manual_seed(42)
        with torch.autocast(CIHAZ) if CUDA_AVAILABLE else torch.no_grad():
            sonuc = pipe(
                prompt=SABIT_PROMPT,
                num_inference_steps=sch_bilgi["adim"],
                guidance_scale=7.5,
                height=GORUNTU_H, width=GORUNTU_W,
                generator=generator,
            )
        img_arr = np.array(sonuc.images[0])
    else:
        seed_mod = {"ddim": 0, "dpm": 5, "euler": 10}[sch_bilgi["turu"]]
        img_arr = goruntu_uret_sim(SABIT_PROMPT + " warm", 42 + seed_mod,
                                   steps=sch_bilgi["adim"])
    sch_goruntuleri.append(img_arr)
    print(f"  {sch_adi:<28}  ✅ üretildi")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 6: IMG2IMG
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: img2img — Başlangıç Görüntüsünden Dönüşüm")
print("─" * 65)
print("""
  strength parametresi:
    0.0 → Tamamen orijinal görüntü (hiç değişmez)
    0.5 → Yarı-yarıya (önerilen)
    1.0 → Tamamen yeni (text2img ile eşdeğer)

  Kullanım Alanı: eskiz → gerçekçi, gece → gündüz,
                  yaz → kış, siyah-beyaz → renkli
""")

# Başlangıç görüntüsü: programatik olarak oluştur
np.random.seed(99)
def eskiz_goruntu_olustur(width=512, height=512):
    """Basit bir eskiz benzeri görüntü oluşturur."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 230
    # Dağ silueti
    for x in range(width):
        t  = x / width
        y1 = int(height * (0.6 - 0.25 * np.sin(t * np.pi)))
        y2 = int(height * (0.7 - 0.15 * np.sin(t * 2 * np.pi + 1)))
        img[y1:y1+3, x] = [60, 60, 60]
        img[y2:y2+2, x] = [90, 90, 90]
    # Gökyüzü gradient
    for y in range(int(height * 0.5)):
        ton = int(180 + 75 * y / (height * 0.5))
        img[y, :] = [ton - 30, ton - 20, ton]
    return img

eskiz = eskiz_goruntu_olustur()

STRENGTH_DEGERLERI = [0.2, 0.5, 0.75, 1.0]
img2img_goruntuleri = [eskiz]  # İlk: orijinal eskiz

IMG2IMG_PROMPT = (
    "a photorealistic mountain landscape at golden hour, "
    "dramatic sky, detailed textures, 8K photography"
)

for strength in STRENGTH_DEGERLERI:
    if not SIM_MODE:
        if not hasattr(pipe, "_img2img"):
            img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
            img2img_pipe = img2img_pipe.to(CIHAZ)
        pil_eskiz = Image.fromarray(eskiz).resize((GORUNTU_W, GORUNTU_H))
        generator  = torch.Generator(CIHAZ).manual_seed(42)
        with torch.autocast(CIHAZ) if CUDA_AVAILABLE else torch.no_grad():
            sonuc = img2img_pipe(
                prompt=IMG2IMG_PROMPT,
                image=pil_eskiz,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=25,
                generator=generator,
            )
        img_arr = np.array(sonuc.images[0])
    else:
        # Eskiz ile hedef görüntüyü strength oranında karıştır
        hedef = goruntu_uret_sim(IMG2IMG_PROMPT + " warm", 42, steps=25)
        img_arr = (eskiz.astype(float) * (1 - strength) +
                   hedef.astype(float) * strength).astype(np.uint8)
    img2img_goruntuleri.append(img_arr)
    print(f"  strength={strength:.2f}  ✅ üretildi")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 7: LATENT UZAY İNTERPOLASYON
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Latent Uzay İnterpolasyonu")
print("─" * 65)
print("""
  Yöntem: Prompt embedding'leri arasında interpolasyon
    z_interp = (1−α)·z_prompt1 + α·z_prompt2
    α ∈ [0.0, 0.25, 0.50, 0.75, 1.00]
""")

PROMPT1 = "a serene snowy mountain at dawn, peaceful, cold"
PROMPT2 = "a tropical beach at sunset, warm, colorful, vibrant"
ALFA_DEGERLERI = [0.0, 0.25, 0.5, 0.75, 1.0]

interp_goruntuleri = []
for alfa in ALFA_DEGERLERI:
    if not SIM_MODE:
        # Gerçek embedding interpolasyonu
        with torch.no_grad():
            emb1 = pipe._encode_prompt(PROMPT1, CIHAZ, 1, True, "")[0]
            emb2 = pipe._encode_prompt(PROMPT2, CIHAZ, 1, True, "")[0]
            emb_interp = (1 - alfa) * emb1 + alfa * emb2
        generator = torch.Generator(CIHAZ).manual_seed(42)
        with torch.autocast(CIHAZ) if CUDA_AVAILABLE else torch.no_grad():
            sonuc = pipe(
                prompt_embeds=emb_interp,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=GORUNTU_H, width=GORUNTU_W,
                generator=generator,
            )
        img_arr = np.array(sonuc.images[0])
    else:
        # Simülasyon: 2 görüntüyü alfa oranında karıştır
        img1 = goruntu_uret_sim(PROMPT1, 42, steps=25)
        img2 = goruntu_uret_sim(PROMPT2, 42, steps=25)
        img_arr = ((1 - alfa) * img1.astype(float) +
                   alfa       * img2.astype(float)).astype(np.uint8)
    interp_goruntuleri.append(img_arr)
    print(f"  α={alfa:.2f}  {'kar→kış←yaz' if alfa < 0.5 else 'yaz→sahil' if alfa > 0.5 else 'ara nokta'}  ✅")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 8: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#1C0A00")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                         hspace=0.42, wspace=0.22,
                         top=0.93, bottom=0.05)

def goruntu_goster(ax, img_arr, baslik, alt_metin=None, cerceve_rengi=None):
    ax.imshow(img_arr)
    ax.axis("off")
    ax.set_title(baslik, fontsize=11, fontweight="bold",
                 color="white" if not cerceve_rengi else cerceve_rengi, pad=6)
    if alt_metin:
        ax.text(0.5, -0.04, alt_metin, transform=ax.transAxes,
                fontsize=9, color="#FED7AA", ha="center")
    if cerceve_rengi:
        for spine in ax.spines.values():
            spine.set_edgecolor(cerceve_rengi)
            spine.set_linewidth(3)

# ── PANEL 1: 4 farklı prompt sonucu ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.axis("off")
ax1.set_title("Text-to-Image — 4 Farklı Prompt (25 adım, CFG=7.5)",
              fontsize=13, fontweight="bold", color="white", pad=8)
inner1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :2],
                                          wspace=0.06)
for i, (p, img) in enumerate(zip(PROMTLAR, uretilen_goruntular)):
    ax_tmp = fig.add_subplot(inner1[0, i])
    ax_tmp.imshow(img)
    ax_tmp.axis("off")
    ax_tmp.set_title(p["baslik"], fontsize=10, color="#FED7AA", pad=4)

# ── PANEL 2: CFG Ablasyonu ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
ax2.set_title("guidance_scale Ablasyonu\n(Sabit: seed=42, 25 adım)",
              fontsize=11, fontweight="bold", color="white", pad=6)
inner2 = gridspec.GridSpecFromSubplotSpec(1, len(CFG_DEGERLERI),
                                           subplot_spec=gs[0, 2],
                                           wspace=0.05)
for i, (cfg, img) in enumerate(zip(CFG_DEGERLERI, cfg_goruntuleri)):
    ax_tmp = fig.add_subplot(inner2[0, i])
    renk = ("#22C55E" if cfg == 7.5 else "#EF4444"
            if cfg in [1.0, 15.0] else "#F59E0B")
    ax_tmp.imshow(img)
    ax_tmp.axis("off")
    ax_tmp.set_title(f"w={cfg}", fontsize=8, color=renk, pad=3)

# ── PANEL 3: Adım Sayısı Ablasyonu ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis("off")
ax3.set_title("Inference Steps Ablasyonu\n(CFG=7.5, seed=42)",
              fontsize=11, fontweight="bold", color="white", pad=6)
inner3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 0],
                                           hspace=0.22, wspace=0.08)
for i, (adim, img) in enumerate(zip(ADIM_SAYILARI, adim_goruntuleri)):
    ax_tmp = fig.add_subplot(inner3[i // 2, i % 2])
    renk = "#22C55E" if adim in [20, 50] else "#F59E0B"
    ax_tmp.imshow(img)
    ax_tmp.axis("off")
    ax_tmp.set_title(f"{adim} adım", fontsize=9, color=renk, pad=3)

# ── PANEL 4: Scheduler Karşılaştırması ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
ax4.set_title("Scheduler Karşılaştırması\n(Aynı prompt, benzer kalite)",
              fontsize=11, fontweight="bold", color="white", pad=6)
inner4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1],
                                           wspace=0.06)
sch_etiketler = ["DDIM\n50 adım", "DPM++\n25 adım ✅", "Euler-A\n30 adım"]
for i, (lbl, img) in enumerate(zip(sch_etiketler, sch_goruntuleri)):
    ax_tmp = fig.add_subplot(inner4[0, i])
    renk = "#22C55E" if i == 1 else "#FED7AA"
    ax_tmp.imshow(img)
    ax_tmp.axis("off")
    ax_tmp.set_title(lbl, fontsize=9, color=renk, pad=3)

# ── PANEL 5: img2img şeridi ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
ax5.set_title("img2img — strength=0.0→1.0\n(Eskiz → Fotogerçekçi Dönüşüm)",
              fontsize=11, fontweight="bold", color="white", pad=6)
inner5 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 2],
                                           wspace=0.05)
etiketler = ["Orijinal\n(eskiz)"] + [f"str={s}" for s in STRENGTH_DEGERLERI]
for i, (lbl, img) in enumerate(zip(etiketler, img2img_goruntuleri)):
    ax_tmp = fig.add_subplot(inner5[0, i])
    renk = "#94A3B8" if i == 0 else "#22C55E" if i == 3 else "#FED7AA"
    ax_tmp.imshow(img)
    ax_tmp.axis("off")
    ax_tmp.set_title(lbl, fontsize=8, color=renk, pad=3)

# ── PANEL 6: Latent İnterpolasyon ────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
ax6.axis("off")
ax6.set_title(
    f"Latent Uzay İnterpolasyonu  →  α: 0.0 (Kar Dağı) → 1.0 (Tropikal Sahil)\n"
    f"z_interp = (1−α)·z₁ + α·z₂",
    fontsize=11, fontweight="bold", color="white", pad=6
)
inner6 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2, :2],
                                           wspace=0.05)
for i, (alfa, img) in enumerate(zip(ALFA_DEGERLERI, interp_goruntuleri)):
    ax_tmp = fig.add_subplot(inner6[0, i])
    renk = "#7DD3FC" if alfa == 0 else "#F97316" if alfa == 1.0 else "#FED7AA"
    ax_tmp.imshow(img)
    ax_tmp.axis("off")
    ax_tmp.set_title(f"α={alfa:.2f}", fontsize=10, color=renk,
                     fontweight="bold", pad=4)

# ── PANEL 7: Prompt Karşılaştırma Tablosu ────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor("#2D1200")
ax7.axis("off")
ax7.set_title("Prompt Mühendisliği İpuçları", fontsize=12,
              fontweight="bold", color="white", pad=8)
ipuclari = [
    ("Kalite arttır",  "ultra-detailed, 4K, 8K, photorealistic,\ncinematic lighting, award winning"),
    ("Stil belirt",    "oil painting, watercolor, anime, pencil\ndrawing, impressionist, digital art"),
    ("Negatif prompt", "blurry, low quality, watermark,\nugly, distorted, artifacts, nsfw"),
    ("Kompozisyon",    "rule of thirds, centered, wide angle,\nclose-up portrait, aerial view"),
    ("Işık kalitesi",  "golden hour, dramatic lighting, soft\nshadows, volumetric light, rim light"),
    ("Referans mod.",  "by Greg Rutkowski, by Artgerm,\nTrending on ArtStation, DALL-E style"),
]
for i, (baslik, metin) in enumerate(ipuclari):
    y_pos = 0.94 - i * 0.16
    ax7.text(0.02, y_pos, f"▸ {baslik}:", transform=ax7.transAxes,
             fontsize=9.5, color="#EA580C", fontweight="bold")
    ax7.text(0.02, y_pos - 0.06, metin, transform=ax7.transAxes,
             fontsize=8.5, color="#FED7AA", fontfamily="monospace")

# Ana başlık
fig.suptitle(
    "HAFTA 5 CUMARTESİ — UYGULAMA 01\n"
    "Stable Diffusion: Text-to-Image · CFG Ablasyon · Steps · Scheduler · img2img · İnterpolasyon",
    fontsize=14, fontweight="bold", color="white", y=0.98
)

plt.savefig("h5c_01_stable_diffusion.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h5c_01_stable_diffusion.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Model              : {MODEL_ID}")
print(f"  Cihaz              : {CIHAZ.upper()}")
print(f"  Temel üretim       : {len(PROMTLAR)} farklı prompt")
print(f"  CFG ablasyonu      : {CFG_DEGERLERI}")
print(f"  Adım ablasyonu     : {ADIM_SAYILARI}")
print(f"  Scheduler test     : DDIM / DPM-Solver++ / Euler-A")
print(f"  img2img strength   : {STRENGTH_DEGERLERI}")
print(f"  Interpolasyon      : α ∈ {ALFA_DEGERLERI}")
print(f"  Grafik çıktısı     : h5c_01_stable_diffusion.png")
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print("=" * 65)
