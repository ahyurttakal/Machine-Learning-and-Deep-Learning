"""
=============================================================================
HAFTA 5 CUMARTESİ — UYGULAMA 02
LLM Prompt Mühendisliği — OpenAI API & Yerel Model (llama-cpp-python)
=============================================================================
Kapsam:
  - Zero-Shot / Few-Shot / Chain-of-Thought / Rol Verme karşılaştırması
  - System prompt şablonu tasarımı (Jinja2-style f-string)
  - OpenAI Chat Completions API — tam entegrasyon
  - llama-cpp-python — yerel GGUF model (CPU/GPU)
  - Temperature & Top-P ablasyonu: deterministik → yaratıcı
  - Streaming yanıt implementasyonu
  - RAG (Retrieval-Augmented Generation) temeli: FAISS + embedding
  - Prompt kalite metrikleri: uzunluk, token sayısı, yanıt süresi
  - Kapsamlı görselleştirme (8 panel)
  - OpenAI/llama yoksa tam simülasyon modunda çalışır

Kurulum:
  pip install openai tiktoken numpy matplotlib
  pip install llama-cpp-python  (yerel model için)
  pip install faiss-cpu sentence-transformers  (RAG için)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import json
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

SIM_MODE = not OPENAI_AVAILABLE

print("=" * 65)
print("  HAFTA 5 CUMARTESİ — UYGULAMA 02")
print("  LLM Prompt Mühendisliği")
print("=" * 65)
print(f"  Mod               : {'🔵 Gerçek (OpenAI)' if not SIM_MODE else '🟡 Simülasyon'}")
print(f"  OpenAI            : {'✅' if OPENAI_AVAILABLE else '❌  pip install openai'}")
print(f"  tiktoken          : {'✅' if TIKTOKEN_AVAILABLE else '❌  pip install tiktoken'}")
print(f"  llama-cpp-python  : {'✅' if LLAMA_AVAILABLE else '❌  pip install llama-cpp-python'}")
print(f"  FAISS (RAG)       : {'✅' if RAG_AVAILABLE else '❌  pip install faiss-cpu sentence-transformers'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# YARDIMCI: API İSTEĞİ / SİMÜLASYON
# ─────────────────────────────────────────────────────────────────────────

if not SIM_MODE:
    import os
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-demo"))

def token_say(metin):
    """Yaklaşık token sayısı (gerçekte tiktoken kullanılır)."""
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
            return len(enc.encode(metin))
        except Exception:
            pass
    return max(1, len(metin.split()) * 4 // 3)

def llm_cagir(messages, model="gpt-4o-mini", temperature=0.7,
              max_tokens=400, stream=False):
    """
    OpenAI API çağrısı veya simülasyon.
    Döner: {"content": str, "tokens_in": int, "tokens_out": int, "sure_ms": float}
    """
    t0 = time.time()

    if not SIM_MODE:
        try:
            yanit = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            if stream:
                parcalar = []
                for chunk in yanit:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        parcalar.append(delta)
                        print(delta, end="", flush=True)
                print()
                icerik = "".join(parcalar)
            else:
                icerik = yanit.choices[0].message.content
            sure_ms = (time.time() - t0) * 1000
            girdi_token  = sum(token_say(m["content"]) for m in messages)
            cikti_token  = token_say(icerik)
            return {"content": icerik, "tokens_in": girdi_token,
                    "tokens_out": cikti_token, "sure_ms": sure_ms}
        except Exception as e:
            print(f"  [API Hatası: {e}] Simülasyona geçiliyor.")

    # Simülasyon
    time.sleep(0.05)  # API gecikmesi simülasyonu
    girdi_token = sum(token_say(m["content"]) for m in messages)
    sure_ms     = (time.time() - t0) * 1000 + 120 + np.random.exponential(80)

    # Yanıt içeriği: prompt tipine göre simüle et
    son_kullanici = next((m["content"] for m in reversed(messages)
                          if m["role"] == "user"), "")
    sistem = next((m["content"] for m in messages
                   if m["role"] == "system"), "")

    if "sınıflandır" in son_kullanici.lower() or "sentiment" in sistem.lower():
        icerik = _sim_siniflandirma(son_kullanici, temperature)
    elif "adım" in son_kullanici.lower() or "hesapla" in son_kullanici.lower():
        icerik = _sim_cot(son_kullanici)
    elif "yaz" in son_kullanici.lower() and "kod" in son_kullanici.lower():
        icerik = _sim_kod(son_kullanici, temperature)
    elif "özetle" in son_kullanici.lower() or "özet" in son_kullanici.lower():
        icerik = _sim_ozet(son_kullanici)
    else:
        icerik = _sim_genel(son_kullanici, sistem, temperature)

    cikti_token = token_say(icerik)
    return {"content": icerik, "tokens_in": girdi_token,
            "tokens_out": cikti_token, "sure_ms": sure_ms}

def _sim_siniflandirma(metin, temperature):
    np.random.seed(int(len(metin) + temperature * 100) % 999)
    if any(k in metin.lower() for k in ["harika","mükemmel","güzel","iyi","sevdim"]):
        etiket = "POZİTİF"
    elif any(k in metin.lower() for k in ["berbat","kötü","rezil","beğenmedim","korkunç"]):
        etiket = "NEGATİF"
    else:
        etiket = "NÖTR"
    return f"Duygu Analizi Sonucu: {etiket}\n\nGerekçe: Metindeki ifadeler ve bağlam analiz edildi."

def _sim_cot(metin):
    return ("Adım adım çözüm:\n\n"
            "Adım 1: Problemi tanımla ve verilen bilgileri listele.\n"
            "Adım 2: Uygun formülü veya yöntemi seç.\n"
            "Adım 3: Hesaplamayı gerçekleştir.\n"
            "  → Ara sonuç: ilk değer elde edildi.\n"
            "Adım 4: Sonucu doğrula ve kontrol et.\n\n"
            "Sonuç: Problem başarıyla çözüldü. ✅")

def _sim_kod(metin, temperature):
    np.random.seed(int(temperature * 100))
    return ('def cozum(veri):\n'
            '    """Fonksiyon açıklaması."""\n'
            '    # Girdi doğrulaması\n'
            '    if not veri:\n'
            '        raise ValueError("Boş girdi")\n'
            '    \n'
            '    sonuc = []\n'
            '    for eleman in veri:\n'
            '        # İşlem uygula\n'
            '        sonuc.append(eleman * 2)\n'
            '    \n'
            '    return sonuc\n')

def _sim_ozet(metin):
    return ("Özet:\n\n"
            "Metin, temel kavramları ve aralarındaki ilişkileri açıklamaktadır. "
            "Ana bulgular üç başlık altında özetlenebilir:\n\n"
            "1. Birinci önemli nokta ve bağlamı.\n"
            "2. İkinci önemli nokta ve uygulaması.\n"
            "3. Sonuç ve öneriler.\n\n"
            "Toplam kelime sayısı önemli ölçüde azaltılmıştır.")

def _sim_genel(metin, sistem, temperature):
    np.random.seed(int(len(metin) * temperature * 10) % 9999)
    uzunluk = int(50 + temperature * 80 + np.random.randint(-20, 40))
    kelimeler = ["Analiz", "değerlendirme", "perspektif", "bağlam", "önem",
                 "ilişki", "sonuç", "yöntem", "strateji", "uygulama",
                 "faktör", "etki", "geliştirme", "verimlilik", "kalite"]
    np.random.shuffle(kelimeler)
    return (f"Yanıt: {' '.join(kelimeler[:min(10, uzunluk//5)])}.\n\n"
            f"Detaylı açıklama ve analiz burada yer almaktadır. "
            f"Temperature={temperature:.1f} ile üretilen bu yanıt "
            f"{'daha yaratıcı ve çeşitlidir' if temperature > 1.0 else 'dengeli ve tutarlıdır'}.")


# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: PROMPT TEKNİK KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Prompt Teknik Karşılaştırması")
print("─" * 65)

TEST_METIN = "'Bu ürün beklentilerimin altında kaldı, para değmez.'"

TEKNIKLER = {
    "zero_shot": {
        "ad": "Zero-Shot",
        "messages": [
            {"role": "user",
             "content": f"Şu metni duygu açısından sınıflandır: {TEST_METIN}"}
        ],
    },
    "few_shot": {
        "ad": "Few-Shot",
        "messages": [
            {"role": "user",
             "content": (
                 "Metni duygu açısından sınıflandır. Örnekler:\n\n"
                 "Metin: 'Harika bir ürün, çok beğendim!' → POZİTİF\n"
                 "Metin: 'Berbat, hiç işe yaramıyor.' → NEGATİF\n"
                 "Metin: 'İdare eder, fena sayılmaz.' → NÖTR\n\n"
                 f"Şimdi sınıflandır: {TEST_METIN}"
             )},
        ],
    },
    "cot": {
        "ad": "Chain-of-Thought",
        "messages": [
            {"role": "user",
             "content": (
                 "Aşağıdaki metni duygu açısından sınıflandır.\n"
                 "Adım adım düşün:\n"
                 "1. Metindeki anahtar ifadeleri belirle\n"
                 "2. Her ifadenin olumlu/olumsuz katkısını değerlendir\n"
                 "3. Genel tonu belirle\n"
                 "4. Etiket ver: POZİTİF / NEGATİF / NÖTR\n\n"
                 f"Metin: {TEST_METIN}"
             )},
        ],
    },
    "rol_verme": {
        "ad": "Rol Verme",
        "messages": [
            {"role": "system",
             "content": (
                 "Sen duygu analizi konusunda uzmanlaşmış bir NLP araştırmacısısın. "
                 "Türkçe metinleri analiz edersin. "
                 "Yanıtlarını şu formatta ver:\n"
                 "ETİKET: [POZİTİF/NEGATİF/NÖTR]\n"
                 "GÜVEN: [Düşük/Orta/Yüksek]\n"
                 "GEREKÇE: [kısa açıklama]"
             )},
            {"role": "user",
             "content": f"Analiz et: {TEST_METIN}"},
        ],
    },
}

teknik_sonuclari = {}
print(f"  Test metni: {TEST_METIN}\n")
print(f"  {'Teknik':<20} {'Giriş Token':>12} {'Çıkış Token':>12} {'Süre (ms)':>10}")
print("  " + "-" * 60)

for tid, teknik in TEKNIKLER.items():
    sonuc = llm_cagir(teknik["messages"], temperature=0.3, max_tokens=200)
    teknik_sonuclari[tid] = sonuc
    print(f"  {teknik['ad']:<20} {sonuc['tokens_in']:>12} "
          f"{sonuc['tokens_out']:>12} {sonuc['sure_ms']:>10.0f}")

print()
for tid, teknik in TEKNIKLER.items():
    print(f"  [{teknik['ad']}]")
    for satir in teknik_sonuclari[tid]["content"].split("\n")[:4]:
        if satir.strip():
            print(f"    {satir}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: SYSTEM PROMPT ŞABLONU
# ─────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 2: System Prompt Şablonu Tasarımı")
print("─" * 65)

def sistem_prompt_olustur(rol, uzmanlik, dil, cikti_formati, kisitlamalar=""):
    """Jinja2-style f-string şablonu."""
    return (
        f"Sen {uzmanlik} alanında uzman bir {rol}sun.\n\n"
        f"## Temel Kurallar\n"
        f"- Her zaman {dil} dilinde yanıt ver\n"
        f"- Teknik terimler için kısa açıklama ekle\n"
        f"- Yanıtlarını şu formatta yapılandır: {cikti_formati}\n"
        + (f"- Kısıtlamalar: {kisitlamalar}\n" if kisitlamalar else "") +
        f"\n## Yanıt Formatı\n"
        f"Başlık → Açıklama → Örnek → Özet\n\n"
        f"## Kişilik\n"
        f"Sabırlı, anlaşılır, yapıcı ve merak uyandırıcı bir ton kullan."
    )

SENARYO_SISTEMLER = {
    "Python Eğitmeni": sistem_prompt_olustur(
        rol="Python eğitmeni",
        uzmanlik="Python ve yazılım geliştirme",
        dil="Türkçe",
        cikti_formati="kod bloğu + satır satır açıklama",
        kisitlamalar="Sadece Python 3.10+ kullan, type hints ekle"
    ),
    "Veri Analisti": sistem_prompt_olustur(
        rol="kıdemli veri analisti",
        uzmanlik="veri analizi ve istatistik",
        dil="Türkçe",
        cikti_formati="tablo veya madde listesi",
        kisitlamalar="Sayısal örnekler ve metrikler mutlaka ekle"
    ),
    "Müşteri Destek": sistem_prompt_olustur(
        rol="müşteri hizmetleri temsilcisi",
        uzmanlik="ürün ve hizmet desteği",
        dil="Türkçe",
        cikti_formati="empati → çözüm → sonraki adım",
        kisitlamalar="Olumsuz söz etme, çözüm odaklı ol"
    ),
}

print(f"  {'Senaryo':<20} {'Sistem Token':>13} {'Yanıt Token':>12} {'Süre (ms)':>10}")
print("  " + "-" * 60)

sist_sonuclari = {}
TEST_SORU = "Bir liste içindeki tekrar eden elemanları nasıl kaldırırım?"

for senaryo, sistem_prompt in SENARYO_SISTEMLER.items():
    msgs = [
        {"role": "system",  "content": sistem_prompt},
        {"role": "user",    "content": TEST_SORU},
    ]
    sonuc = llm_cagir(msgs, temperature=0.5, max_tokens=300)
    sist_sonuclari[senaryo] = {
        "sistem_token": token_say(sistem_prompt),
        "sonuc": sonuc,
    }
    print(f"  {senaryo:<20} {token_say(sistem_prompt):>13} "
          f"{sonuc['tokens_out']:>12} {sonuc['sure_ms']:>10.0f}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: TEMPERATURE & TOP-P ABLASYONU
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Temperature & Top-P Ablasyonu")
print("─" * 65)
print("""
  Temperature (T):
    T = 0   → p_i = argmax (tamamen deterministik)
    T = 0.7 → dengeli (önerilen)
    T = 1.5 → çok yaratıcı, bazen tutarsız
    T > 2.0 → genellikle anlamsız

  Top-P (Nucleus Sampling):
    p = 0.1 → Sadece en olası %10 token
    p = 0.9 → Geniş örnekleme uzayı (önerilen)
    p = 1.0 → Tüm kelime dağarcığı
""")

SICAKLIK_DEGERLERI = [0.0, 0.3, 0.7, 1.0, 1.5]
YARATICI_GOREV     = "Yapay zeka hakkında kısa ve özgün bir metafor yaz."

sicaklik_sonuclari = []
print(f"  {'Temperature':>12} {'Token Sayısı':>13} {'Süre (ms)':>10} {'Çeşitlilik'}")
print("  " + "-" * 60)

for T in SICAKLIK_DEGERLERI:
    msgs = [{"role": "user", "content": YARATICI_GOREV}]
    sonuc = llm_cagir(msgs, temperature=T, max_tokens=120)
    # Kelime çeşitliliği: benzersiz kelime oranı
    kelimeler  = sonuc["content"].lower().split()
    cesitlilik = len(set(kelimeler)) / max(len(kelimeler), 1)
    sicaklik_sonuclari.append({
        "T": T, "token": sonuc["tokens_out"],
        "sure": sonuc["sure_ms"], "cesitlilik": cesitlilik,
        "icerik": sonuc["content"],
    })
    yorumlar = {
        0.0: "Deterministik",
        0.3: "Tutarlı",
        0.7: "Dengeli ✅",
        1.0: "Yaratıcı",
        1.5: "Çok Serbest",
    }
    print(f"  {T:>12.1f} {sonuc['tokens_out']:>13} "
          f"{sonuc['sure_ms']:>10.0f}  {yorumlar[T]} ({cesitlilik:.2f})")

# Top-P ablasyonu
TOPP_DEGERLERI = [0.1, 0.5, 0.9, 0.95, 1.0]
topp_sonuclari = []

print()
print(f"  {'Top-P':>7} {'Token':>8} {'Çeşitlilik':>12}")
print("  " + "-" * 32)
for p in TOPP_DEGERLERI:
    msgs  = [{"role": "user", "content": YARATICI_GOREV}]
    sonuc = llm_cagir(msgs, temperature=0.7, max_tokens=120)
    # Top-P simülasyonu: düşük p → daha az çeşitlilik
    sim_cesit = min(0.9, p * 0.7 + np.random.uniform(0.05, 0.15))
    topp_sonuclari.append({"p": p, "cesitlilik": sim_cesit, "token": sonuc["tokens_out"]})
    print(f"  {p:>7.2f} {sonuc['tokens_out']:>8} {sim_cesit:>12.3f}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: STREAMING YANIT
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Streaming Yanıt Implementasyonu")
print("─" * 65)
print("""
  Streaming: Model yanıtını kelime kelime (token token) gönderir.
  Kullanıcı deneyimini iyileştirir — ilk token gecikme süresini azaltır.

  stream=True → Generator döner (SSE: Server-Sent Events)
""")
print("  Streaming yanıt demo:")
print("  " + "─" * 50)
print("  ", end="")

STREAM_GOREV = "Python'da dekoratör tasarım kalıbını tek paragrafta açıkla."
stream_parcalar = []

if not SIM_MODE:
    msgs = [{"role": "user", "content": STREAM_GOREV}]
    sonuc = llm_cagir(msgs, temperature=0.5, max_tokens=150, stream=True)
    stream_icerik = sonuc["content"]
else:
    # Streaming simülasyonu
    sim_icerik = (
        "Dekoratörler, bir fonksiyonun davranışını değiştirmek için "
        "sarmalayan yüksek mertebeden fonksiyonlardır. "
        "@decorator sözdizimi, wrap(func) çağrısının kısaltmasıdır. "
        "Loglama, önbellekleme ve yetki kontrolü gibi çapraz kesen "
        "endişeler için idealdir; orijinal fonksiyon kodunu bozmadan "
        "yeni işlevsellik katmanı ekler. ✅"
    )
    kelimeler = sim_icerik.split()
    for kelime in kelimeler:
        print(kelime, end=" ", flush=True)
        time.sleep(0.02)
    print()
    stream_icerik = sim_icerik

print("  " + "─" * 50)
print(f"  Toplam token: ~{token_say(stream_icerik)}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 5: ÇOKLU TUR KONUŞMA
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: Çoklu Tur Konuşma Yönetimi")
print("─" * 65)
print("""
  LLM durumsuzdur (stateless) — her istekte tam geçmiş gönderilmeli.
  Context penceresi dolduğunda özetleme veya budama gerekir.
""")

class KonusmaYoneticisi:
    """Çoklu tur konuşma durumunu yönetir."""

    def __init__(self, sistem_prompt, max_token_esik=3000):
        self.gecmis       = [{"role": "system", "content": sistem_prompt}]
        self.max_esik     = max_token_esik
        self.tur_sayisi   = 0
        self.toplam_token = token_say(sistem_prompt)

    def mesaj_gonder(self, kullanici_mesaji, **kwargs):
        self.gecmis.append({"role": "user", "content": kullanici_mesaji})
        sonuc = llm_cagir(self.gecmis, **kwargs)
        self.gecmis.append({"role": "assistant", "content": sonuc["content"]})
        self.tur_sayisi   += 1
        self.toplam_token += sonuc["tokens_in"] + sonuc["tokens_out"]

        # Context budama: eşik aşılırsa ortadaki mesajları kaldır
        if self.toplam_token > self.max_esik:
            sistem = self.gecmis[0]
            son_ikili = self.gecmis[-4:]  # son 2 tur koru
            self.gecmis  = [sistem] + son_ikili
            self.toplam_token = sum(token_say(m["content"]) for m in self.gecmis)

        return sonuc["content"]

    def ozet(self):
        return (f"  Tur: {self.tur_sayisi}  "
                f"Mesaj: {len(self.gecmis)}  "
                f"Yaklaşık Token: {self.toplam_token}")

konusma = KonusmaYoneticisi(
    sistem_prompt="Sen yardımsever bir Python programlama asistanısın. Kısa ve net yanıtlar ver.",
    max_token_esik=2000,
)

SORULAR = [
    "Python'da liste comprehension nedir?",
    "Bir örnek göster.",
    "Dict comprehension da aynı mı çalışır?",
]

print(f"  {'Tur':>4}  {'Soru (kısa)':<42} {'Token (çıkış)':>14}")
print("  " + "-" * 64)
for si, soru in enumerate(SORULAR, 1):
    yanit = konusma.mesaj_gonder(soru, temperature=0.4, max_tokens=150)
    print(f"  {si:>4}  {soru[:42]:<42} {token_say(yanit):>14}")

print()
print(konusma.ozet())

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 6: LLAMA-CPP-PYTHON (YEREL MODEL)
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: llama-cpp-python — Yerel GGUF Model")
print("─" * 65)
print("""
  GGUF (GPT-Generated Unified Format):
    Model ağırlıklarını quantize edilmiş formatta saklar.
    Q4_K_M: 4-bit quantization → ~4GB, dengeli kalite/hız
    Q8_0:   8-bit              → ~8GB, yüksek kalite
    F16:    16-bit             → tam kalite, büyük boyut

  Başlatma:
    llm = Llama(
        model_path="./models/llama-3-8b-instruct.Q4_K_M.gguf",
        n_ctx=4096,         # Context penceresi
        n_gpu_layers=35,    # GPU katman sayısı (0=CPU)
        chat_format="llama-3",  # Chat şablonu
    )

  Çıkarım:
    yanit = llm.create_chat_completion(
        messages=[{"role":"user","content":"Merhaba!"}],
        max_tokens=200, temperature=0.7,
    )
    print(yanit["choices"][0]["message"]["content"])
""")

if LLAMA_AVAILABLE:
    MODEL_YOLU = "./models/llama-3-8b-instruct.Q4_K_M.gguf"
    import os
    if os.path.exists(MODEL_YOLU):
        llm_yerel = Llama(
            model_path=MODEL_YOLU,
            n_ctx=4096, n_gpu_layers=35, verbose=False,
        )
        yerel_yanit = llm_yerel.create_chat_completion(
            messages=[{"role": "user",
                       "content": "Türkçe bir yapay zeka tanımı yaz."}],
            max_tokens=150, temperature=0.7,
        )
        print("  Yerel model yanıtı:")
        print("  " + yerel_yanit["choices"][0]["message"]["content"][:200])
    else:
        print(f"  ⚠️  Model dosyası bulunamadı: {MODEL_YOLU}")
        print("  İndirme: huggingface-cli download meta-llama/Llama-3-8B-Instruct "
              "--include '*.gguf'")
else:
    print("  [SIM] Yerel model karşılaştırması:")
    print()
    karsilastirma = [
        ("GPT-4o-mini",  "Bulut", "API key", "Yüksek ✅",  "~1s",  "Yok"),
        ("Llama 3 8B",   "Yerel", "GGUF ~4GB","Orta ✅",   "~3s",  "%100 ✅"),
        ("Mistral 7B",   "Yerel", "GGUF ~4GB","Orta ✅",   "~4s",  "%100 ✅"),
        ("Llama 3 70B",  "Yerel", "GGUF ~40GB","Yüksek ✅", "~15s", "%100 ✅"),
        ("GPT-4o",       "Bulut", "API key", "En Yüksek ✅","~2s",  "Yok"),
    ]
    print(f"  {'Model':<16} {'Tür':<8} {'Gereksinim':<12} {'Kalite':<14} {'Hız':<8} {'Gizlilik'}")
    print("  " + "-" * 72)
    for row in karsilastirma:
        print(f"  {row[0]:<16} {row[1]:<8} {row[2]:<12} {row[3]:<14} {row[4]:<8} {row[5]}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 7: RAG TEMELİ
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: RAG (Retrieval-Augmented Generation) Temeli")
print("─" * 65)
print("""
  RAG Adımları:
    1. Belgeleri chunk'lara böl (512 token örtüşen pencere)
    2. Her chunk için embedding üret
    3. Vektör veritabanına kaydet (FAISS / ChromaDB / Pinecone)
    4. Soru gelince → embedding → en yakın k chunk → bağlam
    5. Prompt = "Bağlam: {chunk} Soru: {soru}" → LLM yanıtlar
""")

# Örnek belge koleksiyonu
BELGELER = [
    "Python, Guido van Rossum tarafından 1991 yılında geliştirilen yüksek seviyeli bir programlama dilidir.",
    "Makine öğrenmesi, bilgisayarların açıkça programlanmadan veriden öğrenmesini sağlar.",
    "Derin öğrenme, insan beyninden ilham alan yapay sinir ağları kullanan ML alt dalıdır.",
    "Transformerlar, dikkat mekanizması kullanan ve NLP'de devrim yaratan modellerdir.",
    "FastAPI, Python 3.6+ type hintslerini kullanan modern, hızlı bir web çerçevesidir.",
    "Docker, uygulamaları container'larda paketleyerek tutarlı çalışma ortamı sağlar.",
    "GPU'lar paralel hesaplama kapasitesiyle derin öğrenme eğitimini dramatik biçimde hızlandırır.",
    "BERT, Google tarafından 2018'de geliştirilen ve NLP'de çığır açan bir transformer modelidir.",
    "Stable Diffusion, latent difüzyon modeli kullanan açık kaynaklı görüntü üretim modelidir.",
    "LangChain, LLM uygulamaları oluşturmak için modüler bileşenler sunan Python kütüphanesidir.",
]

class BaskitRAG:
    """
    FAISS veya numpy tabanlı basit RAG uygulaması.
    Gerçek kullanımda: sentence-transformers + FAISS.
    """
    def __init__(self, belgeler):
        self.belgeler  = belgeler
        # Sahte embedding: TF-IDF benzeri bag-of-words
        self.kelime_listesi = self._kelime_listesi_olustur()
        self.embeddingler   = np.array([
            self._embedding_uret(b) for b in belgeler
        ])
        # L2 normalizasyon
        normlar = np.linalg.norm(self.embeddingler, axis=1, keepdims=True)
        self.embeddingler = self.embeddingler / (normlar + 1e-8)

    def _kelime_listesi_olustur(self):
        kelimeler = set()
        for b in self.belgeler:
            kelimeler.update(b.lower().split())
        return sorted(list(kelimeler))

    def _embedding_uret(self, metin):
        """Basit BoW embedding."""
        vec = np.zeros(len(self.kelime_listesi))
        for kelime in metin.lower().split():
            if kelime in self.kelime_listesi:
                idx      = self.kelime_listesi.index(kelime)
                vec[idx] += 1.0
        return vec

    def ara(self, sorgu, k=3):
        """En yakın k belgeyi döner (cosine similarity)."""
        sorgu_emb = self._embedding_uret(sorgu)
        norm      = np.linalg.norm(sorgu_emb)
        if norm > 0:
            sorgu_emb /= norm
        benzerlikler = self.embeddingler @ sorgu_emb
        en_iyi_idx   = np.argsort(benzerlikler)[::-1][:k]
        return [(self.belgeler[i], float(benzerlikler[i]))
                for i in en_iyi_idx]

    def rag_cevapla(self, soru, k=2):
        """Bağlam al → LLM'e gönder → yanıt döndür."""
        alakali = self.ara(soru, k)
        baglam  = "\n".join(f"[{i+1}] {b}" for i, (b, _) in enumerate(alakali))
        prompt  = (
            f"Aşağıdaki bağlamı kullanarak soruyu yanıtla.\n\n"
            f"Bağlam:\n{baglam}\n\n"
            f"Soru: {soru}\n\n"
            f"Yanıt:"
        )
        msgs  = [{"role": "user", "content": prompt}]
        sonuc = llm_cagir(msgs, temperature=0.3, max_tokens=200)
        return sonuc["content"], alakali

rag = BaskitRAG(BELGELER)

SORULAR_RAG = [
    "Python ne zaman geliştirildi?",
    "Transformer modeller ne işe yarar?",
    "Stable Diffusion nasıl çalışır?",
]

print(f"  {'Soru':<45} {'Top-1 Benzerlik':>16} {'Token':>7}")
print("  " + "-" * 72)

rag_sonuclari = []
for soru in SORULAR_RAG:
    yanit, alakali = rag.rag_cevapla(soru, k=2)
    en_iyi_benzerlik = alakali[0][1] if alakali else 0
    rag_sonuclari.append({
        "soru": soru, "yanit": yanit,
        "alakali": alakali, "benzerlik": en_iyi_benzerlik
    })
    print(f"  {soru[:45]:<45} {en_iyi_benzerlik:>16.4f} {token_say(yanit):>7}")

print()
print("  İlk RAG yanıtı:")
print(f"  Soru   : {rag_sonuclari[0]['soru']}")
print(f"  Bağlam : {rag_sonuclari[0]['alakali'][0][0][:80]}...")
print(f"  Yanıt  : {rag_sonuclari[0]['yanit'][:120]}...")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 8: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#FFF7ED")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.48, wspace=0.36,
                        top=0.93, bottom=0.05)

RENKLER = {"zero_shot":"#EF4444","few_shot":"#F97316",
           "cot":"#D97706","rol_verme":"#0D9488"}
TUR_ADLARI = {"zero_shot":"Zero-Shot","few_shot":"Few-Shot",
              "cot":"CoT","rol_verme":"Rol Verme"}

# ── GRAFİK 1: Teknik Karşılaştırma (bar) ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
tidler  = list(TEKNIKLER.keys())
token_g = [teknik_sonuclari[t]["tokens_in"]  for t in tidler]
token_c = [teknik_sonuclari[t]["tokens_out"] for t in tidler]
sureler = [teknik_sonuclari[t]["sure_ms"]    for t in tidler]
x       = np.arange(len(tidler))
ren_lst = [RENKLER[t] for t in tidler]

ax1.bar(x - 0.2, token_g, 0.35, color=ren_lst, alpha=0.7,
        label="Giriş Token", edgecolor="white")
ax1.bar(x + 0.2, token_c, 0.35,
        color=[r + "AA" for r in ["#EF4444","#F97316","#D97706","#0D9488"]],
        alpha=0.9, label="Çıkış Token", edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels([TUR_ADLARI[t] for t in tidler], fontsize=10)
ax1.set_title("Prompt Tekniği vs Token Kullanımı",
              fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Token Sayısı", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.4)

# ── GRAFİK 2: Teknik Karşılaştırma (süre) ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
bars2 = ax2.barh([TUR_ADLARI[t] for t in tidler], sureler,
                  color=ren_lst, edgecolor="white", alpha=0.85)
for bar, val in zip(bars2, sureler):
    ax2.text(val + 5, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}ms", va="center", fontsize=10, color="#1E293B")
ax2.set_title("Teknik Karşılaştırması\nYanıt Süresi (ms)",
              fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Süre (ms)", fontsize=10)
ax2.grid(axis="x", alpha=0.4)

# ── GRAFİK 3: Temperature vs Çeşitlilik ──────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
T_ler = [r["T"] for r in sicaklik_sonuclari]
cesit = [r["cesitlilik"] for r in sicaklik_sonuclari]
tok   = [r["token"]      for r in sicaklik_sonuclari]
ax3.plot(T_ler, cesit, "o-", color="#EA580C", linewidth=2.5,
         markersize=9, label="Kelime Çeşitliliği")
ax3b = ax3.twinx()
ax3b.bar(T_ler, tok, width=0.18, alpha=0.35, color="#D97706", label="Token Sayısı")
ax3.axvline(x=0.7, color="#22C55E", linestyle="--", linewidth=1.8,
            alpha=0.8, label="Önerilen (0.7)")
ax3.set_title("Temperature vs Çeşitlilik & Token",
              fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Temperature", fontsize=10)
ax3.set_ylabel("Kelime Çeşitliliği (oran)", fontsize=10, color="#EA580C")
ax3b.set_ylabel("Token Sayısı", fontsize=10, color="#D97706")
lines  = ax3.get_lines() + ax3b.patches[:1]
labels = ["Çeşitlilik", "Token", "Önerilen (0.7)"]
ax3.legend(fontsize=9, loc="upper left")
ax3.grid(alpha=0.4)

# ── GRAFİK 4: Top-P Ablasyonu ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
p_ler  = [r["p"]         for r in topp_sonuclari]
cesit4 = [r["cesitlilik"] for r in topp_sonuclari]
ax4.fill_between(p_ler, cesit4, alpha=0.25, color="#6D28D9")
ax4.plot(p_ler, cesit4, "s-", color="#6D28D9", linewidth=2.5,
         markersize=9, label="Çeşitlilik")
ax4.axvline(x=0.9, color="#22C55E", linestyle="--", linewidth=1.8,
            alpha=0.8, label="Önerilen (0.9)")
ax4.set_title("Top-P (Nucleus) vs Çeşitlilik\n(T=0.7 sabit)",
              fontsize=12, fontweight="bold", pad=10)
ax4.set_xlabel("Top-P", fontsize=10)
ax4.set_ylabel("Kelime Çeşitliliği", fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.4)

# ── GRAFİK 5: System Prompt Token Maliyeti ───────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("white")
senaryo_adlar = list(sist_sonuclari.keys())
sist_tokler   = [sist_sonuclari[s]["sistem_token"]          for s in senaryo_adlar]
yanit_tokler  = [sist_sonuclari[s]["sonuc"]["tokens_out"]   for s in senaryo_adlar]
x5            = np.arange(len(senaryo_adlar))
ax5.bar(x5, sist_tokler,  0.45, color="#EA580C", label="Sistem Prompt",
        edgecolor="white")
ax5.bar(x5, yanit_tokler, 0.45, bottom=sist_tokler,
        color="#0D9488", label="Model Yanıtı", edgecolor="white")
ax5.set_xticks(x5)
ax5.set_xticklabels(senaryo_adlar, fontsize=9, rotation=12)
ax5.set_title("System Prompt Token Maliyeti\n(Senaryo Karşılaştırması)",
              fontsize=12, fontweight="bold", pad=10)
ax5.set_ylabel("Token", fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(axis="y", alpha=0.4)

# ── GRAFİK 6: RAG Benzerlik Skoru ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
soru_kisa = [s["soru"][:30] + "…" for s in rag_sonuclari]
benzerlkr = [s["benzerlik"] for s in rag_sonuclari]
bar_renkl = ["#22C55E" if b > 0.5 else "#F97316" if b > 0.3 else "#EF4444"
             for b in benzerlkr]
ax6.barh(soru_kisa, benzerlkr, color=bar_renkl, edgecolor="white")
ax6.axvline(x=0.5, color="#94A3B8", linestyle="--", linewidth=1.5,
            label="İyi eşik=0.5")
ax6.set_title("RAG Retrieval Benzerlik Skoru\n(Cosine Similarity)",
              fontsize=12, fontweight="bold", pad=10)
ax6.set_xlabel("Cosine Benzerlik ↑", fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(axis="x", alpha=0.4)
ax6.set_xlim(0, 1.05)

# ── GRAFİK 7: Konuşma Context Büyümesi ───────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor("white")
tur_sayisi   = list(range(1, len(SORULAR) + 1))
birikimli_tk = []
toplam = 0
for i, soru in enumerate(SORULAR):
    toplam += token_say(soru) + 100  # yanıt yaklaşık
    birikimli_tk.append(toplam)
ax7.fill_between(tur_sayisi, birikimli_tk, alpha=0.2, color="#EA580C")
ax7.plot(tur_sayisi, birikimli_tk, "o-", color="#EA580C",
         linewidth=2.5, markersize=9, label="Birikimli Token")
ax7.axhline(y=2000, color="#EF4444", linestyle="--", linewidth=1.5,
            label="Budama eşiği (2000)")
ax7.set_title("Çoklu Tur Konuşma\nContext Token Büyümesi",
              fontsize=12, fontweight="bold", pad=10)
ax7.set_xlabel("Tur Numarası", fontsize=10)
ax7.set_ylabel("Birikimli Token", fontsize=10)
ax7.set_xticks(tur_sayisi)
ax7.legend(fontsize=10)
ax7.grid(alpha=0.4)

# ── GRAFİK 8: Model Karşılaştırma Radar ──────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1:], projection="polar")
ax8.set_facecolor("#FFFBF5")

KATEGORILER = ["Kalite", "Hız", "Gizlilik", "Maliyet\n(düşük=iyi)", "Kolay\nKurulum"]
N = len(KATEGORILER)
acılar = [n / float(N) * 2 * np.pi for n in range(N)]
acılar += acılar[:1]

modeller_radar = {
    "GPT-4o-mini (Bulut)":  {"degerler": [0.85, 0.90, 0.10, 0.70, 0.95], "renk": "#EA580C"},
    "Llama 3 8B (Yerel)":   {"degerler": [0.75, 0.65, 1.00, 1.00, 0.55], "renk": "#0D9488"},
    "GPT-4o (Bulut)":       {"degerler": [1.00, 0.80, 0.10, 0.30, 0.95], "renk": "#6D28D9"},
}
for model_adi, bilgi in modeller_radar.items():
    degerler = bilgi["degerler"] + bilgi["degerler"][:1]
    ax8.plot(acılar, degerler, "o-", color=bilgi["renk"],
             linewidth=2.2, markersize=7, label=model_adi)
    ax8.fill(acılar, degerler, color=bilgi["renk"], alpha=0.1)

ax8.set_xticks(acılar[:-1])
ax8.set_xticklabels(KATEGORILER, fontsize=10)
ax8.set_ylim(0, 1)
ax8.set_title("Model Karşılaştırması\n(Radar Grafik)",
              fontsize=12, fontweight="bold", pad=20)
ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
ax8.grid(alpha=0.4)

# Ana başlık
fig.suptitle(
    "HAFTA 5 CUMARTESİ — UYGULAMA 02\n"
    "LLM Prompt Mühendisliği: Zero-Shot · Few-Shot · CoT · Streaming · RAG · Model Karşılaştırma",
    fontsize=14, fontweight="bold", color="#1C0A00", y=0.98
)

plt.savefig("h5c_02_llm_prompt.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h5c_02_llm_prompt.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Prompt teknik sayısı      : {len(TEKNIKLER)}  (Zero/Few/CoT/Rol)")
print(f"  Temperature ablasyonu     : {SICAKLIK_DEGERLERI}")
print(f"  Top-P ablasyonu           : {TOPP_DEGERLERI}")
print(f"  Konuşma turu              : {len(SORULAR)}")
print(f"  RAG koleksiyon            : {len(BELGELER)} belge")
print(f"  RAG sorgu sayısı          : {len(SORULAR_RAG)}")
print(f"  Model radar karşılaştırma : {list(modeller_radar.keys())}")
print(f"  Grafik çıktısı            : h5c_02_llm_prompt.png")
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print("=" * 65)
