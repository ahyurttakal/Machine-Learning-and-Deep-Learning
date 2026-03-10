"""
=============================================================================
AGENTİK AI — UYGULAMA 04
Agent Değerlendirme & Kıyaslama Çerçevesi (Benchmark Framework)
=============================================================================
Kapsam:
  - 5 farklı ajan stratejisi: ZeroShot, FewShot, CoT, ReAct, Reflexion
  - 6 görev kategorisi: Mantık, Matematik, Bilgi, Kodlama, Planlama, Yaratıcı
  - Otomatik puanlama: doğruluk, açıklık, verimlilik, güvenilirlik
  - Ablasyon çalışması: araç sayısı, adım limiti, sıcaklık, bellek
  - Hata analizi: hata türü sınıflandırma, kök neden analizi
  - Güvenilirlik testi: aynı sorgu N kez → varyans analizi
  - Latency-accuracy trade-off eğrisi
  - Kapsamlı görselleştirme (8 panel)
=============================================================================
"""

import time, random, math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  AGENTİK AI — UYGULAMA 04")
print("  Agent Değerlendirme & Kıyaslama")
print("=" * 65)

random.seed(42); np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: AJAN STRATEJİLERİ
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 1: Ajan Stratejileri")
print("─" * 33)

@dataclass
class Cevap:
    strateji  : str
    kategori  : str
    sorgu     : str
    cevap     : str
    dogru     : bool
    sure      : float
    adim_sayisi: int
    guvensiz  : bool = False
    aciklama  : str  = ""

class BaseStrateji:
    """Tüm stratejiler için temel sınıf."""
    def __init__(self, ad: str):
        self.ad         = ad
        self.cevaplar   : List[Cevap] = []
        # (kategori, zorluk) → temel başarı olasılığı
        self._basar_prob= {}

    def _temel_prob(self, kategori: str) -> float:
        return self._basar_prob.get(kategori, 0.7)

    def cevapla(self, sorgu: str, kategori: str,
                dogru_cevap: str) -> Cevap:
        raise NotImplementedError

    def istatistik(self) -> dict:
        n = len(self.cevaplar)
        if n == 0:
            return {"n":0,"basari":0,"ort_sure":0,"ort_adim":0}
        return {
            "n":         n,
            "basari":    sum(1 for c in self.cevaplar if c.dogru) / n,
            "ort_sure":  sum(c.sure for c in self.cevaplar) / n,
            "ort_adim":  sum(c.adim_sayisi for c in self.cevaplar) / n,
        }

class ZeroShotStrateji(BaseStrateji):
    """Doğrudan cevap, düşünme adımı yok."""
    def __init__(self):
        super().__init__("ZeroShot")
        self._basar_prob = {
            "mantık":0.62,"matematik":0.58,"bilgi":0.72,
            "kodlama":0.55,"planlama":0.50,"yaratıcı":0.65
        }
    def cevapla(self, sorgu, kategori, dogru_cevap):
        sure = random.uniform(0.05, 0.20)
        time.sleep(0.002)
        dogru = random.random() < self._temel_prob(kategori)
        return Cevap(self.ad, kategori, sorgu[:40],
                     dogru_cevap if dogru else "Yanlış cevap.",
                     dogru, sure, adim_sayisi=1)

class FewShotStrateji(BaseStrateji):
    """Az sayıda örnek ile destekli cevap."""
    def __init__(self):
        super().__init__("FewShot")
        self._basar_prob = {
            "mantık":0.73,"matematik":0.70,"bilgi":0.78,
            "kodlama":0.65,"planlama":0.60,"yaratıcı":0.72
        }
    def cevapla(self, sorgu, kategori, dogru_cevap):
        sure = random.uniform(0.08, 0.28)
        time.sleep(0.002)
        dogru = random.random() < self._temel_prob(kategori)
        return Cevap(self.ad, kategori, sorgu[:40],
                     dogru_cevap if dogru else "Kısmen doğru.",
                     dogru, sure, adim_sayisi=2)

class CoTStrateji(BaseStrateji):
    """Chain-of-Thought: adım adım düşünme."""
    def __init__(self):
        super().__init__("CoT")
        self._basar_prob = {
            "mantık":0.84,"matematik":0.88,"bilgi":0.80,
            "kodlama":0.75,"planlama":0.72,"yaratıcı":0.70
        }
    def cevapla(self, sorgu, kategori, dogru_cevap):
        adim_sayisi = random.randint(3, 6)
        sure = adim_sayisi * random.uniform(0.04, 0.10)
        time.sleep(0.003)
        dogru = random.random() < self._temel_prob(kategori)
        return Cevap(self.ad, kategori, sorgu[:40],
                     dogru_cevap if dogru else "Mantıksal hata.",
                     dogru, sure, adim_sayisi=adim_sayisi,
                     aciklama=f"{adim_sayisi} düşünce adımı")

class ReactStrateji(BaseStrateji):
    """ReAct: Thought + Action + Observation döngüsü."""
    def __init__(self):
        super().__init__("ReAct")
        self._basar_prob = {
            "mantık":0.88,"matematik":0.92,"bilgi":0.90,
            "kodlama":0.88,"planlama":0.85,"yaratıcı":0.78
        }
    def cevapla(self, sorgu, kategori, dogru_cevap):
        dongu_sayisi = random.randint(2, 5)
        adim_sayisi  = dongu_sayisi * 3  # T+A+O her döngü
        sure         = adim_sayisi * random.uniform(0.03, 0.08)
        time.sleep(0.003)
        # Araç kullanımı başarıyı artırır ama yavaşlatır
        dogru = random.random() < self._temel_prob(kategori)
        return Cevap(self.ad, kategori, sorgu[:40],
                     dogru_cevap if dogru else "Araç hatası.",
                     dogru, sure, adim_sayisi=adim_sayisi,
                     aciklama=f"{dongu_sayisi} ReAct döngüsü")

class ReflexionStrateji(BaseStrateji):
    """Reflexion: öz-değerlendirme ve iteratif iyileştirme."""
    def __init__(self):
        super().__init__("Reflexion")
        self._basar_prob = {
            "mantık":0.91,"matematik":0.93,"bilgi":0.88,
            "kodlama":0.92,"planlama":0.90,"yaratıcı":0.85
        }
    def cevapla(self, sorgu, kategori, dogru_cevap):
        yineleme = random.randint(1, 3)
        adim_sayisi = yineleme * random.randint(4, 7)
        sure = adim_sayisi * random.uniform(0.04, 0.09)
        time.sleep(0.003)
        # Refleksion hataları yakalar ve düzeltir
        dogru = random.random() < self._temel_prob(kategori)
        return Cevap(self.ad, kategori, sorgu[:40],
                     dogru_cevap if dogru else "Düzeltildi.",
                     dogru, sure, adim_sayisi=adim_sayisi,
                     aciklama=f"{yineleme} refleksion turu")

STRATEJILER = {
    "ZeroShot":  ZeroShotStrateji(),
    "FewShot":   FewShotStrateji(),
    "CoT":       CoTStrateji(),
    "ReAct":     ReactStrateji(),
    "Reflexion": ReflexionStrateji(),
}
for ad in STRATEJILER:
    print(f"  ✅ {ad}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: BENCHMARK GÖREV SETİ
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 2: Benchmark Görev Seti")
print("─" * 33)

GOREVLER_DB = {
    "mantık": [
        ("Tüm insanlar ölümlüdür. Sokrates insandır. Sonuç?",
         "Sokrates ölümlüdür."),
        ("A > B, B > C ise A ile C ilişkisi?",
         "A > C (geçişkenlik)"),
        ("p VE q doğruysa, p VEYA r'nin değeri?",
         "p VEYA r doğru (p doğru olduğu için)"),
    ],
    "matematik": [
        ("1'den 100'e kadar tek sayıların toplamı?",
         "2500"),
        ("log₂(8) = ?",
         "3"),
        ("3x + 7 = 22 ise x = ?",
         "5"),
    ],
    "bilgi": [
        ("GPT-3'ün kaç parametresi var?",
         "175 milyar"),
        ("Transformer mimarisi ne zaman yayınlandı?",
         "2017"),
        ("Python'u kim geliştirdi?",
         "Guido van Rossum"),
    ],
    "kodlama": [
        ("Python'da list comprehension ile kareler listesi?",
         "[x**2 for x in range(n)]"),
        ("O(1) arama karmaşıklığı hangi yapıda?",
         "Hash tablosu / Dictionary"),
        ("Decorator pattern Python'da nasıl uygulanır?",
         "@fonksiyon sözdizimi"),
    ],
    "planlama": [
        ("Web scraper projesi için adımları listele.",
         "Hedef URL → BeautifulSoup → Parser → Veri kaydet"),
        ("ML pipeline nasıl tasarlanır?",
         "Veri → Ön işleme → Model → Değerlendirme → Deploy"),
        ("Bir API projesi için minimum mimari?",
         "Endpoint → İş mantığı → Veritabanı → Cache → Log"),
    ],
    "yaratıcı": [
        ("'Yapay zeka' için yenilikçi bir metafor yaz.",
         "Açık uçlu yaratıcı yanıt"),
        ("Chatbot için 3 farklı kişilik tasarla.",
         "Açık uçlu tasarım"),
        ("AI ürünü için dikkat çekici slogan öner.",
         "Açık uçlu slogan"),
    ],
}

toplam_gorev = sum(len(v) for v in GOREVLER_DB.values())
print(f"  {'Kategori':<14} {'Görev Sayısı'}")
print("  " + "─" * 28)
for kat, gorevler in GOREVLER_DB.items():
    print(f"  {kat:<14} {len(gorevler)}")
print(f"  {'TOPLAM':<14} {toplam_gorev}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: BENCHMARK ÇALIŞTIRMA
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 3: Benchmark Çalıştırma")
print("─" * 33)

print(f"  {'Strateji':<12} ", end="")
for kat in GOREVLER_DB: print(f"{kat[:7]:<8}", end="")
print(f"{'ORTALAMA':>9}")
print("  " + "─" * 70)

for ad, strateji in STRATEJILER.items():
    kat_basari = {}
    for kategori, gorevler in GOREVLER_DB.items():
        for sorgu, dogru in gorevler:
            c = strateji.cevapla(sorgu, kategori, dogru)
            strateji.cevaplar.append(c)
        kat_bas = sum(1 for c in strateji.cevaplar
                      if c.kategori == kategori and c.dogru) / len(gorevler)
        kat_basari[kategori] = kat_bas
    ist = strateji.istatistik()
    print(f"  {ad:<12} ", end="")
    for kat in GOREVLER_DB:
        print(f"{kat_basari[kat]*100:6.0f}%  ", end="")
    print(f"{ist['basari']*100:8.0f}%")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: ABLASYON ÇALIŞMASI
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 4: Ablasyon Çalışması")
print("─" * 33)

def ablasyon_test(parametre: str, degerler: list,
                  gürültü: float = 0.04) -> Dict:
    sonuclar = {}
    # Temel başarı: ReAct stratejisi için
    temel_basari = {
        "araç_sayısı": [0.62, 0.75, 0.85, 0.90, 0.91],
        "adım_limiti": [0.50, 0.70, 0.85, 0.90, 0.88],
        "sıcaklık":    [0.92, 0.90, 0.85, 0.75, 0.55],
        "bellek":      [0.68, 0.79, 0.87, 0.90, 0.91],
    }
    temel = temel_basari.get(parametre, [0.7]*len(degerler))
    sonuclar[parametre] = {
        "degerler": degerler,
        "basari":   [min(1.0, max(0, b + random.uniform(-gürültü, gürültü)))
                     for b in temel],
        "sure":     [random.uniform(0.1, 2.0) for _ in degerler],
    }
    return sonuclar[parametre]

abl_sonuclar = {}
for param, degerler in [
    ("araç_sayısı", [0, 2, 4, 6, 8]),
    ("adım_limiti", [2, 4, 6, 8, 12]),
    ("sıcaklık",    [0.0, 0.3, 0.6, 0.9, 1.2]),
    ("bellek",      [0, 512, 2048, 8192, 32768]),
]:
    abl_sonuclar[param] = ablasyon_test(param, degerler)
    ort_bas = sum(abl_sonuclar[param]["basari"]) / len(degerler)
    print(f"  {param:<15} → Ort. başarı: {ort_bas:.2f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: HATA ANALİZİ
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 5: Hata Analizi")
print("─" * 33)

HATA_TIPLERI = {
    "Bilgi Eksikliği":    {"oran":0.28, "renk":"#EF4444"},
    "Yanlış Akıl Yürütme":{"oran":0.22, "renk":"#F97316"},
    "Araç Başarısızlığı": {"oran":0.18, "renk":"#F59E0B"},
    "Bağlam Kaybı":       {"oran":0.15, "renk":"#8B5CF6"},
    "Aşırı Üretim":       {"oran":0.10, "renk":"#3B82F6"},
    "Diğer":              {"oran":0.07, "renk":"#64748B"},
}

# Strateji bazlı hata dağılımı (simüle)
hata_matris = {
    "ZeroShot":  [35, 28, 10, 12, 8, 7],
    "FewShot":   [28, 24, 12, 10, 9, 7],
    "CoT":       [18, 22, 10,  8, 6, 6],
    "ReAct":     [12, 15, 20,  5, 4, 4],
    "Reflexion": [ 8, 12, 15,  4, 3, 3],
}

print(f"  {'Strateji':<12} ", end="")
for ht in HATA_TIPLERI: print(f"{ht[:8]:<10}", end="")
print()
print("  " + "─" * 65)
for strateji, sayilar in hata_matris.items():
    toplam = sum(sayilar)
    print(f"  {strateji:<12} ", end="")
    for s in sayilar:
        print(f"{s/toplam*100:7.0f}%   ", end="")
    print()

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 6: GÜVENİLİRLİK TESTİ (Varyans Analizi)
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 6: Güvenilirlik Testi (N=20 Tekrar)")
print("─" * 33)

guven_testi = {}
TEST_SORGU  = "GPT-4 ile Claude 3 farkı nedir?"
N_TEKRAR    = 20

for ad, strateji in STRATEJILER.items():
    sonuclar = [random.random() < strateji._temel_prob("bilgi")
                for _ in range(N_TEKRAR)]
    oran     = sum(sonuclar) / N_TEKRAR
    std      = math.sqrt(oran*(1-oran)/N_TEKRAR)
    guven_testi[ad] = {"oran": oran, "std": std, "sonuclar": sonuclar}
    print(f"  {ad:<12} Başarı: {oran:.2f} ± {std:.3f}  "
          f"{'█' * int(oran*20)}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 7: GÖRSELLEŞTİRME (8 PANEL)
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 7: Görselleştirme (8 panel)")
print("─" * 33)

BG = "#0D1117"; CARD = "#161B22"; GRID = "#21262D"; TEXT = "#C9D1D9"; MUTED = "#8B949E"
STRAT_RENKLER = {
    "ZeroShot":"#64748B","FewShot":"#1A6FD8",
    "CoT":"#F5A623","ReAct":"#10C98F","Reflexion":"#7B52E8"
}
KAT_RENKLER = {
    "mantık":"#1A6FD8","matematik":"#0FBCCE","bilgi":"#7B52E8",
    "kodlama":"#F5A623","planlama":"#10C98F","yaratıcı":"#E879A0"
}

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.38,
                         top=0.93, bottom=0.05)

# ── G1: Strateji × Kategori ısı haritası ─────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(CARD)
kategoriler_lst = list(GOREVLER_DB.keys())
strat_lst       = list(STRATEJILER.keys())
matris1 = np.zeros((len(strat_lst), len(kategoriler_lst)))
for i, (ad, strateji) in enumerate(STRATEJILER.items()):
    for j, kat in enumerate(kategoriler_lst):
        kat_cevaplar = [c for c in strateji.cevaplar if c.kategori == kat]
        if kat_cevaplar:
            matris1[i,j] = sum(c.dogru for c in kat_cevaplar) / len(kat_cevaplar)
im1 = ax1.imshow(matris1, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax1.set_xticks(range(len(kategoriler_lst)))
ax1.set_xticklabels(kategoriler_lst, fontsize=10, color=TEXT)
ax1.set_yticks(range(len(strat_lst)))
ax1.set_yticklabels(strat_lst, fontsize=10, color=TEXT)
for i in range(len(strat_lst)):
    for j in range(len(kategoriler_lst)):
        ax1.text(j, i, f"{matris1[i,j]*100:.0f}%",
                 ha="center", va="center", fontsize=9.5,
                 color="white" if matris1[i,j]<0.6 else "black")
plt.colorbar(im1, ax=ax1, fraction=0.03)
ax1.set_title("Strateji × Kategori Başarı Haritası (%)", fontsize=12,
              fontweight="bold", color=TEXT, pad=8)
ax1.tick_params(colors=MUTED)
for sp in ax1.spines.values(): sp.set_color(GRID)

# ── G2: Latency-Accuracy scatter ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(CARD)
for ad, strateji in STRATEJILER.items():
    ist = strateji.istatistik()
    ax2.scatter(ist["ort_sure"]*1000, ist["basari"]*100,
                s=250, color=STRAT_RENKLER[ad], zorder=5,
                edgecolors="#30363D", linewidth=1.5, label=ad)
    ax2.annotate(ad, (ist["ort_sure"]*1000, ist["basari"]*100),
                 textcoords="offset points", xytext=(6,4),
                 fontsize=8.5, color=TEXT)
ax2.set_title("Latency – Accuracy\nTrade-off", fontsize=12, fontweight="bold",
              color=TEXT, pad=8)
ax2.set_xlabel("Ort. Süre (ms)", fontsize=10, color=MUTED)
ax2.set_ylabel("Başarı Oranı (%)", fontsize=10, color=MUTED)
ax2.tick_params(colors=MUTED)
ax2.grid(alpha=0.3, color=GRID)
for sp in ax2.spines.values(): sp.set_color(GRID)

# ── G3: Ablasyon — araç sayısı ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD)
a3 = abl_sonuclar["araç_sayısı"]
ax3.plot(a3["degerler"], [b*100 for b in a3["basari"]],
         "o-", color="#10C98F", lw=2.5, ms=8, markerfacecolor="#0D9488")
ax3.fill_between(a3["degerler"], [b*100-3 for b in a3["basari"]],
                 [b*100+3 for b in a3["basari"]], alpha=0.15, color="#10C98F")
ax3.set_title("Ablasyon: Araç Sayısı\nvs Başarı Oranı", fontsize=12,
              fontweight="bold", color=TEXT, pad=8)
ax3.set_xlabel("Araç Sayısı", fontsize=10, color=MUTED)
ax3.set_ylabel("Başarı (%)", fontsize=10, color=MUTED)
ax3.tick_params(colors=MUTED)
ax3.grid(alpha=0.3, color=GRID)
for sp in ax3.spines.values(): sp.set_color(GRID)

# ── G4: Ablasyon — sıcaklık ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD)
a4 = abl_sonuclar["sıcaklık"]
ax4.plot(a4["degerler"], [b*100 for b in a4["basari"]],
         "s-", color="#F5A623", lw=2.5, ms=8)
ax4.axvspan(0, 0.3, alpha=0.1, color="#10C98F", label="Deterministik")
ax4.axvspan(0.3, 0.8, alpha=0.1, color="#1A6FD8", label="Dengeli")
ax4.axvspan(0.8, 1.2, alpha=0.1, color="#EF4444", label="Yaratıcı")
ax4.set_title("Ablasyon: Sıcaklık (Temperature)\nvs Başarı Oranı", fontsize=12,
              fontweight="bold", color=TEXT, pad=8)
ax4.set_xlabel("Sıcaklık", fontsize=10, color=MUTED)
ax4.set_ylabel("Başarı (%)", fontsize=10, color=MUTED)
ax4.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD)
ax4.tick_params(colors=MUTED)
ax4.grid(alpha=0.3, color=GRID)
for sp in ax4.spines.values(): sp.set_color(GRID)

# ── G5: Hata türü dağılımı (yığılmış çubuk) ──────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(CARD)
hata_renk_listesi = [v["renk"] for v in HATA_TIPLERI.values()]
hata_etiketler    = list(HATA_TIPLERI.keys())
strat_isimler     = list(hata_matris.keys())
alt = np.zeros(len(strat_isimler))
for j, (ht, renk) in enumerate(zip(hata_etiketler, hata_renk_listesi)):
    degerler = [hata_matris[s][j] for s in strat_isimler]
    ax5.bar(strat_isimler, degerler, bottom=alt,
            color=renk, label=ht[:12], edgecolor=BG, linewidth=0.5)
    alt += np.array(degerler)
ax5.set_title("Strateji Başına Hata\nTürü Dağılımı", fontsize=12,
              fontweight="bold", color=TEXT, pad=8)
ax5.set_ylabel("Hata Sayısı", fontsize=10, color=MUTED)
ax5.legend(fontsize=7.5, labelcolor=TEXT, facecolor=CARD,
           loc="upper right", bbox_to_anchor=(1.55, 1))
ax5.tick_params(colors=TEXT, rotation=15)
ax5.set_facecolor(CARD)
for sp in ax5.spines.values(): sp.set_color(GRID)

# ── G6: Güvenilirlik — hata çubukları ───────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_facecolor(CARD)
strat_isimler6 = list(guven_testi.keys())
oranlar6 = [guven_testi[s]["oran"]*100 for s in strat_isimler6]
std6     = [guven_testi[s]["std"]*100  for s in strat_isimler6]
renkler6 = [STRAT_RENKLER[s] for s in strat_isimler6]
ax6.bar(strat_isimler6, oranlar6, yerr=std6, color=renkler6,
        capsize=7, edgecolor=GRID, alpha=0.85,
        error_kw=dict(elinewidth=2, ecolor=TEXT, capthick=2))
ax6.set_title(f"Güvenilirlik Testi\n(N={N_TEKRAR} tekrar, hata çubukları)", fontsize=12,
              fontweight="bold", color=TEXT, pad=8)
ax6.set_ylabel("Başarı Oranı (%)", fontsize=10, color=MUTED)
ax6.tick_params(colors=TEXT, rotation=15)
ax6.grid(axis="y", alpha=0.3, color=GRID)
for sp in ax6.spines.values(): sp.set_color(GRID)

# ── G7: Radar — çok boyutlu performans ───────────────────────
ax7 = fig.add_subplot(gs[2, 1], projection="polar")
ax7.set_facecolor(CARD)
radar_boyutlar = ["Doğruluk","Hız","Güvenilirlik","Açıklama","Verimlilik"]
N7 = len(radar_boyutlar)
acılar7 = [n/float(N7)*2*np.pi for n in range(N7)] + [0]
radar_deger = {
    "ZeroShot":  [0.65, 0.96, 0.60, 0.35, 0.92],
    "FewShot":   [0.74, 0.88, 0.70, 0.55, 0.82],
    "CoT":       [0.83, 0.72, 0.80, 0.88, 0.70],
    "ReAct":     [0.90, 0.60, 0.88, 0.82, 0.60],
    "Reflexion": [0.93, 0.50, 0.94, 0.90, 0.52],
}
for ad, deger in radar_deger.items():
    d = deger + [deger[0]]
    ax7.plot(acılar7, d, "o-", color=STRAT_RENKLER[ad], lw=2, ms=4, label=ad)
    ax7.fill(acılar7, d, alpha=0.07, color=STRAT_RENKLER[ad])
ax7.set_xticks(acılar7[:-1])
ax7.set_xticklabels(radar_boyutlar, fontsize=8.5, color=TEXT)
ax7.set_ylim(0,1)
ax7.set_title("Çok Boyutlu Performans\nRadarı", fontsize=11,
              fontweight="bold", color=TEXT, pad=20)
ax7.tick_params(colors=MUTED)
ax7.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD,
           loc="lower right", bbox_to_anchor=(1.45, -0.05))
ax7.set_facecolor(CARD)

# ── G8: Bellek ablasyonu ──────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(CARD)
a8 = abl_sonuclar["bellek"]
bellek_kisa = ["0","512","2K","8K","32K"]
ax8.semilogx([max(d,1) for d in a8["degerler"]],
             [b*100 for b in a8["basari"]],
             "D-", color="#E879A0", lw=2.5, ms=8)
ax8.fill_between([max(d,1) for d in a8["degerler"]],
                 [b*100-2.5 for b in a8["basari"]],
                 [b*100+2.5 for b in a8["basari"]],
                 alpha=0.15, color="#E879A0")
ax8.set_xscale("log"); ax8.set_xticks([1,512,2048,8192,32768])
ax8.set_xticklabels(bellek_kisa, fontsize=9, color=TEXT)
ax8.set_title("Ablasyon: Bellek Boyutu\nvs Başarı Oranı", fontsize=12,
              fontweight="bold", color=TEXT, pad=8)
ax8.set_xlabel("Bellek Boyutu (token)", fontsize=10, color=MUTED)
ax8.set_ylabel("Başarı (%)", fontsize=10, color=MUTED)
ax8.tick_params(colors=MUTED)
ax8.grid(alpha=0.3, color=GRID, which="both")
for sp in ax8.spines.values(): sp.set_color(GRID)

fig.suptitle(
    "AGENTİK AI — UYGULAMA 04  |  Agent Değerlendirme & Kıyaslama\n"
    "ZeroShot · FewShot · CoT · ReAct · Reflexion  ×  6 Kategori  ×  Ablasyon",
    fontsize=14, fontweight="bold", color=TEXT, y=0.98
)
plt.savefig("agentic_04_degerlendirme.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ agentic_04_degerlendirme.png kaydedildi")
plt.close()

print()
print("=" * 65)
print("  ÖZET")
print(f"  Strateji sayısı    : {len(STRATEJILER)}")
print(f"  Kategori sayısı    : {len(GOREVLER_DB)}")
print(f"  Toplam değerlendirme: {sum(len(s.cevaplar) for s in STRATEJILER.values())}")
print(f"  Güvenilirlik N     : {N_TEKRAR}")
print(f"  Ablasyon parametresi: {len(abl_sonuclar)}")
print()
print(f"  {'Strateji':<12} {'Başarı':>8} {'Ort Süre':>10} {'Ort Adım':>10}")
print("  " + "─" * 44)
for ad, s in STRATEJILER.items():
    ist = s.istatistik()
    print(f"  {ad:<12} {ist['basari']*100:7.1f}%  {ist['ort_sure']*1000:9.1f}ms  {ist['ort_adim']:9.1f}")
print()
print(f"  Grafik             : agentic_04_degerlendirme.png")
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("=" * 65)
