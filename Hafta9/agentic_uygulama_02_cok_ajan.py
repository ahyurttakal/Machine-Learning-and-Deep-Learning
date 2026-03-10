"""
=============================================================================
AGENTİK AI — UYGULAMA 02
Çok-Ajan Sistemi (Multi-Agent System) — Orkestratör + Uzman Ajanlar
=============================================================================
Kapsam:
  - Orkestratör ajan: görevi alt-görevlere böler, ajanları yönlendirir
  - 5 uzman ajan: Araştırmacı, Kodlayıcı, Analist, Yazıcı, Doğrulayıcı
  - AjanMesaj dataclass — mesajlaşma protokolü
  - AutoGen benzeri ConversableAgent mimarisi
  - Görev kuyruğu, bağımlılık grafiği, öncelik sırası
  - Ajan arası iletişim logu ve ısı haritası
  - Gantt şeması, radar analizi, performans metrikleri
  - 8-panel görselleştirme
=============================================================================
"""

import time, random, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import List, Dict, Optional
from enum import Enum
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  AGENTİK AI — UYGULAMA 02")
print("  Çok-Ajan Sistemi (Multi-Agent System)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: PROTOKOL VE ROL TANIMLARI
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 1: Mesaj Protokolü ve Ajan Rolleri")
print("─" * 65)

class AjanRol(Enum):
    ORKESTRATOR = "🎯 Orkestratör"
    ARASTIRMACI = "🔍 Araştırmacı"
    KODLAYICI   = "💻 Kodlayıcı"
    ANALIST     = "📊 Analist"
    YAZICI      = "✍️  Yazıcı"
    DOGRULAYICI = "✅ Doğrulayıcı"

@dataclass
class AjanMesaj:
    gonderen : str
    alici    : str
    icerik   : str
    tur      : str   = "bilgi"   # bilgi | gorev | sonuc | hata | onay
    oncelik  : int   = 2
    zaman    : float = field(default_factory=time.time)
    meta     : dict  = field(default_factory=dict)
    mesaj_id : str   = field(default_factory=lambda: f"MSG-{random.randint(1000,9999)}")

@dataclass
class AltGorev:
    gorev_id    : str
    aciklama    : str
    atanan_ajan : str
    oncelik     : int = 2
    bagli       : List[str] = field(default_factory=list)
    durum       : str = "bekliyor"
    sonuc       : Optional[str] = None
    baslangic   : float = field(default_factory=time.time)
    bitis       : Optional[float] = None

for rol in AjanRol:
    print(f"  {rol.value}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: AJAN SINIFI HİYERARŞİSİ
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 2: ConversableAgent Sınıfları")
print("─" * 65)

class ConversableAgent:
    def __init__(self, ad, rol, sistem_msg, verbose=False):
        self.ad, self.rol, self.verbose = ad, rol, verbose
        self.sistem_msg  = sistem_msg
        self.gelen       = deque()
        self.giden_log   = []
        self.tamamlanan  = 0
        self.hata_sayisi = 0
        self.toplam_sure = 0.0

    def mesaj_gonder(self, alici, icerik, tur="bilgi", meta=None):
        m = AjanMesaj(self.ad, alici, icerik, tur, meta=meta or {})
        self.giden_log.append(m); return m

    def _sim(self, mn=0.05, mx=0.25):
        t = random.uniform(mn, mx); time.sleep(t); return t

    def isle(self, gorev: AltGorev) -> str:
        raise NotImplementedError


class ArastirmaciAjan(ConversableAgent):
    def isle(self, g):
        self.toplam_sure += self._sim(0.10, 0.30)
        kb = {"python":"Python 1991, GVR. OOP, yorumlanan.",
              "ai":"AI evrimi: Kural→ML→DL→LLM→Agentic.",
              "rag":"RAG: Retrieval-Augmented Generation — vektör DB + LLM.",
              "agent":"Agent: Algıla→Planla→Hareket döngüsü."}
        for k, v in kb.items():
            if k in g.aciklama.lower():
                self.tamamlanan += 1
                return f"Araştırma: {v}"
        self.tamamlanan += 1
        return f"Araştırma tamamlandı: '{g.aciklama[:35]}' kaynaklar derlendi."


class KodlayiciAjan(ConversableAgent):
    def isle(self, g):
        self.toplam_sure += self._sim(0.15, 0.40)
        sablonlar = {
            "api":    "def api_cagri(url):\n    import requests\n    return requests.get(url).json()",
            "analiz": "import pandas as pd\ndf = pd.read_csv('veri.csv')\nprint(df.describe())",
            "sinif":  "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100)",
        }
        for k, v in sablonlar.items():
            if k in g.aciklama.lower():
                self.tamamlanan += 1
                return f"Kod üretildi:\n```python\n{v}\n```"
        self.tamamlanan += 1
        return "Kod üretildi:\n```python\ndef coz(x):\n    return {'sonuc': x}\n```"


class AnalistAjan(ConversableAgent):
    def isle(self, g):
        self.toplam_sure += self._sim(0.10, 0.35)
        np.random.seed(42); v = np.random.randn(50)
        self.tamamlanan += 1
        return f"Analiz: n=50, ort={v.mean():.2f}, std={v.std():.2f}, max={v.max():.2f}"


class YaziciAjan(ConversableAgent):
    def isle(self, g):
        self.toplam_sure += self._sim(0.12, 0.28)
        self.tamamlanan += 1
        return f"Rapor taslağı: '{g.aciklama[:30]}' — 3 bölüm (Giriş, Analiz, Sonuç)"


class DogrulayiciAjan(ConversableAgent):
    def isle(self, g):
        self.toplam_sure += self._sim(0.05, 0.20)
        if random.random() > 0.15:
            self.tamamlanan += 1
            return "✅ Doğrulama geçti: Çıktı tutarlı."
        self.hata_sayisi += 1
        return "⚠️ Doğrulama uyarısı: Revizyon gerekli."


print("  ✅ ArastirmaciAjan, KodlayiciAjan, AnalistAjan, YaziciAjan, DogrulayiciAjan")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: ORKESTRATÖRaj
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 3: Orkestratör — Koordinasyon Motoru")
print("─" * 65)

class OrkestratörAjan(ConversableAgent):
    def __init__(self, verbose=True):
        super().__init__("Orkestratör", AjanRol.ORKESTRATOR, "Koordine et.", verbose)
        self.ajanlar : Dict[str, ConversableAgent] = {
            "Araştırmacı": ArastirmaciAjan("Araştırmacı", AjanRol.ARASTIRMACI, "Araştır"),
            "Kodlayıcı"  : KodlayiciAjan  ("Kodlayıcı",   AjanRol.KODLAYICI,   "Kodla"),
            "Analist"    : AnalistAjan    ("Analist",     AjanRol.ANALIST,     "Analiz et"),
            "Yazıcı"     : YaziciAjan     ("Yazıcı",      AjanRol.YAZICI,      "Yaz"),
            "Doğrulayıcı": DogrulayiciAjan("Doğrulayıcı", AjanRol.DOGRULAYICI, "Doğrula"),
        }
        self.iletisim_mat = defaultdict(int)
        self.gorev_log   : List[AltGorev] = []
        self.mesaj_log   : List[AjanMesaj] = []
        self._sayac = 0

    def _kaydet(self, m: AjanMesaj):
        self.mesaj_log.append(m)
        self.iletisim_mat[(m.gonderen, m.alici)] += 1

    def _alt_gorevler(self, gorev: str) -> List[AltGorev]:
        self._sayac += 1; p = f"AG{self._sayac}"
        return [
            AltGorev(f"{p}-1", f"Araştır: {gorev[:38]}",   "Araştırmacı", oncelik=3),
            AltGorev(f"{p}-2", f"Kod üret: {gorev[:38]}",  "Kodlayıcı",   oncelik=2, bagli=[f"{p}-1"]),
            AltGorev(f"{p}-3", f"Analiz et: {gorev[:38]}", "Analist",     oncelik=2, bagli=[f"{p}-1"]),
            AltGorev(f"{p}-4", f"Rapor yaz: {gorev[:38]}", "Yazıcı",      oncelik=1, bagli=[f"{p}-2",f"{p}-3"]),
            AltGorev(f"{p}-5", f"Doğrula: {gorev[:38]}",   "Doğrulayıcı", oncelik=3, bagli=[f"{p}-4"]),
        ]

    def calistir(self, ana_gorev: str) -> dict:
        t0 = time.time()
        if self.verbose: print(f"\n  🎯 GÖREV: {ana_gorev}")
        alt = self._alt_gorevler(ana_gorev)
        self.gorev_log.extend(alt)

        for a in alt:
            m = self.mesaj_gonder(a.atanan_ajan, f"Görev: {a.aciklama}", "gorev")
            self._kaydet(m)
            if self.verbose: print(f"  📤 {a.atanan_ajan:<16} ← {a.aciklama[:45]}")

        tamamlandi = set(); sonuclar = {}
        for _ in range(4):
            for a in alt:
                if a.durum == "tamamlandi": continue
                if all(b in tamamlandi for b in a.bagli):
                    a.durum = "devam"
                    a.sonuc = self.ajanlar[a.atanan_ajan].isle(a)
                    a.durum = "tamamlandi"; a.bitis = time.time()
                    tamamlandi.add(a.gorev_id); sonuclar[a.gorev_id] = a.sonuc
                    m2 = AjanMesaj(a.atanan_ajan,"Orkestratör",
                                   f"Tamam [{a.gorev_id}]: {a.sonuc[:40]}","sonuc")
                    self._kaydet(m2)
                    if self.verbose: print(f"  ✅ {a.atanan_ajan:<16} → {a.sonuc[:55]}")

        sure = time.time()-t0; self.tamamlanan += 1
        if self.verbose: print(f"\n  ⏱️  Süre: {sure:.3f}s | "
                               f"{len(tamamlandi)}/{len(alt)} alt-görev tamamlandı")
        return {"gorev":ana_gorev,"alt_gorevler":alt,"sonuclar":sonuclar,
                "sure":sure,"tamamlanan":len(tamamlandi)}

ork = OrkestratörAjan(verbose=True)
print("  ✅ OrkestratörAjan + 5 uzman ajan başlatıldı")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: GÖREV ÇALIŞTIRIM
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 4: Görev Simülasyonları")
print("─" * 65)

GOREVLER = [
    "Python ile makine öğrenmesi uygulaması geliştir ve belgele",
    "E-ticaret veri seti analizi ve tahmin modeli oluştur",
    "RAG sistemi için API entegrasyonu pipeline inşa et",
    "Müşteri memnuniyet raporu: analiz, görselleştirme, öneri",
]
tum_sonuclar = [ork.calistir(g) for g in GOREVLER]

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: METRİKLER
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 5: Performans Metrikleri")
print("─" * 65)

print(f"  Toplam görev    : {len(GOREVLER)}")
print(f"  Alt-görev say.  : {len(ork.gorev_log)}")
print(f"  Toplam mesaj    : {len(ork.mesaj_log)}")
print(f"\n  {'Ajan':<16} {'Tamamlanan':>12} {'Hata':>6} {'Ort ms':>9}")
print("  " + "─" * 48)
for ad, a in ork.ajanlar.items():
    ort = a.toplam_sure / max(a.tamamlanan,1) * 1000
    print(f"  {ad:<16} {a.tamamlanan:>12} {a.hata_sayisi:>6} {ort:>8.1f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 6: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 6: Görselleştirme (8 panel)")
print("─" * 65)

AJAN_ADLARI = ["Orkestratör","Araştırmacı","Kodlayıcı","Analist","Yazıcı","Doğrulayıcı"]
AJAN_RENK   = {"Orkestratör":"#F59E0B","Araştırmacı":"#3B82F6","Kodlayıcı":"#10B981",
               "Analist":"#A78BFA","Yazıcı":"#F472B6","Doğrulayıcı":"#34D399"}
N = len(AJAN_ADLARI)

# İletişim matrisi
imat = np.zeros((N, N))
for (g, a), cnt in ork.iletisim_mat.items():
    gi = AJAN_ADLARI.index(g) if g in AJAN_ADLARI else 0
    ai = AJAN_ADLARI.index(a) if a in AJAN_ADLARI else 0
    imat[gi, ai] += cnt

uzman_adlar = list(ork.ajanlar.keys())

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22,18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.44,wspace=0.36,top=0.93,bottom=0.05)

# G1 — İletişim ısı haritası
ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor("#161B22")
im  = ax1.imshow(imat, cmap="YlOrRd", aspect="auto")
ax1.set_xticks(range(N)); ax1.set_xticklabels([a[:6] for a in AJAN_ADLARI],
    rotation=45,ha="right",fontsize=8.5,color="#C9D1D9")
ax1.set_yticks(range(N)); ax1.set_yticklabels([a[:9] for a in AJAN_ADLARI],
    fontsize=8.5,color="#C9D1D9")
for i in range(N):
    for j in range(N):
        if imat[i,j]>0: ax1.text(j,i,int(imat[i,j]),ha="center",va="center",
                                  fontsize=10,color="black")
plt.colorbar(im,ax=ax1,shrink=0.85)
ax1.set_title("Ajan İletişim Matrisi",fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)

# G2 — İş yükü
ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor("#161B22")
tam = [ork.ajanlar[a].tamamlanan for a in uzman_adlar]
ren2= [AJAN_RENK.get(a,"#64748B") for a in uzman_adlar]
ax2.bar(uzman_adlar,tam,color=ren2,edgecolor="#30363D",alpha=0.88)
ax2.set_title("Ajan İş Yükü\n(Tamamlanan)",fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)
ax2.set_ylabel("Görev",fontsize=9,color="#8B949E")
ax2.tick_params(colors="#8B949E",axis="x",rotation=20)
ax2.grid(axis="y",alpha=0.3,color="#30363D")
ax2.set_facecolor("#161B22")
for sp in ax2.spines.values(): sp.set_color("#30363D")

# G3 — Görev süreleri
ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor("#161B22")
ax3.barh([f"G{i+1}" for i in range(len(tum_sonuclar))],
         [s["sure"] for s in tum_sonuclar],
         color="#0FBCCE",edgecolor="#30363D",alpha=0.85)
ax3.set_title("Ana Görev Süreleri",fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)
ax3.set_xlabel("Saniye",fontsize=9,color="#8B949E")
ax3.tick_params(colors="#8B949E")
ax3.grid(axis="x",alpha=0.3,color="#30363D")
ax3.set_facecolor("#161B22")
for sp in ax3.spines.values(): sp.set_color("#30363D")

# G4 — Bağımlılık grafiği
ax4 = fig.add_subplot(gs[1,:2]); ax4.set_facecolor("#161B22")
ax4.set_xlim(-0.5,4.5); ax4.set_ylim(-0.5,2.5); ax4.axis("off")
ax4.set_title("Alt-Görev Bağımlılık Grafiği — Görev 1",
              fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)
ag1 = tum_sonuclar[0]["alt_gorevler"]
pos = {ag1[0].gorev_id:(0.5,2.0), ag1[1].gorev_id:(1.5,1.0),
       ag1[2].gorev_id:(2.5,1.0), ag1[3].gorev_id:(2.0,0.0), ag1[4].gorev_id:(3.5,0.0)}
dr  = {"tamamlandi":"#22C55E","bekliyor":"#64748B","hata":"#EF4444","devam":"#F59E0B"}
for ag in ag1:
    x,y = pos[ag.gorev_id]
    ax4.add_patch(mpatches.FancyBboxPatch((x-.45,y-.32),.9,.64,
        boxstyle="round,pad=0.06",facecolor=dr.get(ag.durum,"#64748B"),
        edgecolor="#0D1117",linewidth=2,alpha=0.85))
    ax4.text(x,y+.12,ag.atanan_ajan[:10],ha="center",va="center",
             fontsize=9,color="white",fontweight="bold")
    ax4.text(x,y-.12,ag.gorev_id,ha="center",va="center",fontsize=7.5,color="#D1D5DB")
    for b in ag.bagli:
        if b in pos:
            x0,y0=pos[b]
            ax4.annotate("",xy=(x-.02,y+.33),xytext=(x0+.02,y0-.33),
                arrowprops=dict(arrowstyle="->",color="#3B82F6",lw=1.8,mutation_scale=14))
ax4.legend(handles=[mpatches.Patch(facecolor=v,label=k.capitalize())
    for k,v in dr.items()],loc="lower right",fontsize=9,
    labelcolor="#C9D1D9",facecolor="#161B22")

# G5 — Ortalama süre
ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor("#161B22")
ort_ms = [ork.ajanlar[a].toplam_sure/max(ork.ajanlar[a].tamamlanan,1)*1000 for a in uzman_adlar]
ax5.bar(uzman_adlar,ort_ms,color=ren2,edgecolor="#30363D",alpha=0.88)
ax5.set_title("Ort. İşlem Süresi\n(ms/görev)",fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)
ax5.set_ylabel("ms",fontsize=9,color="#8B949E")
ax5.tick_params(colors="#8B949E",axis="x",rotation=20)
ax5.grid(axis="y",alpha=0.3,color="#30363D")
ax5.set_facecolor("#161B22")
for sp in ax5.spines.values(): sp.set_color("#30363D")

# G6 — Mesaj türü
ax6 = fig.add_subplot(gs[2,0]); ax6.set_facecolor("#161B22")
tc = defaultdict(int)
for m in ork.mesaj_log: tc[m.tur]+=1
ax6.pie(list(tc.values()),labels=list(tc.keys()),
    colors=["#F59E0B","#10B981","#3B82F6","#EF4444","#A78BFA"][:len(tc)],
    autopct="%1.0f%%",startangle=90,
    textprops={"fontsize":9,"color":"#C9D1D9"},
    wedgeprops={"edgecolor":"#0D1117","linewidth":2})
ax6.set_title("Mesaj Türü Dağılımı",fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)

# G7 — Radar
ax7 = fig.add_subplot(gs[2,1],projection="polar"); ax7.set_facecolor("#161B22")
cats  = ["Hız","Kalite","Güvenilirlik","Verimlilik","Esneklik"]
ang   = [n/float(len(cats))*2*np.pi for n in range(len(cats))]+[0]
radar_vals = {"Araştırmacı":[.80,.90,.88,.75,.85],"Kodlayıcı":[.70,.95,.90,.80,.92],
              "Analist":[.85,.88,.92,.90,.78],"Yazıcı":[.75,.85,.80,.85,.70],
              "Doğrulayıcı":[.95,.92,.97,.88,.80]}
radar_renk = ["#3B82F6","#10B981","#A78BFA","#F472B6","#34D399"]
for (ad,vals),ren in zip(radar_vals.items(),radar_renk):
    v = vals+[vals[0]]
    ax7.plot(ang,v,"o-",color=ren,linewidth=1.8,markersize=5,label=ad)
    ax7.fill(ang,v,alpha=0.08,color=ren)
ax7.set_xticks(ang[:-1]); ax7.set_xticklabels(cats,fontsize=8,color="#C9D1D9")
ax7.set_ylim(0,1); ax7.set_facecolor("#161B22")
ax7.legend(loc="upper right",bbox_to_anchor=(1.38,1.18),
           fontsize=8,labelcolor="#C9D1D9",facecolor="#161B22")
ax7.set_title("Ajan Performans Radar",fontsize=11,fontweight="bold",color="#C9D1D9",pad=20)
ax7.tick_params(colors="#8B949E")

# G8 — Gantt
ax8 = fig.add_subplot(gs[2,2]); ax8.set_facecolor("#161B22")
random.seed(7)
bas_zamanlar = [0,.25,.25,.55,.85]
for i,ag in enumerate(tum_sonuclar[0]["alt_gorevler"]):
    dur = random.uniform(.15,.35)
    ax8.barh(i,dur,left=bas_zamanlar[i],
             color=AJAN_RENK.get(ag.atanan_ajan,"#64748B"),
             edgecolor="#0D1117",height=0.7,alpha=0.88)
    ax8.text(bas_zamanlar[i]+dur/2,i,ag.atanan_ajan[:8],
             ha="center",va="center",fontsize=8,color="white",fontweight="bold")
ax8.set_yticks(range(5)); ax8.set_yticklabels([f"AG-{i+1}" for i in range(5)],
    fontsize=9,color="#C9D1D9")
ax8.set_title("Alt-Görev Gantt\nZaman Çizelgesi",fontsize=11,fontweight="bold",color="#C9D1D9",pad=8)
ax8.set_xlabel("Süre (s)",fontsize=9,color="#8B949E")
ax8.tick_params(colors="#8B949E")
ax8.grid(axis="x",alpha=0.3,color="#30363D")
ax8.set_facecolor("#161B22")
for sp in ax8.spines.values(): sp.set_color("#30363D")

fig.suptitle(
    "AGENTİK AI — UYGULAMA 02  |  Çok-Ajan Sistemi\n"
    "Orkestratör · Uzman Ajanlar · İletişim · Koordinasyon",
    fontsize=14,fontweight="bold",color="#C9D1D9",y=0.98)

plt.savefig("agentic_02_cok_ajan.png",dpi=150,bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ agentic_02_cok_ajan.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  Uzman ajan     : {len(ork.ajanlar)}")
print(f"  Toplam görev   : {len(GOREVLER)}")
print(f"  Alt-görev      : {len(ork.gorev_log)}")
print(f"  Toplam mesaj   : {len(ork.mesaj_log)}")
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print("=" * 65)
