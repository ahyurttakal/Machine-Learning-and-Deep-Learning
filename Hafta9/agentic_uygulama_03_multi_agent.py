"""
=============================================================================
AGENTİK AI — UYGULAMA 03
Çok-Ajan Sistemi + LangChain/AutoGen Tarzı Entegrasyon
=============================================================================
Kapsam:
  - Orkestratör + Uzman Ajan mimarisi (AutoGen tarzı)
  - 6 uzman ajan: Araştırma, Yazma, Kod, Analiz, Eleştiri, Koordinasyon
  - Message Passing protokolü: mesaj kuyruğu, tip sistemi
  - Grup Sohbeti (GroupChat): yuvarlak masa tartışması
  - LangChain tarzı Chain: SequentialChain, ParallelChain
  - DAG iş akışı: bağımlılık çözümleme, topological sort
  - Ajan sağlık durumu: heartbeat, yük dengeleme
  - Kapsamlı görselleştirme (8 panel)
=============================================================================
"""

import time, random, uuid, json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  AGENTİK AI — UYGULAMA 03")
print("  Çok-Ajan Sistemi + LangChain/AutoGen Tarzı")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: MESAJ SİSTEMİ
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 1: Mesaj Sistemi (Message Passing Protokolü)")
print("─" * 65)

@dataclass
class Mesaj:
    gonderen:   str
    alici:      str
    tip:        str          # "task" | "result" | "critique" | "broadcast"
    icerik:     str
    oncelik:    int = 1      # 1=normal, 2=yüksek, 3=kritik
    meta:       dict = field(default_factory=dict)
    mesaj_id:   str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    zaman:      float = field(default_factory=time.time)

    def __repr__(self):
        return f"Mesaj({self.gonderen}→{self.alici} [{self.tip}]: {self.icerik[:40]})"

class MesajKuyrugu:
    """Öncelikli mesaj kuyruğu."""
    def __init__(self):
        self._kuyruk: list[Mesaj] = []
        self.gecmis: list[Mesaj]  = []

    def ekle(self, mesaj: Mesaj):
        self._kuyruk.append(mesaj)
        self._kuyruk.sort(key=lambda m: -m.oncelik)

    def al(self) -> Optional[Mesaj]:
        if self._kuyruk:
            m = self._kuyruk.pop(0)
            self.gecmis.append(m)
            return m
        return None

    def hepsi_isle(self):
        isle = []
        while self._kuyruk:
            isle.append(self.al())
        return isle

kuyruk = MesajKuyrugu()
test_mesajlar = [
    Mesaj("orkestrator", "arastirma", "task", "LLM güvenlik trendlerini araştır", oncelik=2),
    Mesaj("orkestrator", "yazici",    "task", "Bulgular için rapor taslağı hazırla"),
    Mesaj("arastirma",  "orkestrator","result","Araştırma tamamlandı: 5 trend bulundu", oncelik=2),
    Mesaj("yazici",     "elestirel",  "critique","Rapor taslağı hazır, gözden geçir"),
    Mesaj("orkestrator","broadcast",  "broadcast","Tüm ajanlara: Deadline 5 dakika",  oncelik=3),
]
for m in test_mesajlar:
    kuyruk.ekle(m)

print(f"  Kuyruktaki mesaj sayısı: {len(kuyruk._kuyruk)}")
islenen = kuyruk.hepsi_isle()
print(f"  {'Mesaj ID':<10} {'Gönderen→Alıcı':<30} {'Tip':<12} {'Öncelik'}")
print("  " + "-" * 65)
for m in islenen:
    print(f"  {m.mesaj_id:<10} {m.gonderen+'→'+m.alici:<30} {m.tip:<12} {m.oncelik}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: UZMAN AJAN SINIFI
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: Uzman Ajan Sınıfı")
print("─" * 65)

class UzmanAjan:
    """AutoGen ConversableAgent tarzı uzman ajan."""

    def __init__(self, isim: str, rol: str, yetenekler: list, max_tur: int = 5):
        self.isim       = isim
        self.rol        = rol
        self.yetenekler = yetenekler
        self.max_tur    = max_tur
        self.kuyruk     = MesajKuyrugu()
        self.sohbet_geçmisi: list[dict] = []
        self.gorev_sayisi = 0
        self.basarili     = 0
        self.toplam_sure  = 0.0
        self.aktif        = True
        self.son_heartbeat = time.time()
        self._yuk         = 0      # 0-100

    @property
    def yuk(self) -> int:
        return self._yuk

    def mesaj_al(self, mesaj: Mesaj):
        self.kuyruk.ekle(mesaj)

    def isle(self, mesaj: Mesaj) -> Mesaj:
        """Görevi işle ve sonuç mesajı üret."""
        t0 = time.perf_counter()
        self.gorev_sayisi += 1
        self._yuk = min(100, self._yuk + random.randint(10, 35))

        # Rol bazlı simüle işleme
        if self.rol == "araştırmacı":
            cikti = f"[{self.isim}] Araştırma sonucu: '{mesaj.icerik[:30]}...' için 3 kaynak analiz edildi."
        elif self.rol == "yazıcı":
            cikti = f"[{self.isim}] Rapor taslağı oluşturuldu. 500 kelime, 3 bölüm."
        elif self.rol == "kod geliştirici":
            cikti = f"[{self.isim}] Kod yazıldı. 25 satır Python, 2 fonksiyon."
        elif self.rol == "veri analisti":
            cikti = f"[{self.isim}] Analiz tamamlandı. Ortalama: {random.uniform(0.7, 0.95):.3f}"
        elif self.rol == "eleştirmen":
            cikti = f"[{self.isim}] İnceleme: {'Onaylandı' if random.random() > 0.3 else 'Revizyon gerekli'}."
        else:
            cikti = f"[{self.isim}] İşlem tamamlandı: {mesaj.icerik[:40]}"

        sure = time.perf_counter() - t0
        self.toplam_sure += sure
        self.basarili    += 1
        self._yuk = max(0, self._yuk - random.randint(5, 15))
        self.son_heartbeat = time.time()

        gecmis_kayit = {"rol": "assistant", "icerik": cikti, "sure": sure}
        self.sohbet_geçmisi.append(gecmis_kayit)

        return Mesaj(
            gonderen=self.isim,
            alici=mesaj.gonderen,
            tip="result",
            icerik=cikti,
            meta={"sure": sure, "kaynak_gorev": mesaj.mesaj_id}
        )

    def ozet(self) -> dict:
        return {
            "isim": self.isim, "rol": self.rol,
            "gorev": self.gorev_sayisi,
            "basari": self.basarili,
            "ort_sure": self.toplam_sure / max(self.gorev_sayisi, 1),
            "yuk": self._yuk,
        }

AJAN_TANIMLARI = [
    ("ArastirmaAjani", "araştırmacı",     ["web_ara","bilgi_sorgula","dosya_oku"]),
    ("YaziciAjani",    "yazıcı",          ["metin_yaz","duzenle","ozet"]),
    ("KodAjani",       "kod geliştirici", ["python_calistir","test_yaz","hata_ayikla"]),
    ("AnalizAjani",    "veri analisti",   ["istatistik","grafik","sql_sorgu"]),
    ("ElestirmenAjani","eleştirmen",      ["inceleme","puan_ver","geri_bildirim"]),
    ("KoordAjani",     "koordinatör",     ["planlama","kaynak_dagit","ilerleme_takip"]),
]

ajanlar = {isim: UzmanAjan(isim, rol, yetenekler)
           for isim, rol, yetenekler in AJAN_TANIMLARI}

print(f"  {'İsim':<20} {'Rol':<20} {'Yetenekler'}")
print("  " + "-" * 65)
for isim, ajan in ajanlar.items():
    print(f"  {isim:<20} {ajan.rol:<20} {', '.join(ajan.yetenekler[:2])}…")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: ORKESTRATÖR
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Orkestratör — Görev Dağıtımı")
print("─" * 65)

class Orkestrator:
    """Görevleri uzman ajanlara yönlendiren orkestratör."""

    ROL_HARITASI = {
        "araştır": "ArastirmaAjani",
        "yaz":     "YaziciAjani",
        "kod":     "KodAjani",
        "analiz":  "AnalizAjani",
        "incele":  "ElestirmenAjani",
        "planla":  "KoordAjani",
    }

    def __init__(self, ajanlar: dict):
        self.ajanlar  = ajanlar
        self.log      = []
        self.dag_grafik = defaultdict(list)

    def _ajan_sec(self, gorev: str) -> str:
        gorev_l = gorev.lower()
        for anahtar, ajan_isim in self.ROL_HARITASI.items():
            if anahtar in gorev_l:
                return ajan_isim
        # En az yüklü ajanı seç
        return min(self.ajanlar, key=lambda a: self.ajanlar[a].yuk)

    def gorevi_dagit(self, gorev: str, bağımli: list = None) -> dict:
        ajan_isim = self._ajan_sec(gorev)
        ajan      = self.ajanlar[ajan_isim]
        mesaj     = Mesaj("orkestrator", ajan_isim, "task", gorev)
        ajan.mesaj_al(mesaj)
        sonuc_msg = ajan.isle(mesaj)

        kayit = {
            "gorev":     gorev,
            "ajan":      ajan_isim,
            "sure":      sonuc_msg.meta["sure"],
            "sonuc":     sonuc_msg.icerik[:60],
            "basarili":  True,
            "bagimli":   bağımli or [],
        }
        self.log.append(kayit)

        if bağımli:
            for b in bağımli:
                self.dag_grafik[b].append(gorev)

        print(f"  [{ajan_isim}] {gorev[:45]:<45} → ✅ ({sonuc_msg.meta['sure']*1000:.1f}ms)")
        return kayit

orkestrator = Orkestrator(ajanlar)

PROJE_GOREVLERI = [
    ("Agentic AI pazar araştırması yap",               []),
    ("Araştırma bulgularını analiz et",                ["Agentic AI pazar araştırması yap"]),
    ("Analiz sonuçları için Python kodu yaz",          ["Araştırma bulgularını analiz et"]),
    ("Bulgular için kapsamlı rapor yaz",               ["Araştırma bulgularını analiz et"]),
    ("Kodu incele ve test yaz",                        ["Analiz sonuçları için Python kodu yaz"]),
    ("Raporu incele ve geri bildirim ver",             ["Bulgular için kapsamlı rapor yaz"]),
    ("Proje ilerleme durumunu planla ve raporla",      ["Raporu incele ve geri bildirim ver"]),
]

print(f"  {'Görev':<48} {'Durum'}")
print("  " + "-" * 65)
proje_log = []
for gorev, bagimlilar in PROJE_GOREVLERI:
    kayit = orkestrator.gorevi_dagit(gorev, bagimlilar)
    proje_log.append(kayit)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: GRUP SOHBETİ (GroupChat)
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Grup Sohbeti (GroupChat — AutoGen tarzı)")
print("─" * 65)

class GroupChat:
    """Yuvarlak masa tartışması."""
    def __init__(self, ajanlar: list, max_tur: int = 3):
        self.ajanlar  = ajanlar
        self.max_tur  = max_tur
        self.mesajlar = []

    def baslat(self, baslangic_mesaj: str) -> list:
        print(f"  💬 Başlangıç: {baslangic_mesaj[:60]}")
        self.mesajlar = [{"rol": "system", "icerik": baslangic_mesaj}]

        for tur in range(self.max_tur):
            print(f"\n  ── Tur {tur+1} ──")
            for ajan in self.ajanlar:
                onceki = self.mesajlar[-1]["icerik"]
                yanit  = f"[{ajan.isim}] Tur {tur+1}: '{onceki[:30]}...' hakkında görüşüm: " + \
                         random.choice([
                             "Bu yaklaşım mantıklı, destekliyorum.",
                             "Ek bilgi gerekebilir, araştırmalıyız.",
                             "Revizyon öneririm, bazı noktalar eksik.",
                             "Harika analiz, uygulayabiliriz.",
                         ])
                self.mesajlar.append({"rol": ajan.isim, "icerik": yanit})
                print(f"    {ajan.isim:<20}: {yanit[:55]}")
        return self.mesajlar

gc_ajanlar = [ajanlar["ArastirmaAjani"], ajanlar["YaziciAjani"], ajanlar["ElestirmenAjani"]]
gc = GroupChat(gc_ajanlar, max_tur=2)
gc_log = gc.baslat("Agentic AI sistemlerinde güvenlik en büyük endişe kaynağı mıdır?")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: LangChain Tarzı CHAIN YAPISI
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: LangChain Tarzı Chain Yapısı")
print("─" * 65)

class BaseChain:
    def __init__(self, isim: str):
        self.isim = isim
        self.sure = 0.0

    def calistir(self, girdi: dict) -> dict:
        raise NotImplementedError

class LLMChain(BaseChain):
    """Basit LLM çağrısı zinciri."""
    def __init__(self, isim: str, prompt_sablonu: str):
        super().__init__(isim)
        self.prompt_sablonu = prompt_sablonu

    def calistir(self, girdi: dict) -> dict:
        t0 = time.perf_counter()
        prompt = self.prompt_sablonu.format(**girdi)
        cikti  = f"[LLM] Yanıt: '{prompt[:40]}...' → üretildi."
        self.sure = time.perf_counter() - t0
        return {"cikti": cikti, "prompt": prompt[:60], "zincir": self.isim}

class SequentialChain(BaseChain):
    """Zincirleri sırayla çalıştırır (LangChain SequentialChain)."""
    def __init__(self, isim: str, zincirler: list):
        super().__init__(isim)
        self.zincirler = zincirler

    def calistir(self, girdi: dict) -> dict:
        t0     = time.perf_counter()
        sonuc  = girdi.copy()
        geçmiş = []
        for zincir in self.zincirler:
            ara = zincir.calistir(sonuc)
            sonuc.update(ara)
            geçmiş.append({"zincir": zincir.isim, "sure": zincir.sure})
        self.sure = time.perf_counter() - t0
        return {"sonuc": sonuc, "gecmis": geçmiş, "toplam_sure": self.sure}

class ParallelChain(BaseChain):
    """Zincirleri paralel çalıştırır."""
    def __init__(self, isim: str, zincirler: list):
        super().__init__(isim)
        self.zincirler = zincirler

    def calistir(self, girdi: dict) -> dict:
        t0     = time.perf_counter()
        sonuclar = {}
        for zincir in self.zincirler:
            sonuclar[zincir.isim] = zincir.calistir(girdi.copy())
        self.sure = time.perf_counter() - t0
        return {"paralel_sonuclar": sonuclar, "toplam_sure": self.sure}

# Chain oluşturma
arama_chain  = LLMChain("AramaChain",   "Şu konuyu ara: {konu}")
analiz_chain = LLMChain("AnalizChain",  "Şunu analiz et: {konu}")
rapor_chain  = LLMChain("RaporChain",   "Rapor yaz: {cikti}")
ozet_chain   = LLMChain("OzetChain",    "Özetle: {cikti}")

seq_chain = SequentialChain("ArastirmaSeq", [arama_chain, analiz_chain, rapor_chain])
par_chain = ParallelChain("KarsilastirmaParallel", [analiz_chain, ozet_chain])

print("  ── Sequential Chain ──")
seq_sonuc = seq_chain.calistir({"konu": "Agentic AI framework karşılaştırması"})
for g in seq_sonuc["gecmis"]:
    print(f"  {g['zincir']:<20} → süre: {g['sure']*1000:.2f}ms")
print(f"  Toplam süre: {seq_sonuc['toplam_sure']*1000:.2f}ms")

print()
print("  ── Parallel Chain ──")
par_sonuc = par_chain.calistir({"konu": "LLM güvenlik açıkları"})
for zincir_adi, s in par_sonuc["paralel_sonuclar"].items():
    print(f"  {zincir_adi:<20} → süre: {analiz_chain.sure*1000:.2f}ms")
print(f"  Toplam süre: {par_sonuc['toplam_sure']*1000:.2f}ms")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 6: DAG İŞ AKIŞI — TOPOLOJİK SIRALAMA
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: DAG İş Akışı — Topological Sort")
print("─" * 65)

class DAGIsAkisi:
    def __init__(self):
        self.dugumler  = {}         # isim → ajan
        self.kenarlar  = defaultdict(list)  # onceki → [sonraki]
        self.giris_der = defaultdict(int)   # girdi derecesi

    def dugum_ekle(self, isim: str, ajan):
        self.dugumler[isim] = ajan

    def baglanti_ekle(self, kaynak: str, hedef: str):
        self.kenarlar[kaynak].append(hedef)
        self.giris_der[hedef] += 1
        if kaynak not in self.giris_der:
            self.giris_der[kaynak] = 0

    def topolojik_siralama(self) -> list:
        """Kahn algoritması ile topolojik sıralama."""
        in_deg  = self.giris_der.copy()
        kuyruk  = deque([n for n in self.dugumler if in_deg.get(n, 0) == 0])
        sıralama = []
        while kuyruk:
            n = kuyruk.popleft()
            sıralama.append(n)
            for komsu in self.kenarlar.get(n, []):
                in_deg[komsu] -= 1
                if in_deg[komsu] == 0:
                    kuyruk.append(komsu)
        return sıralama

    def calistir(self) -> list:
        sıra  = self.topolojik_siralama()
        log   = []
        print(f"  Çalışma sırası: {' → '.join(sıra)}")
        for ad in sıra:
            ajan = self.dugumler[ad]
            m    = Mesaj("dag", ad, "task", f"DAG görevi: {ad}")
            sonuc = ajan.isle(m)
            log.append({"dugum": ad, "sure": sonuc.meta["sure"]})
            print(f"  ✅ {ad:<20} tamamlandı ({sonuc.meta['sure']*1000:.1f}ms)")
        return log

dag = DAGIsAkisi()
dag_dugumler = {
    "Araştır":   ajanlar["ArastirmaAjani"],
    "Analiz":    ajanlar["AnalizAjani"],
    "KodYaz":    ajanlar["KodAjani"],
    "Rapor":     ajanlar["YaziciAjani"],
    "İncele":    ajanlar["ElestirmenAjani"],
    "Koordine":  ajanlar["KoordAjani"],
}
for isim, ajan in dag_dugumler.items():
    dag.dugum_ekle(isim, ajan)

baglantılar = [
    ("Araştır","Analiz"), ("Araştır","KodYaz"),
    ("Analiz","Rapor"),   ("KodYaz","İncele"),
    ("Rapor","İncele"),   ("İncele","Koordine"),
]
for k, h in baglantılar:
    dag.baglanti_ekle(k, h)

dag_log = dag.calistir()

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 7: AJAN SAĞLIK DURUMU
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Ajan Sağlık Durumu & Yük Dengesi")
print("─" * 65)

print(f"  {'Ajan':<22} {'Görev':<8} {'Başarı':<10} {'Ort Süre(ms)':<14} {'Yük%'}")
print("  " + "-" * 65)
ozet_listesi = []
for isim, ajan in ajanlar.items():
    oz = ajan.ozet()
    ozet_listesi.append(oz)
    print(f"  {isim:<22} {oz['gorev']:<8} {oz['basari']:<10} "
          f"{oz['ort_sure']*1000:<14.2f} {oz['yuk']}%")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 8: GÖRSELLEŞTİRME (8 panel)
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.36,
                        top=0.93, bottom=0.05)

PALETA = ["#1A6FD8","#0FBCCE","#7B52E8","#F5A623","#10C98F","#E879A0"]

def ax_stili(ax, baslik):
    ax.set_facecolor("#161B22")
    ax.set_title(baslik, fontsize=11, fontweight="bold", color="#C9D1D9", pad=8)
    ax.tick_params(colors="#8B949E")
    ax.grid(alpha=0.25, color="#30363D")
    for sp in ax.spines.values(): sp.set_color("#30363D")

ajan_isimleri = [o["isim"] for o in ozet_listesi]
gorev_sayilari = [o["gorev"] for o in ozet_listesi]
basari_sayilari = [o["basari"] for o in ozet_listesi]
ort_sureler = [o["ort_sure"] * 1000 for o in ozet_listesi]
yuk_yuzdeleri = [o["yuk"] for o in ozet_listesi]
kisa_isimler = [a.replace("Ajani","") for a in ajan_isimleri]

# ── G1: Ajan görev dağılımı ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(kisa_isimler, gorev_sayilari, color=PALETA, edgecolor="#30363D", alpha=0.85)
ax_stili(ax1, "Ajan Başına Görev Sayısı")
ax1.set_ylabel("Görev", fontsize=10, color="#8B949E")
ax1.tick_params(axis="x", rotation=25)

# ── G2: Yük dengesi ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(kisa_isimler, yuk_yuzdeleri, color=PALETA, edgecolor="#30363D", alpha=0.85)
ax2.axhline(np.mean(yuk_yuzdeleri), color="#F5A623", ls="--", lw=1.8,
            label=f"Ort={np.mean(yuk_yuzdeleri):.0f}%")
ax_stili(ax2, "Ajan Yük Dağılımı (%)")
ax2.set_ylabel("Yük (%)", fontsize=10, color="#8B949E")
ax2.tick_params(axis="x", rotation=25)
ax2.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# ── G3: Ortalama görev süresi ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.barh(kisa_isimler, ort_sureler, color=PALETA, edgecolor="#30363D", alpha=0.85)
ax_stili(ax3, "Ortalama Görev Süresi (ms)")
ax3.set_xlabel("ms", fontsize=10, color="#8B949E")

# ── G4: Ajan topoloji grafiği (DAG) ──────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.set_facecolor("#161B22")
ax4.set_xlim(0, 10); ax4.set_ylim(0, 4); ax4.axis("off")
ax4.set_title("DAG İş Akışı — Ajan Bağımlılık Grafiği", fontsize=11,
              fontweight="bold", color="#C9D1D9", pad=8)

dag_pos = {
    "Araştır":  (1.2, 2.0),
    "Analiz":   (3.4, 3.0),
    "KodYaz":   (3.4, 1.0),
    "Rapor":    (5.8, 3.0),
    "İncele":   (7.2, 2.0),
    "Koordine": (9.0, 2.0),
}
dag_renkler = {"Araştır":"#0FBCCE","Analiz":"#1A6FD8","KodYaz":"#7B52E8",
               "Rapor":"#F5A623","İncele":"#E879A0","Koordine":"#10C98F"}
for bagl in baglantılar:
    x1, y1 = dag_pos[bagl[0]]
    x2, y2 = dag_pos[bagl[1]]
    ax4.annotate("", xy=(x2-0.32, y2), xytext=(x1+0.32, y1),
                 arrowprops=dict(arrowstyle="->", color="#94A3B8",
                                 lw=1.8, mutation_scale=18,
                                 connectionstyle="arc3,rad=0.1"))
for ad, (x, y) in dag_pos.items():
    box = mpatches.FancyBboxPatch(
        (x-0.60, y-0.36), 1.20, 0.72,
        boxstyle="round,pad=0.06",
        facecolor=dag_renkler[ad], edgecolor="#0D1117", linewidth=2, alpha=0.9
    )
    ax4.add_patch(box)
    ax4.text(x, y, ad, ha="center", va="center",
             fontsize=10, fontweight="bold", color="white")

# ── G5: Mesaj tipi dağılımı ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
mesaj_tipler = {"task": 7, "result": 7, "critique": 3, "broadcast": 2}
wedge_colors = [PALETA[0], PALETA[1], PALETA[2], PALETA[3]]
ax5.pie(list(mesaj_tipler.values()), labels=list(mesaj_tipler.keys()),
        colors=wedge_colors, autopct="%1.0f%%",
        textprops={"fontsize":9, "color":"#C9D1D9"},
        wedgeprops={"edgecolor":"#0D1117","linewidth":2})
ax5.set_facecolor("#161B22")
ax5.set_title("Mesaj Tipi Dağılımı", fontsize=11, fontweight="bold",
              color="#C9D1D9", pad=8)

# ── G6: Grup sohbeti tur analizi ─────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
gc_ajan_isimleri = [m["rol"] for m in gc_log[1:] if "rol" in m]
gc_ajan_sayac    = {}
for m in gc_log[1:]:
    if m.get("rol") not in ("system",):
        gc_ajan_sayac[m.get("rol","?")] = gc_ajan_sayac.get(m.get("rol","?"), 0) + 1
ax6.bar(list(gc_ajan_sayac.keys()), list(gc_ajan_sayac.values()),
        color=PALETA[:len(gc_ajan_sayac)], edgecolor="#30363D", alpha=0.85)
ax_stili(ax6, "GroupChat — Ajan Mesaj Sayısı")
ax6.set_ylabel("Mesaj Sayısı", fontsize=10, color="#8B949E")
ax6.tick_params(axis="x", rotation=15)

# ── G7: Sequential vs Parallel süre ─────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
karsilastirma = {
    "Sequential": seq_sonuc["toplam_sure"] * 1000,
    "Parallel": par_sonuc["toplam_sure"] * 1000,
    "Tek Ajan": (sum(ort_sureler) * 2),
}
bars7 = ax7.bar(list(karsilastirma.keys()), list(karsilastirma.values()),
                color=[PALETA[0], PALETA[1], PALETA[2]],
                edgecolor="#30363D", alpha=0.85)
ax_stili(ax7, "Çalışma Modu Süre Karşılaştırması (ms)")
ax7.set_ylabel("Süre (ms)", fontsize=10, color="#8B949E")
for b, v in zip(bars7, karsilastirma.values()):
    ax7.text(b.get_x()+b.get_width()/2, v*1.04, f"{v:.2f}",
             ha="center", fontsize=10, color="#C9D1D9")

# ── G8: Ajan rol radar ────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2], projection="polar")
ax8.set_facecolor("#161B22")
N = len(kisa_isimler)
acılar = [n/N * 2*np.pi for n in range(N)] + [0]
# simüle yetenek skorları
radar_vals = [0.92, 0.85, 0.88, 0.90, 0.78, 0.83, 0.92]
ax8.plot(acılar, radar_vals, "o-", color="#0FBCCE", lw=2, markersize=6)
ax8.fill(acılar, radar_vals, alpha=0.15, color="#0FBCCE")
ax8.set_xticks(acılar[:-1])
ax8.set_xticklabels(kisa_isimler, fontsize=8, color="#C9D1D9")
ax8.set_ylim(0, 1)
ax8.set_title("Ajan Yetkinlik Radar", fontsize=11,
              fontweight="bold", color="#C9D1D9", pad=20)
ax8.tick_params(colors="#8B949E")
ax8.set_facecolor("#161B22")

fig.suptitle(
    "AGENTİK AI — UYGULAMA 03  |  Çok-Ajan Sistemi + LangChain/AutoGen Tarzı\n"
    "Message Passing · Orkestratör · GroupChat · Sequential/Parallel Chain · DAG",
    fontsize=13, fontweight="bold", color="#C9D1D9", y=0.98
)
plt.savefig("agentic_03_multi_agent.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ agentic_03_multi_agent.png kaydedildi")
plt.close()

print()
print("=" * 65)
print("  ÖZET")
print(f"  Ajan sayısı     : {len(ajanlar)}")
print(f"  Toplam görev    : {sum(o['gorev'] for o in ozet_listesi)}")
print(f"  DAG düğümleri   : {len(dag_dugumler)}")
print(f"  GroupChat tur   : {gc.max_tur}")
print(f"  Grafik          : agentic_03_multi_agent.png")
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print("=" * 65)
