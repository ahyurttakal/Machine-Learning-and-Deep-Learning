"""
=============================================================================
AGENTİK AI — UYGULAMA 04
LangChain & AutoGen Entegrasyonu — Framework Karşılaştırması
=============================================================================
Kapsam:
  - LangChain: Chain, Agent, Memory, Tool, LCEL pipe mimarisi (simüle)
  - AutoGen: ConversableAgent, UserProxy, GroupChat (simüle)
  - Her iki framework'ün API tasarım prensipleri ve kullanım farkları
  - LangChain LCEL zinciri: prompt | llm | parser
  - AutoGen çok-ajan sohbeti: planlayıcı ↔ uygulayıcı ↔ eleştirmen
  - Ortak görev üzerinde benchmark karşılaştırması
  - Bellek sistemleri: ConversationBufferMemory, VectorStoreMemory
  - 8-panel görselleştirme: mimari, performans, konuşma akışı
=============================================================================
"""

import re, time, json, random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Iterator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  AGENTİK AI — UYGULAMA 04")
print("  LangChain & AutoGen Entegrasyonu")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: LANGCHAIN MİMARİSİ SİMÜLASYONU
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 1: LangChain — Chain, Tool, Memory, LCEL")
print("─" * 65)

# ── 1a. Temel soyutlamalar ────────────────────────────────────────

class BaseMessage:
    def __init__(self, content: str, role: str = "human"):
        self.content = content; self.role = role
    def __repr__(self): return f"{self.role.upper()}: {self.content[:50]}"

class HumanMessage(BaseMessage):
    def __init__(self, content): super().__init__(content, "human")

class AIMessage(BaseMessage):
    def __init__(self, content): super().__init__(content, "ai")

class SystemMessage(BaseMessage):
    def __init__(self, content): super().__init__(content, "system")


class ChatPromptTemplate:
    """LangChain ChatPromptTemplate simülasyonu."""
    def __init__(self, messages: list):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages: list):
        return cls(messages)

    def format_messages(self, **kwargs) -> List[BaseMessage]:
        sonuclar = []
        for tur, sablon in self.messages:
            icerik = sablon
            for k, v in kwargs.items():
                icerik = icerik.replace(f"{{{k}}}", str(v))
            if tur == "system":  sonuclar.append(SystemMessage(icerik))
            elif tur == "human": sonuclar.append(HumanMessage(icerik))
            else:                sonuclar.append(AIMessage(icerik))
        return sonuclar

    def __or__(self, other):
        """LCEL pipe operatörü: prompt | llm"""
        return LCELChain([self, other])


class SimChatLLM:
    """Simüle edilmiş LLM (gerçek API yerine)."""
    MODEL = "gpt-4o-sim"

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        time.sleep(random.uniform(0.02, 0.08))
        son_human = next((m.content for m in reversed(messages)
                         if isinstance(m, HumanMessage)), "")
        yanit = self._uret(son_human)
        return AIMessage(yanit)

    def _uret(self, girdi: str) -> str:
        giris = girdi.lower()
        if "özet" in giris or "özetle" in giris:
            return f"Özet: {girdi[:80]}... temel kavramlar ele alındı."
        if "kod" in giris or "python" in giris:
            return "```python\ndef coz(x):\n    return x * 2\nprint(coz(21))\n```"
        if "analiz" in giris:
            return f"Analiz: Girdi değerlendirildi. Güçlü yanlar: A, B. Zayıf yanlar: C."
        if "plan" in giris:
            return "Plan:\n1. Araştırma yap\n2. Tasarım oluştur\n3. Uygula\n4. Test et\n5. Dağıt"
        return f"LLM yanıtı: '{girdi[:50]}' için uygun çıktı üretildi."

    def stream(self, messages: list) -> Iterator[str]:
        yanit = self.invoke(messages).content
        for kelime in yanit.split():
            yield kelime + " "
            time.sleep(0.001)

    def __or__(self, other): return LCELChain([self, other])


class StrOutputParser:
    """LangChain StrOutputParser simülasyonu."""
    def invoke(self, message) -> str:
        return message.content if hasattr(message, "content") else str(message)
    def __or__(self, other): return LCELChain([self, other])


class LCELChain:
    """
    LangChain Expression Language (LCEL) — pipe (|) zinciri.
    Her bileşen bir sonrakine çıktısını iletir.
    """
    def __init__(self, adimlar: list):
        self.adimlar   = adimlar
        self.cagri_log = []

    def invoke(self, girdi, **kwargs) -> Any:
        guncel = girdi
        for adim in self.adimlar:
            t0 = time.perf_counter()
            if isinstance(adim, ChatPromptTemplate):
                if isinstance(guncel, dict):
                    guncel = adim.format_messages(**guncel)
                else:
                    guncel = adim.format_messages(input=str(guncel))
            elif hasattr(adim, "invoke"):
                guncel = adim.invoke(guncel, **kwargs)
            sure = (time.perf_counter() - t0) * 1000
            self.cagri_log.append({"adim": type(adim).__name__,
                                    "sure_ms": sure, "cikti_tip": type(guncel).__name__})
        return guncel

    def __or__(self, other):
        return LCELChain(self.adimlar + ([other] if not isinstance(other, LCELChain)
                                          else other.adimlar))

    def batch(self, girdiler: list) -> list:
        return [self.invoke(g) for g in girdiler]


# ── 1b. LangChain Bellek ─────────────────────────────────────────

class ConversationBufferMemory:
    """LangChain ConversationBufferMemory simülasyonu."""
    def __init__(self, max_token=2000):
        self.gecmis: List[BaseMessage] = []
        self.max_token = max_token

    def ekle(self, human_msg: str, ai_msg: str):
        self.gecmis.append(HumanMessage(human_msg))
        self.gecmis.append(AIMessage(ai_msg))
        # Kayan pencere
        while self._token_tahmin() > self.max_token and len(self.gecmis) > 2:
            self.gecmis.pop(0)

    def _token_tahmin(self):
        return sum(len(m.content.split()) * 1.3 for m in self.gecmis)

    def yukle(self) -> str:
        return "\n".join(f"{m.role.upper()}: {m.content}" for m in self.gecmis[-6:])

    def __len__(self): return len(self.gecmis)


# ── 1c. LangChain Tool ────────────────────────────────────────────

class LCTool:
    """LangChain @tool dekoratörü yerine temel sınıf."""
    def __init__(self, ad: str, acik: str, func: Callable):
        self.ad, self.acik, self.func = ad, acik, func
        self.cagri_sayisi = 0

    def run(self, girdi: str) -> str:
        self.cagri_sayisi += 1
        try:
            return str(self.func(girdi))
        except Exception as e:
            return f"Araç hatası: {e}"


# ── 1d. LangChain ReAct Agent ─────────────────────────────────────

class LangChainAgent:
    """LangChain AgentExecutor simülasyonu."""
    def __init__(self, llm, tools: List[LCTool], memory: ConversationBufferMemory,
                 verbose=True):
        self.llm, self.tools, self.bellek = llm, tools, memory
        self.verbose   = verbose
        self.cagri_log = []
        self._arac_map = {t.ad: t for t in tools}

    def _arac_sec(self, girdi: str) -> Optional[LCTool]:
        for arac in self.tools:
            if any(k in girdi.lower()
                   for k in arac.ad.replace("_"," ").split()):
                return arac
        return self.tools[0]  # varsayılan

    def invoke(self, girdi: str) -> dict:
        t0 = time.time()
        if self.verbose: print(f"\n  [LangChain] 🔵 Girdi: {girdi[:55]}")

        # Bellek bağlamı
        baglam = self.bellek.yukle()

        # Araç seçimi
        arac   = self._arac_sec(girdi)
        t_arac = time.perf_counter()
        sonuc  = arac.run(girdi)
        arac_sure = (time.perf_counter() - t_arac) * 1000

        # LLM ile nihai yanıt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sen yardımcı bir asistansın. Bağlam: {baglam}"),
            ("human",  "{girdi}\nAraç çıktısı: {arac_cikti}"),
        ])
        zincir  = prompt | SimChatLLM() | StrOutputParser()
        yanit   = zincir.invoke({"baglam": baglam[:200], "girdi": girdi,
                                  "arac_cikti": sonuc[:100]})
        self.bellek.ekle(girdi, yanit)

        sure = (time.time() - t0) * 1000
        kayit = {"girdi":girdi[:40],"arac":arac.ad,"arac_sure_ms":arac_sure,
                 "toplam_sure_ms":sure,"yanit":yanit[:60]}
        self.cagri_log.append(kayit)

        if self.verbose:
            print(f"  [LangChain] ⚙️  Araç: {arac.ad}  →  {sonuc[:45]}")
            print(f"  [LangChain] ✅ Yanıt: {yanit[:55]}  ({sure:.1f}ms)")
        return kayit


# ── 1e. LCEL Zincirleri ──────────────────────────────────────────

llm = SimChatLLM()

ozet_zinciri = (
    ChatPromptTemplate.from_messages([
        ("system", "Verilen metni en fazla 2 cümlede özetle."),
        ("human",  "{metin}"),
    ])
    | llm
    | StrOutputParser()
)

kod_zinciri = (
    ChatPromptTemplate.from_messages([
        ("system", "Python kodu yaz. Yalnızca kod bloğu döndür."),
        ("human",  "Görev: {gorev}"),
    ])
    | llm
    | StrOutputParser()
)

analiz_zinciri = (
    ChatPromptTemplate.from_messages([
        ("system", "Veriyi analiz et. Güçlü ve zayıf yönleri belirt."),
        ("human",  "Veri: {veri}"),
    ])
    | llm
    | StrOutputParser()
)

print("  ✅ LCEL zincirleri: ozet_zinciri | kod_zinciri | analiz_zinciri")

# LangChain araçları
lc_tools = [
    LCTool("web_arama",   "Web araması",     lambda x: f"Arama sonucu: {x[:30]}... bulundu."),
    LCTool("hesaplama",   "Matematik",       lambda x: str(eval(re.sub(r'[^0-9+\-*/().]','',x) or "0"))),
    LCTool("dosya_okuma", "Dosya oku",       lambda x: f"Dosya içeriği: {x} — veri hazır."),
    LCTool("kod_calistir","Kod çalıştır",    lambda x: f"Çıktı: {eval(x) if re.match(r'^[\d+\-*/\s()]+$',x) else 'OK'}"),
]
bellek  = ConversationBufferMemory(max_token=2000)
lc_ajan = LangChainAgent(llm, lc_tools, bellek, verbose=True)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: AUTOGEN MİMARİSİ SİMÜLASYONU
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 2: AutoGen — ConversableAgent, GroupChat")
print("─" * 65)

@dataclass
class AutoGenMesaj:
    gonderen  : str
    alici     : str
    icerik    : str
    tur       : str   = "text"   # text | code | result | terminate
    tur_no    : int   = 0
    zaman     : float = field(default_factory=time.time)

class ConversableAgent:
    """AutoGen ConversableAgent simülasyonu."""
    MAX_TUR = 8

    def __init__(self, ad: str, sistem_msg: str,
                 insan_girisi: bool = False, verbose: bool = True):
        self.ad            = ad
        self.sistem_msg    = sistem_msg
        self.insan_girisi  = insan_girisi
        self.verbose       = verbose
        self.konusma_gecmisi: List[AutoGenMesaj] = []
        self.cagri_sayisi  = 0
        self.toplam_sure   = 0.0
        self._llm          = SimChatLLM()

    def uret(self, mesajlar: List[AutoGenMesaj], tur: int = 0) -> str:
        """LLM ile yanıt üret."""
        self.cagri_sayisi += 1
        t0 = time.perf_counter()

        son_icerik = mesajlar[-1].icerik if mesajlar else ""

        # Rol bazlı yanıt simülasyonu
        ad_lower = self.ad.lower()
        if "planlayıcı" in ad_lower or "planner" in ad_lower:
            yanit = (f"Tur {tur} — Plan:\n"
                     f"1. Problemi analiz et: '{son_icerik[:30]}'\n"
                     f"2. Alt görevlere böl\n3. Her birini uygula\n4. Sonuçları birleştir")
        elif "uygulayıcı" in ad_lower or "executor" in ad_lower:
            yanit = (f"Uygulama:\n```python\n"
                     f"def coz_{tur}(x):\n"
                     f"    # {son_icerik[:30]}\n"
                     f"    return x ** 2 + {tur}\n```\n"
                     f"Çalıştırıldı. Sonuç: OK")
        elif "eleştirmen" in ad_lower or "critic" in ad_lower:
            if tur >= 2:
                yanit = "TERMINATE — Çözüm yeterli ve doğrulandı."
            else:
                yanit = (f"Geri bildirim (Tur {tur}): Plan mantıklı. "
                         f"Ancak hata yönetimi eksik. Revize edilmeli.")
        elif "kullanıcı" in ad_lower or "user" in ad_lower:
            yanit = f"Görev: '{son_icerik[:40]}' üzerinde çalışın."
        else:
            yanit = f"{self.ad}: '{son_icerik[:40]}' için işlenmiş yanıt."

        self.toplam_sure += (time.perf_counter() - t0) * 1000
        return yanit

    def mesaj_al_ve_yanıtla(self, mesaj: AutoGenMesaj,
                             tur: int = 0) -> Optional[AutoGenMesaj]:
        self.konusma_gecmisi.append(mesaj)
        icerik = self.uret(self.konusma_gecmisi, tur)

        if self.verbose:
            print(f"  [{tur}] {self.ad:<18} → {icerik[:60]}")

        if "TERMINATE" in icerik:
            return AutoGenMesaj(self.ad, mesaj.gonderen, icerik, "terminate", tur)

        return AutoGenMesaj(self.ad, mesaj.gonderen, icerik, "text", tur)


class GroupChat:
    """AutoGen GroupChat simülasyonu."""
    def __init__(self, ajanlar: List[ConversableAgent],
                 max_tur: int = 6, verbose: bool = True):
        self.ajanlar   = ajanlar
        self.max_tur   = max_tur
        self.verbose   = verbose
        self.mesaj_log : List[AutoGenMesaj] = []
        self.bitis_neden = ""

    def baslat(self, baslangic_mesaj: str,
               baslatan: ConversableAgent) -> List[AutoGenMesaj]:
        ilk = AutoGenMesaj(baslatan.ad, "Grup", baslangic_mesaj, "text", 0)
        self.mesaj_log.append(ilk)

        if self.verbose:
            print(f"\n  🚀 GroupChat başlatıldı: {baslangic_mesaj[:55]}")
            print(f"  Katılımcılar: {[a.ad for a in self.ajanlar]}")
            print()

        for tur in range(self.max_tur):
            # Round-robin konuşma sırası
            ajan = self.ajanlar[tur % len(self.ajanlar)]
            son_mesaj = self.mesaj_log[-1]
            yanit = ajan.mesaj_al_ve_yanıtla(son_mesaj, tur + 1)

            if yanit:
                self.mesaj_log.append(yanit)
                if yanit.tur == "terminate":
                    self.bitis_neden = "TERMINATE sinyali"
                    if self.verbose:
                        print(f"\n  🏁 Konuşma sonlandı: {self.bitis_neden}")
                    break
        else:
            self.bitis_neden = f"Max tur ({self.max_tur}) aşıldı"
            if self.verbose:
                print(f"\n  ⏹️  {self.bitis_neden}")

        return self.mesaj_log


# AutoGen ajanları
planlayici  = ConversableAgent("Planlayıcı",  "Görevi planla ve böl.",    verbose=True)
uygulayici  = ConversableAgent("Uygulayıcı",  "Kodu uygula ve çalıştır.", verbose=True)
elestirmen  = ConversableAgent("Eleştirmen",  "Çözümü değerlendir.",      verbose=True)
kullanici   = ConversableAgent("Kullanıcı",   "Görevi ver ve izle.",
                                insan_girisi=True, verbose=True)

grp = GroupChat([planlayici, uygulayici, elestirmen], max_tur=6, verbose=True)
print("  ✅ AutoGen GroupChat hazır: Planlayıcı + Uygulayıcı + Eleştirmen")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: ORTAK GÖREV ÜZERINDE BENCHMARK
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 3: Ortak Görevler — LangChain vs AutoGen")
print("─" * 65)

GOREVLER = [
    "Müşteri verilerini analiz et ve tahmin modeli oluştur",
    "Python ile RAG pipeline kodu yaz ve belgele",
    "E-ticaret satış raporunu özetle ve önerileri listele",
    "Veri temizleme ve ön işleme pipeline tasarla",
]

# LangChain çalıştırımı
print("\n  === LangChain Agent ===")
lc_sonuclar = [lc_ajan.invoke(g) for g in GOREVLER]

# AutoGen GroupChat çalıştırımı
print("\n  === AutoGen GroupChat ===")
ag_sonuclar = []
for gorev in GOREVLER:
    grp2 = GroupChat([planlayici, uygulayici, elestirmen], max_tur=4, verbose=True)
    mesajlar = grp2.baslat(gorev, kullanici)
    ag_sonuclar.append({
        "gorev":       gorev,
        "mesaj_sayisi": len(mesajlar),
        "tur_sayisi":  max((m.tur_no for m in mesajlar), default=0),
        "sure_ms":     sum(a.toplam_sure for a in [planlayici,uygulayici,elestirmen]),
        "bitis":       grp2.bitis_neden,
    })

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: METRİKLER
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 4: Karşılaştırmalı Metrikler")
print("─" * 65)

print(f"\n  {'Framework':<14} {'Görev':>6} {'Ort. Süre (ms)':>15} {'Araç/LLM Çağrısı':>18}")
print("  " + "─" * 58)

lc_ort_sure = np.mean([r["toplam_sure_ms"] for r in lc_sonuclar])
ag_ort_sure = np.mean([r["sure_ms"] for r in ag_sonuclar])
lc_cagri    = sum(t.cagri_sayisi for t in lc_tools)
ag_cagri    = sum(a.cagri_sayisi for a in [planlayici,uygulayici,elestirmen])

print(f"  {'LangChain':<14} {len(GOREVLER):>6} {lc_ort_sure:>15.1f} {lc_cagri:>18}")
print(f"  {'AutoGen':<14} {len(GOREVLER):>6} {ag_ort_sure:>15.1f} {ag_cagri:>18}")

# LCEL zincir zamanlama
print("\n  LCEL Zincir Performansı:")
for zincir_ad, zincir, girdi in [
    ("ozet_zinciri",  ozet_zinciri,  {"metin":"Bu bir uzun metin özetidir. Agentic AI çok önemli."}),
    ("kod_zinciri",   kod_zinciri,   {"gorev":"Fibonacci dizisi hesapla"}),
    ("analiz_zinciri",analiz_zinciri,{"veri":"[85, 92, 78, 96, 88, 73, 91]"}),
]:
    t0 = time.perf_counter()
    yanit = zincir.invoke(girdi)
    sure  = (time.perf_counter()-t0)*1000
    print(f"    {zincir_ad:<20} → {yanit[:50]}  ({sure:.1f}ms)")

# Bellek istatistikleri
print(f"\n  Bellek (ConversationBuffer):")
print(f"    Mesaj sayısı   : {len(bellek)}")
print(f"    Tahmini token  : {bellek._token_tahmin():.0f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: GÖRSELLEŞTİRME (8 panel)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 5: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.38,
                        top=0.93, bottom=0.05)

LC_RENK = "#3B82F6"; AG_RENK = "#F59E0B"
GOREV_KISA = [f"G{i+1}" for i in range(len(GOREVLER))]

# G1 — Süre karşılaştırması
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#161B22")
lc_sureler = [r["toplam_sure_ms"] for r in lc_sonuclar]
ag_sureler = [r["sure_ms"] for r in ag_sonuclar]
x1 = np.arange(len(GOREVLER)); w1 = 0.35
ax1.bar(x1-w1/2, lc_sureler, w1, label="LangChain", color=LC_RENK, edgecolor="#30363D", alpha=0.88)
ax1.bar(x1+w1/2, ag_sureler, w1, label="AutoGen",   color=AG_RENK, edgecolor="#30363D", alpha=0.88)
ax1.set_xticks(x1); ax1.set_xticklabels(GOREV_KISA, color="#C9D1D9", fontsize=10)
ax1.set_title("Süre Karşılaştırması\n(ms / görev)", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax1.set_ylabel("ms", fontsize=10, color="#8B949E")
ax1.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax1.tick_params(colors="#8B949E")
ax1.grid(axis="y", alpha=0.3, color="#30363D")
ax1.set_facecolor("#161B22")
for sp in ax1.spines.values(): sp.set_color("#30363D")

# G2 — LangChain araç kullanımı
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#161B22")
arac_adlari = [t.ad for t in lc_tools]
arac_sayilari= [t.cagri_sayisi for t in lc_tools]
renk2 = ["#3B82F6","#10B981","#A78BFA","#F472B6"]
ax2.bar(arac_adlari, arac_sayilari, color=renk2, edgecolor="#30363D", alpha=0.88)
ax2.set_title("LangChain\nAraç Kullanımı", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax2.set_ylabel("Çağrı Sayısı", fontsize=10, color="#8B949E")
ax2.tick_params(colors="#8B949E", axis="x", rotation=20)
ax2.grid(axis="y", alpha=0.3, color="#30363D")
ax2.set_facecolor("#161B22")
for sp in ax2.spines.values(): sp.set_color("#30363D")

# G3 — AutoGen mesaj sayısı
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#161B22")
ag_mesajlar = [r["mesaj_sayisi"] for r in ag_sonuclar]
ax3.bar(GOREV_KISA, ag_mesajlar, color=AG_RENK, edgecolor="#30363D", alpha=0.88)
ax3.set_title("AutoGen\nMesaj Sayısı / Görev", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax3.set_ylabel("Mesaj", fontsize=10, color="#8B949E")
ax3.tick_params(colors="#8B949E")
ax3.grid(axis="y", alpha=0.3, color="#30363D")
ax3.set_facecolor("#161B22")
for sp in ax3.spines.values(): sp.set_color("#30363D")

# G4 — LCEL Zincir Mimarisi (şema)
ax4 = fig.add_subplot(gs[1, :2]); ax4.set_facecolor("#161B22")
ax4.set_xlim(0, 10); ax4.set_ylim(0, 3); ax4.axis("off")
ax4.set_title("LCEL Zincir Mimarisi  —  prompt | llm | parser",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
lcel_bilesenleri = [
    ("ChatPromptTemplate", "#3B82F6", "Şablon\ndoldur"),
    ("SimChatLLM",         "#10B981", "LLM\nçağrısı"),
    ("StrOutputParser",    "#A78BFA", "String\nayrıştır"),
    ("Sonuç",              "#F59E0B", "Final\nçıktı"),
]
xs4 = [1.2, 3.5, 5.8, 8.1]
for (ad, renk, alt), x in zip(lcel_bilesenleri, xs4):
    ax4.add_patch(mpatches.FancyBboxPatch(
        (x-.85, 1.2), 1.7, 1.0, boxstyle="round,pad=0.08",
        facecolor=renk, edgecolor="#0D1117", linewidth=2, alpha=0.85))
    ax4.text(x, 1.88, ad.replace("Sim",""), ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")
    ax4.text(x, 1.42, alt, ha="center", va="center",
             fontsize=8.5, color="#E5E7EB")
    if x < xs4[-1]:
        ax4.annotate("", xy=(x+.88, 1.7), xytext=(x+.88+0.01, 1.7),
                     xycoords="data", textcoords="data")

for i in range(len(xs4)-1):
    ax4.annotate("", xy=(xs4[i+1]-.87, 1.70), xytext=(xs4[i]+.87, 1.70),
                 arrowprops=dict(arrowstyle="-|>", color="#94A3B8",
                                  lw=2.2, mutation_scale=20))
ax4.text(5.0, 0.5, "| (pipe) operatörü ile bileşenleri zincirle — chain = prompt | llm | parser",
         ha="center", va="center", fontsize=10, color="#8B949E", style="italic")

# G5 — AutoGen konuşma akışı (görev 1)
ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor("#161B22")
ax5.set_xlim(0, 3); ax5.set_ylim(-0.5, 6.5); ax5.axis("off")
ax5.set_title("AutoGen Konuşma\nAkışı", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
AJAN_RENK = {"Kullanıcı":"#64748B","Planlayıcı":"#3B82F6","Uygulayıcı":"#10B981","Eleştirmen":"#EF4444"}

# GroupChat'in ilk görev mesajlarını al
grp_ilk = GroupChat([
    ConversableAgent("Planlayıcı",  "Planla.", verbose=False),
    ConversableAgent("Uygulayıcı",  "Uygula.", verbose=False),
    ConversableAgent("Eleştirmen",  "Değerlendir.", verbose=False),
], max_tur=4, verbose=False)
msgs_viz = grp_ilk.baslat(GOREVLER[0], ConversableAgent("Kullanıcı","",verbose=False))

for j, msg in enumerate(msgs_viz[:6]):
    renk = AJAN_RENK.get(msg.gonderen, "#64748B")
    y_pos = 5.5 - j
    ax5.add_patch(mpatches.FancyBboxPatch(
        (0.1, y_pos-.30), 2.8, 0.58, boxstyle="round,pad=0.06",
        facecolor=renk, edgecolor="#0D1117", linewidth=1.5, alpha=0.80))
    ax5.text(1.5, y_pos+0.05, f"[{msg.tur_no}] {msg.gonderen}: {msg.icerik[:28]}…",
             ha="center", va="center", fontsize=7.5, color="white")
    if j < len(msgs_viz[:6])-1:
        ax5.annotate("", xy=(1.5, y_pos-.32), xytext=(1.5, y_pos-.32-.08),
                     arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5))

# G6 — Bellek kullanımı zaman serisi
ax6 = fig.add_subplot(gs[2, 0]); ax6.set_facecolor("#161B22")
bellek_boyutlari = list(range(2, len(bellek)+1, 2)) or [0]
token_tahminleri = [i * 12 for i in bellek_boyutlari]
ax6.plot(bellek_boyutlari, token_tahminleri, "o-", color=LC_RENK, linewidth=2.5,
         markersize=7, markerfacecolor="#fff", markeredgecolor=LC_RENK)
ax6.axhline(y=2000, color="#EF4444", linestyle="--", linewidth=1.5, label="Max Token Sınırı")
ax6.set_title("Konuşma Belleği\nToken Büyümesi", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax6.set_xlabel("Mesaj Sayısı", fontsize=10, color="#8B949E")
ax6.set_ylabel("Tahmini Token", fontsize=10, color="#8B949E")
ax6.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax6.tick_params(colors="#8B949E")
ax6.grid(alpha=0.3, color="#30363D")
ax6.set_facecolor("#161B22")
for sp in ax6.spines.values(): sp.set_color("#30363D")

# G7 — Framework özellik karşılaştırma radar
ax7 = fig.add_subplot(gs[2, 1], projection="polar"); ax7.set_facecolor("#161B22")
cats7 = ["Kolaylık","Esneklik","Performans","Topluluk","Belgeleme","Araç Zeng."]
N7    = len(cats7)
ang7  = [n/float(N7)*2*np.pi for n in range(N7)] + [0]
lc_puan = [0.88, 0.92, 0.80, 0.95, 0.90, 0.93] + [0.88]
ag_puan = [0.80, 0.95, 0.85, 0.88, 0.82, 0.88] + [0.80]
ax7.plot(ang7, lc_puan, "o-", color=LC_RENK, linewidth=2, markersize=6, label="LangChain")
ax7.fill(ang7, lc_puan, alpha=0.15, color=LC_RENK)
ax7.plot(ang7, ag_puan, "s-", color=AG_RENK, linewidth=2, markersize=6, label="AutoGen")
ax7.fill(ang7, ag_puan, alpha=0.15, color=AG_RENK)
ax7.set_xticks(ang7[:-1]); ax7.set_xticklabels(cats7, fontsize=8, color="#C9D1D9")
ax7.set_ylim(0, 1); ax7.set_facecolor("#161B22")
ax7.legend(loc="upper right", bbox_to_anchor=(1.35, 1.18),
           fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax7.set_title("Framework Karşılaştırma\nRadar", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=20)
ax7.tick_params(colors="#8B949E")

# G8 — LLM çağrı karşılaştırması
ax8 = fig.add_subplot(gs[2, 2]); ax8.set_facecolor("#161B22")
kategoriler8 = ["Araç\nÇağrısı","LLM\nÇağrısı","Bellek\nYükle","Toplam\nSüre(x10ms)"]
lc_degerler8 = [lc_cagri, len(lc_sonuclar)*2, len(bellek)//2, lc_ort_sure/10]
ag_degerler8 = [ag_cagri, ag_cagri,           len(ag_sonuclar), ag_ort_sure/10]
x8 = np.arange(len(kategoriler8)); w8 = 0.35
ax8.bar(x8-w8/2, lc_degerler8, w8, label="LangChain", color=LC_RENK, edgecolor="#30363D", alpha=0.88)
ax8.bar(x8+w8/2, ag_degerler8, w8, label="AutoGen",   color=AG_RENK, edgecolor="#30363D", alpha=0.88)
ax8.set_xticks(x8); ax8.set_xticklabels(kategoriler8, color="#C9D1D9", fontsize=9)
ax8.set_title("Kaynak Kullanımı\nKarşılaştırması", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax8.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax8.tick_params(colors="#8B949E")
ax8.grid(axis="y", alpha=0.3, color="#30363D")
ax8.set_facecolor("#161B22")
for sp in ax8.spines.values(): sp.set_color("#30363D")

fig.suptitle(
    "AGENTİK AI — UYGULAMA 04  |  LangChain & AutoGen Entegrasyonu\n"
    "LCEL Zinciri · GroupChat · Bellek · Framework Karşılaştırması",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98
)
plt.savefig("agentic_langchain_autogen.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ agentic_langchain_autogen.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print("  ÖZET")
print(f"  LangChain araç sayısı  : {len(lc_tools)}")
print(f"  AutoGen ajan sayısı    : 3 (Planlayıcı + Uygulayıcı + Eleştirmen)")
print(f"  Toplam görev           : {len(GOREVLER)}")
print(f"  LangChain ort. süre    : {lc_ort_sure:.1f} ms")
print(f"  AutoGen ort. süre      : {ag_ort_sure:.1f} ms")
print(f"  Grafik                 : agentic_langchain_autogen.png")
print("  ✅ LangChain & AutoGen TAMAMLANDI")
print("=" * 65)
