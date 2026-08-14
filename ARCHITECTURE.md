# RL + LLM Mimarisi — Fazlara Ayrılmış Uygulama Planı

> Bu dosya `rules.md` (kurallar) ve önceki `STRATEGY_PLAN.md`'nin (ilk taslak) yerini
> alan, **son ve uygulamaya hazır** plan. Pazartesi–Cuma, 3 kişi, Colab Pro.
> Her faz "ne, neden, kim, gece mi gündüz mü uygun" sorularını cevaplıyor.

---

## 0. Mimarinin özeti (değişmedi, referans için)

```
ASU_FROZEN_TEACHER (V(s), gate_1, gate_2)  →  Φ potansiyel fonksiyonu  →  PBRS ödül  →  RL (PPO/DDQN)
        │                                                                                  ↑
        └──→  Etiketli veri  →  Gemma QLoRA (LLM)  →  aksiyon önerisi ── look-back advice ─┘
        │
        └──→  Deterministik davranış katmanı (trade fiyatlama, hapis kararı, açık artırma,
              ev biriktirme) — RL/LLM'e ÖĞRETİLMİYOR, sabit kural olarak koşuyor
```

**Gerekçe (tekrar, kısaca):** Saf RL 5 günde rastgeleyi geçme garantisi vermiyor
(`TRAINING_RESULTS.md`: 2000 oyunda %0 final win-rate). Aksiyon uzayının %94'ü trade —
GNOME-p3 paper'ı (arXiv 2103.00683) bu tür çarpık dağılımın saf DRL'yi zayıflattığını
gösteriyor. Çözüm: ASU'nun hazır değer fonksiyonunu Φ olarak kullanıp PBRS (Ng et al.
1999 — optimal politikayı bozmadan öğrenmeyi hızlandırma teoremi) ile RL'yi
hızlandırmak, LLM'i hem davranış klonlama kaynağı hem de ek şekillendirme sinyali
olarak kullanmak (Bhambri ve ark. 2024, aynı ASU grubu).

---

## 1. Stratejiler — önem sırasına göre

Önce tüm konuştuklarımızı tek bir öncelik listesinde topluyoruz. Sıralama, **etki /
efor** oranına göre — P0 olmadan hiçbir şey çalışmaz, P2 olmadan da ortada bir ürün
teslim edilir.

### P0 — Temel, engelleyici (bunlar olmadan hiçbir şey anlamlı çalışmaz)

| # | Strateji | Neden P0 |
|---|---|---|
| 1 | **PBRS: Φ(s) = ASU `V(s)`** | Tüm RL eğitiminin çekirdeği |
| 2 | **ASU `gate_1`/`gate_2` ile nakit güvenliği** | Zaten kodda hazır, iflas riskini doğrudan azaltıyor, entegre etmemek için sebep yok |
| 3 | **Aksiyon uzayı daraltma** (trade'ler öğrenilen politikanın dışında) | GNOME-p3 bulgusu — bunsuz RL muhtemelen yakınsamıyor |
| 4 | **Board iniş-frekansı kalibrasyonu** (zar-yürüyüşü simülasyonu) | Ucuz (GPU yok), Φ ağırlıklarının gerçeğe uygunluğunu doğruluyor |

### P1 — Yüksek değer, orta efor

| # | Strateji | Not |
|---|---|---|
| 5 | **LLM QLoRA distillation** (ASU teacher etiketli veri + "neden" alanı) | Warm-start + look-back advice için kaynak |
| 6 | **Look-back advice şekillendirmesi** (LLM önerisiyle örtüşme bonusu) | Wiewiora ve ark. 2003 formülü, policy-invariant |
| 7 | **Tren istasyonu/kamu hizmeti de-önceliklendirme** | Şampiyon tavsiyesi + kart etkisi olmadığı için burada daha da geçerli; sadece ağırlık ayarı |
| 8 | **Trade fiyat kademesi eskalasyonu** (%75 dene → tekel-kritikse %100/%125'e çık) | Ucuz, deterministik kural |
| 9 | **Hapis kararı: erken oyunda kefalet öde, geç oyunda ödeme** | Ucuz, deterministik kural — round/max_rounds oranına göre koşullandır |
| 10 | **Çeşitli rakip havuzuna karşı eğitim** | Kör teslim olduğu için (bkz. §6) tek tip rakibe aşırı uyumu önlüyor |

### P2 — İyi olur, zaman kalırsa

| # | Strateji | Not |
|---|---|---|
| 11 | **Ev biriktirme / kıtlık yaratma** (32 ev stoğu, otele yükseltmeden biriktir) | Etkili ama trade/inşa kararlarının olgunlaşmasını bekliyor |
| 12 | **Açık artırma arbitrajı** (ucuz kazan → tekel-ihtiyacı olan oyuncuya %125'e sat) | Yeni fikir, aşağıda §5'te detaylı değerlendirdim |
| 13 | **Lider/takipçi varyans asimetrisi** | İnce ayar, temel politika oturduktan sonra anlamlı |
| 14 | **Lideri hedef alma (açık artırmada)** | İnce ayar, aynı sebep |
| 15 | **3 vs 4 ev eşiği ince ayarı** | Sabit kural değil, eğitim sonuçlarına göre kalibre edilecek |
| 16 | **İpotek kaldırma maliyetini `int(mortgage×1.1)` olarak hesapla, `round`/tam ondalık değil** | Kod bunu aşağı yuvarlıyor (`env.py`), kural değişmiyor — sadece Φ/reward hesaplarınızın kodla bire bir eşleşmesi için |

**Kapsam dışı bırakılan / düzeltilen fikirler** (önemli, unutulmasın):
- ~~Açık artırmada rakip teklif sırasını izleyerek bilgi toplama~~ — **kısmen geçerli**:
  bir açık artırma sürerken güncel en yüksek teklif/lider herkese görünür (state
  vektöründe var), o kısım geçerli kalıyor. Ama **başka oyuncular arasındaki (bize hiç
  değmeyen) trade tekliflerini göremiyoruz** — bunu doğru tespit ettiniz, o kısım kapsam
  dışı.
- ~~PSRO / bilinen rakiplere özel karşı-strateji~~ — **turnuva kör teslim olduğu için
  (bkz. §6) uygulanamaz**, yerine #10 (çeşitli havuz eğitimi) kullanılıyor.

---

## 2. Fazlar (Pazartesi → Cuma)

### Faz 0 — Kurulum ve kalibrasyon (Pazartesi gündüz)
**Kim:** Herkes birlikte, kısa.
**İş:** Roller netleşir. §1.4'teki zar-yürüyüşü simülasyonu koşulur (GPU gerekmez).
`monopoly_bench` eval iskeleti hazırlanır. `CLAUDE.md` repoya eklenir (görev
tanımları için).
**Gece uygunluğu:** Hayır — bu faz etkileşimli, karar gerektiriyor.

### Faz 1 — Çekirdek PBRS + veri üretimi (Pazartesi gece → Salı gündüz)
**Kim A (RL):** ASU `V(s)`'i Φ yapan PBRS wrapper'ını yaz, `gate_1`/`gate_2`'yi entegre
et, aksiyon uzağını daralt (trade'leri dışarı al). İlk PBRS-PPO/DDQN eğitimini başlat.
**Kim B (LLM):** ASU teacher ile self-play'den etiketli veri üretim script'i — bu
**gece boyunca sürebilir**, GPU gerektirmiyorsa CPU'da paralel koşturulabilir.
**Kim C:** Faz 0'daki eval iskeletini genişlet, ilk PBRS checkpoint'ini test etmeye
hazırlan.
**Gece uygunluğu:** **Evet, ideal.** PBRS-PPO eğitimi ve veri üretimi, başlatıp
bırakabileceğiniz, uzun süren işler — Pazartesi gecesi için birebir uygun (aşağıda §6).

### Faz 2 — LLM distillation (Salı gece → Çarşamba gündüz)
**Kim B:** Toplanan veriyle Gemma QLoRA fine-tune (küçük varyantla başla, T4'te).
**Kim A:** Behavior-cloning warm-start'ı PPO aktörüne entegre et.
**Kim C:** İlk LLM checkpoint'ini `monopoly_bench`'te test et.
**Gece uygunluğu:** **Evet.** QLoRA fine-tune, gece boyu bırakılabilecek klasik bir iş —
checkpoint'li olduğundan sabah kaldığı yerden devam edilebilir.

### Faz 3 — Davranışsal katman (Çarşamba gündüz → Perşembe gündüz)
**Kim C (öncülük):** P1/P2 listesindeki deterministik kuralları tek tek ekle:
tren istasyonu ağırlığı, trade kademe eskalasyonu, hapis erken/geç kuralı, ev
biriktirme, açık artırma arbitrajı, lider/takipçi asimetrisi, lideri hedef alma.
**Kim A:** Look-back advice şekillendirmesini entegre et (LLM önerisiyle örtüşme
bonusu), ikinci PBRS-PPO iterasyonuna başla.
**Kim B:** Veri kalitesini iyileştir (LLM'in hatalı karar verdiği durumları hedefle),
gerekirse A100 ile daha büyük modele geç.
**Gece uygunluğu:** **Kısmen.** İkinci PBRS-PPO iterasyonu ve büyük model fine-tune
gece için uygun; davranışsal kural yazımı/hata ayıklama gündüz yapılmalı (hızlı
geri bildirim gerekiyor).

### Faz 4 — Entegrasyon ve geniş ölçekli değerlendirme (Perşembe gündüz → gece)
**Herkes:** Tüm bileşenleri `monopoly_bench`'te birleştir, §1'deki tüm P0/P1
stratejilerinin gerçekten devrede olduğunu doğrula. Çeşitli rakip havuzuna karşı
final eğitim/ince-ayar koşusu başlat.
**Gece uygunluğu:** **Evet, en kritik gece.** Final koşu + 500+ oyunluk istatistiksel
karşılaştırma — Perşembe gecesi boyunca sürmeli, Cuma sabahı sonuçla uyanın.

### Faz 5 — Donma ve teslim (Cuma)
**Herkes:** Son checkpoint seçimi, `rules.md` + bu dosyanın son hallerinin repoya
eklenmesi, kısa rapor (hangi paper/kural hangi bileşende kullanıldı), sunum provası.
**Gece uygunluğu:** Hayır — tampon gün, sürprizlere karşı boş bırakın.

---

## 3. Gece antrenman fikri — değerlendirme

**Kısaca: iyi fikir, kullanın.** Ama iki şeye dikkat:

1. **Hangi işler gece için uygun, hangileri değil:**
   - ✅ Uygun: PBRS-PPO/DDQN eğitimi, QLoRA fine-tune, büyük ölçekli veri üretimi/
     self-play, geniş çaplı (yüzlerce oyunluk) değerlendirme koşuları — hepsi
     "başlat, bırak, sabah bak" tipi işler.
   - ❌ Uygun değil: davranışsal kural yazımı, prompt/şema iterasyonu, hata ayıklama —
     bunlar hızlı geri bildirim istiyor, gece boşa gidebilir.

2. **Colab'ın oturum-kopması riskini ilk gece test edin.** Compute biriminiz kalsa
   bile, bazı durumlarda tarayıcı/sekme kapanınca oturum kesilebiliyor. Pazartesi
   gecesi **önce küçük, önemsiz bir "dummy" iş** (örn. 10 dakikalık bir eğitim
   döngüsü) başlatıp sabah hâlâ çalışıyor mu diye kontrol edin — büyük bir işi ilk
   kez gece denemeden önce bunu doğrulamak, olası bir geceyi boşa harcamamak için
   ucuz bir sigorta. Mevcut `training_guard.py` + checkpoint/`--resume` altyapınız
   zaten oturum kesintilerine karşı hazır — sık checkpoint alacak şekilde
   ayarladığınızdan emin olun (bir oturum ortasında koparsa bile en son
   checkpoint'ten devam edebilesiniz).
3. **3 kişi, 3 hesap, 3 paralel gece işi.** Herkes yatmadan önce farklı bir iş
   kuyruğa alırsa (biri PBRS-PPO, biri QLoRA, biri veri üretimi), sabah üç işin de
   sonucunu birlikte değerlendirip günün geri kalanını ona göre planlarsınız —
   bu, 5 günlük bütçeye pratikte 1-1.5 gün daha eklemek gibi.

---

## 4. Açık artırma arbitrajı — yeni fikrinizin değerlendirmesi

"Açık artırmada ucuzdan alıp başka bir oyuncuya pahalıya satalım" fikri **kurallara
uygun ve gerçekten çalışabilir**, çünkü:

- Açık artırma fiyatı, o mülkte kaç oyuncunun rekabet ettiğine bağlı — kimse
  istemiyorsa çok ucuza (hatta ilk artışla, tek teklifçi olarak) kazanabilirsiniz.
- Satış teklifi ise **liste fiyatının** %75/%100/%125'i üzerinden hesaplanıyor —
  açık artırmada ödediğiniz fiyatla değil. Yani ucuza kazanıp liste fiyatının
  %125'ine satmak, aradaki farkın tamamı kâr.
- En güçlü senaryo: mülk **sizin için değersiz** ama **başka bir oyuncunun tekelini
  tamamlıyor** — o oyuncu için %125 fiyatı bile tekelin getirisinin yanında küçük
  kalır (champion mantığıyla aynı: tekel değeri, fiyat farkını fazlasıyla aşar),
  yani kabul etme ihtimalleri yüksek.

**Dikkat edilecek:** Bunu öğrenilen politikaya bırakmak yerine, §1'deki P2 listesine
deterministik bir kural olarak ekleyin: "açık artırmada bana gereksiz ama bir
rakibin grup-tamamlayıcısı olan bir mülk kazanılırsa → o rakibe %125 sat teklifi
gönder." Basit, ucuz, ve önceki tüm bulgularla (tekel değeri >> fiyat farkı) tutarlı.

---

## 5. Turnuva formatı hatırlatması

Kör teslim, ~1000 maç, sabit 6 ajan havuzundan her maçta 4'ü oynuyor. Bu yüzden:
- Rakibe özel karşı-strateji **yok** — çeşitli havuza karşı sağlam/genel eğitim (§1,
  madde 10) önceliklidir.
- Büyük N (1000 maç) şansı eritir — yüksek varyanslı taktiklere (aşırı spite bidding
  gibi) ana strateji olarak güvenmeyin, tutarlı pozitif-EV'li politika önceliklidir.

---

## 6. İpotek kaldırma yuvarlaması (P2 #16 detayı)

Kural değişmiyor, sadece uygulama hassasiyeti: `env.py`, ipotek kaldırma maliyetini
`int(mortgage_value × 1.1)` olarak hesaplıyor — yani `.5` çıkan sonuçları **aşağı**
yuvarlıyor, `round()` değil. `constants.py`'deki mortgage değerlerini taradığımızda bu
yalnızca üç mülkte gerçek bir fark yaratıyor: **Electric Company** ve **Water Works**
(mortgage 75 → 82.5 yerine 82) ve **Park Place** (mortgage 175 → 192.5 yerine 192).
Fark mülk başına yalnızca **0.5$** — oyunu değiştirecek bir arbitraj değil, ama iki yerde
önemli:
1. **Φ/reward hesaplarınız** bu maliyeti kullanıyorsa (örn. net-worth veya nakit-akışı
   tabanlı bir shaping terimi), formülü kodla birebir eşleştirin (`int(...)`, `round(...)`
   değil) — aksi halde değer fonksiyonunuzda üç mülke özel, gerçek dışı küçük bir sapma
   oluşur.
2. Bu üç mülkü ipotekten çıkarmak, listedeki diğer tüm mülklere göre **teorik olarak en
   ucuz** işlem (oransal değil ama mutlak 0.5$ daha ucuz) — davranışsal katmanda bir
   önceliklendirme kuralı yazacaksanız bu sıralamayı gerçek maliyetlerden türetin, elle
   yuvarlanmış bir tablodan değil.

## 7. İlk training koşusu sonrası backlog — etki/efor önceliklendirmesi

İlk PPO hibrit koşusunun (5000 oyun, Fixed-A/B/C'ye karşı) win-rate eğrisi
düz kaldı (~%0-1, hiç trend yok) — sonraki iterasyon için konuşulan fikirlerin
etki/efor'a göre sıralaması:

| Sıra | Fikir | Efor | Gerekçe |
|---|---|---|---|
| 1 | 3-ev eşiği Φ terimi | Düşük | Gerçek Monopoly ekonomisi — rent 3. evde ciddi sıçrıyor. Mevcut `_color_group_progress`'e ek bir terim olarak ucuza eklenir. |
| 2 | Hapishane geç-oyun sığınağı | Düşük | Mevcut hybrid-fixed-rule kalıbına (`fixed_trade_offer_decision` gibi) uyuyor — tahta yoğunluğuna göre "çık/kal" kararı fixed rule'a bağlanır, Φ'ye dokunmaz. |
| 3 | Reddedilen tekliflere ceza | Düşük | Riskli — düz bir ceza ajanı her teklifi kabul etmeye itebilir. Uygulanacaksa "kötü reddetme"yi hedeflemeli, "reddetme"yi değil. |
| 4 | Turuncu grup ağırlıklandırma | Düşük | Ucuz ama etkisi belirsiz — muhtemelen zaten monopoly_term üzerinden dolaylı öğrenilir. |
| 5 | İstasyon/kamu hizmeti açık artırma flip'i | Orta-Yüksek | Trade-teklif başlatma şu an tamamen fixed rule'a bağlı (sadece kendi grubu tamamlanıyorsa teklif verir) — flip stratejisi trade mantığını genişletmek demek. |
| 6 | 32 ev tavanı / rakip engelleme | Orta | Modellemesi zor, rakip davranışına bağımlı, getirisi belirsiz. |
| 7 | PSRO / popülasyon eğitimi | Yüksek | En sağlam ama en pahalı — baseline çalışınca zaman kalırsa değerlendirilecek. |
