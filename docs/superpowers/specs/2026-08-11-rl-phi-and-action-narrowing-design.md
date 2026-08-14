# RL Φ Genişletmesi + Aksiyon Uzayı Daraltma — Tasarım

> Durum: onaylandı, uygulanacak (2026-08-11 gecesi).

## Bağlam

Yarışma kuralları ASU'nun kod/output'unun reward-shaping potansiyeli (Φ) veya
davranış-klonlama etiketi olarak kullanılmasını yasaklıyor. Bu, `ARCHITECTURE.md`'nin
P0 #1 (Φ(s)=ASU V(s)) ve P0 #2 (ASU gate_1/gate_2) maddelerini geçersiz kılıyor. ASU
yalnızca bir rakip/benchmark olarak kullanılabilir.

Bu doküman, o iki maddenin yerine geçecek — tamamen bağımsız, kendi yazdığımız —
reward-shaping ve aksiyon-daraltma tasarımını tanımlar. Amaç: bu gece Colab'da bir
eğitim koşusu başlatmak ve sonucundan ne geliştirilebileceğini görmek.

## 1. Φ (PBRS potansiyeli) genişletmesi — `monopoly_game_engine/env.py`

Mevcut `MonopolyEnv._compute_reward` (satır ~1047) relative net-worth potansiyelini
hesaplıyor, `[-1, 1]` aralığına clip'li, ASU'dan bağımsız. Bu korunuyor ve iki yeni
terimle genişletiliyor:

- **Monopoly-yakınlık terimi**: her renk grubunda `(sahip_olunan_oran)²` toplamı.
  Kendi payımız eksi rakiplerin ortalaması, `[-1, 1]`'e clip. ASU'nun
  `2**eksik_mülk` iskonto/dice-simülasyonlu yaklaşımından tamamen farklı, kasıtlı
  olarak daha basit bir formül.
- **Nakit-güvenlik terimi**: borçtaysa `-1`, değilse cash için doğrusal bir tampon
  skoru (`clip((cash - 50) / 300, -1, 1)` gibi). Kendi payımız eksi rakip ortalaması.

```
Φ_total = clip(0.6 * net_worth_terimi + 0.25 * monopoly_terimi + 0.15 * nakit_terimi, -1, 1)
```

Ağırlıklar başlangıç değeri; eğitim sonrası gözlemlere göre kalibre edilebilir.

`train.py`'deki `potential_delta()` hâlâ `env._compute_reward(agent_pid)`'i çağırıyor
— değişiklik tamamen `_compute_reward` içinde, çağıran kod değişmiyor.

## 2. Aksiyon uzayı daraltma — `monopoly_game_engine/agent_ppo.py`

Hybrid mod şu an yalnızca `BUY_PROPERTY` ve `ACCEPT_TRADE`'i kural katmanına
devrediyor. Bunu genişletip **tüm trade ailesini** (252 buy-trade + 252 sell-trade +
2268 exchange = 2772 aksiyon) `nn_allowed`'dan çıkarıyoruz:

- Teklif kabul/red kararı: mevcut `fixed_accept_trade_decision` (zaten ASU'suz).
- Teklif başlatma kararı (yeni): kendi basit kuralımız — sadece bir tekel
  tamamlanıyorsa veya karşı tarafın tekelini engelliyorsak teklif ver, fiyat
  kademesi `75% → 100% → 125%` eskalasyonuyla (ARCHITECTURE.md P1 #8, ASU'dan
  bağımsız, deterministik).

Bu, öğrenilen politikanın aksiyon alanını ~186 aksiyona indiriyor (GNOME-p3
bulgusuyla tutarlı: trade'ler aksiyon uzayının çoğunu oluşturuyor ve saf RL'yi
zayıflatıyor).

## 3. Eğitim — Colab

- `train.py` ile `n_games≈4000-5000`, Fixed-A/B/C'ye karşı (mevcut trainer).
- Sık `checkpoint_every` (oturum kopmasına karşı sigorta).
- Bu geceki koşuda ASU training opponent pool'unda **yok** — sadece eval
  aşamasında karşılaştırma için kullanılacak, sinyale hiç girmiyor.

## 4. Değerlendirme

- `evaluate()` ile Fixed-A/B/C'ye karşı win-rate.
- Ayrı bir eval koşusu: ASU'ya karşı (izin dahilinde, sadece ölçüm).
- Sonuçlardan çıkan gözlemler bir sonraki iterasyon için not edilecek
  (`ARCHITECTURE.md` P1/P2 listesine göre önceliklendirme).

## Kapsam dışı (bu gece için)

- ASU'yu herhangi bir şekilde training sinyaline sokmak — yasak.
- MonopolyZero pipeline'ı — bu gece PPO-Plus üzerinde çalışıyoruz.
- LLM/Gemma entegrasyonu — RL bitince ayrı faz.
