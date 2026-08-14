# PPO-Plus Monopoly Ruleset (`ppo-plus-v2`)

> Source of truth: `monopoly_game_engine/{constants,state,actions,env}.py`.
> This document explains the same ruleset as the code — not a simplification of it.
> If code and document ever disagree, **the code wins**; please open an issue/PR to fix this file.

---

# ENGLISH

## 1. What `ppo-plus-v2` is

`ppo-plus-v2` is the single canonical game engine shared by every learning path in this
repo: DDQN and PPO play against it directly through `env.py`; CFR (`classic_cfr.py`) clones
and explores the same engine. Algorithm-specific policy/checkpoint code stays separate —
only the **rules** are shared.

This is a **classic-board research ruleset**, not a certified reproduction of the official
Monopoly rulebook. Several mechanics are deliberately simplified so the environment has a
bounded, fully-enumerable action space (see §9) — those simplifications are listed
explicitly in §8 so nobody mistakes this for "real" Monopoly.

## 2. Board, players, money

- Standard **US 40-square board** (`BOARD` in `constants.py`): 22 real-estate color
  properties, 4 railroads, 2 utilities, Go, Jail/Just Visiting, Free Parking, Go To Jail,
  Income Tax, Luxury Tax, 3× Chance, 3× Community Chest.
- **4 players** (`NUM_PLAYERS = 4`), fixed at engine level — not currently configurable per
  game.
- Starting cash: **$1,500** per player.
- Passing (not landing exactly on) Go pays **$200**. Landing on Go pays no separate bonus
  beyond the pass-Go salary (there's only one Go-salary payment path in the code).
- **Income Tax** (square 4): flat **$200**, paid from cash only (see §7 for the "cash-only"
  rule).
- **Luxury Tax** (square 38): flat **$100**, cash only.
- **Free Parking** collects nothing and pays nothing — no jackpot.
- **Chance and Community Chest squares have no effect.** The card text lists exist in
  `constants.py` for reference/flavor, but landing on a Chance/Community Chest square is a
  no-op in `_handle_landing`. There is no shuffled deck, so Get Out of Jail Free cards are
  never normally introduced into play (see the note on the `gooj_card` flag in §5).

## 3. Turn structure (phase state machine)

Every player's turn moves through explicit phases (`state.py: PHASES`):

1. **`pre_roll`** — the active player may, before rolling: mortgage/unmortgage, sell houses
   or hotels, build houses/hotels, sell an unimproved property back to the bank, make or
   respond to trade offers, pay bail / use a Get-Out-of-Jail-Free card if in jail, or simply
   end this phase (`END_TURN` moves to `post_roll`).
2. **`post_roll`** — the player must roll the dice first (`ROLL_DICE`), unless still in jail
   with a bail/card option available. After the dice resolve and any owed rent/tax is
   settled, the player may buy the property they landed on (if unowned and affordable),
   mortgage/sell property to raise cash, or end the turn.
   - If the landed-on property is unowned and the player does **not** buy it, ending the
     turn immediately starts an **auction** for that property (see §6).
   - If the roll was doubles, the extra roll is granted here (see §4) instead of ending the
     turn.
3. **`out_of_turn`** — after the active player's turn resolves, every other non-bankrupt
   player gets one micro-turn, in seating order, where they may make trade offers or
   respond to a trade addressed to them, then must end their micro-turn. Only after every
   player has passed through `out_of_turn` does play advance to the next active player's
   `pre_roll`.
4. **`auction`** — entered only when an unowned property goes to auction (see §6); play
   returns to `post_roll` (if an extra roll is pending) or to `out_of_turn` once the auction
   resolves.

The engine enforces this state machine strictly: `get_allowed_actions()` returns only the
actions that are legal for the exact phase/player/situation right now. Ending a turn
prematurely, buying a property you can't afford, or declaring bankruptcy while solvent are
all outside the legal-action mask and cannot be executed.

## 4. Dice and doubles

- Two six-sided dice, rolled by the active player.
- Rolling doubles (not in jail) grants **one extra roll** this turn (`extra_roll_pending`),
  taken immediately after the current roll's landing/auction resolves.
- **Three consecutive doubles** in one turn send the player directly to Jail — the move is
  cancelled, no landing effect applies, and the extra-roll chain is reset.
- **Doubles rolled while in jail** release the player immediately, but — unlike normal
  doubles — do **not** grant an extra roll; the player simply moves normally on that roll.

## 5. Jail

- A player enters jail by landing on **Go To Jail**, by rolling **three consecutive
  doubles**, or (rarely, since cards are inert — see §2) never via a card in this ruleset.
- While in jail, at the start of `post_roll` (before rolling), the player may:
  - **Use a Get-Out-of-Jail-Free card** (`gooj_card` flag), if they hold one, or
  - **Pay bail** (**$50**), if they can afford it, or
  - Otherwise, simply roll.
- Rolling in jail:
  - **Doubles** → released immediately, moves normally, **no extra roll** (see §4).
  - **Non-doubles**, jail turn count **< 3** → stays in jail, turn ends immediately.
  - **Non-doubles**, jail turn count **≥ 3** (i.e. the third failed attempt) → forced to pay
    bail (deducting up to the full $50, capped at whatever cash the player has) and is
    released, then moves normally on that roll.
- Because Chance/Community Chest are inert, a player can only acquire a Get-Out-of-Jail-Free
  card through another player's bankruptcy transferring their card to a creditor (§7) — the
  card cannot currently be drawn from a deck.

## 6. Properties, rent, and auctions

- **Real estate (22 squares)**: base rent, or double base rent if the owner holds the
  complete color group with zero houses; rent tiers thereafter for 1–4 houses and a hotel,
  exactly as printed on the standard board (`PROPERTIES` in `constants.py`).
- **Railroads (4 squares)**: rent scales with how many railroads the same owner holds —
  $25 / $50 / $100 / $200 for 1/2/3/4 owned.
- **Utilities (2 squares)**: rent = dice roll from the current move **×4** if the owner holds
  one utility, **×10** if they hold both.
- **Monopoly (complete color group)**: unlocks house/hotel building on every property in
  that group and doubles the unimproved base rent. Owning a partial group does not.
- **Landing on an unowned property**: the player may buy it at list price if they can afford
  it. If they decline (or can't afford it) and then end the turn, the property is put up for
  **auction** among all non-bankrupt players. Per the code (`env.py: _start_auction`), the
  bidding order starts with the **active (landing) player themselves**, then continues around
  the table in turn order — it does **not** start with the player to their left. **[QUESTION
  FOR ORGANIZERS: is active-player-bids-first the intended rule, or should it be
  left-of-active as in traditional Monopoly? First-bidder position is a real strategic edge,
  worth confirming before relying on it.]**
  - **Auction bidding**: each bidder may **pass** (drops out) or **raise the current high
    bid** by a fixed increment: **+$1, +$10, +$50, or +$100** (only increments the bidder can
    actually afford are offered). Bidding rotates to the next remaining bidder after each
    action; the auction ends when only one bidder remains (or nobody bids at all, in which
    case the property stays unsold this turn) and that bidder pays their high bid to the
    bank for the property.
  - There is no separate "declined the option to buy → skip auction" path — decline always
    routes to auction once the turn ends.
- **Landing on your own property**: no effect.
- **Landing on a property owned by someone else**: rent is charged automatically, capped at
  whatever cash the paying player actually has (see §7 for what happens when that's not
  enough).
- **Mortgaging**: an unimproved, wholly-owned property (0 houses) can be mortgaged for its
  listed mortgage value; a mortgaged property earns no rent for its owner. Unmortgaging
  costs the mortgage value **plus 10% interest** (`cost = mortgage_value × 1.1`, verified in
  `env.py`) — same as traditional Monopoly rules.
- **Building houses/hotels**: only on a **complete color group**, only on an unmortgaged
  property, only while the shared bank supply of **32 houses / 12 hotels** has stock left,
  and only if the player can afford the listed house price for that color tier. **An
  even-building rule *is* enforced** (`env.py: _is_least_developed`, gating both
  `improve_house` and `improve_hotel`): a build action is only legal on the property that
  currently has the *fewest* houses within its color group, so you cannot pile houses onto
  one property while a group-mate has fewer. This was added after the ruleset was first
  written up — earlier notes in this doc said the opposite; the code is authoritative. There
  is still no matching sell-side enforcement, and no separate "building auction" for scarce
  houses — first affordable legal build action wins among the least-developed properties.
- **Selling houses/hotels**: sells back to the bank at **half the listed house price**,
  returning the physical piece(s) to the shared supply.
- **Selling an unimproved property back to the bank**: allowed at any time pre-roll (or when
  raising emergency cash) for its **mortgage value**, as long as it carries 0 houses. This is
  a research-engine convenience; traditional Monopoly does not offer this and relies on
  mortgages/trades instead (see §8).

## 7. Debt, cash-only tax, and bankruptcy

- **Rent and tax payments are capped at the payer's current cash on hand.** If a player owes
  more than they have in cash, the shortfall becomes an **explicit debt**: a
  `debt_player` / `debt_creditor` / `debt_amount` record is created (creditor is the property
  owner for rent; there is no creditor for tax — tax payments simply deduct up to available
  cash with no rollover debt).
- **Jail's forced bail-on-third-turn payment** is likewise capped at cash on hand — it will
  never send the player into debt by itself.
- While in debt, the indebted player's only legal actions become **liquidation** moves:
  sell houses/hotels, mortgage properties, or sell unimproved properties to the bank — in
  that priority order the engine offers them — until the debt is cleared or the player has
  nothing left to sell/mortgage, at which point **Declare Bankruptcy** becomes the only legal
  action.
- **Bankruptcy to a player creditor** (unpaid rent to another player): all of the bankrupt
  player's houses/hotels are first liquidated to the bank at half price (cash added to the
  bankrupt player before the transfer), any remaining debt is settled from that cash, then
  the **creditor receives**: all of the bankrupt player's remaining cash, all of their
  properties, and their Get-Out-of-Jail-Free card if they held one. **Each property keeps
  its current mortgage status on transfer** (`env.py: _do_bankrupt`, creditor branch never
  resets `mortgaged`): a property that was mortgaged arrives mortgaged (the new owner must
  pay it off at the usual ×1.1 to earn rent from it again), a property that was unmortgaged
  arrives unmortgaged. This differs from the bank-creditor case below, where properties are
  explicitly reset to unmortgaged.
- **Bankruptcy to the bank** (no player creditor — e.g. insolvent on a tax payment): the
  bankrupt player's properties return to the bank **unmortgaged** and become available for
  purchase/auction again; no player receives them directly.
- A bankrupt player is marked `bankrupt=True`, is skipped in all future turns and
  out-of-turn phases, and their only legal action for the rest of the game is `DO_NOTHING`.

## 8. Trades

- A player may, during their own `pre_roll` or during their own `out_of_turn` micro-turn,
  offer a trade to another player who has not yet acted this round (out-of-turn offers only
  target players who haven't taken their micro-turn yet, to avoid re-litigating a turn that
  already passed).
- Three trade shapes, all restricted to **unimproved (0-house) properties**:
  - **Buy offer**: cash for one of the other player's properties, at **75%, 100%, or 125%**
    of that property's list price (`TRADE_CASH_LEVELS`).
  - **Sell offer**: one of your properties for cash, at the same three price levels.
  - **Exchange offer**: one specific property of yours for one specific property of theirs,
    no cash involved.
- The action space only offers a trade action if the proposer can actually afford the cash
  side of it (or, for a sell offer, if the target can afford it) at generation time.
- A recipient may **Accept** or **Decline** a pending trade addressed to them; only one
  outgoing trade offer per player can be pending at a time.
- The fixed (non-learning) opponent agents use their own configured "buying personality"
  when deciding how to bid in auctions and whether to accept trades — that logic lives in
  `agents_fixed.py`, not in the shared engine rules.

## 9. Bank supply limits

- The bank holds a **finite** supply of **32 houses** and **12 hotels** shared by all
  players. Building is blocked once the relevant supply hits zero; selling
  houses/hotels/bankruptcy liquidation returns pieces to the shared pool immediately.
- There is no equivalent scarcity mechanic for cash, deeds, or Get-Out-of-Jail-Free cards —
  the bank's cash supply is treated as unlimited.

## 10. Game end

- The game ends immediately when only **one non-bankrupt player** remains — that player
  wins outright.
- If no one has gone bankrupt after **200 rounds** (`max_rounds`, configurable per game
  instance), the game ends in a **capped draw-off**: the player with the greatest
  **simulator net worth** (cash + a computed property/building value, not simply the sum of
  face prices — see `calculate_net_worth()` in `state.py`) is declared the winner. This
  200-round cap and net-worth tiebreak are research controls added for training stability,
  not part of traditional Monopoly (see §11).

## 11. Deliberate differences from traditional Monopoly (read this before assuming a "bug")

These are **intentional** simplifications baked into `ppo-plus-v2` for a bounded, learnable
action space — not oversights:

1. Chance and Community Chest squares have no card effect at all; no shuffled deck exists,
   so Get-Out-of-Jail-Free cards only change hands via bankruptcy transfer, never via a draw.
2. Houses/hotels require a complete color group to build, and **even-building across the
   group *is* enforced** (build only on the group's currently-least-developed property), but
   there is **no separate building-auction** for houses when supply is scarce.
3. **Unimproved properties can be sold directly back to the bank** for mortgage value —
   traditional rules only offer mortgaging or player-to-player trades, not a bank buy-back.
4. Mortgage/building legality is checked **per individual deed**, not by re-verifying every
   color-group constraint from the official rulebook on each action.
5. Trades use a **bounded, discretized action space**: cash offers are only 75%/100%/125% of
   list price, and exchanges are always exactly one property for one property — free-form
   cash amounts or multi-property packages are not representable.
6. **Income Tax, Luxury Tax, and the forced third-turn jail bail are all capped at cash on
   hand** and never create a rollover debt or a forced liquidation phase by themselves
   (unlike unpaid rent, which does create explicit debt).
7. The **200-round cap** and the **net-worth tiebreak** for capped games are research
   controls for bounding episode length during training, not traditional Monopoly rules.

## 12. Public interfaces and compatibility (for reference)

- **Observation vector**: 300 floats (`STATE_DIM`). The original 240-value prefix
  (`BASE_STATE_DIM`) — per-player position/cash/jail/GOOJ-card features plus per-property
  owner/mortgage/monopoly/improvement features — is preserved unchanged. The remaining 60
  values add: turn phase, whose turn/who's active, roll status, doubles streak, last dice,
  bank house/hotel supply, per-player bankruptcy and jail-turn progress, turn order,
  outstanding debt and its creditor, active auction property/high bid/leader/bidder set,
  extra-roll-pending flag, and incoming/outgoing trade context.
- **Action space**: 2,958 discrete actions (`ACTION_SPACE_SIZE`), broken down as:
  - 9 global actions (do nothing, end turn, roll dice, buy property, use GOOJ card, pay
    bail, declare bankruptcy, accept trade, decline trade)
  - 28 + 28 mortgage / unmortgage
  - 22 + 22 build house / build hotel
  - 22 + 22 sell house / sell hotel
  - 28 sell property to bank
  - 252 buy-trade offers (3 other players × 28 properties × 3 price levels)
  - 252 sell-trade offers (same breakdown)
  - 2,268 exchange offers (3 other players × 28 × 27 property pairs)
  - 5 auction actions (pass, +$1, +$10, +$50, +$100)
- **Checkpoints** record the ruleset identifier, observation dimension, and action
  dimension. A checkpoint trained under a different ruleset/shape fails to load with an
  explicit incompatibility error rather than silently loading into the wrong network/table
  shape.

---

# TÜRKÇE

## 1. `ppo-plus-v2` nedir?

`ppo-plus-v2`, bu depodaki tüm öğrenme yollarının (DDQN, PPO, CFR) paylaştığı **tek ve
kanonik oyun motorudur**. DDQN ve PPO, `env.py` üzerinden motoru doğrudan oynar; CFR
(`classic_cfr.py`) ise aynı motoru klonlayıp keşfeder. Algoritmaya özgü politika/checkpoint
kodu ayrı tutulur — yalnızca **kurallar** ortaktır.

Bu, resmi Monopoly kural kitabının sertifikalı bir uygulaması değil, **klasik tahta üzerine
kurulu bir araştırma kural setidir**. Bazı mekanikler, ortamın sınırlı ve tamamen
numaralandırılabilir bir eylem alanına (bkz. §9) sahip olması için kasıtlı olarak
basitleştirilmiştir — bu basitleştirmeler, "gerçek" Monopoly ile karıştırılmaması için §8'de
açıkça listelenmiştir.

## 2. Tahta, oyuncular, para

- Standart **ABD 40 kareli tahta** (`constants.py` içindeki `BOARD`): 22 emlak (renkli
  grup) mülkü, 4 tren istasyonu, 2 kamu hizmeti (elektrik/su), Başlangıç (Go), Hapishane /
  Sadece Ziyarette, Ücretsiz Park, Doğrudan Hapse Git, Gelir Vergisi, Lüks Vergi, 3× Şans
  (Chance), 3× Kamu Fonu (Community Chest).
- **4 oyuncu** (`NUM_PLAYERS = 4`), motor seviyesinde sabit — şu an oyun başına
  yapılandırılamıyor.
- Başlangıç nakiti: oyuncu başına **1.500$**.
- Başlangıç karesinden **geçmek** (üzerine tam basmak değil) **200$** öder. Başlangıç
  karesine tam basmak, geçiş maaşının ötesinde ayrı bir bonus ödemez (kodda tek bir
  Başlangıç-maaşı ödeme yolu vardır).
- **Gelir Vergisi** (kare 4): sabit **200$**, yalnızca nakitten ödenir (nakit sınırı için
  §7'ye bakın).
- **Lüks Vergi** (kare 38): sabit **100$**, yalnızca nakit.
- **Ücretsiz Park** hiçbir şey biriktirmez ve hiçbir şey ödemez — jackpot yoktur.
- **Şans ve Kamu Fonu karelerinin hiçbir etkisi yoktur.** Kart metinleri `constants.py`
  içinde referans/atmosfer amacıyla listelenmiştir, ancak bir Şans/Kamu Fonu karesine
  basmak `_handle_landing` içinde hiçbir işlem yapmaz. Karıştırılmış bir kart destesi
  bulunmadığından, Hapisten Çıkış Kartı normal oyun akışında asla bu şekilde
  kazanılamaz (bkz. §5'teki `gooj_card` bayrağı notu).

## 3. Sıra yapısı (faz durum makinesi)

Her oyuncunun sırası açık fazlardan geçer (`state.py: PHASES`):

1. **`pre_roll` (zar atmadan önce)** — aktif oyuncu, zar atmadan önce şunları yapabilir:
   ipotek verme/kaldırma, ev/otel satma, ev/otel inşa etme, geliştirilmemiş bir mülkü
   bankaya geri satma, takas teklifi yapma/yanıtlama, hapisteyse kefalet ödeme / Hapisten
   Çıkış Kartı kullanma veya bu fazı basitçe bitirme (`END_TURN` → `post_roll`'a geçer).
2. **`post_roll` (zar attıktan sonra)** — oyuncu önce zar atmak zorundadır (`ROLL_DICE`),
   yalnızca hapisteyse ve kefalet/kart seçeneği varsa istisna olur. Zarlar belirlendikten ve
   varsa kira/vergi borcu ödendikten sonra, oyuncu üzerine bastığı mülkü satın alabilir
   (sahipsizse ve gücü yetiyorsa), nakit toplamak için ipotek/satış yapabilir ya da sırayı
   bitirebilir.
   - Üzerine basılan mülk sahipsizse ve oyuncu onu **satın almazsa**, sırayı bitirmek o mülk
     için hemen bir **açık artırma** başlatır (bkz. §6).
   - Zar çift gelmişse, ekstra zar hakkı burada verilir (bkz. §4), sıra bitmez.
3. **`out_of_turn` (sıra dışı)** — aktif oyuncunun sırası tamamlandıktan sonra, iflas etmemiş
   diğer her oyuncu, oturma sırasına göre birer mikro-sıra alır; bu sırada takas teklifi
   yapabilir veya kendilerine yöneltilmiş bir teklife yanıt verebilirler, ardından
   mikro-sıralarını bitirmek zorundadırlar. Yalnızca tüm oyuncular `out_of_turn` fazından
   geçtikten sonra oyun bir sonraki aktif oyuncunun `pre_roll` fazına geçer.
4. **`auction` (açık artırma)** — yalnızca sahipsiz bir mülk açık artırmaya çıktığında
   girilir (bkz. §6); açık artırma sonuçlandığında oyun, bekleyen bir ekstra zar varsa
   `post_roll`'a, yoksa `out_of_turn`'a döner.

Motor bu durum makinesini sıkı biçimde uygular: `get_allowed_actions()` yalnızca o anki
tam faz/oyuncu/duruma göre yasal olan eylemleri döndürür. Sırayı erken bitirmek, gücü
yetmeyen bir mülkü satın almak veya ödeme gücü varken iflas ilan etmek — bunların hepsi
yasal-eylem maskesinin dışındadır ve çalıştırılamaz.

## 4. Zar ve çiftler

- Aktif oyuncu tarafından atılan iki adet altı yüzlü zar.
- Çift gelmesi (hapiste değilken) bu sırada **bir ekstra zar hakkı** verir
  (`extra_roll_pending`); bu hak, mevcut zarın basma/açık artırma sonucu belirlendikten
  hemen sonra kullanılır.
- Bir sırada **üst üste üç kez çift gelmesi**, oyuncuyu doğrudan hapse gönderir — hareket
  iptal edilir, hiçbir basma etkisi uygulanmaz ve ekstra-zar zinciri sıfırlanır.
- **Hapisteyken çift atmak** oyuncuyu anında serbest bırakır, ancak normal çiftlerin aksine
  **ekstra zar hakkı vermez**; oyuncu o zarla sadece normal şekilde ilerler.

## 5. Hapishane

- Bir oyuncu hapse şu şekillerde girer: **Doğrudan Hapse Git** karesine basarak, bir sırada
  **üst üste üç çift** atarak veya (kartlar etkisiz olduğu için — bkz. §2 — bu kural setinde
  neredeyse hiç) bir kart aracılığıyla.
- Hapisteyken, `post_roll` fazının başında (zar atmadan önce) oyuncu şunları yapabilir:
  - Elinde varsa bir **Hapisten Çıkış Kartı kullanmak** (`gooj_card` bayrağı), veya
  - Gücü yetiyorsa **kefalet ödemek** (**50$**), veya
  - Aksi halde, sadece zar atmak.
- Hapiste zar atmak:
  - **Çift** → anında serbest bırakılır, normal şekilde ilerler, **ekstra zar yok** (bkz.
    §4).
  - **Çift değil**, hapis tur sayısı **< 3** → hapiste kalır, sıra hemen biter.
  - **Çift değil**, hapis tur sayısı **≥ 3** (yani üçüncü başarısız deneme) → kefalet
    ödemeye zorlanır (elindeki nakit kadarıyla sınırlı, en fazla 50$ düşülür) ve serbest
    bırakılır, ardından o zarla normal şekilde ilerler.
- Şans/Kamu Fonu kareleri etkisiz olduğundan, bir oyuncu bir Hapisten Çıkış Kartını yalnızca
  başka bir oyuncunun iflası sırasında alacaklıya devredilmesiyle elde edebilir (§7) — kart
  şu an bir desteden çekilerek kazanılamaz.

## 6. Mülkler, kira ve açık artırmalar

- **Emlak (22 kare)**: temel kira; sahip, sıfır evle tam renkli grubun tamamına sahipse
  temel kiranın iki katı; sonrasında 1-4 ev ve otel için standart tahtada yazan kira
  seviyeleri (`constants.py` içindeki `PROPERTIES`).
- **Tren istasyonları (4 kare)**: kira, aynı sahibin kaç tren istasyonuna sahip olduğuna
  göre ölçeklenir — 1/2/3/4 istasyon için 25$ / 50$ / 100$ / 200$.
- **Kamu hizmetleri (2 kare)**: kira = o hamledeki zar toplamı **×4** (sahip bir kamu
  hizmetine sahipse) veya **×10** (her ikisine de sahipse).
- **Tekel (tam renkli grup)**: o gruptaki her mülkte ev/otel inşasının kilidini açar ve
  geliştirilmemiş temel kirayı ikiye katlar. Kısmi bir gruba sahip olmak bunu sağlamaz.
- **Sahipsiz bir mülke basmak**: oyuncu, gücü yetiyorsa liste fiyatından satın alabilir.
  Reddederse (veya gücü yetmezse) ve ardından sırayı bitirirse, mülk iflas etmemiş tüm
  oyuncular arasında **açık artırmaya** çıkarılır. Koda göre (`env.py: _start_auction`),
  teklif sırası **aktif (basan) oyuncunun kendisiyle** başlar, ardından masa etrafında sıra
  düzeninde devam eder — solundaki oyuncuyla **başlamaz**. **[ORGANİZATÖRLERE SORULACAK:
  ilk teklifi aktif oyuncunun vermesi kasıtlı bir kural mı, yoksa geleneksel Monopoly'deki
  gibi solundaki oyuncuyla mı başlaması gerekiyor? İlk teklifçi olmak gerçek bir stratejik
  avantaj, güvenmeden önce teyit edilmeli.]**
  - **Açık artırma teklifi**: her teklifçi **pas geçebilir** (çekilir) veya mevcut en
    yüksek teklifi sabit bir artışla yükseltebilir: **+1$, +10$, +50$ veya +100$** (yalnızca
    teklifçinin gerçekten karşılayabileceği artışlar sunulur). Her eylemden sonra sıra bir
    sonraki kalan teklifçiye geçer; açık artırma yalnızca bir teklifçi kaldığında (veya hiç
    teklif gelmediğinde — bu durumda mülk bu sıra satılmadan kalır) biter ve o teklifçi en
    yüksek teklifini bankaya ödeyerek mülkü alır.
  - "Satın alma seçeneğini reddet → açık artırmayı atla" diye ayrı bir yol yoktur; sıra
    bittiğinde reddetme her zaman açık artırmaya yönlendirir.
- **Kendi mülkünüze basmak**: hiçbir etkisi yoktur.
- **Başkasına ait bir mülke basmak**: kira otomatik olarak tahsil edilir, ödeyen oyuncunun
  gerçekte sahip olduğu nakitle sınırlıdır (bu yeterli olmadığında ne olacağı için bkz. §7).
- **İpotek verme**: geliştirilmemiş (0 evli), tamamen sahip olunan bir mülk, listelenen
  ipotek değeri karşılığında ipotek edilebilir; ipotekli bir mülk sahibi için kira
  kazandırmaz. İpoteği kaldırmak, ipotek değerinin **%10 faiziyle birlikte** geri
  ödenmesini gerektirir (`cost = ipotek_değeri × 1.1`, `env.py`'de doğrulandı) — geleneksel
  Monopoly kuralıyla aynı.
- **Ev/otel inşası**: yalnızca **tam bir renkli grupta**, yalnızca ipotekli olmayan bir
  mülkte, yalnızca paylaşılan banka stoğunda **32 ev / 12 otel** kaldığı sürece ve oyuncu o
  renk kademesi için listelenen ev fiyatını karşılayabiliyorsa mümkündür. **Eşit inşa kuralı
  uygulanır** (`env.py: _is_least_developed`, hem `improve_house` hem `improve_hotel`
  eylemlerini kısıtlar): bir inşa eylemi yalnızca kendi renk grubunda o an **en az** eve
  sahip mülkte yasaldır — bir mülke ev yığıp grup arkadaşını geride bırakamazsınız. Bu kural
  ruleset ilk yazıldıktan sonra eklendi; bu doc'un önceki sürümü tam tersini söylüyordu, kod
  esas alınmalıdır. Satış tarafında eşdeğer bir zorunluluk hâlâ yok, kıt evler için ayrı bir
  "inşaat açık artırması" da yok — en az gelişmiş mülkler arasında karşılanabilir ilk yasal
  inşa eylemi kazanır.
- **Ev/otel satma**: bankaya listelenen ev fiyatının **yarısına** geri satılır, fiziksel
  parça(lar) paylaşılan stoğa geri döner.
- **Geliştirilmemiş bir mülkü bankaya geri satmak**: 0 evi olduğu sürece, zar atmadan önce
  herhangi bir zamanda (veya acil nakit toplarken) **ipotek değeri** karşılığında izin
  verilir. Bu bir araştırma-motoru kolaylığıdır; geleneksel Monopoly bunu sunmaz, bunun
  yerine ipotek/takas kullanır (bkz. §8).

## 7. Borç, yalnızca-nakit vergi ve iflas

- **Kira ve vergi ödemeleri, ödeyen oyuncunun eldeki nakdiyle sınırlıdır.** Bir oyuncu
  eldeki nakdinden fazlasını borçluysa, eksik kısım açık bir **borç** haline gelir:
  `debt_player` / `debt_creditor` / `debt_amount` kaydı oluşturulur (kira için alacaklı
  mülk sahibidir; vergi için alacaklı yoktur — vergi ödemeleri sadece mevcut nakit kadar
  düşülür, devreden borç oluşmaz).
- **Hapiste üçüncü turda zorunlu kefalet ödemesi** de aynı şekilde eldeki nakitle
  sınırlıdır — bu tek başına oyuncuyu borca sokmaz.
- Borçluyken, borçlu oyuncunun tek yasal eylemleri **tasfiye** hamleleri olur: ev/otel
  satmak, mülk ipotek etmek veya geliştirilmemiş mülkleri bankaya satmak — motorun bu
  öncelik sırasıyla sunduğu — borç kapanana veya oyuncunun satacak/ipotek edecek bir şeyi
  kalmayana kadar; bu noktada **İflas İlan Et** tek yasal eylem haline gelir.
- **Bir oyuncu alacaklıya iflas** (başka bir oyuncuya ödenmemiş kira): iflas eden
  oyuncunun tüm ev/otelleri önce bankaya yarı fiyattan tasfiye edilir (nakit, devirden önce
  iflas eden oyuncuya eklenir), kalan borç bu nakitten ödenir, ardından **alacaklı şunları
  alır**: iflas eden oyuncunun kalan tüm nakdi, tüm mülkleri ve varsa Hapisten Çıkış Kartı.
  **Her mülk devredilirken mevcut ipotek durumunu korur** (`env.py: _do_bankrupt`, alacaklı
  dalı `mortgaged` bayrağını hiç sıfırlamaz): ipotekli bir mülk ipotekli olarak gelir (yeni
  sahip kira kazanmaya başlamadan önce yine ×1,1 ile ipoteği kaldırmalıdır), ipoteksiz bir
  mülk ipoteksiz olarak gelir. Bu, aşağıdaki bankaya-iflas durumundan farklıdır; orada
  mülkler açıkça ipoteksize sıfırlanır.
- **Bankaya iflas** (oyuncu alacaklısı yok — örn. bir vergi ödemesinde ödeme gücü
  kalmaması): iflas eden oyuncunun mülkleri bankaya **ipoteksiz** olarak döner ve tekrar
  satın alma/açık artırmaya açık hale gelir; hiçbir oyuncu bunları doğrudan almaz.
- İflas eden bir oyuncu `bankrupt=True` olarak işaretlenir, sonraki tüm sıralarda ve
  sıra-dışı fazlarda atlanır ve oyunun geri kalanında tek yasal eylemi `DO_NOTHING`'dir.

## 8. Takaslar

- Bir oyuncu, kendi `pre_roll` fazında veya kendi `out_of_turn` mikro-sırasında, bu turda
  henüz hareket etmemiş başka bir oyuncuya takas teklif edebilir (sıra-dışı teklifler
  yalnızca henüz mikro-sırasını almamış oyuncuları hedefler; böylece zaten geçmiş bir sıra
  yeniden tartışmaya açılmaz).
- Üç takas şekli vardır, hepsi **geliştirilmemiş (0 evli) mülklerle** sınırlıdır:
  - **Satın alma teklifi**: diğer oyuncunun bir mülkü karşılığında nakit, o mülkün liste
    fiyatının **%75, %100 veya %125**'i (`TRADE_CASH_LEVELS`).
  - **Satış teklifi**: sizin bir mülkünüz karşılığında nakit, aynı üç fiyat seviyesinde.
  - **Değişim teklifi**: sizin belirli bir mülkünüz karşılığında onların belirli bir
    mülkü, nakit dahil değil.
- Eylem alanı, bir takas eylemini yalnızca teklif eden kişi nakit tarafını gerçekten
  karşılayabiliyorsa (ya da bir satış teklifinde hedef karşılayabiliyorsa) üretim anında
  sunar.
- Bir alıcı, kendisine yöneltilmiş bekleyen bir teklifi **Kabul Et** veya **Reddet**
  edebilir; her oyuncunun aynı anda bekleyen yalnızca bir giden teklifi olabilir.
- Sabit (öğrenmeyen) rakip ajanlar, açık artırmalarda nasıl teklif vereceklerine ve
  takasları kabul edip etmeyeceklerine karar verirken kendi yapılandırılmış "satın alma
  kişiliklerini" kullanır — bu mantık paylaşılan motor kurallarında değil,
  `agents_fixed.py` içindedir.

## 9. Banka stok sınırları

- Banka, tüm oyuncular tarafından paylaşılan **sınırlı** bir stoğa sahiptir: **32 ev** ve
  **12 otel**. İlgili stok sıfıra düştüğünde inşaat engellenir; ev/otel satışı veya iflas
  tasfiyesi parçaları anında paylaşılan havuza geri döndürür.
- Nakit, tapu veya Hapisten Çıkış Kartları için eşdeğer bir kıtlık mekanizması yoktur —
  bankanın nakit stoğu sınırsız kabul edilir.

## 10. Oyun sonu

- Yalnızca **iflas etmemiş bir oyuncu** kaldığında oyun hemen biter — o oyuncu doğrudan
  kazanır.
- **200 tur** (`max_rounds`, oyun örneği başına yapılandırılabilir) sonunda kimse iflas
  etmediyse, oyun **sınırlı bir eşitlik bozma** ile biter: en yüksek **simülatör net
  değerine** (nakit + hesaplanmış mülk/bina değeri — sadece yüzeysel fiyatların toplamı
  değil, bkz. `state.py` içindeki `calculate_net_worth()`) sahip oyuncu kazanan ilan edilir.
  Bu 200 tur sınırı ve net-değer eşitlik bozma, geleneksel Monopoly'nin bir parçası değil,
  eğitim kararlılığı için eklenmiş araştırma kontrolleridir (bkz. §11).

## 11. Geleneksel Monopoly'den kasıtlı farklar (bunu "hata" sanmadan önce okuyun)

Bunlar, `ppo-plus-v2` içine sınırlı ve öğrenilebilir bir eylem alanı için **kasıtlı olarak**
gömülmüş basitleştirmelerdir — gözden kaçma değildir:

1. Şans ve Kamu Fonu karelerinin hiçbir kart etkisi yoktur; karıştırılmış bir deste
   bulunmadığından, Hapisten Çıkış Kartları yalnızca iflas devriyle el değiştirir, asla bir
   çekilişle değil.
2. Ev/otel inşası için tam bir renkli grup gerekir ve **grup genelinde eşit inşa
   zorunludur** (yalnızca o an en az geliştirilmiş mülke inşa edilebilir), ancak stok
   kıtken evler için ayrı bir **inşaat açık artırması yoktur**.
3. **Geliştirilmemiş mülkler doğrudan bankaya ipotek değerinden geri satılabilir** —
   geleneksel kurallar yalnızca ipotek veya oyuncular arası takas sunar, bankaya geri
   satım sunmaz.
4. İpotek/inşaat yasallığı, resmi kural kitabındaki her renk-grubu kısıtını her eylemde
   yeniden doğrulamak yerine, **her tapu için ayrı ayrı** kontrol edilir.
5. Takaslar **sınırlı, ayrık bir eylem alanı** kullanır: nakit teklifleri yalnızca liste
   fiyatının %75/%100/%125'idir ve değişimler her zaman tam olarak bir mülk karşılığında
   bir mülktür — serbest biçimli nakit miktarları veya çok mülklü paketler temsil
   edilemez.
6. **Gelir Vergisi, Lüks Vergi ve hapiste zorunlu üçüncü tur kefaleti, hepsi eldeki
   nakitle sınırlıdır** ve tek başlarına devreden bir borç veya zorunlu bir tasfiye fazı
   oluşturmazlar (ödenmemiş kiranın aksine, o açık borç oluşturur).
7. Sınırlı oyunlar için **200 tur sınırı** ve **net-değer eşitlik bozma**, eğitim
   sırasında bölüm uzunluğunu sınırlamak için araştırma kontrolleridir, geleneksel
   Monopoly kuralları değildir.

## 12. Genel arayüzler ve uyumluluk (referans için)

- **Gözlem vektörü**: 300 float (`STATE_DIM`). Orijinal 240 değerlik önek
  (`BASE_STATE_DIM`) — oyuncu başına konum/nakit/hapis/GOOJ-kartı özellikleri artı mülk
  başına sahip/ipotek/tekel/geliştirme özellikleri — değişmeden korunur. Kalan 60 değer
  şunları ekler: sıra fazı, sıradaki/aktif oyuncu, zar atma durumu, çift serisi, son zar,
  banka ev/otel stoğu, oyuncu başına iflas ve hapis-tur ilerlemesi, sıra düzeni, bekleyen
  borç ve alacaklısı, aktif açık artırma mülkü/en yüksek teklif/lider/teklifçi kümesi,
  ekstra-zar-bekliyor bayrağı ve gelen/giden takas bağlamı.
- **Eylem alanı**: 2.958 ayrık eylem (`ACTION_SPACE_SIZE`), şu şekilde dağılır:
  - 9 genel eylem (hiçbir şey yapma, sırayı bitir, zar at, mülk satın al, GOOJ kartı
    kullan, kefalet öde, iflas ilan et, takası kabul et, takası reddet)
  - 28 + 28 ipotek ver / ipotek kaldır
  - 22 + 22 ev inşa et / otel inşa et
  - 22 + 22 ev sat / otel sat
  - 28 mülkü bankaya sat
  - 252 satın alma-takas teklifi (3 diğer oyuncu × 28 mülk × 3 fiyat seviyesi)
  - 252 satış-takas teklifi (aynı dağılım)
  - 2.268 değişim teklifi (3 diğer oyuncu × 28 × 27 mülk çifti)
  - 5 açık artırma eylemi (pas, +1$, +10$, +50$, +100$)
- **Checkpoint'ler**, kural seti kimliğini, gözlem boyutunu ve eylem boyutunu kaydeder.
  Farklı bir kural seti/boyut altında eğitilmiş bir checkpoint, sessizce yanlış ağ/tablo
  boyutuna yüklenmek yerine açık bir uyumsuzluk hatasıyla yüklenemez.
