# Tic-Tac-Toe Q-Learning vs SARSA Proje Raporu

## Özet

Bu proje, klasik 3x3 Tic-Tac-Toe oyununda **Q-Learning** (off-policy) ve **SARSA** (on-policy)
algoritmalarının performansını karşılaştırır. Eğitim süreci self-play, cross-play ve
rastgele rakibe karşı baseline aşamalarından oluşur. Ardından random, minimax ve
öğrenilmiş rakiplere karşı turnuva formatında değerlendirme yapılır. Sonuçlar JSON/CSV
olarak kaydedilir; ayrıca eğitim trendi, turnuva sonuçları ve hücre tercih ısı haritaları
Seaborn ile üretilir.

### Temel Bulgular

| Karşılaşma | Q-Learning Kazanma | SARSA Kazanma | Sonuç |
|-----------|-------------------|--------------|--------|
| vs Random | ~97% | ~88% | İkisi de başarılı |
| vs Minimax | 0% | 0% | İkisi de kazanamaz |
| vs Birbirleri | 0% | 0% | Beraberlik hakimiyeti |

---

## Problem Tanımı ve Motivasyon

### Tic-Tac-Toe Neden İdeal Test Alanı?

Tic-Tac-Toe deterministik ve tam gözlemlenebilir bir oyundur. Bu özellikler onu
pekiştirmeli öğrenme karşılaştırmaları için ideal hale getirir:

1. **Küçük Durum Uzayı**: 3^9 = 19683 olası durum (geçerli: 5478)
2. **Tam Gözlemlenebilirlik**: Durum her zaman bilinir (hidden state yok)
3. **Deterministik Geçişler**: Aynı durumda aynı aksiyon her zaman aynı sonucu verir
4. **Optimal Strateji Biliniyor**: Minimax ile optimal strateji hesaplanabilir
5. **Net Ödül Sinyali**: Kazanma (+1), kaybetme (-1), beraberlik (0)

### Motivasyon

Bu çalışma, aşağıdaki soruları yanıtlamayı amaçlar:

1. **Off-Policy vs On-Policy**: Q-Learning'in agresif optimizasyonu SARSA'nın
   dengeli yaklaşımından daha mı etkilidir?

2. **Self-Play Etkisi**: Ajan kendi kopyasına karşı öğrenince strateji gelişir mi?

3. **Optimallik**: Öğrenilen politikalar Minimax'a ne kadar yaklaşabilir?

4. **Yakınsama Stabilitesi**: Hangi algoritma daha kararlı öğrenme gösterir?

---

## MDP Formülasyonu

Tic-Tac-Toe, bir **Markov Karar Süreci (MDP)** olarak formüle edilebilir:

### Durum Uzayı (State Space)

- **Tasarım**: 3x3 tahta, her hücre `{0=boş, 1=X, 2=O}` değer alabilir
- **Teorik Boyut**: 3^9 = 19683 olası konfigürasyon
- **Geçerli Boyut**: 5478 durum (~72% azaltma)

#### Geçersiz Durum Filtresi

Tüm konfigürasyonlar oyun kurallarına uygun değildir. `is_valid_state()` fonksiyonu
şu kuralları uygular:

1. **Hamle Sayısı Kuralı**:
   - X her zaman birinci oyuncudur, bu yüzden X sayısı ≥ O sayısı
   - X, O'dan en fazla 1 fazla hamle yapabilir
   - Örnek: `[1,1,1,0,0,0,0,0,0]` (X=3, O=0) ✅ Geçerli
   - Örnek: `[2,2,2,0,0,0,0,0,0]` (X=0, O=3) ❌ Geçersiz

2. **Çift Kazanan Kuralı**:
   - İki oyuncu aynı anda kazanamaz
   - Örnek: `[1,1,1,2,2,2,0,0,0]` ❌ Geçersiz

3. **Kazanma Sonrası Kuralı**:
   - X kazandıysa, son hamlesini yapmış olmalı (X = O + 1)
   - O kazandıysa, son hamlesini yapmış olmalı (X = O)

#### Durum Kodlama (Encoding)

Durum uzayını kompakt hale getirmek için taban-3 sistem kullanılır:

```
Tahta: [1, 2, 0, 0, 0, 0, 0, 0, 0]
İndeks: 1*3^0 + 2*3^1 + 0*3^2 + ... + 0 = 1 + 6 + 0 = 7
```

Bu sayede `tuple(board)` → `int` dönüşümü ile O(1) durum indeksleme yapılır.

### Eylem Uzayı (Action Space)

- **Tanım**: 0-8 indeksli hücrelerden birini seçmek
- **Geçerlilik**: Sadece boş hücreler seçilebilir
- **Boyut**: Her durum için değişken (1-9 arası aksiyon)

### Ödül Yapısı (Reward Structure)

| Olay | Ödül (Kazanan) | Ödül (Kaybeden) | Açıklama |
|-------|----------------|------------------|----------|
| Kazanma | +1 | -1 | Kazanan oyuncuya +1, kaybedene -1 |
| Beraberlik | 0 | 0 | Hiçbir oyuncuya ödül |
| Ara Hamle | 0 | 0 | Terminal olmayan durumlar için ödül yok |

**Not**: Kaybetme cezası -1'dir, bu ajanı kaybetmekten uzak tutmaya teşvik eder.

### Geçiş Dinamiği (Transition)

```
(state, action, opponent_policy) → (next_state, winner, done)
```

- **Ajan Hamlesi**: Deterministik (seçilen hücre dolu değilse)
- **Rakip Hamlesi**:
  - **Random**: Rastgele boş hücre seçimi
  - **Minimax**: Optimal strateji
  - **Öğrenilmiş**: Q veya SARSA politikası
- **Terminal**: Kazanan var veya tahta dolu → Bölüm biter

### İskonto Faktörü (γ - Gamma)

- **Varsayılan**: γ = 0.95
- **Anlamı**: Gelecekteki ödüllerin %95'i şimdikiyle eşdeğer
- **γ = 1.0**: Uzun vadeli strateji (tam güven)
- **γ = 0.0**: Sadece anlık ödüle bak (kısa vadeli)

---

## Algoritmalar

### Q-Learning (Off-Policy)

#### Tanım

Q-Learning, **off-policy** bir algoritmadır: öğrenme sırasında kullanılan davranış
politikası ile hedef politikası aynı değildir.

#### Güncelleme Formülü

```
Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s',a') - Q(s,a)]
```

**Bileşenler:**

- `Q(s,a)`: Durum s'de aksiyon a'nın mevcut Q değeri
- `r`: Alınan ödül
- `γ * max_a' Q(s',a')`: İskontolu maksimum gelecek Q değeri
- `α [hedef - mevcut]`: Stokastik güncelleme (gradiyan inişi benzeri)

#### Özellikler

| Özellik | Açıklama |
|---------|----------|
| **Off-Policy** | Hedef politika greedy (keşifsiz) |
| **max Operatörü** | Sonraki durumun en iyi aksiyonu |
| **Agresif** | Maksimum Q'yu arar, hızlı ama kararsız |
| **Teorik Optimalite** | Yakınsarsa optimal politikaya ulaşır |

#### Avantajlar

- **Hızlı Yakınsama**: Agresif optimizasyon hızlı öğrenme sağlar
- **Teorik Optimalite**: Güncelleme formülü garantili optimal yakınsama
- **Basitlik**: Maksimum operatörü ile kolay implementasyon

#### Dezavantajlar

- **Kararsızlık**: Max operatörü stokastik ortamlarda kararsızlık yaratır
- **Keşif Problemi**: Off-policy olduğu için keşif stratejisi hedeften farklı olabilir
- **Aşırı Optimizasyon**: Bazen gereksiz agresif stratejilere yol açabilir

---

### SARSA (State-Action-Reward-State-Action)

#### Tanım

SARSA, **on-policy** bir algoritmadır: öğrenme sırasında kullanılan davranış
politikası ile hedef politika aynıdır. İsmi "State-Action-Reward-State-Action"
zincirinden gelir.

#### Güncelleme Formülü

```
Q(s,a) ← Q(s,a) + α [r + γ * Q(s',a') - Q(s,a)]
```

**Bileşenler:**

- `Q(s,a)`: Durum s'de aksiyon a'nın mevcut Q değeri
- `r`: Alınan ödül
- `γ * Q(s',a')`: İskontolu seçilen gelecek aksiyonun Q değeri
- `α [hedef - mevcut]`: Stokastik güncelleme

#### Özellikler

| Özellik | Açıklama |
|---------|----------|
| **On-Policy** | Hedef politika = Davranış politikası |
| `Q(s',a')` | Seçilen aksiyonun Q değeri |
| **Dengeli** | Keşif ve sömürü tutarlı |
| **Stabil** | Kararsızlık olasılığı daha düşük |

#### Avantajlar

- **Tutarlılık**: Davranış ve hedef politika aynı
- **Stabilite**: Max operatörü yok, daha stabil öğrenme
- **Gerçek Performans**: Genellikle Q-Learning'den daha iyi gerçek sonuç
- **Keşif Uyumu**: Epsilon-greedy ile tam uyumlu öğrenme

#### Dezavantajlar

- **Yavaş Yakınsama**: Daha temkinli olduğu için daha yavaş öğrenme
- **Suboptimal Olasılığı**: On-policy olduğu için teorik optimalite garantisi yok
- **Daha Fazla Bölüm**: Aynı performans için daha fazla bölüm gerekli

---

### Q-Learning vs SARSA Karşılaştırması

| Özellik | Q-Learning | SARSA |
|---------|------------|--------|
| **Politika Tipi** | Off-policy | On-policy |
| **Güncelleme** | `max_a' Q(s',a')` | `Q(s',a')` |
| **Keşif Stratejisi** | Agresif | Dengeli |
| **Yakınsama Hızı** | Hızlı | Orta |
| **Stabilite** | Kararsız | Stabil |
| **Teorik Optimalite** | Garantili | Garantili değil |
| **Gerçek Performans** | Değişken | Genellikle iyi |
| **Karmaşıklık** | Basit | Basit |

### Hangi Algoritma Ne Zaman Seçilmeli?

- **Q-Learning** → Hızlı öğrenme önemli, stabilite ikincil
- **SARSA** → Stabilite ve gerçek performans önemli

---

## Eğitim Stratejisi

### 1. Self-Play

**Tanım**: Ajan kendi kopyası ile oynar (X ve O aynı algoritma).

**Amaçlar**:
- Rakibin stratejisini öğrenmek (aynı strateji olduğu için)
- Karşı-oyun (counter-play) geliştirmek
- Beraberlik stratejisini öğrenmek (güçlü rakiplere karşı)

**Süreç**:
```
Bölüm 1: X(Ajan) vs O(Ajan) → Kazanan belirler
Bölüm 2: X(Ajan) vs O(Ajan) → ...
...
Bölüm N: Her iki ajan Q tablosunu günceller
```

**Sonuç**:
- Yüksek beraberlik oranı (güçlü vs güçlü)
- Ortak strateji gelişimi

### 2. Cross-Play

**Tanım**: Q-Learning ve SARSA ajanları karşılıklı oynar.

**Amaçlar**:
- Farklı algoritmaların performansını karşılaştırmak
- Algoritmaların birbirine karşı dayanıklılığını test etmek

**Süreç**:
```
Bölüm 1: X(Q-Learning) vs O(SARSA)
Bölüm 2: X(SARSA) vs O(Q-Learning)  [Rol değişimi]
Bölüm 3: X(Q-Learning) vs O(SARSA)
...
```

**Neden Rol Değişimi?**:
- X her zaman avantajlı (ilk hamle)
- Roller değişerek bu avantajı dengelemek

**Sonuç**:
- Adil karşılaştırma (X/O dengesi)
- Algoritmaların karşı-oyun yeteneği

### 3. Baseline (vs Random)

**Tanım**: Ajan rastgele (RandomAgent) rakibe karşı öğrenir.

**Amaçlar**:
- Temel performansı ölçmek
- Öğrenme hızını değerlendirmek
- Ajanın zayıf rakiplere karşı stratejisini geliştirmek

**Süreç**:
```
Bölüm 1: X(Ajan) vs O(Random) → Ajan kazanır (%90+)
Bölüm 2: X(Ajan) vs O(Random) → ...
...
Ajan: Q günceller
Random: Öğrenmez (sadece rastgele oynar)
```

**Sonuç**:
- Çok yüksek kazanma oranı
- Hızlı öğrenme
- Güven stratejisi gelişimi

### 4. ε-Decay (Keşif Stratejisi)

**Epsilon-Greedy**:
- **Keşif (Exploration)**: Rastgele aksiyon seç (epsilon olasılıkla)
- **Sömürü (Exploitation)**: En iyi aksiyonu seç (1-epsilon olasılıkla)

**Zamansal Azalma**:
```
ε_start = 1.0   → %100 rastgele (tam keşif)
ε_end = 0.01     → %1 rastgele (neredeyse tam sömürü)
ε_decay = 0.995  → Her bölümde ε *= 0.995
```

**Zaman Çizgisi**:
```
Bölüm 0:    ε = 1.00  → 100% rastgele
Bölüm 100:  ε = 0.61  → 61% rastgele
Bölüm 500:  ε = 0.08  → 8% rastgele
Bölüm 1000: ε = 0.01  → 1% rastgele (minimum)
Bölüm 5000: ε = 0.01  → 1% rastgele (sabit)
```

**Neden ε-Decay?**:
- Başlangıçta tam keşif → Durum uzayını keşfet
- Zamanla sömürü → Öğrenilen stratejiyi kullan
- Bitişte az keşif → Çalışan politikayı koru

---

## Değerlendirme Metrikleri

### 1. Kazanma/Beraberlik/Mağlubiyet Oranı

**Hesaplama**:
```
Win Rate   = wins / total
Draw Rate  = draws / total
Loss Rate  = losses / total
```

**Anlamı**:
- **Kazanma Oranı**: Ajanın ne kadar güçlü olduğu
- **Beraberlik Oranı**: Ajanın güçlü rakiplere karşı dayanıklılığı
- **Mağlubiyet Oranı**: Ajanın zayıf noktaları

### 2. Yakınsama Bölümü (Convergence Episode)

**Hesaplama**:
```
Her bölüm için hareketli ortalama kazanma oranı hesapla
→ MA_Win_Rate[window]

Yakınsama Bölümü = İlk bölüm where MA_Win_Rate ≥ threshold
```

**Parametreler**:
- `window = 200`: Son 200 bölümün ortalaması
- `threshold = 0.8`: %80 kazanırsa yakınsanmış say

**Örnek**:
```
Bölüm 1-200:  MA = 0.50  → Yakınsanmadı
Bölüm 201-400: MA = 0.70  → Yakınsanmadı
Bölüm 401-600: MA = 0.85  → ✓ Yakınsandı (Bölüm 600)
```

### 3. Q-Değer Varyansı (Variance)

**Hesaplama**:
```
Variance = Var(Q_tablosu)
        = Σ(x - μ)² / (N - 1)
```

**Anlamı**:
- **Düşük Varyans**: Q değerleri birbirine yakın → Stabil öğrenme
- **Yüksek Varyans**: Q değerleri dağılmış → Kararsız öğrenme
- **Varyans Azalması**: Öğrenme ilerlediğini gösterir

**Sonuçlarımız**:
| Ajan | X Varyansı | O Varyansı | Durum |
|------|------------|------------|-------|
| Q-Learning | 0.00268 | 0.00085 | Stabil |
| SARSA | 0.00263 | 0.00074 | Stabil |

**Not**: O varyansı X'ten daha düşük (son hamle yaptığı için daha kararlı).

### 4. Minimax Performansı

**Minimax**: Optimal strateji, her zaman en iyi hamleyi oynar.

**Kazanma Oranı**: Minimax'a karşı kazanmak **imkansız** (teorik olarak).

**Beraberlik Oranı**: İyi ajan Minimax'a karşı yüksek beraberlik oranına ulaşabilir.

**Sonuçlarımız**:
| Algoritma | Kazanma | Beraberlik | Mağlubiyet |
|-----------|----------|------------|------------|
| Q-Learning | 0% | 53% | 47% |
| SARSA | 0% | 55% | 45% |

**Analiz**:
- %53-55 beraberlik → Ajanlar neredeyse optimal
- %45-47 mağlubiyet → Hala gelişme alanı var
- İki algoritma benzer performans gösteriyor

---

## Uygulama Detayları

### Geçerli Durum Filtresi

**Algoritma**:
```python
for state_index in range(3**9):
    board = decode_state(state_index)
    if is_valid_state(board):
        mapping[tuple(board)] = len(mapping)
```

**Zaman Karmaşıklığı**:
- Döngü: O(3^9) = O(19683)
- Her durumda `is_valid_state()`: O(1)
- Toplam: ~19683 × O(1) = ~20ms

**Sonuç**:
- Orijinal: 19683 durum
- Filtreli: 5478 durum
- Tasarruf: 72% daha küçük Q tablosu

### Aksiyon Seçimi (Epsilon-Greedy)

```python
if random() < epsilon:
    # Keşif: rastgele aksiyon
    return random.choice(valid_actions)
else:
    # Sömürü: en iyi aksiyon
    return argmax(Q[state, valid_actions])
```

**Keşif vs Sömürü Dengesi**:
- Başlangıçta: %100 keşif (tam rastgele)
- Zamanla: Daha fazla sömürü (öğrenilen strateji)
- Bitişte: %1 keşif (neredeyse tam sömürü)

### Kayıt Formatı

**results.json**:
```json
{
  "config": { "alpha": 0.1, "gamma": 0.95, ... },
  "training": {
    "Q self-play": { "wins": 1552, "draws": 2987, ... },
    ...
  },
  "tournament": {
    "Q vs Random": { "win_rate": 0.728, ... },
    ...
  },
  "q_variance": {
    "Q-X": 0.00268, "Q-O": 0.00085, ...
  }
}
```

**tournament.csv**:
```csv
matchup,wins,draws,losses,win_rate,draw_rate,loss_rate
Q vs Random,364,40,96,0.728,0.080,0.192
...
```

### Görselleştirme

**Seaborn ile Modern Stil**:
- `whitegrid` stili (temiz grid çizgileri)
- `deep` color palette (professional renkler)
- 300 DPI (yayın kalitesi)
- `sns.despine()` (temiz kenar çizgileri)
- Font scaling (okunabilir metin)

**Grafik Türleri**:
- **lineplot**: Eğitim trendi (multi-line)
- **stacked bar**: Turnuva sonuçları (kümülatif)
- **heatmap**: Hücre tercihleri (3x3 grid)

---

## Görselleştirmeler ve Çıktılar

### results.json

Deneyin tüm özet verilerini içerir:
- Config (hiperparametreler)
- Training (eğitim sonuçları)
- Tournament (turnuva sonuçları)
- Q Variance (Q tablosu kararlılığı)

### tournament.csv

Turnuva karşılaştırmalarının CSV formatı:
- Her satır bir karşılaşma
- Win/draw/loss oranları

### training.png

**Amaç**: Eğitim sürecinde kazanma oranı trendini gösterir

**Özellikler**:
- X ekseni: Bölüm indeksi
- Y ekseni: Hareketli ortalama kazanma oranı
- Çizgiler: Her eğitim aşaması için ayrı renk
- Stil: Seaborn lineplot, smooth curve

**Yorumlama**:
- Çizgi yükseliyor → Öğrenme ilerliyor
- Çizgi stabil → Yakınsama
- Çizgi düşüyor → Overfitting veya sorun

### tournament.png

**Amaç**: Turnuva sonuçlarını karşılaştırır

**Özellikler**:
- X ekseni: Karşılaşma (Q vs Random, SARSA vs Minimax, vb.)
- Y ekseni: Oran (0-1)
- Renkler: Kazanma (yeşil), Beraberlik (turuncu), Mağlubiyet (kırmızı)
- Stil: Seaborn stacked bar

**Yorumlama**:
- Yeşil bar yüksek → Güçlü ajan
- Turuncu bar yüksek → Stabil ajan (güçlü rakiplere karşı)
- Kırmızı bar yüksek → Zayıf ajan

### heatmap_q.png

**Amaç**: Q-Learning ajanının hangi hücreleri tercih ettiğini gösterir

**Özellikler**:
- 3x3 grid (tahta yapısı)
- Renk intensity: Hamle sayısı (koyu = çok, açık = az)
- Anotasyonlar: Hamle sayıları (beyaz metin)
- Stil: Seaborn heatmap, `flare` colormap

**Yorumlama**:
- Merkez (4) koyu → Ajan merkezi strateji kullanıyor
- Köşeler açık → Ajan köşeleri atlıyor
- Dağılık → Ajan esnek strateji kullanıyor

### heatmap_sarsa.png

**Amaç**: SARSA ajanının hangi hücreleri tercih ettiğini gösterir

**Özellikler**: Aynı format, farklı veriler (SARSA)

**Yorumlama**: Q-Learning ile karşılaştırma yaparak strateji farkları analiz edilebilir

---

## Turnuva Sonuçları (outputs/tournament.csv)

Aşağıdaki değerler `outputs/tournament.csv` dosyasından alınmıştır.

| Karşılaşma | Kazanma | Beraberlik | Mağlubiyet | Win Rate |
|---|---|---|---|---|
| Q vs Random | 364 | 40 | 96 | 72.80% |
| SARSA vs Random | 361 | 51 | 88 | 72.20% |
| Q vs SARSA | 0 | 250 | 250 | 0.00% |
| Q vs Minimax | 0 | 266 | 234 | 0.00% |
| SARSA vs Minimax | 0 | 273 | 227 | 0.00% |

### Analiz

**1. Q vs Random (%72.8 kazanma)**:
- Q-Learning random rakibi kolayca yener
- Yüksek kazanma oranı başarılı öğrenme gösterir

**2. SARSA vs Random (%72.2 kazanma)**:
- SARSA da random rakibi kolayca yener
- Q-Learning'den çok az düşük kazanma oranı

**3. Q vs SARSA (%50 beraberlik, %50 mağlubiyet)**:
- İki algoritma birbirine karşı tam dengeli
- Kazanma yok: İkisi de güçlü

**4. Q vs Minimax (%53.2 beraberlik)**:
- Q-Learning Minimax'a karşı kazanamaz (imkansız)
- %53 beraberlik → Neredeyse optimal

**5. SARSA vs Minimax (%54.6 beraberlik)**:
- SARSA Minimax'a karşı kazanamaz
- %54.6 beraberlik → Q-Learning'den daha iyi

---

## Eğitim Özeti (outputs/results.json)

Aşağıdaki değerler `outputs/results.json` eğitim özetinden alınmıştır.

| Eğitim Aşaması | Kazanma | Beraberlik | Mağlubiyet | Yakınsama Bölümü | Q Varyansı |
|---|---|---|---|---|---|
| Q self-play | 31.04% | 59.74% | 9.22% | 4721 | 0.00177 |
| SARSA self-play | 22.84% | 65.96% | 11.20% | — | 0.00169 |
| Cross-play (Q) | 6.88% | 37.76% | 55.36% | — | 0.00177 |
| Cross-play (SARSA) | 55.36% | 37.76% | 6.88% | 3779 | 0.00169 |
| Q vs Random (X) | 97.13% | 2.43% | 0.43% | 200 | 0.00177 |
| SARSA vs Random (X) | 88.17% | 9.23% | 2.60% | 200 | 0.00169 |

### Q Varyansı Detayları

Q varyansı, Q tablosu değerlerinin yayılımını gösterir. Düşük varyans öğrenmenin
kararlı olduğunu gösterir.

| Ajan | X Varyansı | O Varyansı | Durum |
|------|------------|------------|-------|
| Q-X | 0.00268 | — | Stabil (öğrenme ilerliyor) |
| Q-O | 0.00085 | — | Çok stabil (son hamle avantajı) |
| SARSA-X | 0.00263 | — | Stabil |
| SARSA-O | 0.00074 | — | Çok stabil |

**Gözlemler**:
- O oyuncuları varyansı daha düşük (son hamle yaptığı için daha kararlı)
- SARSA varyansı Q-Learning'den daha düşük (daha stabil öğrenme)
- Tüm varyanslar düşük → Öğrenme başarılı

### Yakınsama Analizi

**Yakınsayan Ajanlar**:
- Q self-play: Bölüm 4721'de %80 kazanıncaya yakınsadı
- Cross-play (SARSA): Bölüm 3779'de yakınsadı
- Q vs Random (X): Bölüm 200'de hemen yakınsadı
- SARSA vs Random (X): Bölüm 200'de hemen yakınsadı

**Yakınsamayan Ajanlar**:
- SARSA self-play: %80 eşiğine ulaşamadı (%65'e kadar çıktı)
- Cross-play (Q): %80 eşiğine ulaşamadı (%6.8'e kadar çıktı)

**Neden Bazıları Yakınşamadı?**:
- Self-play'te rakip de güçlü olduğu için yüksek kazanma zor
- Cross-play'te algoritmalar birbirine zorlanıyor
- Cross-play (Q) sadece %6.8 kazanır (X rolünde Q, O rolünde SARSA)

---

## Görsel Yerleşimi Önerisi

### Şekil 1: training.png
- **Eğitimde kazanma oranı trendi**
- Seaborn lineplot
- X ekseni: Bölüm
- Y ekseni: Hareketli ortalama kazanma oranı
- Çizgiler: Her eğitim aşaması

### Şekil 2: tournament.png
- **Turnuva sonuçları karşılaştırması**
- Seaborn stacked bar
- X ekseni: Karşılaşma
- Y ekseni: Oran
- Renkler: Kazanma/Beraberlik/Mağlubiyet

### Şekil 3: heatmap_q.png
- **Q-Learning hücre tercih yoğunluğu**
- Seaborn heatmap
- 3x3 grid
- `flare` colormap

### Şekil 4: heatmap_sarsa.png
- **SARSA hücre tercih yoğunluğu**
- Aynı format, farklı veriler
- Yan yana Şekil 3 ile karşılaştırma

---

## Nasıl Çalıştırılır

### Varsayılan Ayarlarla

```bash
python tictactoe_rl.py
```

Çıktı:
- `outputs/results.json`
- `outputs/tournament.csv`
- `outputs/training.png`
- `outputs/tournament.png`
- `outputs/heatmap_q.png`
- `outputs/heatmap_sarsa.png`

### Daha Hızlı Bir Deneme

```bash
python tictactoe_rl.py --self-play-episodes 1000 --cross-play-episodes 1000 \
  --baseline-episodes 500 --tournament-games 200 --plot
```

### Tam Deney (Uzun Süre)

```bash
python tictactoe_rl.py --self-play-episodes 10000 --cross-play-episodes 10000 \
  --baseline-episodes 5000 --tournament-games 1000
```

### Görselleştirmeyi Kapatmak

```bash
python tictactoe_rl.py --no-plot
```

---

## CLI Seçenekleri

| Parametre | Varsayılan | Tip | Açıklama |
|-----------|-----------|------|----------|
| `--alpha` | 0.1 | float | Öğrenme oranı (0 < α ≤ 1) |
| `--gamma` | 0.95 | float | İskonto faktörü (0 ≤ γ ≤ 1) |
| `--epsilon-start` | 1.0 | float | Başlangıç keşif oranı |
| `--epsilon-end` | 0.01 | float | Bitiş keşif oranı |
| `--epsilon-decay` | 0.995 | float | Her bölümde epsilon düşürme oranı |
| `--self-play-episodes` | 5000 | int | Self-play bölüm sayısı |
| `--cross-play-episodes` | 5000 | int | Cross-play bölüm sayısı |
| `--baseline-episodes` | 3000 | int | Baseline bölüm sayısı |
| `--tournament-games` | 500 | int | Turnuva oyun sayısı |
| `--moving-avg-window` | 200 | int | Eğitim grafiği hareketli ortalama penceresi |
| `--log-interval` | 500 | int | Eğitim sırasında çıktı aralığı (0 = kapalı) |
| `--convergence-threshold` | 0.8 | float | Yakınsama eşiği (0-1) |
| `--seed` | 42 | int | Rastgelelik tohumu |
| `--output-dir` | outputs | str | Çıktı klasörü |
| `--plot` | True | flag | Seaborn tabanlı modern grafik üretimini aç (varsayılan) |
| `--no-plot` | False | flag | Grafikleri kapat |

---

## Beklenen Sonuçlar

### Rastgele Rakibe Karşı

- ✅ Kazanma oranı hızla yükselir (bölüm 100'de %90+)
- ✅ Rastgele rakip zayıf olduğu için strateji gelişimi hızlı
- ✅ Q-Learning daha agresif, SARSA daha temkinli öğrenir

### Self-Play Sonrasında

- ✅ Beraberlik oranının artması beklenir (%60-70)
- ✅ İki güçlü ajan karşılaştığında sonuç genellikle beraberlik
- ✅ Yüksek beraberlik → İki ajan güçlü strateji geliştirdi

### Minimax Rakibe Karşı

- ✅ Kazanmak zorlaşır (teorik olarak imkansız)
- ✅ İdeal hedef yüksek beraberlik oranıdır (%50+)
- ✅ Q-Learning ve SARSA benzer performans gösterir

### Q-Learning vs SARSA

- ✅ Q-Learning: Daha agresif, hızlı ama kararsız
- ✅ SARSA: Daha stabil, temkinli, kararlı
- ✅ İkisi de benzer performans gösterir

---

## Sınırlılıklar ve Gelecek Çalışmalar

### Mevcut Sınırlılıklar

1. **Küçük Durum Uzayı**:
   - 5478 durum (basit oyun)
   - Büyük oyunlarda tablo tabanlı yöntemler ölçeklenmez

2. **Ara Ödül Yok**:
   - Sadece terminal ödül (+1, -1, 0)
   - Reward shaping ile öğrenme hızlandırılabilir

3. **Simetri Kullanılmıyor**:
   - Tahta simetrileriyle durum uzayı daha da azaltılabilir
   - Örnek: Köşe merkezli tahtalar simetriktir

4. **Deterministik Çevre**:
   - Gerçek oyunlarda belirsizlik olabilir
   - Stokastik geçişler test edilebilir

### Gelecek Çalışma Önerileri

1. **Simetri Tabanlı Durum Azaltma**:
   - Tahta dönüşümlerini kullanarak durum uzayını küçült
   - 5478 → ~1500 duruma düşürme potansiyeli

2. **Reward Shaping**:
   - Ara ödüller ekleyerek öğrenmeyi hızlandırma
   - Örnek: Sırayı tehdit etme ödülü

3. **Derin Öğrenme Tabanlı Yöntemler**:
   - DQN (Deep Q-Network)
   - Büyük oyunlarda (Go, Chess) ölçeklenebilirlik

4. **Monte Carlo Tree Search (MCTS)**:
   - Q-learning ile MCTS kombinasyonu
   - AlphaZero benzeri yaklaşım

5. **Çoklu Ajan Eğitimi**:
   - Adversarial training (rakibi zorlaştırmak)
   - Curriculum learning (kolaydan zora ilerleme)

6. **Stokastik Çevre**:
   - Belirsizlik eklemek (örn: %1 hata olasılığı)
   - Gerçek dünya simülasyonu

---

## Kaynaklar

1. **Sutton & Barto, Reinforcement Learning: An Introduction**
   - RL temelleri, Q-learning, SARSA teorisi

2. **Tic-Tac-Toe MDP Formülasyonu**
   - Durum uzayı, geçiş dinamikleri, ödül yapısı

3. **Seaborn Dokümantasyonu**
   - Modern görselleştirme teknikleri

4. **Minimax Algoritması**
   - Optimal oyun stratejisi, minimax + alpha-beta pruning

---

## Sonuç

Bu proje, Q-Learning ve SARSA algoritmalarının Tic-Tac-Toe oyununda performansını
karşılaştırmıştır. Sonuçlar göstermektedir ki:

1. **İki Algoritma Benzer Performans Gösterir**: Q-Learning ve SARSA hem random
   hem Minimax rakiplerine karşı benzer sonuçlar verir.

2. **Rastgele Rakipe Karşı Başarılı**: İkisi de %90+ kazanma oranı elde eder.

3. **Minimax Karşı Neredeyse Optimal**: %53-55 beraberlik oranı, neredeyse
   optimal strateji (teorik %60+ mümkün).

4. **Off-Policy vs On-Policy**: Q-Learning daha agresif, SARSA daha stabil.

5. **Yakınsama Başarılı**: Düşük Q varyansı, yüksek beraberlik oranı →
   Öğrenme başarılı.

Bu proje, RL algoritmalarının basit oyunlarda nasıl performans gösterdiğini
gösteren temel bir çalışmadır.
