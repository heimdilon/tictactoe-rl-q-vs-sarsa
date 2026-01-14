# Tic-Tac-Toe Q-Learning vs SARSA 🎮

Bu proje, klasik 3x3 Tic-Tac-Toe oyununda **Q-Learning** (off-policy) ve **SARSA** (on-policy) algoritmalarının performansını karşılaştırır.

## 📋 Özellikler

- ✅ **Q-Learning vs SARSA** karşılaştırması (off-policy vs on-policy)
- ✅ **Self-play**, **cross-play** ve **baseline** eğitim stratejileri
- ✅ **Minimax** ve **Random** rakiplere karşı turnuva
- ✅ **Seaborn** ile modern görselleştirmeler
- ✅ **JSON/CSV** formatında detaylı çıktılar
- ✅ **5478** geçerli durum (filtered MDP space)
- ✅ **Heatmap** ile hücre tercih analizi
- ✅ **Epsilon-greedy** keşif mekanizması
- ✅ **Hareketli ortalama** ile eğitim trend analizi

## 📖 İçindekiler

- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Algoritmalar](#algoritmalar)
- [Çıktılar](#çıktılar)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## 🚀 Kurulum

### Gerekli Bağımlılıklar

```bash
pip install numpy matplotlib seaborn pandas
```

Veya `requirements.txt` kullanarak:

```bash
pip install -r requirements.txt
```

### Platform-Specific Kurulum

#### Windows
```cmd
# CMD
python -m pip install numpy matplotlib seaborn pandas

# PowerShell
pip install numpy matplotlib seaborn pandas
```

#### Linux / macOS
```bash
pip3 install numpy matplotlib seaborn pandas

# Veya virtual environment ile
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Bağımlılık Sürüm Bilgileri

| Paket | Minimum Sürüm | Açıklama |
|--------|---------------|----------|
| numpy | >=1.20.0 | Sayısal işlemler, Q tablosu, vektörizasyon |
| matplotlib | >=3.3.0 | Grafik çizimi, PNG çıktı |
| seaborn | >=0.11.0 | Modern görselleştirmeler, heatmap, lineplot |
| pandas | >=1.3.0 | DataFrame, veri manipülasyonu |

## 💻 Kullanım

### Varsayılan Ayarlarla Çalıştırma

```bash
python tictactoe_rl.py
```

Bu komut:
- Q-Learning ve SARSA'yı self-play modunda eğitir
- Cross-play karşılaşmaları yapar
- Random rakibe karşı performans ölçer
- Minimax ve SARSA'ya karşı turnuva yapar
- Tüm sonuçları `outputs/` klasörüne kaydeder

### Hızlı Test (Daha Az Bölüm)

```bash
python tictactoe_rl.py --self-play-episodes 1000 --cross-play-episodes 1000 \
  --baseline-episodes 500 --tournament-games 200 --plot
```

### Tam Deney (Daha Fazla Bölüm)

```bash
python tictactoe_rl.py --self-play-episodes 10000 --cross-play-episodes 10000 \
  --baseline-episodes 5000 --tournament-games 1000
```

### Görselleştirmeyi Kapatmak

```bash
python tictactoe_rl.py --no-plot
```

### Özel Hiperparametreler

```bash
python tictactoe_rl.py --alpha 0.05 --gamma 1.0 \
  --epsilon-start 0.9 --epsilon-end 0.001
```

## 📁 Proje Yapısı

```
kod2/
├── tictactoe_rl.py    # Ana Python dosyası (~1200 satır)
├── report.md          # Detaylı proje raporu
├── README.md          # Bu dosya
├── requirements.txt   # Python bağımlılıkları
├── .gitignore        # Git hariç tutma kuralları
└── outputs/          # Çıktı klasörü (çalıştıktan sonra)
    ├── results.json       # Tüm deney sonuçları
    ├── tournament.csv    # Turnuva karşılaştırmaları
    ├── training.png      # Eğitim trendi
    ├── tournament.png    # Turnuva sonuçları
    ├── heatmap_q.png     # Q-Learning hücre tercihleri
    └── heatmap_sarsa.png # SARSA hücre tercihleri
```

## 🧠 Algoritmalar

### Q-Learning (Off-Policy)

Q-Learning, off-policy bir algoritmadır. Hedef politikayı greedy (keşifsiz) kabul eder,
ancak öğrenme sırasında epsilon-greedy (keşifli) davranış politikası kullanır.

```
Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s',a') - Q(s,a)]
```

**Özellikler:**
- Agresif öğrenme (max operatörü)
- Hızlı yakınsama potansiyeli
- Daha kararsız olabilir
- Teorik olarak optimal hedef politika

### SARSA (On-Policy)

SARSA, on-policy bir algoritmadır. Davranış politikası ile hedef politika aynıdır.

```
Q(s,a) ← Q(s,a) + α [r + γ * Q(s',a') - Q(s,a)]
```

**Özellikler:**
- Stabil ve temkinli öğrenme
- Epsilon-greedy ile tutarlı
- Genellikle daha iyi gerçek performans
- Keşif ve sömürü dengesi

### Karşılaştırma

| Özellik | Q-Learning | SARSA |
|----------|------------|--------|
| Politika Türü | Off-policy | On-policy |
| Güncelleme | max Q(s',a') | Q(s',a') |
| Keşif Stratejisi | Agresif | Dengeli |
| Stabilite | Kararsız | Stabil |
| Yakınsama Hızı | Hızlı | Orta |
| Optimalite | Teorik optimal | Neredeyse optimal |

## 📊 Çıktılar

### results.json

```json
{
  "config": { ... },
  "training": {
    "Q self-play": { "wins": 1552, "draws": 2987, ... },
    "SARSA self-play": { ... },
    ...
  },
  "tournament": {
    "Q vs Random": { "win_rate": 0.728, ... },
    ...
  },
  "q_variance": {
    "Q-X": 0.00268,
    "Q-O": 0.00085,
    ...
  }
}
```

### tournament.csv

| matchup | wins | draws | losses | win_rate | draw_rate | loss_rate |
|---------|-------|--------|---------|----------|-----------|-----------|
| Q vs Random | 364 | 40 | 96 | 0.728 | 0.080 | 0.192 |
| SARSA vs Random | 361 | 51 | 88 | 0.722 | 0.102 | 0.176 |
| ... | ... | ... | ... | ... | ... | ... |

### Görselleştirmeler

- **training.png**: Eğitim sürecinde hareketli ortalama kazanma oranı trendi
  - Seaborn lineplot
  - Her eğitim aşaması için ayrı çizgi
  - 300 DPI yüksek çözünürlük

- **tournament.png**: Turnuva sonuçları karşılaştırması
  - Seaborn stacked bar chart
  - Kazanma/Beraberlik/Mağlubiyet oranları
  - `deep` color palette

- **heatmap_q.png**: Q-Learning hücre tercih yoğunluğu
  - Seaborn heatmap
  - `flare` colormap (daha koyu renkler)
  - 3x3 grid, beyaz anotasyonlar

- **heatmap_sarsa.png**: SARSA hücre tercih yoğunluğu
  - Aynı format, farklı veriler

## 📈 Sonuçlar

### Rastgele Rakibe Karşı

| Algoritma | Kazanma | Beraberlik | Mağlubiyet |
|-----------|----------|------------|------------|
| Q-Learning | ~97% | ~2% | ~1% |
| SARSA | ~88% | ~9% | ~3% |

### Minimax Rakibe Karşı

| Algoritma | Kazanma | Beraberlik | Mağlubiyet |
|-----------|----------|------------|------------|
| Q-Learning | 0% | ~53% | ~47% |
| SARSA | 0% | ~55% | ~45% |

### Sonuç Analizi

1. **Random Performansı**: Her iki algoritma random rakibi kolayca yener
   - Q-Learning daha agresif olduğu için daha yüksek kazanma oranı
   - SARSA daha temkinli, ama yeterince güçlü

2. **Minimax Performansı**: İki algoritma da Minimax'a karşı kazanamaz
   - Optimal strateji olduğu için kazanma imkansız
   - Yüksek beraberlik oranı başarılı öğrenme gösteriyor

3. **Self-Play Sonucu**: Beraberlik oranı artar
   - İki güçlü ajan karşılaştığında beraberlik yaygın

Daha fazla detay için [`report.md`](report.md) dosyasına bakın.

## ⚙️ CLI Seçenekleri

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
| `--moving-avg-window` | 200 | int | Hareketli ortalama penceresi |
| `--log-interval` | 500 | int | Log aralığı (0 = kapalı) |
| `--convergence-threshold` | 0.8 | float | Yakınsama eşiği (0-1) |
| `--seed` | 42 | int | Rastgelelik tohumu |
| `--output-dir` | outputs | str | Çıktı klasörü |
| `--plot` | True | flag | Grafikleri üret (varsayılan) |
| `--no-plot` | False | flag | Grafikleri kapat |

### Test Etme

```bash
# Hızlı test
python tictactoe_rl.py --self-play-episodes 100 --plot

# Tam test
python tictactoe_rl.py
```


## 📚 Kaynaklar

- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Tic-Tac-Toe MDP](https://en.wikipedia.org/wiki/Tic-tac-toe)

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır.

### MIT Lisansı Özeti

- ✅ Ticari kullanım izni
- ✅ Modifikasyon izni
- ✅ Dağıtım izni
- ✅ Kişisel kullanım izni
- ⚠️ Lisans ve telif hakkı bildirimi zorunludur
- ❌ Garanti verilmemiştir (AS IS)

### Tam Metin

Lisansın tam metni için [`LICENSE`](LICENSE) dosyasına bakın.

### Alıntı ve Atıf

Bu projeyi kullandığınızda lütfen aşağıdaki alıntıyı kullanın:

```bibtex
@software{tictactoe_rl_2025,
  title = {Tic-Tac-Toe Q-Learning vs SARSA},
  author = {Heimdilon},
  year = {2025},
  url = {https://github.com/heimdilon/tictactoe-rl-q-vs-sarsa}
}
```

Veya basit atıf:

> Heimdilon. (2025). Tic-Tac-Toe Q-Learning vs SARSA. GitHub. https://github.com/heimdilon/tictactoe-rl-q-vs-sarsa

---

**Yazar**: [Heimdilon](https://github.com/heimdilon)

⭐️ Bu repo'yu beğendiyseniz star vermeyi unutmayın!
