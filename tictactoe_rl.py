"""
Tic-Tac-Toe: Q-Learning vs SARSA Performans Karşılaştırması
===========================================================

Bu proje, klasik 3x3 Tic-Tac-Toe oyununda Q-Learning (off-policy) ve SARSA (on-policy)
algoritmalarının performansını karşılaştıran tek dosyalık bir pekiştirmeli öğrenme uygulamasıdır.

MDP (Markov Karar Süreci) Tanımı:
--------------------------------
- Durum Uzayı (State Space): 3x3 tahta; her hücre 0 (boş), 1 (X), veya 2 (O) değer alabilir.
  Teorik olarak 3^9 = 19683 olası durum olsa da, oyun kurallarına uygun olmayan durumlar
  (hamle sayısı tutarsızlıkları, çift kazanan durumlar vb.) filtrelenerek 5478 geçerli durum elde edilir.

- Eylem Uzayı (Action Space): 0-8 indeksli boş hücrelerden birini seçmek. Yalnızca mevcut
  tahta durumunda geçerli olan eylemler (boş hücreler) kullanılabilir.

- Ödül Yapısı (Reward Structure):
  * Kazanma: +1 (kazanan oyuncuya)
  * Kaybetme: -1 (kaybeden oyuncuya, terminal durumda)
  * Beraberlik: 0 (hiçbir oyuncuya)
  * Ara hamleler: 0 (terminal olmayan durumlar için ödül yok)

- Geçiş Dinamiği (Transition): Ajanın hamlesi deterministiktir; rakip hamlesi ise
  random, minimax veya öğrenilmiş politika ile belirlenir.

Eğitim Stratejileri:
------------------
1. Self-Play: Q-Learning ve SARSA ajanları kendi kopyalarına karşı oynar ve öğrenir.
2. Cross-Play: Q ve SARSA ajanları karşılıklı oynar; X/O rolleri dönüşümlü değişir.
3. Baseline: Rastgele rakibe karşı öğrenme performansı ölçülür.
4. ε-Greedy Keşif: Epsilon 1.0'dan 0.01'e decay ile azaltılır.

Değerlendirme Metrikleri:
-----------------------
- Kazanma/Beraberlik/Mağlubiyet Oranı: Eğitim ve turnuva aşamaları için hesaplanır.
- Yakınsama Bölümü: Hareketli ortalama kazanma oranı belirli eşik üstüne çıktığında kaydedilir.
- Q-Değer Varyansı: Q tablosunun kararlılığının basit bir göstergesi (düşük = stabil).
- Minimax Performansı: Optimal rakibe karşı turnuva sonuçları (en zor test).

Çıktılar:
--------
- results.json: Deneyin tüm özet verileri (config, training, tournament, q_variance)
- tournament.csv: Turnuva karşılaştırmaları (CSV formatı)
- training.png: Eğitim sürecinde hareketli ortalama kazanma oranı trendi (seaborn lineplot)
- tournament.png: Turnuva kazanma/beraberlik/mağlubiyet oranları (seaborn stacked bar)
- heatmap_q.png: Q-Learning ajanının hücre tercih yoğunluğu (seaborn heatmap, flare colormap)
- heatmap_sarsa.png: SARSA ajanının hücre tercih yoğunluğu (seaborn heatmap, flare colormap)

Kullanım:
--------
python tictactoe_rl.py                    # Varsayılan ayarlar
python tictactoe_rl.py --plot            # Grafikleri açık (varsayılan)
python tictactoe_rl.py --no-plot         # Grafikleri kapalı
python tictactoe_rl.py --self-play-episodes 10000  # Daha fazla bölüm
"""

# ============================================================================
# IMPORTS (Zorunlu ve Opsiyonel)
# ============================================================================

import argparse  # Komut satırı argümanlarını ayrıştırmak için
import csv  # CSV formatında çıktı yazmak için
import json  # JSON formatında çıktı yazmak için
import random  # Rastgelelik ve epsilon-greedy keşif için
from dataclasses import asdict, dataclass  # Veri sınıfları için
from pathlib import Path  # Dosya yolları için

import numpy as np  # Sayısal işlemler, Q tablosu, vektörizasyon

# Opsiyonel kütüphaneler - yoksa None olarak ayarla (graceful degradation)
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# ============================================================================
# SABİTLER (Oyun Kuralları ve MDP Tanımı)
# ============================================================================

# WIN_LINES: Tüm olası kazanma kombinasyonları (3 satır, 3 sütun, 2 çapraz)
# Tic-Tac-Toe'da bir oyuncu, bu hücrelerden birini tamamen işgal ederse kazanır.
WIN_LINES = (
    # Yatay çizgiler (satırlar)
    (0, 1, 2),  # Üst satır
    (3, 4, 5),  # Orta satır
    (6, 7, 8),  # Alt satır
    # Dikey çizgiler (sütunlar)
    (0, 3, 6),  # Sol sütun
    (1, 4, 7),  # Orta sütun
    (2, 5, 8),  # Sağ sütun
    # Çapraz çizgiler
    (0, 4, 8),  # Sol üst -> Sağ alt
    (2, 4, 6),  # Sağ üst -> Sol alt
)

# POWER_3: Taban-3 sayı sisteminden ondalık sayıya dönüşüm için katsayılar
# Örnek: [1, 2, 0, ...] = 1*3^0 + 2*3^1 + 0*3^2 + ... = 1 + 6 + 0 + ...
# Bu sayede tahta durumunu tek bir tamsayı ile kodlayabiliriz.
POWER_3 = [3**i for i in range(9)]

# DRAW: Beraberlik durumunu temsil eden kod (kazanan oyuncu yoksa)
# check_winner() fonksiyonunda 3 değerini kullanarak beraberliği tanımlarız.
DRAW = 3


def decode_state(state_index):
    """
    Taban-3 kodlanmış durum indeksini 3x3 tahta dizisine dönüştürür.

    Bu fonksiyon, durumu compact bir tam sayıdan (0-19682) 9 elemanlı
    bir listeye [0-8] dönüştürür. Her hücre değeri: 0=boş, 1=X, 2=O.

    Örnek:
        >>> decode_state(0)        # [0, 0, 0, 0, 0, 0, 0, 0, 0] (boş tahta)
        >>> decode_state(1)        # [1, 0, 0, 0, 0, 0, 0, 0, 0] (X sol üstte)
        >>> decode_state(13)       # [1, 1, 1, 0, 0, 0, 0, 0, 0] (birinci satır X)

    Argümanlar:
        state_index (int): 3^9 olası durumun indeksi (0-19682)

    Dönüş:
        list[int]: 9 elemanlı tahta dizisi, her eleman 0, 1 veya 2
    """
    board = [0] * 9
    for index in range(9):
        board[index] = (state_index // POWER_3[index]) % 3
    return board


def has_winner(board, player):
    """
    Belirtilen oyuncunun bu durumda kazanıp kazanmadığını kontrol eder.

    Oyunun bittiğini kontrol etmek için 8 kazanma çizgisinden herhangi birini
    tamamen işgal edilip edilmediğine bakarız.

    Argümanlar:
        board (list[int]): 9 elemanlı tahta dizisi
        player (int): Kontrol edilecek oyuncu (1=X veya 2=O)

    Dönüş:
        bool: Oyuncu kazanırsa True, aksi halde False

    Örnek:
        >>> has_winner([1,1,1,0,0,0,0,0,0], 1)  # X üst satırı tamamladı
        True
        >>> has_winner([1,2,0,2,1,0,0,0,0], 1)  # Oyun devam ediyor
        False
    """
    return any(board[a] == board[b] == board[c] == player for a, b, c in WIN_LINES)


def is_valid_state(board):
    """
    Bir tahta durumunun oyun kurallarına göre geçerli olup olmadığını kontrol eder.

    Tic-Tac-Toe'da tüm 3^9 konfigürasyonlar geçerli değildir. Bu fonksiyon
    şu kuralları uygulayarak geçersiz durumları filtreler:

    1. Hamle Sayısı Kuralı: X her zaman birinci oyuncu olduğu için,
       - X sayısı O sayısından büyük olmalı (x >= o)
       - X sayısı O sayısından en fazla 1 fazla olabilir (x - o <= 1)
       Örnek: O=5, X=3 -> Geçersiz (O, X'ten fazla hamle yapmış)

    2. Çift Kazanan Kuralı: Her iki oyuncu aynı anda kazanamaz.
       Örnek: XXX ve OOO aynı anda tahtada -> Geçersiz durum

    3. Kazanma Sonrası Hamle Kuralı: Bir oyuncu kazandıktan sonra
       oyun devam edemez:
       - X kazandıysa, X'in hamlesi O'dan bir fazla olmalı (x = o + 1)
       - O kazandıysa, X ve O eşit hamle yapmalı (x = o)

    Argümanlar:
        board (list[int]): 9 elemanlı tahta dizisi

    Dönüş:
        bool: Durum geçerliyse True, aksi halde False

    Örnek:
        >>> is_valid_state([1,1,1,0,0,0,0,0,0])  # X kazandı, x=3, o=0 ✓
        True
        >>> is_valid_state([2,2,2,0,0,0,0,0,0])  # O kazandı ama x=0, o=3 ✗
        False
    """
    x_count = board.count(1)
    o_count = board.count(2)

    # Kural 1: Hamle sayısı tutarlı olmalı
    # O, X'ten fazla hamle yapamaz (X her zaman başlar)
    if o_count > x_count:
        return False
    # X, O'dan en fazla 1 hamle fazla yapabilir
    if x_count - o_count > 1:
        return False

    x_win = has_winner(board, 1)
    o_win = has_winner(board, 2)

    # Kural 2: İki oyuncu aynı anda kazanamaz
    if x_win and o_win:
        return False

    # Kural 3a: X kazandıysa, son hamlesini yapmış olmalı (x = o + 1)
    if x_win and x_count != o_count + 1:
        return False

    # Kural 3b: O kazandıysa, son hamlesini yapmış olmalı (x = o)
    if o_win and x_count != o_count:
        return False

    return True


def generate_valid_state_mapping():
    """
    Geçerli oyun durumları için kompakt indeksleme haritası oluşturur.

    Bu fonksiyon tüm 3^9 = 19683 olası durum tarar ve oyun kurallarına göre
    geçersiz olanları filtreler. Geçerli durumlar için sıralı bir indeks atayarak
    Q tablosunun boyutunu optimize eder.

    Neden Gerekli?
    ---------------
    - Orijinal durum uzayı: 19683 durum (3^9)
    - Geçerli durum uzayı: 5478 durum (is_valid_state() ile filtreleme)
    - Tasarruf: ~72% daha küçük Q tablosu (19683*9 vs 5478*9 = 177147 vs 49302 hücre)

    Algoritma:
    ----------
    1. Her state_index (0-19682) için tahta konfigürasyonunu decode et
    2. is_valid_state() ile geçerli olup olmadığını kontrol et
    3. Geçerli ise {tahta_tuple: indeks} çiftini mapping'e ekle
    4. Geçersiz ise atla

    Dönüş:
        dict[tuple[int]]: Tahta konfigürasyonu -> indeks eşleşmesi

    Not:
        Bu fonksiyon modül yüklendiğinde bir kez çalıştırılır ve STATE_INDEX
        global değişkeninde saklanır. Sonuçta tüm oyuncular aynı mapping'i kullanır.
    """
    mapping = {}
    for state_index in range(3**9):
        board = decode_state(state_index)
        if is_valid_state(board):
            # tuple(board) hashlenebilir olduğu için dict key olarak kullanılabilir
            mapping[tuple(board)] = len(mapping)
    return mapping


STATE_INDEX = generate_valid_state_mapping()
VALID_STATE_COUNT = len(STATE_INDEX)


@dataclass
class Config:
    """
    Deney yapılandırması için dataclass.

    Tüm eğitim, değerlendirme ve görselleştirme ayarlarını tek bir yerde toplar.
    CLI argümanları ile bu değerler override edilebilir.

    Hyperparameter Açıklamaları:
    -------------------------

    Öğrenme Oranı (α - alpha):
        Q değerlerinin güncelleme hızını belirler.
        - Yüksek (0.5+): Hızlı öğrenme ama kararsızlık
        - Düşük (0.01-0.1): Yavaş ama kararlı öğrenme
        - Varsayılan: 0.1 (dengeli)

    İndirgeme Faktörü (γ - gamma):
        Gelecekteki ödüllerin şu anki değeri üzerindeki etkisi.
        - 1.0: Gelecek ödüllere tam güven (uzun vadeli strateji)
        - 0.95: Gelecekle ilgili biraz belirsizlik (daha gerçekçi)
        - 0.0: Sadece anlık ödüle bak (kısa vadeli)
        - Varsayılan: 0.95 (Tic-Tac-Toe için ideal)

    Epsilon-Greedy Keşif (ε):
        Rastgele aksiyon seçme olasılığı (keşif vs sömürü dengesi).
        - epsilon_start = 1.0: %100 rastgele başla (tam keşif)
        - epsilon_end = 0.01: %1 rastgele bitir (çalışan politika)
        - epsilon_decay = 0.995: Her bölümde epsilon * 0.995

    Bölüm Sayıları:
        - self_play_episodes: Ajan kendi kopyasına karşı (5000)
        - cross_play_episodes: Q ve SARSA karşılıklı (5000)
        - baseline_episodes: Random rakibe karşı (3000)
        - tournament_games: Turnuva değerlendirmesi (500)
    """

    # --- Hyperparameters ---
    alpha: float = 0.1  # Öğrenme oranı: Q güncelleme hızı
    gamma: float = 0.95  # İndirgeme faktörü: Gelecek ödül ağırlığı

    # --- Epsilon-Greedy Keşif ---
    epsilon_start: float = 1.0  # Başlangıç: %100 rastgele (tam keşif)
    epsilon_end: float = 0.01  # Bitiş: %1 rastgele (neredeyse tam sömürü)
    epsilon_decay: float = 0.995  # Her bölümde epsilon *= 0.995

    # --- Eğitim Ayarları ---
    self_play_episodes: int = 5000  # Self-play bölümleri (X ve O aynı algoritma)
    cross_play_episodes: int = 5000  # Cross-play bölümleri (Q vs SARSA)
    baseline_episodes: int = 3000  # Random rakibe karşı bölümler

    # --- Değerlendirme Ayarları ---
    tournament_games: int = 500  # Turnuva oyun sayısı (her karşılaşma için)

    # --- Görselleştirme ve Logging ---
    moving_avg_window: int = 200  # Hareketli ortalama pencere genişliği
    log_interval: int = 500  # Eğitim log aralığı (0 = kapalı)
    convergence_threshold: float = 0.8  # Yakınsama eşiği (80% kazanç)

    # --- Diğer ---
    seed: int = 42  # Rastgelelik tohumu (tekrarlanabilirlik için)
    output_dir: str = "outputs"  # Çıktı klasörü


@dataclass
class AgentPair:
    name: str
    agent_x: "Agent"
    agent_o: "Agent"


# Oyuncu değişimi: X (1) <-> O (2).
def opponent(player):
    return 2 if player == 1 else 1


# Aksiyon uzayı: boş hücre indeksleri.
def valid_actions(board):
    return [index for index, value in enumerate(board) if value == 0]


# Terminal kontrolü: kazanan, beraberlik veya devam.
def check_winner(board):
    if has_winner(board, 1):
        return 1
    if has_winner(board, 2):
        return 2
    if all(value != 0 for value in board):
        return DRAW
    return 0


# Durum uzayı kodlaması: geçerli tahta konfigürasyonu indekse çevrilir.
# Not: STATE_INDEX sadece geçerli (legal) durumları içerir.
def encode_state(board):
    return STATE_INDEX[tuple(board)]


# Eğitim sırasında tekrar eden özet işleri için ortak fonksiyon.
def record_training_summary(label, scores, config, training_log, histories):
    # Hareketli ortalama ve yakınsama bilgilerini tek noktadan hesaplanır.
    summary = summarize_scores(scores)
    summary["convergence_episode"] = convergence_episode(
        scores, config.moving_avg_window, config.convergence_threshold
    )
    training_log[label] = summary
    histories[label] = moving_average(scores, config.moving_avg_window)
    return summary


# Ajan çiftini yapılandıran yardımcı: Q-Learning veya SARSA için kullanılır.
def build_agent_pair(name, agent_cls, prefix, n_states, config):
    # Aynı hiperparametrelerle X ve O oyuncuları oluşturulur.
    return AgentPair(
        name,
        agent_cls(
            f"{prefix}-X",
            n_states,
            config.alpha,
            config.gamma,
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        ),
        agent_cls(
            f"{prefix}-O",
            n_states,
            config.alpha,
            config.gamma,
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        ),
    )


class TicTacToeEnv:
    # Ortamın MDP geçişi: (tahta, aksiyon, oyuncu) -> (yeni_tahta, kazanan, done)
    def reset(self):
        return [0] * 9

    def step(self, board, action, player):
        if board[action] != 0:
            raise ValueError("Invalid action: cell already occupied")
        new_board = list(board)
        new_board[action] = player
        winner = check_winner(new_board)
        done = winner != 0
        return new_board, winner, done


class Agent:
    """
    Tüm ajanlar için taban (base) sınıf.

    Bu soyut sınıf, öğrenen (Q-Learning, SARSA) ve öğrenmeyen (Random, Minimax)
    ajanlar için ortak arayüz tanımlar. Polimorfizm sayesinde farklı ajan türleri
    aynı şekilde kullanılabilir.

    Arayüz Metotları:
    -----------------
    - select_action(state, valid_moves, board, player, explore): Aksiyon seç
    - update(state, action, reward, next_state, next_action, done): Öğrenme güncellemesi
    - decay_epsilon(): Keşif oranı azaltma (epsilon-greedy için)
    - reset_pending(): Bekleyen durum/aksiyonu sıfırla

    Öznitelikler:
    -------------
    - name: Ajanın tanımlayıcı adı (örn: "Q-Learning", "SARSA")
    - is_learning: Ajan öğrenme yapıyor mu (True) yoksa sadece oynuyor mu (False)
    - pending_state: SARSA için önceki durum (terminal olmayan adımlar)
    - pending_action: SARSA için önceki aksiyon (terminal olmayan adımlar)
    """

    name = "Agent"
    is_learning = False

    def __init__(self):
        """
        Ajanı başlatır.

        SARSA ve Q-Learning için "pending" (bekleyen) durum/aksiyon takibi gerekir.
        Çünkü terminal olmayan adımlar için ödül 0 varsayarız ve güncellemeyi
        bir sonraki aksiyona kadar erteleriz.
        """
        self.pending_state = None
        self.pending_action = None

    def select_action(self, state, valid_moves, board, player, explore=True):
        """
        Verilen durumda aksiyon seçer (alt sınıflar tarafından override edilir).

        Argümanlar:
            state (int): Encode edilmiş durum indeksi
            valid_moves (list[int]): Geçerli aksiyonlar (boş hücreler)
            board (list[int]): Geçerli tahta dizisi
            player (int): Ajanın oyuncu numarası (1=X, 2=O)
            explore (bool): Keşif modu (epsilon-greedy için)

        Dönüş:
            int: Seçilen aksiyon indeksi (0-8)

        Not:
            Bu metot alt sınıflarda (QLearningAgent, SarsaAgent, RandomAgent, vb.)
            farklı şekillerde uygulanır.
        """
        raise NotImplementedError

    def update(
        self, state, action, reward, next_state=None, next_action=None, done=False
    ):
        """
        Ajanın Q tablosunu günceller (öğrenen ajanlar için).

        Öğrenmeyen ajanlar (Random, Minimax) için bu metot bir şey yapmaz.
        Öğrenen ajanlar için Q güncelleme formülünü uygular:
        - Q-Learning: Q(s,a) += α [r + γ * max_a' Q(s',a') - Q(s,a)]
        - SARSA: Q(s,a) += α [r + γ * Q(s',a') - Q(s,a)]

        Argümanlar:
            state (int): Güncellenecek durum
            action (int): Güncellenecek aksiyon
            reward (float): Alınan ödül
            next_state (int|None): Sonraki durum (None ise terminal)
            next_action (int|None): Sonraki aksiyon (SARSA için)
            done (bool): Bölüm bitti mi?

        Dönüş:
            None: (yan etkisi olarak Q tablosu güncellenir)
        """
        return None

    def decay_epsilon(self):
        """
        Keşif oranını (epsilon) azaltır.

        Öğrenmeyen ajanlar için bu metot bir şey yapmaz.
        Epsilon-greedy ajanlar için epsilon *= epsilon_decay uygular.
        """
        return None

    def reset_pending(self):
        """
        Bekleyen durum/aksiyonları sıfırlar.

        Her bölüm başlangıcında çağrılır.
        """
        self.pending_state = None
        self.pending_action = None


class BaseLearningAgent(Agent):
    """
    Öğrenen ajanlar için ortak taban sınıf.

    Bu sınıf, Q-Learning ve SARSA algoritmaları için paylaşılan altyapıyı sağlar:
    - Q tablosu (değer fonksiyonu)
    - Epsilon-greedy aksiyon seçimi
    - Epsilon decay (keşif oranı azaltma)
    - Q güncelleme için gerekli hiperparametreler

    Q Tablosu Yapısı:
    ----------------
    - Boyut: (n_states, 9) → (5478, 9) for Tic-Tac-Toe
    - Q[s, a]: Durum s'de aksiyon a'nın beklenen kümülatif ödülü
    - Başlangıç: Sıfır matris (tüm değerler 0)
    - Öğrenme sürecinde güncellenir (stokastik yaklaşım)

    Epsilon-Greedy Stratejisi:
    ------------------------
    - Keşif (exploration): rastgele aksiyon seç, epsilon olasılıkla
    - Sömürü (exploitation): en yüksek Q değerine sahip aksiyon seç
    - Epsilon başlangıçta yüksektir (1.0), zamanla düşer (0.01)
    """

    is_learning = True

    def __init__(
        self, name, n_states, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay
    ):
        """
        Öğrenen ajanı başlatır.

        Argümanlar:
            name (str): Ajanın tanımlayıcı adı
            n_states (int): Durum uzayı boyutu (geçerli durum sayısı)
            alpha (float): Öğrenme oranı (0 < alpha ≤ 1)
            gamma (float): İndirgeme faktörü (0 ≤ gamma ≤ 1)
            epsilon_start (float): Başlangıç keşif oranı (0 ≤ epsilon ≤ 1)
            epsilon_end (float): Bitiş keşif oranı (0 ≤ epsilon ≤ 1)
            epsilon_decay (float): Her bölümde epsilon *= decay (0 < decay ≤ 1)
        """
        super().__init__()
        self.name = name

        # --- Hiperparametreler ---
        self.alpha = alpha  # Q güncelleme adım büyüklüğü
        self.gamma = gamma  # Gelecek ödül iskonto faktörü

        # --- Epsilon-Greedy Keşif ---
        self.epsilon = epsilon_start  # Mevcut keşif oranı
        self.epsilon_end = epsilon_end  # Minimum keşif oranı (zamana limit)
        self.epsilon_decay = epsilon_decay  # Her bölümde epsilon düşürme oranı

        # --- Q Tablosu (Değer Fonksiyonu) ---
        # Q[s, a] = durum s'de aksiyon a'nın beklenen kümülatif ödülü
        # 9 aksiyon: tüm hücreler için, geçersizler aksiyon seçiminde filtrelenir
        self.q = np.zeros((n_states, 9), dtype=np.float32)

    def select_action(self, state, valid_moves, board, player, explore=True):
        # Keşif: epsilon olasılığıyla rastgele aksiyon.
        if explore and random.random() < self.epsilon:
            return random.choice(valid_moves)
        # Sömürü: mevcut Q değerlerinden en iyisini seç.
        q_values = self.q[state]
        best_value = max(q_values[action] for action in valid_moves)
        best_actions = [
            action for action in valid_moves if q_values[action] == best_value
        ]
        return random.choice(best_actions)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class QLearningAgent(BaseLearningAgent):
    """
    Q-Learning (Off-Policy) ajanı.

    Q-Learning, off-policy bir algoritmadır: öğrenme sırasında kullanılan davranış
    politikası (epsilon-greedy) ile hedef politikası (greedy) aynı değildir.
    Güncelleme, bir sonraki durumdaki maksimum Q değerini kullanır.

    Q-Learning Güncelleme Formülü:
    ------------------------------
    Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s',a') - Q(s,a)]

    where:
    - s: Mevcut durum
    - a: Mevcut aksiyon
    - r: Alınan ödül
    - s': Sonraki durum
    - max_a' Q(s',a'): Sonraki durumun en iyi aksiyonunun Q değeri
    - α: Öğrenme oranı
    - γ: İndirgeme faktörü

    Off-Policy Avantajları:
    ----------------------
    - Hedef politika greedy, bu nedenle teorik olarak optimal
    - Gelecek seçimler için max operatörü (agresif optimizasyon)
    - SARSA'dan daha hızlı yakınsayabilir (ama daha kararsız olabilir)
    """

    def update(
        self, state, action, reward, next_state=None, next_action=None, done=False
    ):
        """
        Q tablosunu Q-Learning formülü ile günceller.

        Q-Learning, sonraki durumdaki Maksimum Q değerini kullanır.
        Bu, algoritmanın off-policy olmasını sağlar (greedy hedef politika).

        Argümanlar:
            state (int): Güncellenecek durum indeksi
            action (int): Güncellenecek aksiyon indeksi
            reward (float): Alınan ödül
            next_state (int|None): Sonraki durum (None = terminal)
            next_action (int|None): Kullanılmaz (off-policy)
            done (bool): Bölüm bitti mi?

        Güncelleme Adımları:
        --------------------
        1. Mevcut Q değerini al: Q(s,a)
        2. Hedef değeri hesapla:
           - Terminal: hedef = reward
           - Devam: hedef = reward + γ * max_a' Q(s',a')
        3. Q değerini güncelle: Q(s,a) ← Q(s,a) + α [hedef - Q(s,a)]
        """
        current = self.q[state, action]

        if done or next_state is None:
            # Terminal durum: ödül son
            target = reward
        else:
            # Devam: gelecekteki maksimum Q değerini kullan
            # max_a' Q(s',a') → bir sonraki durumun en iyi aksiyonu
            target = reward + self.gamma * np.max(self.q[next_state])

        # Stokastik gradyan inişi benzeri güncelleme
        self.q[state, action] = current + self.alpha * (target - current)


class SarsaAgent(BaseLearningAgent):
    """
    SARSA (State-Action-Reward-State-Action) On-Policy ajanı.

    SARSA, on-policy bir algoritmadır: öğrenme sırasında kullanılan davranış
    politikası ile hedef politikası aynıdır. Güncelleme, bir sonraki durumdaki
    seçilen aksiyonun Q değerini kullanır.

    SARSA Güncelleme Formülü:
    -------------------------
    Q(s,a) ← Q(s,a) + α [r + γ * Q(s',a') - Q(s,a)]

    where:
    - s: Mevcut durum
    - a: Mevcut aksiyon
    - r: Alınan ödül
    - s': Sonraki durum
    - a': Sonraki durumda seçilen aksiyon (epsilon-greedy)
    - α: Öğrenme oranı
    - γ: İndirgeme faktörü

    On-Policy Avantajları:
    ---------------------
    - Davranış politikası ile öğrenilen politika aynı
    - Epsilon-greedy keşif ile tutarlı öğrenme
    - Kararlı ve temkinli öğrenme (Q-Learning'den daha az agresif)
    - Çoğu durumda gerçek performansı daha iyi
    """

    def update(
        self, state, action, reward, next_state=None, next_action=None, done=False
    ):
        """
        Q tablosunu SARSA formülü ile günceller.

        SARSA, sonraki durumda seçilen aksiyonun Q değerini kullanır.
        Bu, algoritmanın on-policy olmasını sağlar (davranış = hedef politika).

        Argümanlar:
            state (int): Güncellenecek durum indeksi
            action (int): Güncellenecek aksiyon indeksi
            reward (float): Alınan ödül
            next_state (int|None): Sonraki durum (None = terminal)
            next_action (int|None): Sonraki durumda seçilen aksiyon
            done (bool): Bölüm bitti mi?

        Güncelleme Adımları:
        --------------------
        1. Mevcut Q değerini al: Q(s,a)
        2. Hedef değeri hesapla:
           - Terminal: hedef = reward
           - Devam: hedef = reward + γ * Q(s',a')
        3. Q değerini güncelle: Q(s,a) ← Q(s,a) + α [hedef - Q(s,a)]

        Not:
            next_action SARSA için kritiktir çünkü algoritma isminden gelen
            State-Action-Reward-State-Action zinciri gereklidir.
        """
        current = self.q[state, action]

        if done or next_state is None or next_action is None:
            # Terminal durum: ödül son
            target = reward
        else:
            # Devam: seçilen sonraki aksiyonun Q değerini kullan
            # Q(s',a') → epsilon-greedy ile seçilen aksiyonun değeri
            target = reward + self.gamma * self.q[next_state, next_action]

        # Stokastik gradyan inişi benzeri güncelleme
        self.q[state, action] = current + self.alpha * (target - current)


class RandomAgent(Agent):
    name = "Random"

    def select_action(self, state, valid_moves, board, player, explore=True):
        return random.choice(valid_moves)


class MinimaxAgent(Agent):
    name = "Minimax"

    def __init__(self):
        super().__init__()
        self.cache = {}

    def select_action(self, state, valid_moves, board, player, explore=True):
        return minimax_action(board, player, self.cache)


def minimax_action(board, player, cache):
    best_value = -2
    best_actions = []
    for action in valid_actions(board):
        next_board = list(board)
        next_board[action] = player
        value = -minimax_value(next_board, opponent(player), cache)
        if value > best_value:
            best_value = value
            best_actions = [action]
        elif value == best_value:
            best_actions.append(action)
    return random.choice(best_actions)


def minimax_value(board, player, cache):
    key = (tuple(board), player)
    if key in cache:
        return cache[key]
    winner = check_winner(board)
    if winner != 0:
        if winner == player:
            return 1
        if winner == DRAW:
            return 0
        return -1
    best_value = -2
    for action in valid_actions(board):
        next_board = list(board)
        next_board[action] = player
        value = -minimax_value(next_board, opponent(player), cache)
        best_value = max(best_value, value)
        if best_value == 1:
            break
    cache[key] = best_value
    return best_value


def score_from_winner(winner, player_id):
    # Skor metrikleri: kazanma 1, beraberlik 0, kaybetme -1.
    if winner == DRAW:
        return 0
    if winner == player_id:
        return 1
    return -1


def play_episode(
    env,
    agent_x,
    agent_o,
    train_x=True,
    train_o=True,
    explore_x=True,
    explore_o=True,
    track_actions=None,
    track_player_id=None,
):
    # Tek bölüm simülasyonu: iki ajan sırayla hamle yapar.
    # MDP akışı: (durum, aksiyon, oyuncu) -> (yeni_durum, ödül, done)
    board = env.reset()
    agent_x.reset_pending()
    agent_o.reset_pending()
    player = 1

    while True:
        agent = agent_x if player == 1 else agent_o
        train_agent = train_x if player == 1 else train_o
        explore = explore_x if player == 1 else explore_o

        state = encode_state(board)
        moves = valid_actions(board)
        action = agent.select_action(state, moves, board, player, explore=explore)

        # Isı haritası için hamle sayımı: izlenen oyuncunun hücre seçimi kaydedilir.
        if track_actions is not None and track_player_id == player:
            track_actions[action] += 1

        # Ara güncelleme: önceki hamle için ödül 0 kabul edilir (Q-Learning ve SARSA).
        # Böylece terminal ödül gelene kadar beklemek zorunda kalmayız.
        if train_agent and agent.pending_state is not None:
            agent.update(
                agent.pending_state,
                agent.pending_action,
                reward=0.0,
                next_state=state,
                next_action=action,
                done=False,
            )
            agent.pending_state = None
            agent.pending_action = None

        board, winner, done = env.step(board, action, player)

        if done:
            if train_agent:
                # Aktif oyuncu için kazanma +1, beraberlik 0; kaybetme cezası rakibe yazılır.
                reward = 1.0 if winner == player else 0.0
                agent.update(
                    state,
                    action,
                    reward=reward,
                    next_state=None,
                    next_action=None,
                    done=True,
                )

            opponent_agent = agent_o if player == 1 else agent_x
            train_opponent = train_o if player == 1 else train_x
            if train_opponent and opponent_agent.pending_state is not None:
                # Rakibin son hamlesi için mağlubiyet -1 ödülü yazılır.
                opp_reward = -1.0 if winner == player else 0.0
                opponent_agent.update(
                    opponent_agent.pending_state,
                    opponent_agent.pending_action,
                    reward=opp_reward,
                    next_state=None,
                    next_action=None,
                    done=True,
                )
                opponent_agent.pending_state = None
                opponent_agent.pending_action = None
            break

        if train_agent:
            # Terminal değilse bir sonraki durum/aksiyon için bekletiriz.
            agent.pending_state = state
            agent.pending_action = action

        player = opponent(player)

    agent_x.reset_pending()
    agent_o.reset_pending()
    return winner


def summarize_scores(scores):
    # Metrikler: galibiyet/beraberlik/mağlubiyet ve oranları.
    total = len(scores)
    wins = sum(1 for score in scores if score == 1)
    draws = sum(1 for score in scores if score == 0)
    losses = total - wins - draws
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total else 0.0,
        "draw_rate": draws / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
    }


def log_training_progress(phase, episode, total, scores, window):
    # Eğitim sırasında ilerleme çıktısı: son pencere üzerinden oranlar.
    if not scores:
        return
    window = min(window, len(scores))
    summary = summarize_scores(scores[-window:])
    print(
        f"[{phase}] Bölüm {episode}/{total}: kazanma {summary['win_rate']:.2%}, "
        f"beraberlik {summary['draw_rate']:.2%}, mağlubiyet {summary['loss_rate']:.2%}"
    )


def log_cross_progress(phase, episode, total, q_scores, sarsa_scores, window):
    # Çapraz eğitimde iki ajanın performansını birlikte yazdır.
    if not q_scores or not sarsa_scores:
        return
    window = min(window, len(q_scores), len(sarsa_scores))
    q_summary = summarize_scores(q_scores[-window:])
    s_summary = summarize_scores(sarsa_scores[-window:])
    print(
        f"[{phase}] Bölüm {episode}/{total}: Q kazanma {q_summary['win_rate']:.2%}, "
        f"beraberlik {q_summary['draw_rate']:.2%} | SARSA kazanma "
        f"{s_summary['win_rate']:.2%}, beraberlik {s_summary['draw_rate']:.2%}"
    )


def convergence_episode(scores, window, threshold):
    # Yakınsama: hareketli ortalama galibiyet oranı eşik üstüne çıkarsa bölüm döner.
    if len(scores) < window:
        return None
    wins = np.array([1 if score == 1 else 0 for score in scores], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    moving_avg = np.convolve(wins, kernel, mode="valid")
    for index, value in enumerate(moving_avg, start=window):
        if value >= threshold:
            return index
    return None


def train_self_play(
    env,
    agent_x,
    agent_o,
    episodes,
    log_interval=0,
    log_window=200,
    label="Self-play",
):
    # Self-play: iki öğrenen ajan karşılıklı oynar.
    # Aynı algoritmanın farklı rolleri (X/O) birlikte öğrenir.
    scores = []
    for episode in range(episodes):
        winner = play_episode(
            env,
            agent_x,
            agent_o,
            train_x=True,
            train_o=True,
            explore_x=True,
            explore_o=True,
        )
        scores.append(score_from_winner(winner, player_id=1))
        agent_x.decay_epsilon()
        agent_o.decay_epsilon()
        if log_interval and (
            (episode + 1) % log_interval == 0 or (episode + 1) == episodes
        ):
            log_training_progress(label, episode + 1, episodes, scores, log_window)
    return scores


def train_vs_random(
    env,
    agent,
    episodes,
    agent_first=True,
    log_interval=0,
    log_window=200,
    label="Baseline",
):
    # Baz çizgi: öğrenen ajan rastgele ajanla oynar.
    # Rastgele ajan öğrenmez; sadece karşılaştırma için kullanılır.
    scores = []
    random_agent = RandomAgent()
    for episode in range(episodes):
        if agent_first:
            winner = play_episode(
                env, agent, random_agent, train_x=True, train_o=False, explore_x=True
            )
            scores.append(score_from_winner(winner, player_id=1))
        else:
            winner = play_episode(
                env, random_agent, agent, train_x=False, train_o=True, explore_o=True
            )
            scores.append(score_from_winner(winner, player_id=2))
        agent.decay_epsilon()
        if log_interval and (
            (episode + 1) % log_interval == 0 or (episode + 1) == episodes
        ):
            log_training_progress(label, episode + 1, episodes, scores, log_window)
    return scores


def train_cross_play(
    env,
    q_pair,
    sarsa_pair,
    episodes,
    log_interval=0,
    log_window=200,
    label="Cross-play",
):
    # Çapraz eğitim: Q-Learning ve SARSA farklı rollerde oynar.
    # Her bölümde X/O rolleri değişir, böylece rol avantajı dengelenir.
    q_scores = []
    sarsa_scores = []
    for episode in range(episodes):
        if episode % 2 == 0:
            winner = play_episode(
                env,
                q_pair.agent_x,
                sarsa_pair.agent_o,
                train_x=True,
                train_o=True,
                explore_x=True,
                explore_o=True,
            )
            q_scores.append(score_from_winner(winner, player_id=1))
            sarsa_scores.append(score_from_winner(winner, player_id=2))
            q_pair.agent_x.decay_epsilon()
            sarsa_pair.agent_o.decay_epsilon()
        else:
            winner = play_episode(
                env,
                sarsa_pair.agent_x,
                q_pair.agent_o,
                train_x=True,
                train_o=True,
                explore_x=True,
                explore_o=True,
            )
            q_scores.append(score_from_winner(winner, player_id=2))
            sarsa_scores.append(score_from_winner(winner, player_id=1))
            sarsa_pair.agent_x.decay_epsilon()
            q_pair.agent_o.decay_epsilon()
        if log_interval and (
            (episode + 1) % log_interval == 0 or (episode + 1) == episodes
        ):
            log_cross_progress(
                label, episode + 1, episodes, q_scores, sarsa_scores, log_window
            )
    return q_scores, sarsa_scores


def evaluate_matchup(env, pair_a, pair_b, games):
    # Değerlendirme: keşif kapalı, salt performans ölçümü.
    # Adil karşılaştırma için her maçta X/O rolleri değiştiririz.
    scores = []
    for game in range(games):
        if game % 2 == 0:
            winner = play_episode(
                env,
                pair_a.agent_x,
                pair_b.agent_o,
                train_x=False,
                train_o=False,
                explore_x=False,
                explore_o=False,
            )
            scores.append(score_from_winner(winner, player_id=1))
        else:
            winner = play_episode(
                env,
                pair_b.agent_x,
                pair_a.agent_o,
                train_x=False,
                train_o=False,
                explore_x=False,
                explore_o=False,
            )
            scores.append(score_from_winner(winner, player_id=2))
    return summarize_scores(scores)


def collect_action_counts(env, agent, opponent_agent, games, agent_first=True):
    # Isı haritası için ajan hamlelerinin hangi hücrelerde yoğunlaştığını ölçer.
    action_counts = np.zeros(9, dtype=np.int32)
    for _ in range(games):
        if agent_first:
            play_episode(
                env,
                agent,
                opponent_agent,
                train_x=False,
                train_o=False,
                explore_x=False,
                explore_o=False,
                track_actions=action_counts,
                track_player_id=1,
            )
        else:
            play_episode(
                env,
                opponent_agent,
                agent,
                train_x=False,
                train_o=False,
                explore_x=False,
                explore_o=False,
                track_actions=action_counts,
                track_player_id=2,
            )
    return action_counts


def q_variance(agent):
    if hasattr(agent, "q"):
        return float(np.var(agent.q))
    return None


def save_json(output_dir, payload, filename="results.json"):
    # Deney çıktısını JSON'a kaydeder.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def save_csv(output_dir, rows, filename="tournament.csv"):
    # Turnuva özetini CSV formatında saklar.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def moving_average(scores, window):
    # Eğitim izleme: galibiyetlerin hareketli ortalaması.
    if len(scores) < window:
        return []
    values = np.array([1 if score == 1 else 0 for score in scores], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def apply_plot_style():
    # Seaborn ile modern stil uygulaması
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            palette="deep",
            font="sans-serif",
            font_scale=1.1,
            rc={
                "figure.figsize": (10, 6),
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            },
        )
        return sns.color_palette("deep")
    return None


def plot_training(histories, output_dir):
    # Görselleştirme: eğitim sürecinde kazanma oranı trendi (Seaborn ile)
    if plt is None:
        return None
    palette = apply_plot_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # DataFrame oluşturma ve seaborn ile çizim
    data = []
    for label, series in histories.items():
        if not series:
            continue
        for i, value in enumerate(series):
            data.append({"Bölüm": i, "Kazanma Oranı": value, "Yöntem": label})

    if data and pd is not None and sns is not None:
        df = pd.DataFrame(data)
        sns.lineplot(
            data=df,
            x="Bölüm",
            y="Kazanma Oranı",
            hue="Yöntem",
            palette="deep",
            linewidth=2,
            alpha=0.9,
            ax=ax,
        )
    elif data:
        # Fallback: pandas/seaborn yoksa manuel çizim
        from itertools import cycle

        colors = palette if palette is not None else ["blue", "orange", "green", "red"]
        valid_items = [(label, series) for label, series in histories.items() if series]
        for (label, series), color in zip(valid_items, cycle(colors)):
            x_values = np.arange(len(series))
            ax.plot(x_values, series, label=label, linewidth=2, alpha=0.9, color=color)
    else:
        # Çizilecek veri yok
        return None

    ax.set_title("Hareketli Ortalama Kazanma Oranı", pad=20, fontweight="bold")
    ax.set_xlabel("Bölüm", labelpad=10)
    ax.set_ylabel("Kazanma Oranı", labelpad=10)
    if len(histories) > 0:
        ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = output_path / "training.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def plot_tournament(tournament, output_dir):
    # Turnuva sonuçları: kazanma/beraberlik/mağlubiyet oranlarını yığılı çubuk gösterir (Seaborn ile)
    if plt is None:
        return None
    palette = apply_plot_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked bar chart için seaborn kullanımı
    if sns is not None and pd is not None:
        # DataFrame oluşturma
        data = []
        for label, summary in tournament.items():
            data.append({"Maç": label, "Sonuç": "Kazanma", "Oran": summary["win_rate"]})
            data.append(
                {"Maç": label, "Sonuç": "Beraberlik", "Oran": summary["draw_rate"]}
            )
            data.append(
                {"Maç": label, "Sonuç": "Mağlubiyet", "Oran": summary["loss_rate"]}
            )

        df = pd.DataFrame(data)

        # Pivot table oluştur ve stack haline getir
        pivot_df = df.pivot(index="Maç", columns="Sonuç", values="Oran")

        # Modern renk paleti
        colors = sns.color_palette("deep", 3)

        # Stacked bar chart
        pivot_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=colors,
            width=0.7,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )
    else:
        # Fallback matplotlib
        labels = list(tournament.keys())
        win_rates = [summary["win_rate"] for summary in tournament.values()]
        draw_rates = [summary["draw_rate"] for summary in tournament.values()]
        loss_rates = [summary["loss_rate"] for summary in tournament.values()]
        positions = np.arange(len(labels))
        colors = palette if palette is not None else ["blue", "orange", "green"]

        ax.bar(
            positions, win_rates, label="Kazanma", color=colors[0], edgecolor="white"
        )
        ax.bar(
            positions,
            draw_rates,
            bottom=win_rates,
            label="Beraberlik",
            color=colors[1],
            edgecolor="white",
        )
        ax.bar(
            positions,
            loss_rates,
            bottom=np.array(win_rates) + np.array(draw_rates),
            label="Mağlubiyet",
            color=colors[2],
            edgecolor="white",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_title("Turnuva Sonuçları", pad=20, fontweight="bold")
    ax.set_ylabel("Oran", labelpad=10)
    ax.set_xlabel("", labelpad=10)
    ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="upper right",
        title="Sonuç",
        title_fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax)

    plt.tight_layout()
    path = output_path / "tournament.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def plot_action_heatmap(action_counts, title, output_dir, filename):
    # Ajanın hangi hücreleri tercih ettiğini gösteren 3x3 ısı haritası (Seaborn ile)
    if plt is None:
        return None
    apply_plot_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    grid = np.array(action_counts, dtype=np.int32).reshape(3, 3)

    fig, ax = plt.subplots(figsize=(6, 5))

    if sns is not None:
        # Modern ısı haritası - koyu colormap (flare) ile beyaz yazı daha okunabilir
        sns.heatmap(
            grid,
            annot=True,
            fmt="d",
            cmap="flare",  # crest yerine flare - daha koyu renkler için beyaz yazı okunabilir
            cbar=True,
            square=True,
            linewidths=2,
            linecolor="white",
            annot_kws={"size": 14, "weight": "bold", "color": "white"},
            cbar_kws={
                "label": "Hamle Sayısı",
                "shrink": 0.85,
            },
            ax=ax,
            vmin=0,
        )
    else:
        # Fallback matplotlib
        im = ax.imshow(grid, cmap="viridis")
        plt.colorbar(im, ax=ax)
        for row_index in range(3):
            for col_index in range(3):
                value = int(grid[row_index, col_index])
                ax.text(
                    col_index,
                    row_index,
                    value,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    weight="bold",
                )

    ax.set_title(title, pad=20, fontweight="bold")
    ax.set_xlabel("Sütun", labelpad=10)
    ax.set_ylabel("Satır", labelpad=10)

    # Tick etiketlerini ayarla
    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_yticks(np.arange(3) + 0.5)
    ax.set_xticklabels([1, 2, 3])
    ax.set_yticklabels([1, 2, 3])

    sns.despine(ax=ax)

    plt.tight_layout()
    path = output_path / filename
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def print_summary(training, tournament):
    print("Eğitim Özeti")
    for label, summary in training.items():
        print(
            f"- {label}: kazanma {summary['win_rate']:.2%}, beraberlik {summary['draw_rate']:.2%}, "
            f"mağlubiyet {summary['loss_rate']:.2%}, yakınsama {summary.get('convergence_episode')}"
        )

    print("\nTurnuva Özeti")
    for label, summary in tournament.items():
        print(
            f"- {label}: kazanma {summary['win_rate']:.2%}, beraberlik {summary['draw_rate']:.2%}, "
            f"mağlubiyet {summary['loss_rate']:.2%}"
        )


def run_experiment(config, plot=False):
    # Deney akışı: ajan kurulumu, eğitim, değerlendirme ve çıktı kaydı.
    # Bu fonksiyon, tüm çıktıları (JSON/CSV/grafikler) aynı isimlerle üretir.
    env = TicTacToeEnv()
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Durum sayısı: geçerli tahta konfigürasyonları (yaklaşık 5.478).
    n_states = VALID_STATE_COUNT
    q_pair = build_agent_pair("Q-Learning", QLearningAgent, "Q", n_states, config)
    sarsa_pair = build_agent_pair("SARSA", SarsaAgent, "S", n_states, config)

    training_log = {}
    histories = {}

    # Self-play eğitimleri: her algoritma kendi kendine öğrenir.
    q_self_scores = train_self_play(
        env,
        q_pair.agent_x,
        q_pair.agent_o,
        config.self_play_episodes,
        log_interval=config.log_interval,
        log_window=config.moving_avg_window,
        label="Q self-play",
    )
    record_training_summary(
        "Q self-play", q_self_scores, config, training_log, histories
    )

    s_self_scores = train_self_play(
        env,
        sarsa_pair.agent_x,
        sarsa_pair.agent_o,
        config.self_play_episodes,
        log_interval=config.log_interval,
        log_window=config.moving_avg_window,
        label="SARSA self-play",
    )
    record_training_summary(
        "SARSA self-play", s_self_scores, config, training_log, histories
    )

    # Çapraz eğitim: Q-Learning ve SARSA farklı rollerde karşılaşır.
    cross_q_scores, cross_s_scores = train_cross_play(
        env,
        q_pair,
        sarsa_pair,
        config.cross_play_episodes,
        log_interval=config.log_interval,
        log_window=config.moving_avg_window,
        label="Cross-play",
    )
    record_training_summary(
        "Cross-play (Q)", cross_q_scores, config, training_log, histories
    )
    record_training_summary(
        "Cross-play (SARSA)", cross_s_scores, config, training_log, histories
    )

    # Baz çizgi: rastgele ajana karşı öğrenme performansı.
    q_random_scores = train_vs_random(
        env,
        q_pair.agent_x,
        config.baseline_episodes,
        agent_first=True,
        log_interval=config.log_interval,
        log_window=config.moving_avg_window,
        label="Q vs Random (X)",
    )
    record_training_summary(
        "Q vs Random (X)", q_random_scores, config, training_log, histories
    )

    s_random_scores = train_vs_random(
        env,
        sarsa_pair.agent_x,
        config.baseline_episodes,
        agent_first=True,
        log_interval=config.log_interval,
        log_window=config.moving_avg_window,
        label="SARSA vs Random (X)",
    )
    record_training_summary(
        "SARSA vs Random (X)", s_random_scores, config, training_log, histories
    )

    random_pair = AgentPair("Random", RandomAgent(), RandomAgent())
    minimax_pair = AgentPair("Minimax", MinimaxAgent(), MinimaxAgent())

    # Turnuva: öğrenen ajanlar, rastgele ve minimax karşılaştırmaları.
    tournament_log = {
        "Q vs Random": evaluate_matchup(
            env, q_pair, random_pair, config.tournament_games
        ),
        "SARSA vs Random": evaluate_matchup(
            env, sarsa_pair, random_pair, config.tournament_games
        ),
        "Q vs SARSA": evaluate_matchup(
            env, q_pair, sarsa_pair, config.tournament_games
        ),
        "Q vs Minimax": evaluate_matchup(
            env, q_pair, minimax_pair, config.tournament_games
        ),
        "SARSA vs Minimax": evaluate_matchup(
            env, sarsa_pair, minimax_pair, config.tournament_games
        ),
    }

    # Q tablosu varyansı: öğrenmenin yayılımını izlemek için basit ölçüt.
    variance_log = {
        "Q-X": q_variance(q_pair.agent_x),
        "Q-O": q_variance(q_pair.agent_o),
        "SARSA-X": q_variance(sarsa_pair.agent_x),
        "SARSA-O": q_variance(sarsa_pair.agent_o),
    }

    payload = {
        "config": asdict(config),
        "training": training_log,
        "tournament": tournament_log,
        "q_variance": variance_log,
    }

    json_path = save_json(config.output_dir, payload)
    csv_rows = [
        {"matchup": label, **summary} for label, summary in tournament_log.items()
    ]
    csv_path = save_csv(config.output_dir, csv_rows)

    plot_paths = []
    if plot:
        training_plot = plot_training(histories, config.output_dir)
        tournament_plot = plot_tournament(tournament_log, config.output_dir)

        # Isı haritaları için ajanların hamle frekansları toplanır.
        q_counts_x = collect_action_counts(
            env,
            q_pair.agent_x,
            random_pair.agent_o,
            config.tournament_games,
            agent_first=True,
        )
        q_counts_o = collect_action_counts(
            env,
            q_pair.agent_o,
            random_pair.agent_x,
            config.tournament_games,
            agent_first=False,
        )
        sarsa_counts_x = collect_action_counts(
            env,
            sarsa_pair.agent_x,
            random_pair.agent_o,
            config.tournament_games,
            agent_first=True,
        )
        sarsa_counts_o = collect_action_counts(
            env,
            sarsa_pair.agent_o,
            random_pair.agent_x,
            config.tournament_games,
            agent_first=False,
        )

        q_heatmap = plot_action_heatmap(
            q_counts_x + q_counts_o,
            "Q-Learning Hücre Tercihleri",
            config.output_dir,
            "heatmap_q.png",
        )
        sarsa_heatmap = plot_action_heatmap(
            sarsa_counts_x + sarsa_counts_o,
            "SARSA Hücre Tercihleri",
            config.output_dir,
            "heatmap_sarsa.png",
        )

        plot_paths = [
            path
            for path in (
                training_plot,
                tournament_plot,
                q_heatmap,
                sarsa_heatmap,
            )
            if path
        ]

    print_summary(training_log, tournament_log)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


def parse_args():
    # CLI ayarları: rapor varsayılanlarıyla uyumlu tutulur.
    # Parametreler eğitimin hızını, keşfi ve çıktıları yönetir.
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Q-Learning vs SARSA")
    parser.add_argument("--alpha", type=float, default=0.1)
    # gamma = 0.95 varsayılan, gerekirse 1.0 denenebilir.
    parser.add_argument("--gamma", type=float, default=0.95)
    # Epsilon aralığı: 1.0 -> 0.01 (decay ile kademeli düşüş).
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    # Bölüm sayıları: self-play, çapraz eğitim ve random baz çizgisi.
    parser.add_argument("--self-play-episodes", type=int, default=5000)
    parser.add_argument("--cross-play-episodes", type=int, default=5000)
    parser.add_argument("--baseline-episodes", type=int, default=3000)
    # Turnuva, eğitim sonrası değerlendirme oyun sayısıdır.
    parser.add_argument("--tournament-games", type=int, default=500)
    # Hareketli ortalama penceresi eğitim grafiğini pürüzsüzleştirir.
    parser.add_argument("--moving-avg-window", type=int, default=200)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Eğitim sırasında çıktı aralığı (0 = kapalı).",
    )
    parser.add_argument("--convergence-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    # Görselleştirme varsayılan olarak açık; gerekirse --no-plot ile kapatılır.
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Eğitim ve turnuva grafikleri üretir (varsayılan).",
    )
    plot_group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Görselleştirmeyi kapatır.",
    )
    parser.set_defaults(plot=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        self_play_episodes=args.self_play_episodes,
        cross_play_episodes=args.cross_play_episodes,
        baseline_episodes=args.baseline_episodes,
        tournament_games=args.tournament_games,
        moving_avg_window=args.moving_avg_window,
        log_interval=args.log_interval,
        convergence_threshold=args.convergence_threshold,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    run_experiment(config, plot=args.plot)


if __name__ == "__main__":
    main()
