# -*- coding: utf-8 -*-
"""
工程常用濾波器（串流式函式庫，只依賴 numpy）
=========================================================
這是什麼：
  把工程上最常用的 9 種「非卡爾曼」濾波器，寫成同一套介面的小類別。
  每個類別的 docstring 都附上：一句話、設計思路（STEP 1、STEP 2…）、
  參數怎麼選、代價（延遲／相位／計算量／記憶體）、適用與不適用的場景。
  （卡爾曼濾波在隔壁 kalman_filter/ 資料夾；本檔的 AlphaBetaFilter 就是它的「穩態簡化版」。）

核心觀念：
  1. 串流式（streaming）：update(x) 一次吃一個取樣、立刻吐一個輸出 y，
     只用「現在與過去」、不偷看未來 → 可以原封不動搬到即時系統 / MCU 上。
  2. 先認雜訊，再挑濾波器：
       高斯白噪（細碎抖動）         → 平均類：MovingAverage、EMA、Biquad 低通
       尖峰脈衝（偶發一兩點大跳）   → 排序類：MedianFilter、HampelFilter
       漂移（慢慢歪掉）             → 互補融合 / 高通：ComplementaryFilter、Biquad 高通
       要估速度、又要少延遲         → 模型類：AlphaBetaFilter、OneEuroFilter
       輸出不能變太快（保護致動器） → RateLimiter
  3. 天下沒有白吃的午餐：因果濾波器「越平滑 ⇔ 越延遲」，只能挑一個平衡點。

操作方式（程式）：
  from filters import EMA, apply
  f = EMA.from_cutoff(fc=1.0, dt=0.01)      # 截止 1 Hz、取樣 100 Hz
  for x in stream:                           # 即時：一次一個取樣
      y = f.update(x)
  ys = apply(f, signal)                      # 批次：整段一次跑完（會先 reset）

需求 / 執行：
  pip install numpy          （scipy 可選，只用在自我測試裡比對 Biquad）
  python filters.py          → 跑自我測試：印出 RMSE / 延遲表，並做 assert 合理性檢查
  搭配動畫：python filters_pyqt6_step.py
"""

from __future__ import annotations

import bisect
import math
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "StreamFilter", "apply",
    "MovingAverage",
    "EMA", "LowPass1st", "alpha_from_cutoff", "cutoff_from_alpha",
    "MedianFilter", "HampelFilter",
    "AlphaBetaFilter", "alpha_beta_from_tracking_index", "tracking_index",
    "benedict_bordner_beta",
    "ComplementaryFilter",
    "Biquad", "BiquadCascade", "rbj_lowpass", "rbj_highpass",
    "butterworth_lowpass", "butterworth_highpass", "butterworth_sos",
    "OneEuroFilter", "RateLimiter",
    "make_demo_signals", "rmse", "xcorr_delay", "step_delay_50", "align_delay",
]


# =========================================================
#  共同介面
# =========================================================
class StreamFilter:
    """
    所有濾波器的共同介面（串流式）：
      update(x) -> y   吃一個新取樣，回傳這一刻的輸出
      reset()          回到「還沒看過任何資料」的狀態
      y                最近一次的輸出（還沒吃過資料時是 None）
    互補濾波例外：update(gyro_rate, accel_angle) 吃兩路輸入。
    """
    y: Optional[float] = None

    def update(self, *args):                     # pragma: no cover - 介面
        raise NotImplementedError

    def reset(self):                             # pragma: no cover - 介面
        raise NotImplementedError

    def __call__(self, *args):
        return self.update(*args)


def apply(filt: StreamFilter, *signals: Sequence[float], reset: bool = True) -> np.ndarray:
    """
    批次 helper：把整段訊號逐點餵進串流濾波器，回傳同長度的輸出陣列。
      apply(EMA(0.2), x)                         # 一路輸入
      apply(ComplementaryFilter(...), gyro, acc) # 兩路輸入（逐點 zip）
    reset=True（預設）會先清空狀態，讓同一個濾波器物件可以重複拿來跑不同訊號。
    """
    if not signals:
        raise ValueError("至少要給一路訊號")
    cols = [np.asarray(s, dtype=float) for s in signals]
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("多路輸入的長度必須一樣")
    if reset:
        filt.reset()
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = filt.update(*(c[i] for c in cols))
    return out


# =========================================================
#  1. 移動平均
# =========================================================
class MovingAverage(StreamFilter):
    """
    移動平均（Moving Average, MA）：輸出 = 最近 N 個取樣的算術平均。

    設計思路
      STEP 1  白噪是「一下正、一下負」，平均起來會互相抵消 → 取最近 N 點平均，
              雜訊標準差降為 1/√N（N=16 → 剩 1/4）。
      STEP 2  笨方法每步重加 N 個數 = O(N)。聰明做法：用「環形緩衝」記住窗口，
              再維護一個「累加和 S」：新點進來 S += x_new、最舊點被擠出 S -= x_old
              → 每步只做 1 加 1 減 1 除，O(1)，跟 N 多大無關。
      STEP 3  暖機期（還沒滿 N 點）：用「目前有幾點就除以幾」，
              不要硬除以 N（否則前 N 點輸出會從 0 慢慢爬上來，看起來像假的步階）。
      STEP 4  浮點誤差累積：S 一直加加減減，捨入誤差會慢慢累積；更慘的是一個超大尖峰
              （例如 1e12）進出窗口後，小數部分早被吃掉，S 再也回不到正確值。
              工程解法：每 recompute_every 步用 math.fsum 重算一次 S（成本 O(N)，平均攤提很小）；
              若輸入是 ADC 整數，直接用整數累加和就完全沒有漂移問題（但要注意整數溢位的位寬）。

    參數怎麼選
      ● N 越大越平滑、但越慢。等效延遲 ≈ (N−1)/2 個取樣（窗口的「重心」在中間）。
      ● 想濾掉頻率 f 的週期干擾（例如 60 Hz 市電）：取 N·dt = 1/f 的整數倍，
        窗口剛好涵蓋整數個週期 → 該頻率被完全抵消（MA 的頻率響應零點）。
      ● 經驗值：先決定「可以接受多少延遲 D（秒）」，N ≈ 2·D/dt + 1。

    代價
      延遲：(N−1)/2 點、線性相位（所有頻率延遲相同，波形不變形）。
      計算：O(1) / 每點；記憶體：N 個取樣（環形緩衝）。
      頻率響應有旁瓣（sinc 形），高頻雜訊壓得不如同延遲的 Biquad 乾淨。

    適用：白噪、資料記錄平滑、顯示用數值（溫度、電壓讀值）、消除固定週期干擾。
    不適用：尖峰脈衝（一個尖峰會被「平均攤開」成一個小方塊，而不是被去掉）；
           需要保留陡峭邊緣（步階會被斜坡化 N 點）。
    """

    def __init__(self, n: int, recompute_every: int = 1000):
        if int(n) < 1:
            raise ValueError("N 必須 ≥ 1")
        self.n = int(n)
        self.recompute_every = int(recompute_every)   # ≤0 代表不重算（只為了示範誤差累積）
        self.reset()

    def reset(self):
        self.buf: List[float] = [0.0] * self.n   # 環形緩衝
        self.idx = 0              # 下一個要寫入的位置（窗口滿時，也就是「最舊那筆」的位置）
        self.count = 0            # 窗口內目前有幾筆（暖機期 < N）
        self.total = 0.0          # 累加和 S
        self._since_recompute = 0
        self.last_in: Optional[float] = None    # 這一步新進的值
        self.last_out: Optional[float] = None   # 這一步被擠出的最舊值（暖機期為 None）
        self.y = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.count < self.n:                 # 暖機：窗口還沒滿，只加不減
            self.last_out = None
            self.count += 1
            self.total += x
        else:                                   # 窗口已滿：加新的、減掉被擠出的最舊值
            self.last_out = self.buf[self.idx]
            self.total += x - self.last_out
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.n
        self.last_in = x
        self._since_recompute += 1
        if self.recompute_every > 0 and self._since_recompute >= self.recompute_every:
            self.total = math.fsum(self.window())   # 定期重算，清掉累積的捨入誤差
            self._since_recompute = 0
        self.y = self.total / self.count
        return self.y

    def window(self) -> List[float]:
        """依時間順序（舊 → 新）回傳窗口內容。"""
        if self.count < self.n:
            return self.buf[:self.count]
        return self.buf[self.idx:] + self.buf[:self.idx]


# =========================================================
#  2. EMA / 一階低通
# =========================================================
def alpha_from_cutoff(fc: float, dt: float) -> float:
    """截止頻率 fc [Hz]、取樣間隔 dt [s] → EMA 的 α = dt / (RC + dt)，RC = 1/(2π·fc)。"""
    if fc <= 0 or dt <= 0:
        raise ValueError("fc 與 dt 都必須 > 0（注意單位：fc 用 Hz、dt 用秒）")
    rc = 1.0 / (2.0 * math.pi * fc)
    return dt / (rc + dt)


def cutoff_from_alpha(alpha: float, dt: float) -> float:
    """反過來：α → 等效截止頻率 fc [Hz]。（α = dt/(RC+dt) ⇒ RC = dt·(1−α)/α）"""
    if not 0.0 < alpha < 1.0:
        raise ValueError("α 必須在 (0, 1) 之間")
    rc = dt * (1.0 - alpha) / alpha
    return 1.0 / (2.0 * math.pi * rc)


class EMA(StreamFilter):
    """
    指數移動平均（Exponential Moving Average）＝ 一階低通 ＝ RC 電路的數位版。
      y[k] = α·x[k] + (1−α)·y[k−1]

    設計思路
      STEP 1  MA 要存 N 個點；想省記憶體 → 只存「上一次的輸出」一個數。
      STEP 2  新輸出 = 舊輸出往新量測「走一小段」：y += α·(x − y)。
              α=1 完全相信新量測（不濾）；α→0 幾乎不動（超平滑、超慢）。
      STEP 3  展開會發現：過去第 m 點的權重是 α(1−α)^m，呈指數衰減 → 名字由來；
              越舊的資料影響越小、但永遠不會完全歸零（無限脈衝響應 IIR）。
      STEP 4  跟 RC 電路對應：電容電壓 dV/dt = (Vin − V)/RC，用 dt 離散化（後向差分）
              就得到 α = dt/(RC+dt)，RC = 1/(2π·fc)。所以「給截止頻率」或「直接給 α」都行。
      STEP 5  暖機：第一筆直接令 y = x（不要從 0 開始，否則開機那一段會慢慢爬升）。

    參數怎麼選
      ● 給 fc：fc 選在「想保留的訊號最高頻率」之上、「雜訊主要頻率」之下。
      ● 給 α：等效平均點數 N_eq ≈ 2/α − 1（α=0.1 約等於 19 點 MA 的平滑度）。
      ● 時間常數 τ = RC ≈ dt·(1−α)/α：步階響應走到 63% 需要約 τ 秒。

    代價
      延遲：低頻群延遲約 τ/dt = (1−α)/α 個取樣；相位是非線性的（不同頻率延遲不同）。
      計算：1 乘 1 加；記憶體：1 個數 —— MCU 上最便宜的濾波器。
      衰減只有 −20 dB/decade（一階），對靠近 fc 的雜訊壓不太下去 → 需要更陡就用 Biquad。

    適用：感測器讀值平滑、控制迴路裡的量測濾波、任何「記憶體很少」的場合。
    不適用：尖峰脈衝（一個尖峰會拖出一條指數尾巴）；需要陡峭截止。
    """

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("α 必須在 (0, 1] 之間")
        self.alpha = float(alpha)
        self.reset()

    @classmethod
    def from_cutoff(cls, fc: float, dt: float) -> "EMA":
        """由截止頻率 fc [Hz] 與取樣間隔 dt [s] 建立。"""
        f = cls(alpha_from_cutoff(fc, dt))
        f.fc, f.dt = fc, dt
        return f

    def reset(self):
        self.y = None
        self.y_prev: Optional[float] = None   # 這一步混合前的舊輸出
        self.x_last: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        self.x_last = x
        if self.y is None:            # 暖機：第一筆直接當輸出
            self.y_prev = x
            self.y = x
        else:
            self.y_prev = self.y
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y_prev
        return self.y


class LowPass1st(EMA):
    """一階低通（用截止頻率指定的 EMA）：LowPass1st(fc, dt)，內容與 EMA 完全相同。"""

    def __init__(self, fc: float, dt: float):
        self.fc, self.dt = float(fc), float(dt)
        super().__init__(alpha_from_cutoff(fc, dt))


# =========================================================
#  3. 中位數濾波
# =========================================================
def _median_of_sorted(s: List[float]) -> float:
    m = len(s)
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


class MedianFilter(StreamFilter):
    """
    中位數濾波（Median Filter）：輸出 = 最近 N 個取樣排序後「正中間」那個。

    設計思路
      STEP 1  尖峰（脈衝雜訊）的特徵是「少數幾點、數值極端」。平均會被它拉走；
              但「排序後取中間」根本不看極端值 → 尖峰被擠到排序的兩端，碰不到中位數。
      STEP 2  同一個道理讓它「保邊緣」：步階前後各是一群相近的值，
              窗口過半換成新位準時中位數就直接跳過去，不會像平均那樣變成斜坡。
      STEP 3  實作：除了按時間順序的窗口（deque），另外維護一份「已排序」的串列；
              新值用二分搜尋插入、最舊值用二分搜尋刪除 → 每步 O(log N) 找位置 + O(N) 搬移，
              比每步重排 O(N log N) 快。（N 很大時可改用雙堆積 / skip list。）
      STEP 4  暖機期：窗口未滿就用現有的點取中位數；N 建議用奇數，中位數才是「真實存在的一點」。

    參數怎麼選
      ● 能去掉的最長連續尖峰 = (N−1)/2 點 → N ≥ 2·(最長尖峰長度) + 1。
      ● N 太大會把「真實的窄峰」也當成尖峰削掉，還會把細節變成階梯狀。常用 3、5、7。

    代價
      延遲：約 (N−1)/2 點（對步階而言）；非線性 → 沒有「頻率響應」可言。
      計算：每步 O(N)（插入/刪除搬移）；記憶體：2N 個數。
      對白噪的平滑效率比 MA 差（同 N 時雜訊變異大約多 57%）。

    適用：尖峰脈衝、ADC 偶發錯讀、超音波/雷射測距的偶發離譜值、需要保留邊緣的訊號。
    不適用：純高斯白噪（用 MA/EMA 更有效率）、尖峰連續超過半個窗口。
    """

    def __init__(self, n: int = 5):
        if int(n) < 1:
            raise ValueError("N 必須 ≥ 1")
        self.n = int(n)
        self.reset()

    def reset(self):
        self.buf: deque = deque()          # 時間順序（舊 → 新）
        self.sorted: List[float] = []      # 同樣內容、由小到大
        self.last_out: Optional[float] = None
        self.x_last: Optional[float] = None
        self.y = None

    def update(self, x: float) -> float:
        x = float(x)
        if len(self.buf) == self.n:                  # 窗口滿了：先把最舊的移出兩份資料
            old = self.buf.popleft()
            del self.sorted[bisect.bisect_left(self.sorted, old)]
            self.last_out = old
        else:
            self.last_out = None
        self.buf.append(x)
        bisect.insort(self.sorted, x)                # 二分搜尋插入，保持排序
        self.x_last = x
        self.y = _median_of_sorted(self.sorted)
        return self.y

    def window(self) -> List[float]:
        return list(self.buf)


# =========================================================
#  4. Hampel 離群值濾波
# =========================================================
class HampelFilter(StreamFilter):
    """
    Hampel 濾波：用「中位數 + MAD」判斷這一點是不是離群值；是才換掉，不是就原值照用。

    設計思路
      STEP 1  中位數濾波會「每一點都換成中位數」→ 連正常點的細節也被改掉。
              Hampel 改成「先判斷、再決定」：只動壞點，好點一個都不碰。
      STEP 2  判斷要有一把「尺」：一般用標準差，但標準差本身會被尖峰撐大（自己騙自己）。
              改用穩健版：MAD = median(|w − median(w)|)，對尖峰幾乎免疫。
      STEP 3  把 MAD 換算成標準差：常態分布下 σ ≈ 1.4826 × MAD（1.4826 = 1/Φ⁻¹(0.75)）。
      STEP 4  規則：|x − med| > k·σ → 離群，輸出 med；否則輸出 x 本身。
              k=3 相當於常態的「三個標準差」，正常點被誤判的機率約 0.3%。
      STEP 5  串流版是「因果」的：拿最近 N 點（含目前這點）當窗口來判斷目前這點，零延遲；
              離線版常用「置中窗口」，判斷更準但要等 (N−1)/2 點的未來資料。
      STEP 6  MAD 可能是 0（訊號完全平坦、或被量化成同一個值）→ 任何一點點偏差都會被判離群。
              工程上給一個下限 min_sigma（例如 1 個 LSB）保護。

    參數怎麼選
      ● N：要讓「窗口內尖峰數 < N/2」，常用 5～11。
      ● k：2 偏嚴（容易誤殺）、3 標準、4～5 寬鬆（只抓很離譜的）。
      ● min_sigma：設成感測器解析度（量化間距）或可接受的雜訊下限。

    代價
      延遲：因果版 0（正常點原值照輸出）；計算：每步排序 O(N log N)；記憶體：N 個數。

    適用：混有偶發尖峰、但又想「保留原始細節」的資料（資料清洗、感測器前處理、
          接在 EMA/卡爾曼之前先剔除壞點）。
    不適用：尖峰密集到超過半個窗口；雜訊本身就很大（門檻跟著變寬，抓不到尖峰）。
    """

    MAD_SCALE = 1.4826

    def __init__(self, n: int = 7, k: float = 3.0, min_sigma: float = 0.0):
        if int(n) < 3:
            raise ValueError("Hampel 的窗口 N 至少要 3")
        self.n = int(n)
        self.k = float(k)
        self.min_sigma = float(min_sigma)
        self.reset()

    def reset(self):
        self.buf: deque = deque(maxlen=self.n)
        self.med: Optional[float] = None
        self.mad: Optional[float] = None
        self.sigma: Optional[float] = None
        self.threshold: Optional[float] = None
        self.deviation: Optional[float] = None
        self.is_outlier = False
        self.n_outliers = 0
        self.x_last: Optional[float] = None
        self.y = None

    def update(self, x: float) -> float:
        x = float(x)
        self.buf.append(x)
        self.x_last = x
        s = sorted(self.buf)
        self.med = _median_of_sorted(s)
        self.mad = _median_of_sorted(sorted(abs(v - self.med) for v in s))
        self.sigma = max(self.MAD_SCALE * self.mad, self.min_sigma)
        self.threshold = self.k * self.sigma
        self.deviation = abs(x - self.med)
        # 暖機：少於 3 點時 MAD 沒有意義，一律當正常點
        self.is_outlier = len(self.buf) >= 3 and self.deviation > self.threshold
        if self.is_outlier:
            self.n_outliers += 1
            self.y = self.med
        else:
            self.y = x
        return self.y

    def window(self) -> List[float]:
        return list(self.buf)


# =========================================================
#  5. Alpha-Beta 追蹤濾波
# =========================================================
def tracking_index(sigma_accel: float, sigma_meas: float, dt: float) -> float:
    """
    追蹤指數 λ = σ_a · dt² / σ_v（Kalata, 1984）。
      σ_a：目標「加速度」的標準差（模型多不可靠）；σ_v：量測雜訊標準差。
    λ 大 → 目標很會亂動 / 量測很準 → 該多相信量測（α、β 變大）。
    """
    return sigma_accel * dt * dt / sigma_meas


def alpha_beta_from_tracking_index(lam: float):
    """
    由追蹤指數 λ 算出「最佳」α、β —— 就是等速模型 + 離散白噪加速度
    的卡爾曼濾波在穩態時的增益（K = [α, β/dt]）：
      r = (4 + λ − √(λ² + 8λ)) / 4,  α = 1 − r²,  β = 2·(1 − r)²
    """
    if lam <= 0:
        raise ValueError("λ 必須 > 0")
    r = (4.0 + lam - math.sqrt(lam * lam + 8.0 * lam)) / 4.0
    return 1.0 - r * r, 2.0 * (1.0 - r) ** 2


def benedict_bordner_beta(alpha: float) -> float:
    """經驗法則（Benedict–Bordner）：給定 α，β = α²/(2−α) 可兼顧雜訊抑制與暫態響應。"""
    return alpha * alpha / (2.0 - alpha)


class AlphaBetaFilter(StreamFilter):
    """
    Alpha-Beta 濾波：同時估「位置 x̂」與「速度 v̂」的固定增益追蹤器（穩態卡爾曼的簡化版）。

    設計思路
      STEP 1  EMA 只有位置，沒有「速度」的概念 → 目標等速移動時永遠落後一截（斜坡延遲）。
              補一個速度狀態，就能先「預測」目標下一刻在哪。
      STEP 2  預測：x̂⁻ = x̂ + v̂·dt（等速模型），v̂⁻ = v̂。
      STEP 3  殘差（創新）：r = z − x̂⁻ ＝「量測跟預測差多少」。
      STEP 4  修正：x̂ = x̂⁻ + α·r、v̂ = v̂ + (β/dt)·r。
              α 決定位置要多相信量測；β 決定速度要多快被殘差修正。
      STEP 5  跟卡爾曼的關係：卡爾曼的增益 K 每步都會變；跑久了 P 收斂、K 變成常數，
              那個常數 K = [α, β/dt] 就是 Alpha-Beta → 省掉矩陣運算，只剩幾個乘加。

    參數怎麼選
      ● 有雜訊統計：λ = tracking_index(σ_a, σ_v, dt)，再 alpha_beta_from_tracking_index(λ)。
      ● 經驗值：先選 α（0.1～0.5），β = benedict_bordner_beta(α) = α²/(2−α)。
      ● 穩定條件：0 < α < 2、0 < β < 4 − 2α；實務上 α < 1、β 比 α 小一個量級。

    代價
      延遲：等速（斜坡）時穩態誤差 = 0；加速時才有誤差。步階會有一點過衝。
      計算：約 6 個乘加；記憶體：2 個數（x̂、v̂）。

    適用：雷達/影像目標追蹤、編碼器位置 → 速度估計、任何「近似等速」的訊號。
    不適用：常常急轉彎/急停（模型錯）、尖峰雜訊（殘差太大會把速度帶歪 → 先過 Hampel）。
    """

    def __init__(self, alpha: float, beta: float, dt: float, v0: float = 0.0):
        if dt <= 0:
            raise ValueError("dt 必須 > 0")
        self.alpha, self.beta, self.dt, self.v0 = float(alpha), float(beta), float(dt), float(v0)
        self.reset()

    def reset(self):
        self.x: Optional[float] = None     # 位置估計 x̂
        self.v = self.v0                   # 速度估計 v̂
        self.x_pred: Optional[float] = None
        self.residual: Optional[float] = None
        self.z_last: Optional[float] = None
        self.y = None

    def update(self, z: float) -> float:
        z = float(z)
        self.z_last = z
        if self.x is None:                 # 暖機：位置直接用第一筆量測、速度用 v0
            self.x = z
            self.v = self.v0
            self.x_pred = z
            self.residual = 0.0
        else:
            self.x_pred = self.x + self.v * self.dt              # STEP 2 預測
            self.residual = z - self.x_pred                       # STEP 3 殘差
            self.x = self.x_pred + self.alpha * self.residual     # STEP 4 修正位置
            self.v = self.v + (self.beta / self.dt) * self.residual  # 修正速度
        self.y = self.x
        return self.y


# =========================================================
#  6. 互補濾波（IMU 姿態角）
# =========================================================
class ComplementaryFilter(StreamFilter):
    """
    互補濾波：把「短期準但會漂」的陀螺儀，和「長期準但很吵」的加速度計，互補成一個角度。
      θ[k] = α·(θ[k−1] + ω[k]·dt) + (1−α)·a[k]

    設計思路
      STEP 1  陀螺儀量的是角速度 ω，積分得到角度：反應快、很平滑，
              但只要有一點點偏差（bias），積分後就變成一條越飄越遠的漂移 → 「低頻爛、高頻好」。
      STEP 2  加速度計用重力方向算角度 a：長期平均是準的，但車體振動讓它很吵 → 「高頻爛、低頻好」。
      STEP 3  所以：陀螺儀那條走「高通」（只取它的變化量）、加速度計那條走「低通」（只取它的長期值），
              兩個濾波器的轉移函數加起來剛好 = 1（HP + LP = 1，「互補」名稱由來）→ 真實訊號不失真。
      STEP 4  寫成一行：先用陀螺儀把上一個角度往前推一步（θ + ω·dt），
              再每步往加速度計的讀值「拉一點點」(1−α)。
      STEP 5  α 由時間常數 τ 決定：α = τ/(τ + dt)。比 τ 還快的變化信陀螺儀、比 τ 還慢的信加速度計。

    參數怎麼選
      ● τ ≈ 0.5～2 s 最常見：τ 太小 → 加速度計的振動雜訊漏進來；
        τ 太大 → 陀螺儀偏差的漂移來不及拉回（穩態誤差約 = bias × τ）。
      ● dt 要用「實際量到的」取樣間隔；dt 抖動大時每步重算 α。

    代價
      延遲：幾乎沒有（高頻靠陀螺儀即時反應）；穩態誤差 ≈ 陀螺偏差 × τ。
      計算：3 乘 2 加；記憶體：1 個角度。

    適用：IMU 姿態角（平衡車、無人機、手勢）、任何「一個快但漂、一個準但吵」的雙感測器融合。
    不適用：大角度 3D 旋轉（要用四元數版本，如 Mahony / Madgwick）；
           持續加速中的載具（加速度計不再只量到重力，長期也不準了）。
    """

    def __init__(self, dt: float, alpha: Optional[float] = None, tau: Optional[float] = None):
        if dt <= 0:
            raise ValueError("dt 必須 > 0")
        if (alpha is None) == (tau is None):
            raise ValueError("alpha 與 tau 請擇一給定")
        self.dt = float(dt)
        if tau is not None:
            self.tau = float(tau)
            self.alpha = self.tau / (self.tau + self.dt)
        else:
            if not 0.0 <= alpha < 1.0:
                raise ValueError("α 必須在 [0, 1) 之間")
            self.alpha = float(alpha)
            self.tau = self.alpha * self.dt / (1.0 - self.alpha)
        self.reset()

    def reset(self):
        self.angle: Optional[float] = None
        self.angle_pred: Optional[float] = None   # 陀螺儀預測：θ_prev + ω·dt
        self.gyro_term: Optional[float] = None    # α × 陀螺儀預測
        self.accel_term: Optional[float] = None   # (1−α) × 加速度計角度
        self.y = None

    def update(self, gyro_rate: float, accel_angle: float) -> float:
        gyro_rate, accel_angle = float(gyro_rate), float(accel_angle)
        if self.angle is None:                     # 暖機：一開始只能相信加速度計
            self.angle = accel_angle
            self.angle_pred = accel_angle
            self.gyro_term = 0.0
            self.accel_term = accel_angle
        else:
            self.angle_pred = self.angle + gyro_rate * self.dt
            self.gyro_term = self.alpha * self.angle_pred
            self.accel_term = (1.0 - self.alpha) * accel_angle
            self.angle = self.gyro_term + self.accel_term
        self.y = self.angle
        return self.y


# =========================================================
#  7. Biquad（二階 IIR）+ Butterworth 係數
# =========================================================
class Biquad(StreamFilter):
    """
    Biquad（二階 IIR 節）：Direct Form II Transposed 實作。
      H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)

    設計思路
      STEP 1  一階（EMA）的衰減只有 −20 dB/decade，太鈍。二階給 −40 dB/decade，
              還能用 Q 值調整截止附近的形狀（平坦、或帶共振峰）。
      STEP 2  係數用 RBJ Audio EQ Cookbook 的公式（類比原型 + 雙線性轉換 + 頻率預扭曲），
              Q = 1/√2 時就是 2 階 Butterworth（通帶最平坦、不過衝的極限）。
      STEP 3  實作用 DF2T，只要兩個延遲暫存器 z1、z2：
                y  = b0·x + z1
                z1 = b1·x − a1·y + z2
                z2 = b2·x − a2·y
              DF2T 在浮點下數值特性最好（狀態量級跟輸出接近），每步 5 乘 4 加。
      STEP 4  要更陡（4 階、6 階…）：不要直接用高階多項式（係數誤差會讓極點跑出單位圓），
              而是串接多個 Biquad（SOS, second-order sections），見 BiquadCascade / butterworth_sos。
      STEP 5  暖機：預設 z1=z2=0（與 scipy.signal.lfilter 相同）；
              steady_init=True 會把狀態設成「輸入一直是第一筆」的穩態，避免開機暴衝。

    參數怎麼選
      ● fc 與 fs 都用 Hz！fc 必須 < fs/2（奈奎斯特）；fc 越接近 fs/2，係數越敏感。
      ● Q：0.707 Butterworth（最平坦）；0.5 臨界阻尼（步階不過衝）；> 0.707 會有共振峰與過衝。
      ● 低通的 fc 選在訊號頻寬之上 2～3 倍，延遲才不會太大。

    代價
      延遲：低頻群延遲 ≈ √2/(2π·fc) 秒（Butterworth 2 階），比同平滑度的 MA 小；非線性相位。
      計算：5 乘 4 加；記憶體：2 個狀態 + 5 個係數。
      定點數（MCU 上 Q15/Q31）實作要注意係數量化與溢位 → 那時 DF1 反而比較好飽和處理。

    適用：要明確規格（截止頻率、衰減斜率）的低通/高通/帶通、音訊、振動訊號、抗混疊後處理。
    不適用：尖峰（IIR 會把尖峰拖成振鈴）；需要線性相位（改用 FIR / 零相位 filtfilt 離線處理）。
    """

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float,
                 steady_init: bool = False):
        self.b0, self.b1, self.b2 = float(b0), float(b1), float(b2)
        self.a1, self.a2 = float(a1), float(a2)
        self.steady_init = bool(steady_init)
        self.reset()

    @classmethod
    def from_ba(cls, b: Sequence[float], a: Sequence[float], steady_init: bool = False) -> "Biquad":
        """由 (b0,b1,b2)、(a0,a1,a2) 建立；會自動除以 a0 正規化。"""
        a0 = float(a[0])
        return cls(b[0] / a0, b[1] / a0, b[2] / a0, a[1] / a0, a[2] / a0, steady_init)

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self.x_last: Optional[float] = None
        self.y = None

    def dc_gain(self) -> float:
        """直流增益 H(1) = Σb / (1 + a1 + a2)。低通應為 1、高通應為 0。"""
        den = 1.0 + self.a1 + self.a2
        return (self.b0 + self.b1 + self.b2) / den if den != 0 else math.inf

    def gain_at(self, f: float, fs: float) -> float:
        """頻率 f [Hz] 的增益 |H(e^{jω})|（ω = 2πf/fs）。"""
        zi = complex(math.cos(2 * math.pi * f / fs), -math.sin(2 * math.pi * f / fs))  # z⁻¹
        num = self.b0 + self.b1 * zi + self.b2 * zi * zi
        den = 1.0 + self.a1 * zi + self.a2 * zi * zi
        return abs(num / den)

    def update(self, x: float) -> float:
        x = float(x)
        if self.y is None and self.steady_init:
            g = self.dc_gain()
            y0 = g * x if math.isfinite(g) else 0.0
            self.z1 = y0 - self.b0 * x
            self.z2 = self.b2 * x - self.a2 * y0
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        self.x_last = x
        self.y = y
        return y

    def coefficients(self) -> Dict[str, float]:
        return dict(b0=self.b0, b1=self.b1, b2=self.b2, a1=self.a1, a2=self.a2)


class BiquadCascade(StreamFilter):
    """多個 Biquad 串接（SOS）：前一節的輸出接到下一節的輸入，用來做 4 階、6 階…的濾波器。"""

    def __init__(self, sections: Sequence[Biquad]):
        if not sections:
            raise ValueError("至少要一節 Biquad")
        self.sections = list(sections)
        self.y = None

    def reset(self):
        for s in self.sections:
            s.reset()
        self.y = None

    def update(self, x: float) -> float:
        for s in self.sections:
            x = s.update(x)
        self.y = x
        return x

    def dc_gain(self) -> float:
        return float(np.prod([s.dc_gain() for s in self.sections]))

    def gain_at(self, f: float, fs: float) -> float:
        return float(np.prod([s.gain_at(f, fs) for s in self.sections]))


def _check_fc(fc: float, fs: float):
    if fs <= 0 or fc <= 0:
        raise ValueError("fc 與 fs 都必須 > 0，而且單位都是 Hz")
    if fc >= fs / 2.0:
        raise ValueError(f"fc={fc} Hz 必須小於 fs/2={fs / 2} Hz（奈奎斯特頻率）"
                         "——常見錯誤：把 fs 填成取樣間隔 dt，或把 fc 填成 rad/s")


def rbj_lowpass(fc: float, fs: float, q: float = 1.0 / math.sqrt(2.0),
                steady_init: bool = False) -> Biquad:
    """RBJ Audio EQ Cookbook 的二階低通。q=1/√2 即 Butterworth。"""
    _check_fc(fc, fs)
    w0 = 2.0 * math.pi * fc / fs
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    b = ((1 - cw) / 2, 1 - cw, (1 - cw) / 2)
    a = (1 + alpha, -2 * cw, 1 - alpha)
    return Biquad.from_ba(b, a, steady_init)


def rbj_highpass(fc: float, fs: float, q: float = 1.0 / math.sqrt(2.0),
                 steady_init: bool = False) -> Biquad:
    """RBJ Audio EQ Cookbook 的二階高通。q=1/√2 即 Butterworth。"""
    _check_fc(fc, fs)
    w0 = 2.0 * math.pi * fc / fs
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    b = ((1 + cw) / 2, -(1 + cw), (1 + cw) / 2)
    a = (1 + alpha, -2 * cw, 1 - alpha)
    return Biquad.from_ba(b, a, steady_init)


def butterworth_lowpass(fc: float, fs: float, steady_init: bool = False) -> Biquad:
    """2 階 Butterworth 低通（RBJ, Q = 1/√2）。fc、fs 單位都是 Hz。"""
    return rbj_lowpass(fc, fs, 1.0 / math.sqrt(2.0), steady_init)


def butterworth_highpass(fc: float, fs: float, steady_init: bool = False) -> Biquad:
    """2 階 Butterworth 高通（RBJ, Q = 1/√2）。fc、fs 單位都是 Hz。"""
    return rbj_highpass(fc, fs, 1.0 / math.sqrt(2.0), steady_init)


def _first_order_section(fc: float, fs: float, kind: str) -> Biquad:
    """奇數階需要的一階節（雙線性轉換 + 預扭曲），用 b2 = a2 = 0 的 Biquad 表示。"""
    k = math.tan(math.pi * fc / fs)
    a1 = (k - 1.0) / (k + 1.0)
    if kind == "lowpass":
        g = k / (1.0 + k)
        return Biquad(g, g, 0.0, a1, 0.0)
    g = 1.0 / (1.0 + k)
    return Biquad(g, -g, 0.0, a1, 0.0)


def butterworth_sos(fc: float, fs: float, order: int, kind: str = "lowpass") -> BiquadCascade:
    """
    任意階 Butterworth = 串接 Biquad（SOS）。
    n 階 Butterworth 的極點在 s 平面的半圓上、彼此等角；每一對共軛極點對應一節 Biquad，
    它的 Q = 1 / (2·cos φ)，φ = 極點與負實軸的夾角：
      偶數階 φ_k = (2k−1)·π/(2n)；奇數階 φ_k = k·π/n，另外多一個一階節。
    （例：4 階 → Q = 0.541、1.307；3 階 → Q = 1.0 + 一階節。）
    """
    if kind not in ("lowpass", "highpass"):
        raise ValueError("kind 只能是 'lowpass' 或 'highpass'")
    if order < 1:
        raise ValueError("order 必須 ≥ 1")
    _check_fc(fc, fs)
    maker = rbj_lowpass if kind == "lowpass" else rbj_highpass
    sections: List[Biquad] = []
    for k in range(1, order // 2 + 1):
        if order % 2 == 0:
            phi = (2 * k - 1) * math.pi / (2 * order)
        else:
            phi = k * math.pi / order
        sections.append(maker(fc, fs, 1.0 / (2.0 * math.cos(phi))))
    if order % 2 == 1:
        sections.append(_first_order_section(fc, fs, kind))
    return BiquadCascade(sections)


# =========================================================
#  8. One Euro Filter
# =========================================================
class OneEuroFilter(StreamFilter):
    """
    One Euro Filter（Casiez 2012）：會依「訊號移動速度」自動調整截止頻率的 EMA。
    慢的時候很平滑（去抖動）、快的時候延遲很小（跟得上）。

    設計思路
      STEP 1  EMA 的兩難：fc 低 → 靜止時不抖，但一動起來就拖很久；fc 高 → 跟得緊，但靜止時一直抖。
      STEP 2  觀察：人眼在「靜止」時最在意抖動，在「快速移動」時最在意延遲。
              那就讓 fc 跟著速度變！
      STEP 3  估速度：dx = (x − y_prev)/dt，再用固定 d_cutoff 的 EMA 把 dx 濾一下（速度本身也很吵）。
      STEP 4  自適應截止：fc = min_cutoff + β·|dx̂|。
      STEP 5  用這個 fc 算 α = dt/(RC+dt)，再做一次普通 EMA：y = α·x + (1−α)·y_prev。

    參數怎麼選（官方建議的調法）
      ● 先令 β = 0，讓目標靜止，把 min_cutoff 往下調到「抖動可接受」為止。
      ● 再快速移動目標，把 β 往上調到「延遲可接受」為止。
      ● β 的單位是「1 / (訊號單位/秒)」：訊號用度、用公尺、用像素，β 要差好幾個數量級！
      ● d_cutoff 通常固定 1 Hz 就好。

    代價
      延遲：靜止時 ≈ min_cutoff 的 EMA；移動時小很多。計算：兩個 EMA + 一次 cos/除法；記憶體：3 個數。
      非線性：β 太大時，雜訊本身產生的 |dx̂| 會把 fc 撐高 → 靜止也開始抖。

    適用：人機互動（滑鼠、觸控、手勢、VR 頭部追蹤）、即時顯示的感測值。
    不適用：需要可預測頻率響應的控制迴路（截止頻率一直在變）、尖峰雜訊（會被當成「快速移動」放行）。
    """

    def __init__(self, dt: float, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        if dt <= 0 or min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("dt、min_cutoff、d_cutoff 都必須 > 0")
        self.dt = float(dt)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.y = None
        self.y_prev: Optional[float] = None
        self.x_last: Optional[float] = None
        self.dx = 0.0          # 原始速度估計
        self.dx_hat = 0.0      # 濾過的速度
        self.cutoff = self.min_cutoff
        self.alpha = alpha_from_cutoff(self.min_cutoff, self.dt)

    def update(self, x: float) -> float:
        x = float(x)
        self.x_last = x
        if self.y is None:                          # 暖機：第一筆直接輸出
            self.y_prev = x
            self.y = x
            return x
        self.y_prev = self.y
        self.dx = (x - self.y_prev) / self.dt                              # STEP 3
        a_d = alpha_from_cutoff(self.d_cutoff, self.dt)
        self.dx_hat = a_d * self.dx + (1.0 - a_d) * self.dx_hat
        self.cutoff = self.min_cutoff + self.beta * abs(self.dx_hat)       # STEP 4
        self.alpha = alpha_from_cutoff(self.cutoff, self.dt)               # STEP 5
        self.y = self.alpha * x + (1.0 - self.alpha) * self.y_prev
        return self.y


# =========================================================
#  9. 斜率限制
# =========================================================
class RateLimiter(StreamFilter):
    """
    斜率限制（Rate Limiter / Slew Rate Limiter）：輸出每秒最多只能變化 max_rate。

    設計思路
      STEP 1  有些時候不是要「去雜訊」，而是「不准變太快」：馬達命令、閥門開度、
              LED 亮度、車速設定——突然跳變會造成衝擊、過電流或不舒服。
      STEP 2  每步算「想走多少」：Δ = x − y_prev。
      STEP 3  算「允許走多少」：上升最多 max_rate·dt、下降最多 max_fall_rate·dt。
      STEP 4  把 Δ 夾（clip）在允許範圍內：y = y_prev + clip(Δ, −down, +up)。
              小變化完全照走（零延遲、零失真），只有大跳變會被「拉成斜坡」。

    參數怎麼選
      ● max_rate = 系統能承受的最大變化率（例如馬達加速度上限、閥門行程速度）。
      ● 上升/下降可以不同（例如溫控：加熱要慢、斷電要快）。
      ● 當作去尖峰用：一個高度 h 的尖峰，輸出最多只會偏 max_rate·dt（但尖峰之後要花時間走回來）。

    代價
      延遲：小訊號 0；大步階需要 |Δ|/max_rate 秒才走到。計算：1 減 1 比較；記憶體：1 個數。
      非線性：白噪「不會被平滑」（每步小抖動都在允許範圍內，照樣通過）。

    適用：致動器命令整形、設定值（setpoint）斜坡化、保護機構、UI 數字平順變化。
    不適用：去除白噪（它根本不管小抖動）；需要準確跟上快速變化的量測。
    """

    def __init__(self, max_rate: float, dt: float, max_fall_rate: Optional[float] = None):
        if max_rate <= 0 or dt <= 0:
            raise ValueError("max_rate 與 dt 都必須 > 0")
        self.max_rate = float(max_rate)
        self.max_fall_rate = float(max_fall_rate) if max_fall_rate is not None else self.max_rate
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.y = None
        self.y_prev: Optional[float] = None
        self.wanted = 0.0        # 想走的 Δ
        self.step = 0.0          # 實際走的
        self.up = self.max_rate * self.dt
        self.down = self.max_fall_rate * self.dt
        self.limited = False

    def update(self, x: float) -> float:
        x = float(x)
        if self.y is None:                 # 暖機：第一筆直接採用（不然會從 0 慢慢爬上來）
            self.y_prev = x
            self.y = x
            self.wanted = self.step = 0.0
            self.limited = False
            return x
        self.y_prev = self.y
        self.wanted = x - self.y_prev                                      # STEP 2
        self.up, self.down = self.max_rate * self.dt, self.max_fall_rate * self.dt   # STEP 3
        self.step = min(max(self.wanted, -self.down), self.up)            # STEP 4
        self.limited = self.step != self.wanted
        self.y = self.y_prev + self.step
        return self.y


# =========================================================
#  示範訊號（自我測試與動畫共用；固定 seed，可重現）
# =========================================================
def make_demo_signals(n: int = 200, dt: float = 0.05, seed: int = 7) -> Dict[str, dict]:
    """
    產生 5 種示範訊號，全部以「秒」定義，所以換 n、dt 形狀不變（總長 T = n·dt）：
      sine  正弦 + 高斯白噪          step  雙步階 + 白噪
      spike 慢正弦 + 步階 + 尖峰脈衝  ramp  平 → 斜坡 → 平 + 白噪
      imu   姿態角：陀螺儀（有偏差 + 偏差漂移）、加速度計角度（很吵）
    每個 dict：name、t、truth、meas（濾波器的輸入）、unit、scale；
    imu 另有 gyro（°/s）、accel（°）、gyro_int（純陀螺積分）；spike 另有 spike_mask。
    """
    t = np.arange(n) * dt
    T = n * dt
    out: Dict[str, dict] = {}

    rng = np.random.default_rng([seed, 1])
    truth = np.sin(2 * np.pi * (2.0 / T) * t)                  # 整段剛好 2 個週期
    out["sine"] = dict(name="正弦+雜訊", t=t, truth=truth,
                       meas=truth + rng.normal(0, 0.25, n), unit="", scale=1.0)

    rng = np.random.default_rng([seed, 2])
    truth = np.where(t < 0.3 * T, 0.0, np.where(t < 0.7 * T, 1.0, 0.4))
    out["step"] = dict(name="步階+雜訊", t=t, truth=truth,
                       meas=truth + rng.normal(0, 0.1, n), unit="", scale=1.0)

    rng = np.random.default_rng([seed, 3])
    truth = 0.4 * np.sin(2 * np.pi * (1.2 / T) * t) + np.where(t < 0.5 * T, 0.0, 0.8)
    meas = truth + rng.normal(0, 0.05, n)
    mask = rng.random(n) < 0.07
    mask[:3] = False                                              # 開頭留乾淨，方便看暖機
    amp = rng.uniform(1.2, 2.0, n) * rng.choice([-1.0, 1.0], n)
    meas = np.where(mask, meas + amp, meas)
    out["spike"] = dict(name="尖峰脈衝", t=t, truth=truth, meas=meas, unit="", scale=1.0,
                        spike_mask=mask)

    rng = np.random.default_rng([seed, 4])
    t0, t1 = 0.2 * T, 0.8 * T
    truth = np.clip((t - t0) / (t1 - t0), 0.0, 1.0) * 1.5
    out["ramp"] = dict(name="斜坡+雜訊", t=t, truth=truth,
                       meas=truth + rng.normal(0, 0.12, n), unit="", scale=1.0)

    rng = np.random.default_rng([seed, 5])
    w1, w2 = 2 * np.pi * (1.5 / T), 2 * np.pi * (4.0 / T)
    angle = 25.0 * np.sin(w1 * t) + 10.0 * np.sin(w2 * t + 1.0)          # 真實角度 [°]
    rate = 25.0 * w1 * np.cos(w1 * t) + 10.0 * w2 * np.cos(w2 * t + 1.0)  # 真實角速度 [°/s]
    bias = 2.0 + 0.3 * math.sqrt(dt) * np.cumsum(rng.normal(0, 1, n))     # 偏差 + 隨機漫步
    gyro = rate + bias + rng.normal(0, 1.0, n)
    accel = angle + rng.normal(0, 6.0, n)
    gyro_int = np.empty(n)
    gyro_int[0] = angle[0]                    # 就算起點給對，積分仍然會漂
    for i in range(1, n):
        gyro_int[i] = gyro_int[i - 1] + gyro[i] * dt
    out["imu"] = dict(name="IMU 姿態角", t=t, truth=angle, meas=accel, gyro=gyro, accel=accel,
                      gyro_int=gyro_int, unit="°", scale=30.0)
    return out


# =========================================================
#  評估工具
# =========================================================
def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def xcorr_delay(y: Sequence[float], ref: Sequence[float], max_lag: int) -> int:
    """互相關估延遲：把 y 往前挪 L 點後跟 ref 最相像（相關最大）→ 延遲 ≈ L 點。"""
    y, ref = np.asarray(y, float), np.asarray(ref, float)
    n = min(len(y), len(ref))
    max_lag = max(0, min(max_lag, n - 2))
    best, best_l = -np.inf, 0
    for lag in range(max_lag + 1):
        a = y[max_lag:n]
        b = ref[max_lag - lag:n - lag]
        a, b = a - a.mean(), b - b.mean()
        den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
        c = float(np.dot(a, b)) / den if den > 0 else -np.inf
        if c > best + 1e-12:
            best, best_l = c, lag
    return best_l


def align_delay(y: Sequence[float], ref: Sequence[float], max_lag: int) -> int:
    """最小平方對齊估延遲：找 L 讓 mean((y[k] − ref[k−L])²) 最小（每個 L 比較的點數相同）。"""
    y, ref = np.asarray(y, float), np.asarray(ref, float)
    n = min(len(y), len(ref))
    max_lag = max(0, min(max_lag, n - 2))
    errs = [float(np.mean((y[max_lag:n] - ref[max_lag - lag:n - lag]) ** 2))
            for lag in range(max_lag + 1)]
    return int(np.argmin(errs))


def step_delay_50(y: Sequence[float], step_idx: int, lo: float = 0.0, hi: float = 1.0) -> Optional[int]:
    """步階 50% 上升時間：從步階發生那一點起，輸出第一次越過 (lo+hi)/2 要幾個取樣。"""
    y = np.asarray(y, float)
    mid = 0.5 * (lo + hi)
    for i in range(step_idx, len(y)):
        if (y[i] >= mid) if hi > lo else (y[i] <= mid):
            return i - step_idx
    return None


# =========================================================
#  自我測試
# =========================================================
def _disp_w(s: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _disp_w(s))


def _steady_kalman_gain(dt: float, sigma_a: float, sigma_v: float) -> np.ndarray:
    """等速模型 + 離散白噪加速度（DWNA）的卡爾曼濾波，迭代 Riccati 到收斂，回傳穩態增益 K。"""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([[0.5 * dt * dt], [dt]])
    Q = G @ G.T * sigma_a ** 2
    H = np.array([[1.0, 0.0]])
    R = np.array([[sigma_v ** 2]])
    P = np.eye(2) * 1e3
    K = np.zeros((2, 1))
    for _ in range(20000):
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K_new = Pp @ H.T @ np.linalg.inv(S)
        P = (np.eye(2) - K_new @ H) @ Pp
        if np.max(np.abs(K_new - K)) < 1e-15:
            K = K_new
            break
        K = K_new
    return K.ravel()


def _self_test():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    fs = 100.0
    dt = 1.0 / fs
    n = 1000                                     # 10 秒
    sig = make_demo_signals(n=n, dt=dt, seed=7)
    print(f"自我測試：fs = {fs:.0f} Hz（dt = {dt} s），每種訊號 {n} 點（{n * dt:.0f} 秒）\n")

    a_ab, b_ab = 0.12, benedict_bordner_beta(0.12)
    makers = [
        ("移動平均 MA(N=21)", lambda: MovingAverage(21)),
        ("EMA(fc=2Hz)", lambda: EMA.from_cutoff(2.0, dt)),
        ("中位數 Median(N=11)", lambda: MedianFilter(11)),
        ("Hampel(N=11,k=3)", lambda: HampelFilter(11, 3.0)),
        (f"Alpha-Beta(α={a_ab})", lambda: AlphaBetaFilter(a_ab, b_ab, dt)),
        ("Biquad 巴特沃斯(2Hz)", lambda: butterworth_lowpass(2.0, fs)),
        ("One Euro(0.5Hz,β=0.1)", lambda: OneEuroFilter(dt, 0.5, 0.1)),
        ("斜率限制(2/s)", lambda: RateLimiter(2.0, dt)),
    ]
    keys = ["sine", "step", "spike", "ramp"]
    step_idx = int(np.argmax(sig["step"]["truth"] >= 0.5))

    results: Dict[str, dict] = {}
    raw = {k: rmse(sig[k]["meas"], sig[k]["truth"]) for k in keys}
    head = (_pad("濾波器", 24) + "".join(_pad(f"{sig[k]['name']}RMSE", 15) for k in keys)
            + _pad("正弦延遲(xcorr)", 17) + "步階50%延遲")
    print(head)
    print("-" * _disp_w(head))
    line = _pad("（原始量測）", 24) + "".join(_pad(f"{raw[k]:.4f}", 15) for k in keys)
    line += _pad(f"{xcorr_delay(sig['sine']['meas'], sig['sine']['truth'], 100)} 點", 17)
    line += f"{step_delay_50(sig['step']['meas'], step_idx)} 點"
    print(line)
    for name, mk in makers:
        f = mk()
        outs = {k: apply(f, sig[k]["meas"]) for k in keys}
        r = {k: rmse(outs[k], sig[k]["truth"]) for k in keys}
        d_sine = xcorr_delay(outs["sine"], sig["sine"]["truth"], 100)
        d_step = step_delay_50(outs["step"], step_idx)
        results[name] = dict(rmse=r, d_sine=d_sine, d_step=d_step, outs=outs)
        line = _pad(name, 24) + "".join(_pad(f"{r[k]:.4f}", 15) for k in keys)
        line += _pad(f"{d_sine} 點", 17) + f"{d_step} 點"
        print(line)

    # ---- IMU：互補濾波 ----
    imu = sig["imu"]
    cf = ComplementaryFilter(dt, tau=1.0)
    y_cf = apply(cf, imu["gyro"], imu["accel"])
    y_ema_acc = apply(EMA.from_cutoff(0.5, dt), imu["accel"])
    r_acc = rmse(imu["accel"], imu["truth"])
    r_gyro = rmse(imu["gyro_int"], imu["truth"])
    r_ema = rmse(y_ema_acc, imu["truth"])
    r_cf = rmse(y_cf, imu["truth"])
    print("\nIMU 姿態角（單位 °）：")
    print(f"  加速度計原始 RMSE = {r_acc:6.2f}°　（很吵）")
    print(f"  純陀螺積分   RMSE = {r_gyro:6.2f}°　（不吵但會漂，最後偏了 "
          f"{imu['gyro_int'][-1] - imu['truth'][-1]:+.1f}°）")
    print(f"  EMA(0.5Hz) 濾加速度計 RMSE = {r_ema:6.2f}°　（延遲 {xcorr_delay(y_ema_acc, imu['truth'], 100)} 點）")
    print(f"  互補濾波(τ=1s, α={cf.alpha:.4f}) RMSE = {r_cf:6.2f}°　"
          f"（延遲 {xcorr_delay(y_cf, imu['truth'], 100)} 點）")

    # =========================================================
    #  合理性檢查（assert）
    # =========================================================
    print("\n合理性檢查：")
    R = {k: v["rmse"] for k, v in results.items()}
    get = lambda prefix: next(v for k, v in results.items() if k.startswith(prefix))  # noqa: E731

    # (1) 平均類能壓白噪
    for prefix in ("移動平均", "EMA", "Biquad", "One Euro"):
        r = get(prefix)["rmse"]["sine"]
        assert r < 0.6 * raw["sine"], (prefix, r)
    print("  ✔ MA / EMA / Biquad / OneEuro 在正弦+白噪上 RMSE 都降到原始的 60% 以下")

    # (2) 排序類能去尖峰，而且比平均類好很多
    r_med, r_ham = get("中位數")["rmse"]["spike"], get("Hampel")["rmse"]["spike"]
    r_ma_sp = get("移動平均")["rmse"]["spike"]
    assert r_med < 0.3 * raw["spike"] and r_ham < 0.3 * raw["spike"], (r_med, r_ham, raw["spike"])
    assert r_med < r_ma_sp and r_ham < r_ma_sp
    print(f"  ✔ 尖峰訊號：原始 {raw['spike']:.3f} → Median {r_med:.3f}、Hampel {r_ham:.3f}"
          f"（MA 只到 {r_ma_sp:.3f}）")

    # (3) Hampel 只換離群點：沒被判離群的點，輸出與輸入完全相同；注入的尖峰幾乎全抓到
    hf = HampelFilter(11, 3.0)
    flags, ys = [], []
    for x in sig["spike"]["meas"]:
        ys.append(hf.update(x))
        flags.append(hf.is_outlier)
    flags, ys = np.array(flags), np.array(ys)
    meas = sig["spike"]["meas"]
    assert np.all(ys[~flags] == meas[~flags])
    mask = sig["spike"]["spike_mask"]
    hit = float(np.mean(flags[mask]))
    false_alarm = float(np.mean(flags[~mask]))
    assert hit > 0.9 and false_alarm < 0.05, (hit, false_alarm)
    print(f"  ✔ Hampel：注入尖峰抓到 {hit:.0%}、正常點誤判 {false_alarm:.1%}，未判離群的點原值保留")

    # (4) MA 與 Median 跟「離線算法」逐點一致
    x = sig["sine"]["meas"]
    y_ma = apply(MovingAverage(21), x)
    ref = np.convolve(x, np.ones(21) / 21, mode="valid")
    assert np.max(np.abs(y_ma[20:] - ref)) < 1e-9
    y_md = apply(MedianFilter(11), x)
    ref = np.array([np.median(x[i - 10:i + 1]) for i in range(10, len(x))])
    assert np.max(np.abs(y_md[10:] - ref)) < 1e-12
    print("  ✔ MovingAverage = np.convolve、MedianFilter = np.median（滿窗口後逐點一致）")

    # (5) MA 浮點誤差：一個 1e12 的尖峰進出窗口後，累加和殘留誤差；定期重算可清掉
    data = np.concatenate([[1e12], np.full(50, 0.123456789)])
    y_bad = apply(MovingAverage(8, recompute_every=0), data)[-1]
    y_good = apply(MovingAverage(8, recompute_every=8), data)[-1]
    err_bad, err_good = abs(y_bad - 0.123456789), abs(y_good - 0.123456789)
    assert err_bad > 1e-6 and err_good < 1e-12, (err_bad, err_good)
    print(f"  ✔ MA 浮點誤差：1e12 尖峰離開窗口後，不重算誤差 {err_bad:.2e}；每 N 步重算 → {err_good:.1e}")

    # (6) Alpha-Beta：追蹤指數公式 = 穩態卡爾曼增益；斜坡上沒有穩態延遲
    for sa, sv in ((1.0, 0.25), (5.0, 0.1), (0.2, 1.0)):
        lam = tracking_index(sa, sv, dt)
        a, b = alpha_beta_from_tracking_index(lam)
        K = _steady_kalman_gain(dt, sa, sv)
        assert abs(a - K[0]) < 1e-6 and abs(b - K[1] * dt) < 1e-6, (a, b, K)
    lam = tracking_index(1.0, 0.25, dt)
    a, b = alpha_beta_from_tracking_index(lam)
    print(f"  ✔ Alpha-Beta：λ={lam:.2e} → α={a:.4f}, β={b:.6f}，與 Riccati 迭代的穩態卡爾曼增益一致（<1e-6）")
    ramp_t = sig["ramp"]["truth"]
    seg = slice(int(0.5 * n), int(0.78 * n))                 # 斜坡中後段（已收斂）
    bias_ab = abs(np.mean(get("Alpha-Beta")["outs"]["ramp"][seg] - ramp_t[seg]))
    bias_ema = abs(np.mean(get("EMA")["outs"]["ramp"][seg] - ramp_t[seg]))
    assert bias_ab < 0.3 * bias_ema, (bias_ab, bias_ema)
    print(f"  ✔ 斜坡穩態誤差：Alpha-Beta {bias_ab:.4f} vs EMA {bias_ema:.4f}（EMA 永遠落後 斜率×τ）")

    # (7) 互補濾波比單用任一感測器都好
    assert r_cf < r_acc and r_cf < r_gyro and r_cf < r_ema, (r_cf, r_acc, r_gyro, r_ema)
    print(f"  ✔ 互補濾波 {r_cf:.2f}° < 加速度計 {r_acc:.2f}°、純陀螺 {r_gyro:.2f}°、EMA {r_ema:.2f}°")

    # (8) 斜率限制：輸出每步變化絕不超過 max_rate·dt
    for k in keys:
        dy = np.abs(np.diff(get("斜率限制")["outs"][k]))
        assert np.max(dy) <= 2.0 * dt + 1e-12
    print(f"  ✔ RateLimiter：所有訊號上 |Δy| ≤ max_rate·dt = {2.0 * dt:.3f}")

    # (9) One Euro：靜止時跟 EMA(min_cutoff) 一樣平滑，但步階延遲小很多
    rest = slice(int(0.1 * n), step_idx)
    em = apply(EMA.from_cutoff(0.5, dt), sig["step"]["meas"])
    d_em, j_em = step_delay_50(em, step_idx), float(np.std(em[rest]))
    sweep = []
    for b in (0.0, 0.1, 0.5):
        oe = apply(OneEuroFilter(dt, 0.5, b), sig["step"]["meas"])
        sweep.append((b, step_delay_50(oe, step_idx), float(np.std(oe[rest]))))
    d_oe, j_oe = sweep[1][1], sweep[1][2]
    assert d_oe < 0.5 * d_em and j_oe < 1.5 * j_em, (d_oe, d_em, j_oe, j_em)
    assert sweep[0][1] == d_em and sweep[2][2] > j_oe          # β=0 退化成 EMA；β 太大靜止也抖
    print(f"  ✔ One Euro(β=0.1)：靜止抖動 {j_oe:.4f}（EMA 0.5Hz：{j_em:.4f}），"
          f"步階延遲 {d_oe} 點（EMA：{d_em} 點）")
    print("      β 掃描（min_cutoff=0.5Hz）：" + "；".join(
        f"β={b}: 延遲 {d} 點、靜止抖動 {j:.4f}" for b, d, j in sweep))

    # (10) Biquad：直流增益、−3 dB 點、與 scipy 逐點比對
    lp, hp = butterworth_lowpass(5.0, fs), butterworth_highpass(5.0, fs)
    assert abs(lp.dc_gain() - 1.0) < 1e-12 and abs(hp.dc_gain()) < 1e-12
    assert abs(lp.gain_at(5.0, fs) - 1 / math.sqrt(2)) < 1e-9
    assert abs(hp.gain_at(5.0, fs) - 1 / math.sqrt(2)) < 1e-9
    c4 = butterworth_sos(5.0, fs, 4)
    assert abs(c4.gain_at(5.0, fs) - 1 / math.sqrt(2)) < 1e-9
    print("  ✔ Biquad：低通 DC 增益 = 1、高通 = 0，在 fc 處都是 −3 dB（1/√2）；4 階 SOS 也是")
    try:
        from scipy import signal as sps
    except ImportError:
        print("  （沒有安裝 scipy，略過 lfilter / sosfilt 逐點比對）")
    else:
        x = sig["sine"]["meas"]
        worst = 0.0
        for maker, q in ((rbj_lowpass, 0.7071067811865476), (rbj_highpass, 0.7071067811865476),
                         (rbj_lowpass, 2.0)):
            bq = maker(3.0, fs, q)
            ref = sps.lfilter([bq.b0, bq.b1, bq.b2], [1.0, bq.a1, bq.a2], x)
            worst = max(worst, float(np.max(np.abs(apply(bq, x) - ref))))
        b_s, a_s = sps.butter(2, 3.0, btype="low", fs=fs)
        bw = butterworth_lowpass(3.0, fs)
        assert np.allclose([bw.b0, bw.b1, bw.b2], b_s / a_s[0], atol=1e-12)
        assert np.allclose([1.0, bw.a1, bw.a2], a_s / a_s[0], atol=1e-12)
        for order, kind in ((4, "lowpass"), (3, "lowpass"), (5, "highpass")):
            sos = sps.butter(order, 3.0, btype="low" if kind == "lowpass" else "high",
                             fs=fs, output="sos")
            ref = sps.sosfilt(sos, x)
            worst = max(worst, float(np.max(np.abs(apply(butterworth_sos(3.0, fs, order, kind), x) - ref))))
        assert worst < 1e-9, worst
        print(f"  ✔ scipy 比對：Biquad vs lfilter、SOS 3/4/5 階 vs butter+sosfilt，"
              f"最大誤差 {worst:.1e} < 1e-9；2 階係數與 scipy.butter 相同")

    print("\n全部通過 ✅")


if __name__ == "__main__":
    _self_test()
