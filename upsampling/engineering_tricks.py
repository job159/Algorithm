# -*- coding: utf-8 -*-
"""
訊號處理 / 嵌入式常用工程小技巧（numpy / 純 python）
=========================================
這是什麼：
  寫量測、控制、感測器程式時，「演算法本身」之外天天會碰到的小零件。
  每個都很短，但每個都有一個「不這樣做就會出事」的理由，寫在 docstring 的設計思路裡。

收錄：
  1. RingBuffer      固定容量環形緩衝；★ 鏡像雙寫，取「最近 N 筆」永遠是連續切片、不搬資料
  2. RunningStats    Welford 線上平均 / 變異數；為什麼不能用 Σx² − (Σx)²/n
  3. SchmittTrigger  遲滯比較器：雜訊在門檻附近不會一直跳
  4. Debouncer       按鍵 / 開關去彈跳：新狀態要連續穩定 N 個樣本才承認
  5. deadband        死區：小於門檻的抖動當作 0（兩種版本：硬切 / 連續）
  6. LinearLUT       查表 + 線性插值（NTC 熱敏電阻 ADC → 溫度）；超出範圍 clamp 並回報
  7. Q15 定點數       轉換、乘法、加法（四捨五入 + 飽和），嵌入式 DSP 常用

需求：pip install numpy
執行：python engineering_tricks.py   （每個技巧都有示範輸出，並用 assert 把關）
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray, list]


# =========================================================
#  1. RingBuffer
# =========================================================
class RingBuffer:
    """
    一句話：固定容量的環形緩衝；寫滿後新資料覆蓋最舊的，取「最近 N 筆」不搬資料。

    設計思路：
      STEP 1  一般 list.append + pop(0) 每次都要整串往前搬 → O(N)；環形緩衝只移動「寫入指標」→ O(1)。
      STEP 2  問題：最近 N 筆可能「繞過陣列尾端」分成兩段，取出時得拼接（= 複製）。
      STEP 3  ★ 鏡像雙寫：配置 2×capacity 的陣列，每筆資料同時寫在 i 與 i+capacity 兩個位置。
              這樣不管指標在哪，「最近 N 筆」一定是一段連續的切片 → 直接回傳 numpy view，零複製。
      STEP 4  回傳的 view 設成唯讀，避免呼叫端不小心改到緩衝內容。
    何時用：滑動平均、最近 1 秒的資料畫圖、觸發前的 pre-trigger 緩衝、FIR 濾波的延遲線。
    代價：記憶體 2 倍、每筆寫 2 次（換來讀取永遠零複製，讀多寫少時很划算）。
          回傳的是 view：之後再寫入，view 看到的內容會變；要保存請自己 .copy()。
    """

    def __init__(self, capacity: int, dtype=float):
        if capacity < 1:
            raise ValueError("capacity 必須 ≥ 1")
        self.capacity = int(capacity)
        self._buf = np.zeros(2 * self.capacity, dtype=dtype)
        self._head = 0            # 下一筆要寫入的位置（0 … capacity−1）
        self._count = 0           # 目前有幾筆有效資料

    def append(self, value) -> None:
        self._buf[self._head] = value
        self._buf[self._head + self.capacity] = value      # 鏡像那一份
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def extend(self, values) -> None:
        for v in np.asarray(values).ravel():
            self.append(v)

    def latest(self, n: Optional[int] = None) -> np.ndarray:
        """最近 n 筆（由舊到新），唯讀 view，不複製。n 省略 = 全部有效資料。"""
        n = self._count if n is None else min(int(n), self._count)
        end = self._head + self.capacity          # 鏡像區裡「最新一筆的下一格」
        view = self._buf[end - n:end]
        view.flags.writeable = False
        return view

    def __len__(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self.capacity

    def mean(self) -> float:
        return float(self.latest().mean()) if self._count else float("nan")


# =========================================================
#  2. RunningStats（Welford）
# =========================================================
class RunningStats:
    """
    一句話：一筆一筆餵資料，隨時拿到平均與變異數；不存原始資料、數值穩定。

    設計思路：
      STEP 1  直覺公式 var = (Σx² − (Σx)²/n)/(n−1) 只要累加兩個和，看似完美……
      STEP 2  ★ 坑：資料有大偏移量時（例如 1e9 ± 幾），Σx² 和 (Σx)²/n 都是 ~1e19 的巨大數字，
              它們的差（真正的變異）卻只有幾十。double 只有約 16 位有效數字，
              兩個幾乎相等的大數相減 → 有效位數全部被抵消（catastrophic cancellation），
              結果可能差很多，甚至算出「負的變異數」。
      STEP 3  Welford：只追蹤「目前平均 mean」與「離均差平方和 M2」，每來一筆：
                delta  = x − mean
                mean  += delta / n
                M2    += delta · (x − mean)     ← 用的是「新」平均
              全程只處理「離平均的差」這種小數字，不會有大數相減。
      STEP 4  merge()：兩組各自統計的結果可以合併（Chan 等人的平行公式），適合多執行緒 / 分段計算。
    何時用：嵌入式即時統計、串流資料、資料太大放不進記憶體。
    代價：每筆多一次除法；比兩個和多記一點狀態（n, mean, M2）。
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def update_many(self, xs) -> None:
        for x in np.asarray(xs, dtype=float).ravel():
            self.update(x)

    @property
    def variance(self) -> float:
        """樣本變異數（除以 n−1）。"""
        return self.M2 / (self.n - 1) if self.n > 1 else float("nan")

    @property
    def variance_population(self) -> float:
        """母體變異數（除以 n）。"""
        return self.M2 / self.n if self.n > 0 else float("nan")

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) if self.n > 1 else float("nan")

    def merge(self, other: "RunningStats") -> "RunningStats":
        out = RunningStats()
        out.n = self.n + other.n
        if out.n == 0:
            return out
        delta = other.mean - self.mean
        out.mean = self.mean + delta * other.n / out.n
        out.M2 = self.M2 + other.M2 + delta ** 2 * self.n * other.n / out.n
        out.min = min(self.min, other.min)
        out.max = max(self.max, other.max)
        return out


def naive_variance(xs) -> float:
    """反面教材：Σx² − (Σx)²/n（大偏移量時會出事）。逐筆累加，模擬嵌入式的寫法。"""
    s = s2 = 0.0
    n = 0
    for x in xs:
        x = float(x)
        s += x
        s2 += x * x
        n += 1
    return (s2 - s * s / n) / (n - 1)


# =========================================================
#  3. SchmittTrigger
# =========================================================
class SchmittTrigger:
    """
    一句話：兩個門檻的比較器 —— 往上要超過 high 才變 ON，往下要低於 low 才變 OFF。

    設計思路：
      STEP 1  單一門檻比較器：x > th 就 ON。訊號在 th 附近帶雜訊時，會在 ON/OFF 之間瘋狂來回跳。
      STEP 2  ★ 遲滯（hysteresis）：把門檻拆成 high 與 low（high > low），中間那段叫「遲滯帶」。
              在遲滯帶裡「維持原狀態」—— 雜訊幅度只要小於 high − low，就不會誤觸發。
      STEP 3  狀態 = 記憶：輸出取決於「現在的值 + 之前的狀態」。
    何時用：數位輸入去雜訊、溫控器（開到 25.5°C 關、降到 24.5°C 再開）、過零偵測、電池低電量警告。
    代價：反應延遲 —— 真的越過中間門檻時，要再多走 (high−low)/2 才會切換。
    """

    def __init__(self, low: float, high: float, initial: bool = False):
        if not low < high:
            raise ValueError("必須 low < high")
        self.low, self.high = float(low), float(high)
        self.state = bool(initial)

    def update(self, x: float) -> bool:
        if self.state and x < self.low:
            self.state = False
        elif not self.state and x > self.high:
            self.state = True
        return self.state

    def process(self, xs) -> np.ndarray:
        return np.array([self.update(x) for x in np.asarray(xs, dtype=float).ravel()])


# =========================================================
#  4. Debouncer
# =========================================================
class Debouncer:
    """
    一句話：機械開關按下時會「彈跳」好幾次；新狀態要連續穩定 n_stable 個樣本才承認。

    設計思路：
      STEP 1  機械接點閉合瞬間會反覆碰撞，幾毫秒內讀到 0101100111… 一次按鍵被算成好幾次。
      STEP 2  記住「目前承認的狀態」與「跟它不同的值已經連續出現幾次（counter）」。
      STEP 3  讀到跟承認狀態相同 → counter 歸零；不同 → counter+1；counter 達 n_stable → 切換。
      STEP 4  n_stable × 取樣週期 = 去彈跳時間，通常 5~20 ms。
    何時用：按鍵、極限開關、繼電器回授、任何「機械接點」的數位輸入。
    代價：反應延遲 n_stable 個樣本；比遲滯更適合 0/1 訊號（Schmitt 適合類比值）。
    """

    def __init__(self, n_stable: int = 3, initial: bool = False):
        if n_stable < 1:
            raise ValueError("n_stable 必須 ≥ 1")
        self.n_stable = int(n_stable)
        self.state = bool(initial)
        self.counter = 0

    def update(self, raw) -> bool:
        if bool(raw) == self.state:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.n_stable:
                self.state = bool(raw)
                self.counter = 0
        return self.state

    def process(self, raws) -> np.ndarray:
        return np.array([self.update(r) for r in raws])


# =========================================================
#  5. deadband
# =========================================================
def deadband(x: ArrayLike, width: float, continuous: bool = True):
    """
    一句話：|x| ≤ width（半寬）就當作 0；搖桿飄移、控制器微小誤差不要一直修正。

    設計思路：
      STEP 1  感測器零點總有一點飄（搖桿放開讀到 0.03），直接拿去控制馬達會一直微微抖動。
      STEP 2  硬切版（continuous=False）：|x| ≤ w → 0，否則原值。缺點：在 ±w 處輸出從 0 突然跳到 w。
      STEP 3  ★ 連續版（continuous=True，預設）：sign(x)·(|x| − w)。死區外整條往內縮 w，
              輸出從 0 平滑長出來，沒有跳躍 → 控制迴路比較不會在邊界來回抖。
    何時用：搖桿、PID 誤差小於容忍度時不動作、避免致動器為了雜訊一直出力。
    代價：死區內的小訊號完全看不到（定位精度變差）；連續版會讓整體增益在小訊號區偏小。
    """
    x_arr = np.asarray(x, dtype=float)
    if continuous:
        y = np.sign(x_arr) * np.maximum(np.abs(x_arr) - width, 0.0) + 0.0   # +0.0 把 −0.0 變成 0.0
    else:
        y = np.where(np.abs(x_arr) <= width, 0.0, x_arr)
    return float(y) if np.ndim(x) == 0 else y


# =========================================================
#  6. LinearLUT
# =========================================================
class LinearLUT:
    """
    一句話：用一張「校正表」+ 線性插值，把感測器的原始讀值換成物理量；超出表格就夾住並回報。

    設計思路：
      STEP 1  很多感測器沒有好用的公式（或公式要算 log / exp，MCU 算不動）→ 事先量一張表。
      STEP 2  查詢時用二分搜尋找到夾住 x 的兩個表格點 xs[i] ≤ x ≤ xs[i+1]（O(log n)）。
      STEP 3  兩點之間拉直線：y = ys[i] + (x − xs[i]) · (ys[i+1] − ys[i]) / (xs[i+1] − xs[i])。
      STEP 4  ★ 超出範圍不外插：clamp 到表格端點，並回傳 out_of_range 旗標。
              因為超出範圍通常代表「感測器斷線或短路」，外插只會算出離譜但看起來合理的數字。
      STEP 5  表格 x 若是遞減（NTC：越熱 ADC 越小），建構時自動翻轉成遞增。
    何時用：熱敏電阻、熱電偶、非線性壓力計、伽瑪校正、馬達效率表。
    代價：曲線彎得厲害的區段誤差較大 → 在彎曲處把表格點放密一點即可（不必等間距）。
    """

    def __init__(self, xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.shape != ys.shape or xs.ndim != 1 or len(xs) < 2:
            raise ValueError("xs、ys 必須是等長的一維陣列，且至少 2 點")
        d = np.diff(xs)
        if np.all(d < 0):
            xs, ys = xs[::-1], ys[::-1]
        elif not np.all(d > 0):
            raise ValueError("xs 必須嚴格單調（全遞增或全遞減）")
        self.xs, self.ys = xs, ys

    def __call__(self, x: ArrayLike) -> Tuple[Union[float, np.ndarray], Union[bool, np.ndarray]]:
        xq = np.asarray(x, dtype=float)
        out = (xq < self.xs[0]) | (xq > self.xs[-1])
        xc = np.clip(xq, self.xs[0], self.xs[-1])
        i = np.clip(np.searchsorted(self.xs, xc, side="right") - 1, 0, len(self.xs) - 2)
        x0, x1 = self.xs[i], self.xs[i + 1]
        y0, y1 = self.ys[i], self.ys[i + 1]
        y = y0 + (xc - x0) * (y1 - y0) / (x1 - x0)
        if np.ndim(x) == 0:
            return float(y), bool(out)
        return y, out


# ---------- NTC 範例用的物理模型（只用來產生表格與標準答案） ----------
NTC_R0, NTC_T0, NTC_B = 10_000.0, 298.15, 3950.0     # 10 kΩ @25°C、B = 3950
R_PULLUP, ADC_MAX = 10_000.0, 4095                  # 上拉 10 kΩ、12-bit ADC


def ntc_adc_from_temp(t_c):
    """溫度 → ADC 碼（NTC 接在下臂：越熱電阻越小、ADC 越小）。"""
    t_k = np.asarray(t_c, dtype=float) + 273.15
    r = NTC_R0 * np.exp(NTC_B * (1.0 / t_k - 1.0 / NTC_T0))
    return ADC_MAX * r / (R_PULLUP + r)


def ntc_temp_from_adc(adc):
    """ADC 碼 → 溫度（Beta 公式的精確反解，當標準答案用；MCU 上通常算不起這個 log）。"""
    adc = np.asarray(adc, dtype=float)
    r = R_PULLUP * adc / (ADC_MAX - adc)
    return 1.0 / (1.0 / NTC_T0 + np.log(r / NTC_R0) / NTC_B) - 273.15


# =========================================================
#  7. Q15 定點數
# =========================================================
Q15_ONE = 1 << 15          # 32768 代表 1.0（但 1.0 本身表示不了，最大是 32767/32768）
Q15_MAX, Q15_MIN = 32767, -32768


def to_q15(x: ArrayLike):
    """
    一句話：浮點 [-1, 1) → Q15 整數（int16 範圍），四捨五入 + 飽和。

    設計思路：
      STEP 1  Q15 = 1 個符號位元 + 15 個小數位元，數值 = 整數 / 32768，範圍 [−1, 1 − 2⁻¹⁵]。
      STEP 2  乘 32768 後「四捨五入」（floor(v + 0.5)）→ 誤差最多半個 LSB；直接截斷會一律偏小。
      STEP 3  ★ 飽和：超出範圍就夾在 32767 / −32768。不飽和的話 1.0 會變成 −32768（= −1.0），
              正負顛倒，在控制迴路裡就是災難。
    何時用：把濾波器係數、增益、ADC 正規化值放進沒有浮點單元的 MCU / DSP。
    代價：解析度固定 2⁻¹⁵ ≈ 3.05e-5，而且只能表示 [−1, 1)；超出範圍的係數要先縮放（例如改用 Q14）。
    """
    q = np.floor(np.asarray(x, dtype=float) * Q15_ONE + 0.5)
    q = np.clip(q, Q15_MIN, Q15_MAX).astype(np.int32)
    return int(q) if np.ndim(x) == 0 else q


def from_q15(q: ArrayLike):
    v = np.asarray(q, dtype=float) / Q15_ONE
    return float(v) if np.ndim(q) == 0 else v


def q15_mul(a: ArrayLike, b: ArrayLike, rounding: bool = True):
    """
    一句話：兩個 Q15 相乘 → Q15（先用 32 位元乘、加半個 LSB 四捨五入、右移 15、再飽和）。

    設計思路：
      STEP 1  Q15 × Q15 = Q30，放在 32 位元裡（int16 × int16 一定放得下 int32）。
      STEP 2  要變回 Q15 就右移 15 位。直接右移 = 往 −∞ 截斷，平均偏差 −0.5 LSB，
              在累加 / 遞迴濾波器裡會一直累積成直流偏移。
      STEP 3  ★ 四捨五入：先加 1<<14（= 0.5 LSB）再右移，偏差平均變 0。
      STEP 4  ★ 唯一會溢位的情況：(−1) × (−1) = +1，Q15 表示不了 → 飽和成 32767。
    注意：Python / numpy 的 >> 對負數是算術右移（往 −∞），C 語言在 C++20 以前是「實作定義」，
          移植時要確認編譯器行為（大部分 DSP / ARM 編譯器也是算術右移）。
    何時用：定點 FIR / IIR、PID、混音 —— 任何「係數 × 訊號」的地方。
    代價：每次乘法多一次加法 + 比較（飽和）；很多 DSP、ARM Cortex-M4 有飽和運算指令（例如 QADD、SSAT）可以代勞。
    """
    p = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    if rounding:
        p = p + (1 << 14)
    r = np.clip(p >> 15, Q15_MIN, Q15_MAX)
    return int(r) if np.ndim(r) == 0 else r.astype(np.int32)


def q15_add(a: ArrayLike, b: ArrayLike):
    """飽和加法：結果夾在 [−32768, 32767]，不會「繞回」變號（DSP 指令 QADD 就是做這件事）。"""
    s = np.clip(np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64), Q15_MIN, Q15_MAX)
    return int(s) if np.ndim(s) == 0 else s.astype(np.int32)


def _wrap16(v: int) -> int:
    """模擬沒有飽和的 int16 溢位（繞回）。"""
    return ((int(v) + 32768) % 65536) - 32768


# =========================================================
#  示範
# =========================================================
def _demo() -> None:
    line = "=" * 70
    print(line)
    print(" engineering_tricks.py 示範")
    print(line)

    # ---------- 1. RingBuffer ----------
    print("\n[1] RingBuffer（容量 5，鏡像雙寫）")
    rb = RingBuffer(5)
    for v in range(1, 9):
        rb.append(v)
    last3 = rb.latest(3)
    print(f"    依序寫入 1~8 → 全部 {rb.latest().tolist()}、最近 3 筆 {last3.tolist()}、平均 {rb.mean():.1f}")
    print(f"    最近 3 筆是不是 view（沒複製）？ {np.shares_memory(last3, rb._buf)}；"
          f"可寫嗎？ {last3.flags.writeable}")
    assert rb.latest().tolist() == [4, 5, 6, 7, 8] and last3.tolist() == [6, 7, 8]
    assert np.shares_memory(last3, rb._buf) and not last3.flags.writeable
    rb2 = RingBuffer(4)
    rb2.extend([1, 2])
    assert rb2.latest().tolist() == [1, 2] and not rb2.is_full

    # ---------- 2. RunningStats ----------
    print("\n[2] RunningStats（Welford）vs 天真公式 Σx² − (Σx)²/n")
    base = np.array([4.0, 7.0, 13.0, 16.0])
    print("    資料 = 偏移量 + [4, 7, 13, 16]，真正的樣本變異數 = 30")
    print(f"    {'偏移量':>8s} | {'天真公式':>14s} | {'Welford':>10s} | {'numpy 兩趟法':>12s}")
    for off in (0.0, 1e6, 1e8, 1e9):
        data = base + off
        rs = RunningStats()
        rs.update_many(data)
        nv = naive_variance(data)
        print(f"    {off:>8.0e} | {nv:>14.4f} | {rs.variance:>10.4f} | {np.var(data, ddof=1):>12.4f}")
        assert abs(rs.variance - 30.0) < 1e-6
    big = np.random.default_rng(0).normal(1e9, 1.0, 10000)
    rs = RunningStats()
    rs.update_many(big)
    nv = naive_variance(big)
    print(f"    1e9 ± N(0,1) 一萬筆：天真 {nv:.4f}、Welford {rs.variance:.4f}、numpy {np.var(big, ddof=1):.4f}")
    assert abs(rs.variance - np.var(big, ddof=1)) < 1e-6
    assert abs(naive_variance(base + 1e9) - 30.0) > 1.0      # 天真公式在這裡確實壞掉
    a, b = RunningStats(), RunningStats()
    a.update_many(big[:3000])
    b.update_many(big[3000:])
    m = a.merge(b)
    print(f"    分兩段統計再 merge：{m.variance:.6f}（整段一次算：{rs.variance:.6f}）")
    # mean ≈ 1e9 時 double 的最小刻度約 1.2e-7，所以平均值用相對誤差比
    assert math.isclose(m.variance, rs.variance, rel_tol=1e-6)
    assert math.isclose(m.mean, rs.mean, rel_tol=1e-14)

    # ---------- 3. SchmittTrigger ----------
    print("\n[3] SchmittTrigger：緩慢正弦 + 雜訊，門檻 0")
    rng = np.random.default_rng(1)
    t = np.linspace(0, 2, 2000)
    sig = np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.08, t.size)
    simple = sig > 0
    st = SchmittTrigger(low=-0.3, high=0.3).process(sig)
    n_simple = int(np.count_nonzero(np.diff(simple.astype(int))))
    n_st = int(np.count_nonzero(np.diff(st.astype(int))))
    print(f"    單一門檻比較器切換 {n_simple} 次；Schmitt（low=−0.3, high=+0.3）切換 {n_st} 次"
          f"（理想 = 2 個週期 × 每週期 2 次 = 4 次）")
    assert n_st <= 4 < n_simple

    # ---------- 4. Debouncer ----------
    print("\n[4] Debouncer（n_stable = 3）")
    raw = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    db = Debouncer(3).process(raw)
    print("    原始： " + "".join(str(v) for v in raw))
    print("    去彈跳：" + "".join(str(int(v)) for v in db))
    rising = int(np.count_nonzero(np.diff(np.array(raw)) == 1))
    print(f"    原始訊號有 {rising} 個上升緣（會被當成 {rising} 次按鍵）；去彈跳後只有 "
          f"{int(np.count_nonzero(np.diff(db.astype(int)) == 1))} 次")
    assert int(np.count_nonzero(np.diff(db.astype(int)) == 1)) == 1

    # ---------- 5. deadband ----------
    print("\n[5] deadband（半寬 0.1）")
    xs = np.array([-1.0, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.12, 0.3, 1.0])
    print("    x        ：" + " ".join(f"{v:+.2f}" for v in xs))
    print("    硬切版    ：" + " ".join(f"{v:+.2f}" for v in deadband(xs, 0.1, continuous=False)))
    print("    連續版    ：" + " ".join(f"{v:+.2f}" for v in deadband(xs, 0.1)))
    print("    → 硬切版在 0.10 → 0.12 時輸出從 0 跳到 0.12；連續版只從 0 長到 0.02")
    assert deadband(0.05, 0.1) == 0.0 and abs(deadband(0.3, 0.1) - 0.2) < 1e-12
    assert deadband(0.12, 0.1, continuous=False) == 0.12

    # ---------- 6. LinearLUT ----------
    print("\n[6] LinearLUT：NTC 10k / B3950 + 10k 上拉、12-bit ADC")
    temps = np.arange(-20, 101, 10)
    adcs = np.round(ntc_adc_from_temp(temps))
    lut = LinearLUT(adcs, temps)
    print("    校正表（每 10°C 一點）：" + "、".join(f"{int(tc)}°C→{int(a)}" for tc, a in zip(temps, adcs)))
    for code in (3000, 2048, 1000, 300):
        tl, oor = lut(code)
        tx = float(ntc_temp_from_adc(code))
        print(f"    ADC {code:>4d} → 查表 {tl:7.2f}°C（精確公式 {tx:7.2f}°C，誤差 {tl - tx:+.2f}°C）")
    sweep = np.arange(int(adcs.min()), int(adcs.max()) + 1)
    err = lut(sweep)[0] - ntc_temp_from_adc(sweep)
    i_worst = int(np.argmax(np.abs(err)))
    print(f"    全範圍逐碼掃描：最大誤差 {err[i_worst]:+.2f}°C（ADC {sweep[i_worst]}，曲線最彎處；"
          f"那段把表格點放密一點即可）")
    for code in (4090, 5):
        tl, oor = lut(code)
        why = "可能是感測器斷線（開路）" if code > 2000 else "可能是感測器短路"
        print(f"    ADC {code:>4d} → 超出表格範圍={oor}，輸出夾在 {tl:.0f}°C 並回報 → {why}")
        assert oor
    y_arr, oor_arr = lut(np.array([5, 2048, 4090]))
    assert oor_arr.tolist() == [True, False, True]
    assert np.allclose(lut(sweep)[0], np.interp(sweep, lut.xs, lut.ys))
    assert abs(err).max() < 2.0

    # ---------- 7. Q15 ----------
    print("\n[7] Q15 定點數（1.0 ≙ 32768，範圍 [−1, 0.99997]）")
    for v in (0.5, -0.25, 0.70710678, 1.0, -1.0, 1.7):
        q = to_q15(v)
        print(f"    to_q15({v:+.8f}) = {q:>6d} → 還原 {from_q15(q):+.8f}"
              + ("   ← 超出範圍，飽和" if abs(v) >= 1 and v != -1.0 else ""))
    a_, b_ = to_q15(0.5), to_q15(0.5)
    print(f"    0.5 × 0.5 = {from_q15(q15_mul(a_, b_)):.6f}")
    a_, b_ = to_q15(0.9), to_q15(-0.7)
    print(f"    0.9 × (−0.7) = {from_q15(q15_mul(a_, b_)):+.6f}（浮點 −0.63）")
    m1 = q15_mul(Q15_MIN, Q15_MIN)
    print(f"    (−1) × (−1) = {m1}（= {from_q15(m1):.6f}，飽和；不飽和右移後是 32768，int16 放不下會變 −1）")
    s_sat = q15_add(to_q15(0.75), to_q15(0.5))
    s_wrap = _wrap16(to_q15(0.75) + to_q15(0.5))
    print(f"    0.75 + 0.5：飽和加法 = {from_q15(s_sat):+.5f}；沒飽和（int16 繞回）= {from_q15(s_wrap):+.5f}  ← 正數加正數變負數！")
    rng = np.random.default_rng(2)
    qa = rng.integers(Q15_MIN, Q15_MAX + 1, 200_000)
    qb = rng.integers(Q15_MIN, Q15_MAX + 1, 200_000)
    exact = qa.astype(float) * qb / Q15_ONE                  # 以 LSB 為單位的精確值
    e_round = q15_mul(qa, qb, rounding=True) - exact
    e_trunc = q15_mul(qa, qb, rounding=False) - exact
    print(f"    20 萬組隨機乘法的平均誤差（單位 LSB）：四捨五入 {e_round.mean():+.4f}、"
          f"直接截斷 {e_trunc.mean():+.4f}")
    print("    → 截斷有 −0.5 LSB 的系統性偏差，在 IIR 濾波器 / 積分器裡會累積成直流偏移")
    assert to_q15(1.0) == Q15_MAX and to_q15(-1.0) == Q15_MIN and to_q15(1.7) == Q15_MAX
    assert q15_mul(to_q15(0.5), to_q15(0.5)) == to_q15(0.25)
    assert m1 == Q15_MAX and s_sat == Q15_MAX and s_wrap < 0
    assert abs(e_round.mean()) < 0.01 and abs(e_trunc.mean() + 0.5) < 0.01
    assert np.abs(e_round).max() <= 0.5 + 1e-9

    print("\n" + line)
    print(" 全部示範與檢查通過 ✔")
    print(line)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _demo()
