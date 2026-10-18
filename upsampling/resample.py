# -*- coding: utf-8 -*-
"""
上採樣 / 降採樣 / 重取樣 函式庫（只依賴 numpy）
=========================================
這是什麼：
  把「取樣率換來換去」時最常用的工程手法收在一起，每個函式都附上
  一句話、設計思路（STEP 1、STEP 2…）、何時用、代價。
  動畫版請看 upsampling_pyqt6_step.py，道理請看「上採樣與工程技巧_設計思路.md」。

核心觀念（五句話）：
  1. 取樣定理：訊號最高頻率 < fs/2（Nyquist）時，取樣點就「包含了全部資訊」，
     中間的值可以算回來 —— 上採樣其實是在「猜回」本來就存在的點。
  2. 上採樣 = 插零（頻譜出現鏡像）+ 低通（把鏡像砍掉）+ ×L（把被稀釋的能量補回來）。
     線性 / 三次插值是「便宜但不完美的低通」；加窗 sinc FIR 是「認真的低通」。
  3. 降採樣 = 先低通（抗混疊）再每 M 取一；沒先低通，高頻會「折」成假的低頻（混疊）。
  4. 有理數倍 L/M = 先上 L 再降 M；用 polyphase 只算最後要留下的點，不浪費。
  5. 不等間距（抖動、掉包）先找左右鄰居再插值；空洞太大就填 NaN，絕不亂補。
  另外：2D 影像縮放一律用「像素中心對齊」 src = (dst + 0.5)/scale − 0.5。

時間 / 長度約定（全檔一致）：
  * 1D 輸入 x[n] 代表時間 n（以「原始取樣間隔」為單位）。
  * 整數倍上採樣輸出長度 = N·L，第 m 點在時間 m/L（跟 scipy.signal.resample_poly 相同）。
    最後 L−1 點落在 x[N−1] 之後，插值法用「邊界延伸（clamp）」處理。
  * FIR 類方法都已補償群延遲：輸出與輸入在時間上對齊（不會整段往右偏）。

用法：
  from resample import upsample_fir, decimate, resample_rational, align_streams
需求：pip install numpy        （scipy 只在自我測試時拿來比對，沒有也能跑）
執行：python resample.py        （跑自我測試，全部用 assert 把關）
"""
from __future__ import annotations

import math
import sys
from typing import Dict, Optional, Tuple, Union

import numpy as np

WindowSpec = Union[str, Tuple[str, float]]
DEFAULT_WINDOW: WindowSpec = ("kaiser", 5.0)     # 跟 scipy.signal.resample_poly 預設相同


# =========================================================
#  共用小工具
# =========================================================
def _as_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("只接受一維資料（1-D array）")
    return x


def _check_factor(k: int, name: str) -> int:
    if int(k) != k or k < 1:
        raise ValueError(f"{name} 必須是 ≥ 1 的整數，收到 {k!r}")
    return int(k)


def cubic_kernel(s, a: float = -0.5) -> np.ndarray:
    """
    三次卷積（cubic convolution, Keys 1981）的核心函數 W(s)。

    設計思路：
      STEP 1  想要一個「分段三次多項式」當插值權重：只看左右各 2 個點（共 4 點）。
      STEP 2  要求：W(0)=1、W(±1)=W(±2)=0 → 曲線一定穿過原始點（是插值，不是平滑）。
      STEP 3  再要求一階導數連續 → 剩一個自由參數 a。
      STEP 4  a = −0.5 時能把二次多項式完整重建（誤差 O(h³)），這就是 Catmull-Rom。
              Photoshop、OpenCV INTER_CUBIC 用 a = −0.75，銳利一點但過衝更多。
    何時用：upsample_cubic、resize_bicubic 的權重來源。
    代價：|s| 在 1~2 之間是負值 → 會過衝。
    """
    s = np.abs(np.asarray(s, dtype=float))
    w = np.zeros_like(s)
    m1 = s <= 1
    m2 = (s > 1) & (s < 2)
    s1, s2 = s[m1], s[m2]
    w[m1] = (a + 2) * s1 ** 3 - (a + 3) * s1 ** 2 + 1
    w[m2] = a * s2 ** 3 - 5 * a * s2 ** 2 + 8 * a * s2 - 4 * a
    return w


def interp_taps(pos, n: int, method: str = "linear", a: float = -0.5):
    """
    一句話：給「浮點位置 pos」，回傳要用哪幾個原始點、各自的權重。

    設計思路：
      STEP 1  i0 = floor(pos)，frac = pos − i0：pos 落在 x[i0] 與 x[i0+1] 之間 frac 處。
      STEP 2  依方法決定要拿幾個鄰居：
                'zoh'     → 1 點（左邊那點，零階保持）
                'nearest' → 1 點（最近的那點，floor(pos+0.5)）
                'linear'  → 2 點，權重 (1−frac, frac)
                'cubic'   → 4 點 x[i0−1..i0+2]，權重 = cubic_kernel(距離)
      STEP 3  索引超出 [0, n−1] 就 clamp 到邊界（= 把邊界值往外延伸）。
      所有插值型上採樣、2D 縮放都呼叫這一個函式 → GUI 顯示的權重跟真正計算的完全一致。
    何時用：任何「等間距樣本 + 浮點位置」的插值；也很適合拿來印出權重做教學 / 除錯。
    代價：回傳 (點數 × 鄰居數) 的索引與權重陣列，記憶體 = 輸出長度 × K。

    回傳：(raw_idx, idx, w)，形狀都是 (len(pos), 每點用到的鄰居數)
      raw_idx：未 clamp 的原始索引（教學顯示用，可能是 −1 或 n）
      idx    ：clamp 後真正拿來取值的索引
      w      ：權重，每一列總和 = 1
    """
    pos = np.atleast_1d(np.asarray(pos, dtype=float))
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    if method == "zoh":
        raw = i0[:, None]
        w = np.ones((len(pos), 1))
    elif method == "nearest":
        raw = np.floor(pos + 0.5).astype(np.int64)[:, None]
        w = np.ones((len(pos), 1))
    elif method == "linear":
        raw = np.stack([i0, i0 + 1], axis=1)
        w = np.stack([1.0 - frac, frac], axis=1)
    elif method == "cubic":
        offs = np.arange(-1, 3)
        raw = i0[:, None] + offs[None, :]
        w = cubic_kernel(pos[:, None] - raw, a)
    else:
        raise ValueError(f"未知的插值方法：{method!r}")
    idx = np.clip(raw, 0, n - 1)
    return raw, idx, w


def _interp_uniform(x: np.ndarray, pos: np.ndarray, method: str, a: float = -0.5) -> np.ndarray:
    _, idx, w = interp_taps(pos, len(x), method, a)
    return (x[idx] * w).sum(axis=1)


# =========================================================
#  一、整數倍上採樣（L 倍）
# =========================================================
def upsample_zoh(x, L: int) -> np.ndarray:
    """
    一句話：零階保持（Zero-Order Hold）—— 每個樣本原地重複 L 次，變成階梯。

    設計思路：
      STEP 1  新時間軸 t = m/L（m = 0…N·L−1）。
      STEP 2  每個新點直接抄「左邊最近的原始點」：y[m] = x[floor(m/L)]。
    何時用：DAC 輸出本來就是這樣、控制迴路的「保持上一個命令」、只想快速對齊格點。
    代價：頻譜上是乘一個 sinc，鏡像只被壓一點點；而且整體延遲半個原始取樣間隔
          （階梯的「平均位置」偏右），所以拿來當插值誤差最大。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    return np.repeat(x, L)


def upsample_linear(x, L: int) -> np.ndarray:
    """
    一句話：線性插值 —— 相鄰兩點拉一條直線，新點落在線上。

    設計思路：
      STEP 1  t = m/L，找 i0 = floor(t)、frac = t − i0。
      STEP 2  y = (1−frac)·x[i0] + frac·x[i0+1]（離誰近、誰的權重就大）。
    何時用：感測器資料對齊、動畫補間、對品質要求不高又要極便宜的場合。
    代價：等效於「三角形核心」低通，高頻會被削弱（變鈍）、鏡像抑制普通；
          轉折處有尖角（一階導數不連續）。每點 2 次乘法。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    return _interp_uniform(x, np.arange(len(x) * L) / L, "linear")


def upsample_cubic(x, L: int, a: float = -0.5) -> np.ndarray:
    """
    一句話：三次插值（Catmull-Rom / cubic convolution, a = −0.5）—— 看 4 個點畫平滑曲線。

    設計思路：
      STEP 1  t = m/L，i0 = floor(t)。
      STEP 2  取 x[i0−1], x[i0], x[i0+1], x[i0+2] 四點，到 t 的距離分別是 1+f, f, 1−f, 2−f。
      STEP 3  權重 = cubic_kernel(距離)，四個權重加起來 = 1；最外側兩點的權重是「負的」，
              這讓曲線能「預測轉彎」，比線性更貼近真實的帶限訊號。
    何時用：影像縮放、軌跡平滑補點、需要一階導數連續的場合。
    代價：負權重 → 在階躍邊緣會過衝（overshoot / ringing）；每點 4 次乘法。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    return _interp_uniform(x, np.arange(len(x) * L) / L, "cubic", a)


def upsample_zero_stuff(x, L: int) -> np.ndarray:
    """
    一句話：只插零（zero-stuffing）—— 每兩個樣本之間塞 L−1 個 0。教學用，不能直接拿來用！

    設計思路：
      STEP 1  y[m·L] = x[m]，其餘位置 = 0。
      STEP 2  看頻譜：原本 0~fs/2 的頻譜「原封不動」，但在新的取樣率下，
              fs、2fs…附近多出 L−1 份鏡像（imaging）。
      → 所以插零之後一定要接一個截止在「舊 Nyquist」的低通濾波器，見 upsample_fir。
    何時用：教學、驗證（upsample_fir_direct 用它對答案）；實務上不會單獨使用。
    代價：直接用的話，波形會變成一根根尖刺；平均能量也只剩 1/L。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    y = np.zeros(len(x) * L)
    y[::L] = x
    return y


def _make_window(window: WindowSpec, numtaps: int) -> np.ndarray:
    if isinstance(window, (tuple, list)):
        name, param = window[0], float(window[1])
    else:
        name, param = window, None
    name = str(name).lower()
    if name in ("rect", "boxcar", "rectangular", "none"):
        return np.ones(numtaps)
    if name in ("hann", "hanning"):
        return np.hanning(numtaps)
    if name == "hamming":
        return np.hamming(numtaps)
    if name == "blackman":
        return np.blackman(numtaps)
    if name == "kaiser":
        return np.kaiser(numtaps, 5.0 if param is None else param)
    raise ValueError(f"不支援的窗函數：{window!r}")


def design_lowpass_fir(numtaps: int, cutoff: float, window: WindowSpec = DEFAULT_WINDOW) -> np.ndarray:
    """
    一句話：加窗 sinc 法設計線性相位低通 FIR（跟 scipy.signal.firwin 結果相同）。

    參數：cutoff 以 Nyquist 為 1 正規化（0 < cutoff ≤ 1），跟 firwin 一樣。
          例如上採樣 L 倍：新取樣率下「舊 Nyquist」= 1/L → cutoff = 1/L。

    設計思路：
      STEP 1  理想低通的脈衝響應是 sinc：h[n] = cutoff·sinc(cutoff·n)，但它無限長。
      STEP 2  截成 numtaps 點、並以中心 (numtaps−1)/2 為對稱軸 → 線性相位（所有頻率延遲一樣）。
      STEP 3  直接截斷會在頻譜上產生漣波（Gibbs），所以乘一個窗函數（Kaiser/Hamming…）讓兩端平滑歸零。
      STEP 4  正規化成 Σh = 1 → 直流增益剛好 1。
    何時用：任何需要「乾淨低通」的重取樣；numtaps 越多，過渡帶越窄、阻帶越深。
    代價：群延遲 (numtaps−1)/2 個樣本（本檔所有用到它的函式都已自動補償）。
    """
    numtaps = int(numtaps)
    if numtaps < 1:
        raise ValueError("numtaps 必須 ≥ 1")
    if not (0 < cutoff <= 1):
        raise ValueError("cutoff 以 Nyquist=1 正規化，必須在 (0, 1]")
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n)          # np.sinc(x) = sin(πx)/(πx)
    h = h * _make_window(window, numtaps)
    return h / h.sum()


def _fir_for(numtaps: Optional[int], R: int, window: WindowSpec, h) -> np.ndarray:
    """沒給 h 就自己設計：截止 1/R，長度預設 2·10·R+1（跟 resample_poly 相同的經驗值）。"""
    if h is None:
        if numtaps is None:
            numtaps = 2 * 10 * R + 1
        h = design_lowpass_fir(numtaps, 1.0 / R, window)
    h = np.asarray(h, dtype=float)
    if len(h) % 2 == 0:
        raise ValueError("FIR 長度請用奇數：群延遲 (numtaps−1)/2 才會是整數、輸出才能精確對齊")
    return h


def upsample_fir(x, L: int, numtaps: Optional[int] = None, window: WindowSpec = DEFAULT_WINDOW,
                 h=None, pad_mode: str = "edge") -> np.ndarray:
    """
    一句話：正統上採樣 = 插零 + 低通 FIR + ×L 補增益，用 polyphase（多相）實作。

    設計思路：
      STEP 1  概念上先插零：u[k·L] = x[k]、其他 = 0（頻譜出現 L−1 份鏡像）。
      STEP 2  用截止在舊 Nyquist（cutoff = 1/L）的低通 h 濾掉鏡像：v = h * u。
      STEP 3  插零讓平均能量只剩 1/L，所以輸出 ×L 補回增益（直流增益變回 1）。
      STEP 4  群延遲補償：h 長度奇數、延遲 D = (numtaps−1)/2，取 y[n] = v[n + D]，
              輸出第 m 點就精確落在時間 m/L，跟輸入對齊。
      STEP 5  ★ polyphase：u 裡 L 個點有 L−1 個是 0，乘 0 是白做工。
              把 h 拆成 L 個子濾波器 e_p[r] = h[r·L + b_p]，第 p 個「相位」的輸出
              只需要拿子濾波器直接對原始 x 卷積，再把 L 路結果交錯排好。
              跟「插零後再濾波」數學上完全等價（見 upsample_fir_direct 與自我測試），
              但每個輸出點只做 numtaps/L 次乘法 → 省 L 倍計算。
    何時用：音訊、通訊、量測 —— 需要「頻帶內幾乎無失真」的上採樣。
    代價：要選 numtaps / window；兩端有暫態（用 pad_mode 決定邊界外怎麼補，
          'edge' 延伸邊界值、'constant' 補 0 跟 scipy 相同、'wrap' 適合週期訊號）。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    if L == 1:
        return x.copy()
    h = _fir_for(numtaps, L, window, h)
    D = (len(h) - 1) // 2
    N = len(x)
    P = len(h) // L + 2                       # 兩側多補幾個原始樣本，夠子濾波器取用
    xp = np.pad(x, P, mode=pad_mode)
    y = np.empty(N * L)
    for p in range(L):                        # 第 p 個相位：輸出 y[m·L + p]
        a, b = divmod(p + D, L)               # y[mL+p] = Σ_r h[rL+b]·x[m+a−r]
        e = h[b::L]                           # 第 p 個子濾波器（polyphase component）
        c = np.convolve(xp, e)                # 直接對「原始」x 卷積，沒有任何乘 0
        y[p::L] = c[P + a: P + a + N]
    return y * L


def upsample_fir_direct(x, L: int, h, pad_mode: str = "edge") -> np.ndarray:
    """
    一句話：教科書寫法 —— 真的先插零、再整條拿 h 去卷積、再 ×L。只為了跟 polyphase 對答案。

    設計思路：STEP 1 插零 → STEP 2 np.convolve(u, h) → STEP 3 扣掉群延遲 D → STEP 4 ×L。
    何時用：驗證 polyphase 實作是否正確（自我測試裡兩者最大差 ~1e-16）。
    代價：每個輸出點做 numtaps 次乘法，其中 (L−1)/L 是在乘 0，比 polyphase 慢 L 倍。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    h = _fir_for(None, L, DEFAULT_WINDOW, h)
    D = (len(h) - 1) // 2
    N = len(x)
    P = len(h) // L + 2
    u = upsample_zero_stuff(np.pad(x, P, mode=pad_mode), L)
    v = np.convolve(u, h)
    return v[P * L + D: P * L + D + N * L] * L


def upsample_fft(x, L: int) -> np.ndarray:
    """
    一句話：頻域補零 —— FFT 後在高頻端塞零、再用更長的 IFFT 轉回來。

    設計思路：
      STEP 1  X = rfft(x)：拿到 0 ~ fs/2 的頻譜。
      STEP 2  新長度 N·L 的頻譜，前面放 X、後面全補 0（「新增的高頻」一律當作沒有能量）。
      STEP 3  N 為偶數時，原本的 Nyquist 那一格要對半分（它其實代表 ±fs/2 兩邊各一半）。
      STEP 4  irfft 回時域，×L 補回 IFFT 的 1/N 正規化差異。
    何時用：訊號在這一段長度內是「週期的」（例如整數週期的測試音、轉一圈的角度資料），
            這時它就是理想 sinc 插值，誤差接近機器精度。
    代價：把整段當成會循環 → 非週期訊號頭尾會互相干擾（環繞假影 / ringing）；
          要整段資料才能算，不適合即時串流。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    N = len(x)
    X = np.fft.rfft(x)
    Y = np.zeros(N * L // 2 + 1, dtype=complex)
    Y[:len(X)] = X
    if N % 2 == 0 and L > 1:
        Y[N // 2] *= 0.5
    return np.fft.irfft(Y, N * L) * L


# =========================================================
#  二、整數倍降採樣（M 倍）
# =========================================================
def lowpass_aligned(x, h, pad_mode: str = "edge") -> np.ndarray:
    """
    一句話：用 FIR 濾波、並把群延遲扣掉，輸出跟輸入一樣長、時間對齊（離線版「零延遲」）。

    設計思路：
      STEP 1  兩端用 pad_mode 補樣本，避免邊界突然掉到 0。
      STEP 2  full 卷積 c = h * x。
      STEP 3  線性相位 FIR 的延遲剛好是 D = (numtaps−1)/2，取 c[n + D] 就對齊了。
    何時用：離線資料分析、降採樣前的抗混疊、想比較濾波前後波形（不能有時間偏移）。
    代價：需要「未來」D 個樣本 → 即時系統做不到，只能接受延遲。
    """
    x = _as_1d(x)
    h = np.asarray(h, dtype=float)
    if len(h) % 2 == 0:
        raise ValueError("FIR 長度請用奇數")
    D = (len(h) - 1) // 2
    P = len(h)
    c = np.convolve(np.pad(x, P, mode=pad_mode), h)
    return c[P + D: P + D + len(x)]


def decimate_naive(x, M: int) -> np.ndarray:
    """
    一句話：每 M 個取 1 個（直接丟樣本）。會混疊！

    設計思路：STEP 1 y[k] = x[k·M]，就這樣。
    何時用：確定訊號早就沒有 fs/(2M) 以上的成分（例如已經被類比濾波過），才能這樣做。
    代價：任何高於新 Nyquist 的成分會「折回」成假頻 f_alias = |f − k·fs_new|，
          而且一旦折進來就跟真訊號混在一起，事後再也分不開。
    """
    x = _as_1d(x)
    M = _check_factor(M, "M")
    return x[::M].copy()


def decimate(x, M: int, numtaps: Optional[int] = None, window: WindowSpec = DEFAULT_WINDOW,
             h=None, pad_mode: str = "edge") -> np.ndarray:
    """
    一句話：正統降採樣 = 先抗混疊低通（截止在新 Nyquist）再每 M 取 1。

    設計思路：
      STEP 1  新取樣率 fs/M，新 Nyquist = fs/(2M) → 低通 cutoff = 1/M（以舊 Nyquist=1）。
      STEP 2  先濾波：把會折回來的高頻先砍掉（lowpass_aligned，延遲已補償）。
      STEP 3  再每 M 取一。
      ★ 省計算的小技巧：反正 M 個只留 1 個，濾波時只要算「會被留下」的那些點即可
        （resample_rational(x, 1, M) 就是這樣做的，省 M 倍）。這裡為了教學清楚，先全算再取。
    何時用：任何「取樣率太高、想變小」的情況（存檔、傳輸、後段演算法跑不動）。
    代價：高於新 Nyquist 的真實資訊永久消失（這是物理限制，不是演算法的錯）。
    """
    x = _as_1d(x)
    M = _check_factor(M, "M")
    if M == 1:
        return x.copy()
    h = _fir_for(numtaps, M, window, h)
    return lowpass_aligned(x, h, pad_mode)[::M]


# =========================================================
#  三、有理數倍 L/M（例如 44.1 kHz → 48 kHz = 160/147）
# =========================================================
def resample_rational(x, L: int, M: int, numtaps: Optional[int] = None,
                      window: WindowSpec = DEFAULT_WINDOW, h=None,
                      pad_mode: str = "edge") -> np.ndarray:
    """
    一句話：取樣率乘 L/M —— 概念上「上 L → 低通 → 降 M」，實作上只算要留下的點。

    設計思路：
      STEP 1  先用 gcd 約分（例如 48000/44100 → 160/147；320/294 也會變 160/147）。
      STEP 2  一支低通同時負責「去鏡像」與「抗混疊」→ 截止取兩者中較嚴的：cutoff = 1/max(L, M)。
      STEP 3  輸出第 i 點對應高取樣率上的位置 n = i·M（再加群延遲 D）。
      STEP 4  ★ polyphase：位置 n+D 落在第 b = (n+D) mod L 個相位，只要用子濾波器 h[b], h[b+L]…
              對原始 x 做一次內積，就得到這個點 —— 插零的 0 不碰、被丟掉的點也不算。
      STEP 5  ×L 補增益。輸出長度 = ceil(N·L/M)。
    何時用：音訊取樣率轉換、不同設備資料對齊到同一取樣率。
    代價：L、M 很大時（例如 160/147）濾波器很長（預設 2·10·max(L,M)+1 = 3201 點），
          但每個輸出只用約 numtaps/L 點，所以還是很快。
    """
    x = _as_1d(x)
    L = _check_factor(L, "L")
    M = _check_factor(M, "M")
    g = math.gcd(L, M)
    L, M = L // g, M // g
    if L == 1 and M == 1:
        return x.copy()
    h = _fir_for(numtaps, max(L, M), window, h)
    D = (len(h) - 1) // 2
    N = len(x)
    n_out = -(-N * L // M)                    # ceil(N·L/M)
    P = len(h) // L + 2
    xp = np.pad(x, P, mode=pad_mode)
    q = np.arange(n_out, dtype=np.int64) * M + D     # 在「插零後」序列上的位置（含延遲）
    base, b = np.divmod(q, L)                 # 第幾個原始樣本、第幾個相位
    K = -(-len(h) // L)                       # 每個子濾波器最多 K 個係數
    r = np.arange(K)
    hi = b[:, None] + r[None, :] * L          # 子濾波器係數在 h 裡的索引
    coef = np.where(hi < len(h), h[np.minimum(hi, len(h) - 1)], 0.0)
    xi = base[:, None] - r[None, :] + P       # 對應的原始樣本（已加上補邊位移）
    return (coef * xp[xi]).sum(axis=1) * L


# =========================================================
#  四、不等間距 → 等間距（時間戳抖動、掉包）
# =========================================================
def _clean_series(t, x) -> Tuple[np.ndarray, np.ndarray]:
    """去掉 NaN/inf（掉包常被記成 NaN）、依時間排序、重複時間戳只留第一筆。"""
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if t.shape != x.shape:
        raise ValueError("t 與 x 長度不同")
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    t_u, first = np.unique(t, return_index=True)    # unique 會順便排序
    return t_u, x[first]


def bracket(t, t_new):
    """
    一句話：對每個查詢時間，找出夾住它的兩個樣本 t[i0] ≤ t_new ≤ t[i0+1]。

    設計思路：
      STEP 1  t 已排序 → 二分搜尋 searchsorted，O(log n)。
      STEP 2  i0 夾在 [0, n−2]，frac = (t_new − t[i0]) / (t[i0+1] − t[i0])。
      STEP 3  一併回傳 gap = t[i0+1] − t[i0]，讓呼叫端判斷「這段空洞是不是太大」。
    回傳：(i0, frac, gap)，frac 可能 <0 或 >1（代表在資料範圍外）。
    何時用：resample_nonuniform 的核心；GUI 也用它顯示「落在哪兩點之間」。
    代價：t 必須已排序且不重複（_clean_series 會先處理）。
    """
    t = np.asarray(t, dtype=float)
    tq = np.atleast_1d(np.asarray(t_new, dtype=float))
    if len(t) < 2:
        raise ValueError("至少要 2 個樣本才能插值")
    i0 = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2)
    gap = t[i0 + 1] - t[i0]
    frac = (tq - t[i0]) / gap
    return i0, frac, gap


def auto_max_gap(t, factor: float = 2.5) -> float:
    """
    一句話：自動決定 max_gap = factor × 「正常取樣間隔（中位數）」，超過就當作掉包。
    設計思路：STEP 1 算相鄰間隔 → STEP 2 取中位數（不會被少數掉包拉大，平均數會）→ STEP 3 乘上 factor。
    何時用：不知道感測器標稱頻率、或標稱值不可信時。
    代價：如果超過一半的間隔都是掉包，中位數本身就錯了 —— 這時請手動指定 max_gap。
    """
    t = np.sort(np.asarray(t, dtype=float))
    return float(factor * np.median(np.diff(t)))


def resample_nonuniform(t, x, t_new, method: str = "linear",
                        max_gap: Optional[float] = None) -> np.ndarray:
    """
    一句話：把時間戳不整齊（抖動、掉包）的資料，重取樣到指定時間點；空洞太大就給 NaN。

    設計思路：
      STEP 1  清資料：丟掉 NaN、照時間排序、重複時間戳只留一筆。
      STEP 2  每個 t_new 用 bracket() 找左右鄰居與比例 frac。
      STEP 3  依 method 計算：
                'linear'   → (1−frac)·x[i0] + frac·x[i0+1]
                'nearest'  → 比較近的那一點
                'previous' → 最後一個 ≤ t_new 的值（因果、不偷看未來，控制系統常用）
      STEP 4  ★ 不亂插值：t_new 超出資料範圍 → NaN（不外插）；
              左右鄰居相隔 > max_gap → NaN（那段根本沒量到，硬插等於捏造資料）。
              剛好落在某個樣本上則照實回傳。
    何時用：多感測器融合前的前處理、log 檔分析、網路封包時間不準的資料。
    代價：NaN 會往下游傳 —— 下游必須能處理（遮罩、分段、或卡爾曼濾波只做預測不做更新）。
    """
    t, x = _clean_series(t, x)
    tq = np.atleast_1d(np.asarray(t_new, dtype=float))
    y = np.full(tq.shape, np.nan)
    if len(t) == 0:
        return y
    if len(t) == 1:
        y[tq == t[0]] = x[0]
        return y
    i0, frac, gap = bracket(t, tq)
    x0, x1 = x[i0], x[i0 + 1]
    if method == "linear":
        val = (1.0 - frac) * x0 + frac * x1
    elif method == "nearest":
        val = np.where(frac < 0.5, x0, x1)
    elif method == "previous":
        val = np.where(frac >= 1.0, x1, x0)
    else:
        raise ValueError(f"未知的 method：{method!r}")
    inside = (tq >= t[0]) & (tq <= t[-1])
    ok = inside
    if max_gap is not None:
        hit = (tq == t[i0]) | (tq == t[i0 + 1])     # 剛好踩在樣本上 → 那是真的量測值
        ok = ok & ((gap <= max_gap) | hit)
    y[ok] = val[ok]
    return y


def align_streams(streams: Dict[str, Tuple[np.ndarray, np.ndarray]], rate: float,
                  max_gap: Union[None, str, float, Dict[str, Optional[float]]] = "auto",
                  method: str = "linear", t_start: Optional[float] = None,
                  t_end: Optional[float] = None):
    """
    一句話：多個感測器（各自取樣率、各自抖動）→ 對齊到同一條等間距時間軸。

    設計思路：
      STEP 1  每個 stream 先清資料（_clean_series）。
      STEP 2  共同時間範圍取「交集」：t0 = 各 stream 起點的最大值、t1 = 終點的最小值
              （只有一個感測器有資料的時段，對齊也沒意義）。
      STEP 3  時間軸對齊到 1/rate 的整數倍（t = k/rate），不同批次的資料才能直接拼接。
      STEP 4  每個 stream 呼叫 resample_nonuniform；max_gap='auto' → 用該 stream
              「正常間隔中位數 × 2.5」當門檻，超過就 NaN。
    回傳：(t_common, {name: 對齊後的陣列})
    何時用：IMU + GPS + 輪速計、多台儀器同步分析、機器學習前把特徵排成表格。
    代價：低取樣率感測器被「上採樣」到 rate，中間的值是插出來的（不是新資訊）；
          高取樣率的被「降採樣」到 rate 時，若差很多倍，應先低通再對齊以免混疊。
    """
    if rate <= 0:
        raise ValueError("rate 必須 > 0")
    clean = {}
    for name, (t, x) in streams.items():
        tc, xc = _clean_series(t, x)
        if len(tc) < 2:
            raise ValueError(f"stream {name!r} 有效樣本不足 2 筆")
        clean[name] = (tc, xc)
    t0 = max(tc[0] for tc, _ in clean.values()) if t_start is None else float(t_start)
    t1 = min(tc[-1] for tc, _ in clean.values()) if t_end is None else float(t_end)
    k0 = math.ceil(t0 * rate - 1e-9)
    k1 = math.floor(t1 * rate + 1e-9)
    t_common = np.arange(k0, k1 + 1) / rate
    out = {}
    for name, (tc, xc) in clean.items():
        g = max_gap.get(name, "auto") if isinstance(max_gap, dict) else max_gap
        if isinstance(g, str):
            if g != "auto":
                raise ValueError(f"max_gap 只接受數字、None 或 'auto'，收到 {g!r}")
            g = auto_max_gap(tc)
        out[name] = resample_nonuniform(tc, xc, t_common, method, g)
    return t_common, out


# =========================================================
#  五、2D 影像縮放
# =========================================================
def map_coord(dst, n_in: int, n_out: int, align_corners: bool = False):
    """
    一句話：輸出像素索引 dst → 對應到輸入影像的「浮點座標」src。

    ★ 像素中心對齊（預設，OpenCV / PIL / PyTorch align_corners=False 都用這個）：
        把像素想成一個個「小方塊」，索引 i 代表方塊中心，在連續座標上位於 i + 0.5。
        STEP 1  dst 中心的連續座標 = dst + 0.5
        STEP 2  換算到輸入：除以 scale（= n_out / n_in）
        STEP 3  再轉回輸入的索引：− 0.5
        → src = (dst + 0.5) / scale − 0.5
        例：4 倍放大時 dst=0 → src = 0.5/4 − 0.5 = −0.375（在第 0 個像素中心的「左邊」）。
        這樣整張圖的「範圍」剛好 [−0.5, n_in−0.5] 對上 [−0.5, n_out−0.5]，不偏不倚。

    ⚠ align_corners=True 的坑：src = dst·(n_in−1)/(n_out−1)，硬把「第一個與最後一個像素中心」對齊。
        * 實際縮放倍率不是 scale，而是 (n_out−1)/(n_in−1)（6→24 變成 23/5 = 4.6 倍）。
        * 放大後整張圖相當於往外撐了半格；縮小再放大回來會一直位移累積。
        * 訓練模型時前後處理若一邊 True、一邊 False，輸出會差半個像素，很難查。
        → 除非對方明確要求，一律用 False。
    何時用：所有 2D 縮放（resize 系列）的第一步。
    代價：像素中心對齊時，邊緣的 src 會落在 [−0.5, 0) 或 (n_in−1, n_in−0.5]，需要 clamp 處理。
    """
    dst = np.asarray(dst, dtype=float)
    if align_corners:
        if n_out <= 1:
            return np.zeros_like(dst)
        return dst * (n_in - 1) / (n_out - 1)
    return (dst + 0.5) * n_in / n_out - 0.5


def _out_size(n_in: int, s: float) -> int:
    return max(1, int(round(n_in * s)))


def _resize_axis(arr: np.ndarray, axis: int, n_out: int, method: str,
                 align_corners: bool, a: float) -> np.ndarray:
    """沿某一軸做 1D 重取樣（2D 縮放 = 先做列、再做行，可分離 separable）。"""
    n_in = arr.shape[axis]
    pos = map_coord(np.arange(n_out), n_in, n_out, align_corners)
    _, idx, w = interp_taps(pos, n_in, method, a)          # 邊界自動 clamp
    shape = [1] * arr.ndim
    shape[axis] = n_out
    out = np.zeros(arr.shape[:axis] + (n_out,) + arr.shape[axis + 1:])
    for k in range(idx.shape[1]):
        out += np.take(arr, idx[:, k], axis=axis) * w[:, k].reshape(shape)
    return out


def resize(img, scale, method: str = "linear", align_corners: bool = False,
           a: float = -0.5) -> np.ndarray:
    """
    一句話：通用 2D 縮放（method = 'nearest' / 'linear' / 'cubic'），可分離實作。

    設計思路：
      STEP 1  輸出大小 = round(H·sy) × round(W·sx)。
      STEP 2  每個輸出列 / 行，用 map_coord 算出對應的浮點座標（像素中心對齊）。
      STEP 3  用 interp_taps 拿鄰居與權重（超出邊界 clamp = 邊緣像素往外延伸）。
      STEP 4  先沿 y 方向插值、再沿 x 方向插值；雙線性/雙三次的 2D 權重 = wy × wx（外積），
              所以分兩次 1D 做，結果跟直接 2D 加權一樣，但計算量從 K² 降到 2K。
    支援灰階 (H, W) 與彩色 (H, W, C)。
    何時用：放大，或縮小倍率不超過 2 倍的縮放。
    代價：縮小超過 2 倍時每個輸出只看 K 個鄰居 → 會混疊，應先低通（模糊）或用區域平均。
          輸出是 float，雙三次可能超出原本範圍，存回 uint8 前要 clip。
    """
    img = np.asarray(img, dtype=float)
    if img.ndim < 2:
        raise ValueError("影像至少要是 2D")
    sy, sx = (scale, scale) if np.isscalar(scale) else scale
    oh, ow = _out_size(img.shape[0], sy), _out_size(img.shape[1], sx)
    out = _resize_axis(img, 0, oh, method, align_corners, a)
    return _resize_axis(out, 1, ow, method, align_corners, a)


def resize_nearest(img, scale) -> np.ndarray:
    """
    一句話：最近鄰 —— 每個輸出像素直接抄「映射回去最近的那個輸入像素」。
    設計思路：STEP 1 src = (dst+0.5)/scale − 0.5 → STEP 2 四捨五入 → STEP 3 clamp 後取值。
    何時用：像素風、標籤圖 / 分割遮罩（類別編號不能被「平均」成不存在的類別）。
    代價：鋸齒、方塊感；縮小時會跳過像素造成閃爍。
    """
    return resize(img, scale, "nearest")


def resize_bilinear(img, scale, align_corners: bool = False) -> np.ndarray:
    """
    一句話：雙線性 —— 取周圍 2×2 個像素，按距離加權平均。
    設計思路：STEP 1 算 src 浮點座標 → STEP 2 fy、fx = 小數部分 →
              STEP 3 權重 (1−fy)(1−fx)、(1−fy)fx、fy(1−fx)、fy·fx → STEP 4 加權和。
    何時用：最常用的預設（影片縮放、深度學習 resize、貼圖取樣）。
    代價：放大會糊；縮小超過 2 倍時只看 2×2 會混疊，應先模糊或用 area 平均。
    """
    return resize(img, scale, "linear", align_corners)


def resize_bicubic(img, scale, a: float = -0.5, align_corners: bool = False) -> np.ndarray:
    """
    一句話：雙三次 —— 取周圍 4×4 個像素，用 cubic_kernel 權重（含負值）加權。
    設計思路：同雙線性，但每個方向用 4 個鄰居、權重來自 cubic_kernel；2D 權重 = wy ⊗ wx。
    何時用：照片放大、要比雙線性銳利。
    代價：亮暗交界會過衝（出現比原圖更亮/更暗的光暈），存回 uint8 前要 clip 到 0~255；計算量約 4 倍。
    """
    return resize(img, scale, "cubic", align_corners, a)


# =========================================================
#  自我測試
# =========================================================
def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _amp_spectrum(x, fs):
    """整數週期訊號的振幅頻譜：|X|·2/N（單一正弦的峰值 = 它的振幅）。"""
    X = np.fft.rfft(x)
    amp = np.abs(X) * 2 / len(x)
    amp[0] /= 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, amp


def _top_peaks(f, amp, k=2):
    order = np.argsort(amp)[::-1][:k]
    return [(float(f[i]), float(amp[i])) for i in sorted(order, key=lambda i: f[i])]


def _selftest() -> None:
    line = "=" * 70
    print(line)
    print(" resample.py 自我測試")
    print(line)

    # ---------- [1] 上採樣：帶限訊號，各方法對真值的 RMSE ----------
    N, L = 256, 4
    bins, amps, phs = (13, 33, 54), (1.0, 0.5, 0.3), (0.3, 1.1, 2.0)

    def truth(t):
        t = np.asarray(t, dtype=float)
        return sum(A * np.sin(2 * np.pi * k * t / N + ph) for k, A, ph in zip(bins, amps, phs))

    x = truth(np.arange(N))
    t_new = np.arange(N * L) / L
    y_true = truth(t_new)
    inner = slice(32 * L, (N - 32) * L)            # 只比中間，避開兩端暫態
    results = {
        "ZOH 零階保持": upsample_zoh(x, L),
        "線性 linear": upsample_linear(x, L),
        "三次 cubic": upsample_cubic(x, L),
        "插零+FIR": upsample_fir(x, L),
        "FFT 補零": upsample_fft(x, L),
    }
    err = {k: _rmse(v[inner], y_true[inner]) for k, v in results.items()}
    print(f"\n[1] 上採樣 ×{L}：帶限訊號（頻率 {', '.join(f'{k / N:.3f}' for k in bins)} "
          f"週期/樣本，Nyquist = 0.5），中段 RMSE：")
    for k, v in err.items():
        bar = "█" * max(1, int(round((np.log10(v) + 17) * 2))) if v > 0 else ""
        print(f"    {k:<12s} RMSE = {v:.3e}  {bar}")
    for k, v in results.items():
        assert len(v) == N * L, k
    assert err["FFT 補零"] < 1e-9
    assert err["插零+FIR"] < err["三次 cubic"] / 5          # FIR ≪ cubic
    assert err["三次 cubic"] < err["線性 linear"] < err["ZOH 零階保持"]
    print("    ✔ 排序符合預期：FFT ≈ 0、FIR ≪ cubic < linear < ZOH")

    # polyphase 與「插零後再濾波」數學等價
    h = design_lowpass_fir(2 * 10 * L + 1, 1 / L)
    d = np.max(np.abs(upsample_fir(x, L, h=h) - upsample_fir_direct(x, L, h)))
    print(f"    polyphase vs 插零後直接卷積：最大差 {d:.2e}（等價，但 polyphase 少做 {L} 倍乘法）")
    assert d < 1e-12
    # 插值法一定穿過原始點
    for name in ("線性 linear", "三次 cubic", "ZOH 零階保持"):
        assert np.allclose(results[name][::L], x)
    # Catmull-Rom 能完整重建二次多項式（內部）
    q = 0.3 * np.arange(20.0) ** 2 - 2 * np.arange(20.0) + 1
    tq = np.arange(20 * L) / L
    qc = upsample_cubic(q, L)
    sl = slice(L, 18 * L)
    assert np.allclose(qc[sl], 0.3 * tq[sl] ** 2 - 2 * tq[sl] + 1)
    print("    ✔ 插值法都穿過原始點；Catmull-Rom 對二次多項式在內部精確")

    # ---------- [2] 降採樣混疊示範 ----------
    fs, M = 1000.0, 4
    fs_new = fs / M
    tt = np.arange(1000) / fs
    f1, f2 = 50.0, 430.0
    sig = 1.0 * np.sin(2 * np.pi * f1 * tt) + 0.8 * np.sin(2 * np.pi * f2 * tt)
    k = round(f2 / fs_new)
    f_alias = abs(f2 - k * fs_new)
    yn = decimate_naive(sig, M)
    ya = decimate(sig, M, pad_mode="wrap")         # 訊號整數週期 → 用 wrap 補邊最乾淨
    fn, an = _amp_spectrum(yn, fs_new)
    fa, aa = _amp_spectrum(ya, fs_new)
    pn, pa = _top_peaks(fn, an), _top_peaks(fa, aa)
    i_alias = int(np.argmin(np.abs(fn - f_alias)))
    i_f1 = int(np.argmin(np.abs(fn - f1)))
    print(f"\n[2] 降採樣 ÷{M}：fs {fs:.0f} → {fs_new:.0f} Hz，新 Nyquist = {fs_new / 2:.0f} Hz")
    print(f"    原訊號 = 1.0·sin({f1:.0f} Hz) + 0.8·sin({f2:.0f} Hz)")
    print(f"    {f2:.0f} Hz 超過新 Nyquist → 假頻 f_alias = |{f2:.0f} − {k}×{fs_new:.0f}| = {f_alias:.0f} Hz")
    print("    naive（直接每 M 取一）頻譜前兩大峰：" +
          "、".join(f"{f:.0f} Hz（{a:.3f}）" for f, a in pn))
    print("    decimate（先抗混疊）頻譜前兩大峰：" +
          "、".join(f"{f:.0f} Hz（{a:.3f}）" for f, a in pa))
    print(f"    {f_alias:.0f} Hz 處振幅：naive {an[i_alias]:.3f} → 抗混疊 {aa[i_alias]:.2e}；"
          f"{f1:.0f} Hz 處：naive {an[i_f1]:.3f}、抗混疊 {aa[i_f1]:.3f}")
    assert abs(an[i_alias] - 0.8) < 0.01           # 假頻完整出現
    assert aa[i_alias] < 0.01                       # 被抗混疊濾波器壓掉
    assert abs(aa[i_f1] - 1.0) < 0.02               # 有用的 50 Hz 保留
    print("    ✔ naive 出現 0.8 的假頻；抗混疊後假頻 < 0.01，50 Hz 幾乎不變")

    # ---------- [3] 有理數倍 ----------
    fs_a, fs_b = 44100, 48000
    g = math.gcd(fs_b, fs_a)
    Lr, Mr = fs_b // g, fs_a // g
    n_in = 4410                                     # 0.1 秒
    ta = np.arange(n_in) / fs_a
    s_in = np.sin(2 * np.pi * 1000 * ta) + 0.3 * np.sin(2 * np.pi * 7300 * ta)
    s_out = resample_rational(s_in, fs_b, fs_a)     # 函式內部自己約分
    tb = np.arange(len(s_out)) / fs_b
    s_true = np.sin(2 * np.pi * 1000 * tb) + 0.3 * np.sin(2 * np.pi * 7300 * tb)
    e_r = _rmse(s_out[300:-300], s_true[300:-300])
    print(f"\n[3] 有理數倍：{fs_a} → {fs_b} Hz，gcd = {g} → L/M = {Lr}/{Mr}")
    print(f"    輸入 {n_in} 點 → 輸出 {len(s_out)} 點（預期 ceil({n_in}×{Lr}/{Mr}) = "
          f"{-(-n_in * Lr // Mr)}），中段 RMSE = {e_r:.2e}")
    assert len(s_out) == -(-n_in * Lr // Mr) == 4800
    assert len(resample_rational(np.zeros(441), 160, 147)) == 480
    assert np.allclose(resample_rational(s_in[:500], 320, 294), resample_rational(s_in[:500], 160, 147))
    assert e_r < 1e-3
    # resample_rational(x, L, 1) 與 upsample_fir 應一致；(x, 1, M) 與 decimate 應一致
    assert np.allclose(resample_rational(x, L, 1), upsample_fir(x, L))
    assert np.allclose(resample_rational(sig, 1, M), decimate(sig, M))
    print("    ✔ 長度正確、自動約分、與 upsample_fir / decimate 結果一致")

    # ---------- [4] 不等間距 + 斷線 ----------
    rng = np.random.default_rng(7)
    t_raw = np.arange(0, 2.0, 0.01) + rng.uniform(-0.002, 0.002, 200)
    keep = ~((t_raw > 0.8) & (t_raw < 1.1))         # 0.8 ~ 1.1 秒掉包
    t_raw, x_raw = t_raw[keep], 3 * t_raw[keep] + 1  # 線性訊號 → 線性插值應精確
    t_q = np.arange(0, 2.01, 0.02)
    y_q = resample_nonuniform(t_raw, x_raw, t_q, "linear", max_gap=0.05)
    nan_mask = np.isnan(y_q)
    lo, hi = t_raw[t_raw <= 0.8].max(), t_raw[t_raw >= 1.1].min()
    expect_nan = (t_q < t_raw[0]) | (t_q > t_raw[-1]) | ((t_q > lo) & (t_q < hi))
    print(f"\n[4] 不等間距：{len(t_raw)} 個抖動樣本（±2 ms），{lo:.3f}~{hi:.3f} 秒掉包，max_gap = 0.05 s")
    print(f"    重取樣到 50 Hz 共 {len(t_q)} 點：有效 {int((~nan_mask).sum())}、NaN {int(nan_mask.sum())}"
          f"（掉包區 {int(((t_q > lo) & (t_q < hi)).sum())} 點 + 範圍外 "
          f"{int(((t_q < t_raw[0]) | (t_q > t_raw[-1])).sum())} 點）")
    assert np.array_equal(nan_mask, expect_nan)
    assert np.allclose(y_q[~nan_mask], 3 * t_q[~nan_mask] + 1)
    y_nogap = resample_nonuniform(t_raw, x_raw, t_q, "linear", max_gap=None)
    assert not np.isnan(y_nogap[(t_q > lo) & (t_q < hi)]).any()
    # 輸入中有 NaN（掉包記錄）也要能處理
    xr2 = x_raw.copy()
    xr2[5] = np.nan
    assert np.isfinite(resample_nonuniform(t_raw, xr2, [t_raw[5]], "linear", 0.05)).all()
    # 多感測器對齊
    t_gps = np.arange(0.05, 2.0, 0.1)
    t_gps = t_gps[(t_gps < 1.0) | (t_gps > 1.5)]
    tc, al = align_streams({"imu": (t_raw, x_raw), "gps": (t_gps, 2 * t_gps)}, rate=20)
    assert tc[0] >= max(t_raw[0], t_gps[0]) - 1e-12 and tc[-1] <= min(t_raw[-1], t_gps[-1]) + 1e-12
    assert np.allclose(tc * 20, np.round(tc * 20))
    assert np.isnan(al["gps"][(tc > 1.0) & (tc < 1.5)]).all()
    print(f"    align_streams：共同時間軸 {tc[0]:.2f}~{tc[-1]:.2f} s、{len(tc)} 點；"
          f"GPS NaN {int(np.isnan(al['gps']).sum())} 點、IMU NaN {int(np.isnan(al['imu']).sum())} 點")
    print("    ✔ 斷線處與範圍外都是 NaN，其餘對線性訊號精確")

    # ---------- [5] 2D 縮放 ----------
    print("\n[5] 2D 縮放（像素中心對齊）：dst → src 映射（6 → 24，×4）")
    ds = np.arange(6)
    print("    " + "  ".join(f"dst={d}→{s:+.3f}" for d, s in zip(ds, map_coord(ds, 6, 24))))
    const = np.full((6, 6), 7.25)
    for fn_, name in ((resize_nearest, "nearest"), (resize_bilinear, "bilinear"),
                      (resize_bicubic, "bicubic")):
        r = fn_(const, 4)
        assert r.shape == (24, 24) and np.max(np.abs(r - 7.25)) < 1e-12, name
    ii, jj = np.mgrid[0:6, 0:6].astype(float)
    ramp = 2 * ii + 3 * jj
    rb = resize_bilinear(ramp, 4)
    sy = map_coord(np.arange(24), 6, 24)
    Y, X = np.meshgrid(sy, sy, indexing="ij")
    interior = (Y >= 0) & (Y <= 5) & (X >= 0) & (X <= 5)
    assert np.allclose(rb[interior], (2 * Y + 3 * X)[interior])
    assert np.array_equal(resize_nearest(ramp, 4), np.kron(ramp, np.ones((4, 4))))
    rgb = np.dstack([ramp, ramp * 0 + 1, -ramp])
    assert resize_bicubic(rgb, 2).shape == (12, 12, 3)
    ac = map_coord(np.array([0, 23]), 6, 24, align_corners=True)
    print(f"    align_corners=True 時 dst 0/23 → src {ac[0]:.0f}/{ac[1]:.0f}，"
          f"實際倍率 {(24 - 1) / (6 - 1):.1f} 而不是 4（這就是坑）")
    print("    ✔ 常數圖放大仍是常數；雙線性對線性斜坡在內部精確；最近鄰 = 每格複製 4×4")

    # ---------- [6] 與 scipy 對答案（有裝才比） ----------
    try:
        from scipy import signal as sps
    except ImportError:
        print("\n[6] 沒有安裝 scipy，略過比對。")
    else:
        print("\n[6] 與 scipy.signal 比對（寬鬆門檻）")
        h1 = design_lowpass_fir(81, 0.25)
        d_fw = np.max(np.abs(h1 - sps.firwin(81, 0.25, window=("kaiser", 5.0))))
        d_up = np.max(np.abs(upsample_fir(x, L, pad_mode="constant") - sps.resample_poly(x, L, 1)))
        hr = design_lowpass_fir(2 * 10 * 160 + 1, 1 / 160)
        d_ra = np.max(np.abs(resample_rational(s_in, 160, 147, h=hr, pad_mode="constant")
                             - sps.resample_poly(s_in, 160, 147, window=hr)))
        d_dc = np.max(np.abs(resample_rational(sig, 1, M, pad_mode="constant")
                             - sps.resample_poly(sig, 1, M)))
        print(f"    design_lowpass_fir vs firwin          最大差 {d_fw:.1e}")
        print(f"    upsample_fir ×{L}   vs resample_poly    最大差 {d_up:.1e}")
        print(f"    resample_rational 160/147 vs resample_poly 最大差 {d_ra:.1e}")
        print(f"    resample_rational 1/{M}   vs resample_poly 最大差 {d_dc:.1e}")
        assert d_fw < 1e-10 and d_up < 1e-6 and d_ra < 1e-6 and d_dc < 1e-6
        print("    ✔ 與 scipy 一致")

    print("\n" + line)
    print(" 全部測試通過 ✔")
    print(line)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _selftest()
