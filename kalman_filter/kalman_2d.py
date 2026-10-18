# -*- coding: utf-8 -*-
"""
二維卡爾曼濾波器（工程用）—— 函式庫 + 命令列示範
=====================================================
這是什麼：
  一個「拿去就能放進真實專案」的 2D 目標追蹤卡爾曼濾波器（車、無人機、雷達上的光點），
  只依賴 numpy。程式碼照「工程師設計它的思考順序」寫成 STEP 1 ～ STEP 13，
  從上往下讀，就是一次完整的設計過程：先定義問題，再定模型，最後才寫演算法與防呆。

核心觀念（一句話）：
  「預測」（靠運動模型推）和「量測」（感測器量到）兩個都不準；
  卡爾曼濾波器 = 依「誰比較準」給權重做加權平均，權重 K 是從兩邊的不確定度（P 與 R）算出來的，
  不是亂調的。

工程版比教科書版多做的事：
  ● 可變 dt：每筆資料間隔不同也能用（STEP 7）
  ● 閘門 gating：用 NIS 卡方檢定擋掉多路徑跳點等離群值（STEP 9）
  ● Joseph form 更新協方差 + 強制對稱：長時間運作不發散（STEP 11）
  ● 量測遺失（None）→ 只預測不更新（STEP 12）
  ● 連續被拒的保險絲：判定「濾波器跟丟」→ 重新初始化（STEP 9）
  ● NIS 健康監控：不用真值也能判斷 Q / R 設得對不對（STEP 13）

操作方式：
  python kalman_2d.py                  跑模擬，印出 RMSE 比較表（含自我檢查 assert）
  python kalman_2d.py --plot           另外用 matplotlib 畫圖
  python kalman_2d.py --save-plot a.png  把圖存成檔案（不開視窗）
  python kalman_2d.py --help           看所有參數

  在自己的程式裡：
      from kalman_2d import KalmanFilter2D
      kf = KalmanFilter2D(sigma_a=0.8, sigma_z=5.0)
      for z, dt in stream:                 # z = (x, y)；沒收到就給 None
          res = kf.step(z, dt)             # res.update.status: init/accepted/rejected/missing/reinit
          print(kf.position, kf.velocity)

需求： pip install numpy（--plot 另需 matplotlib）
執行： python kalman_2d.py
互動課： python kalman_lessons_pyqt6.py（同資料夾，六堂課，先直覺後公式；第 5 章就是用這支函式庫）
"""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# =====================================================================
#  STEP 1　定義問題：要估計什麼？量得到什麼？
# =====================================================================
#  情境：在 2D 平面上追蹤一個移動目標（車、無人機、雷達上的光點）。
#
#    狀態 x（我們「想知道」的）           量測 z（感測器「給得出」的）
#        x = [px, py, vx, vy]ᵀ               z = [px, py]ᵀ
#        位置 (m) 與速度 (m/s)               只有位置，而且帶雜訊、偶爾亂跳、有時收不到
#
#  關鍵：GPS / 雷達只量得到「位置」，速度根本沒有感測器在量！
#  速度要靠濾波器從「位置隨時間的變化」推出來。
#  直接拿相鄰兩筆位置相減 (z_k − z_{k−1}) / dt 當速度？雜訊會被放大 √2/dt 倍，幾乎不能用
#  （__main__ 的比較表會用數字證明這件事）。
#
#  單位約定（整支程式一律 SI 制）：位置 m、速度 m/s、加速度 m/s²、時間 s。
#  把單位寫死在註解裡，是為了避免「Q 用 km、R 用 m」這種低級但致命的錯。
STATE_DIM = 4                         # [px, py, vx, vy]
MEAS_DIM = 2                          # [px, py]
STATE_NAMES = ("px", "py", "vx", "vy")


def _symmetrize(P: np.ndarray) -> np.ndarray:
    """把浮點誤差造成的微小不對稱抹掉：P = (P + Pᵀ) / 2。"""
    return 0.5 * (P + P.T)


# =====================================================================
#  STEP 2　運動模型 F(dt)：沒有新資訊時，目標「應該」怎麼動？
# =====================================================================
#  選最簡單、最常用的「等速模型」(Constant Velocity, CV)：
#
#      px ← px + vx·dt                 ┌ 1  0  dt  0 ┐
#      py ← py + vy·dt           F  =  │ 0  1  0  dt │
#      vx ← vx                         │ 0  0  1   0 │
#      vy ← vy                         └ 0  0  0   1 ┘
#
#  為什麼敢假設等速？因為模型不必完美 —— 模型「錯多少」交給 STEP 4 的 Q 來描述。
#  dt 寫成參數（不寫死），因為真實感測器的資料間隔常常不固定（STEP 7 會用到）。
def make_F(dt: float) -> np.ndarray:
    """STEP 2：等速（CV）模型的狀態轉移矩陣 F(dt)。"""
    return np.array([[1.0, 0.0, dt, 0.0],
                     [0.0, 1.0, 0.0, dt],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


# =====================================================================
#  STEP 3　量測模型 H：感測器從狀態裡「看到」哪些東西？
# =====================================================================
#  z = H·x + v，  v ~ N(0, R)
#
#      H = ┌ 1  0  0  0 ┐   → 從 4 維狀態裡「挑出」px、py（速度看不到）
#          └ 0  1  0  0 ┘
def make_H() -> np.ndarray:
    """STEP 3：量測矩陣 H（只量位置）。"""
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]])


# =====================================================================
#  STEP 4　過程雜訊 Q(dt, σ_a)：模型「錯多少」？—— 從物理推出來，不是亂調
# =====================================================================
#  CV 模型說「速度不變」，但真實目標會加速、轉彎。把這些「模型沒說到的加速度」
#  當成雜訊 a ~ N(0, σ_a²)，並假設它在每個 dt 區間內是常數、區間之間互相獨立
#  （離散白噪加速度模型 DWNA，又叫 piecewise white noise acceleration）。
#
#  在一個 dt 內，加速度 a 造成的影響（國中物理）：
#      位置多走 ½·a·dt²        速度多了 a·dt
#  寫成矩陣 G（把 2D 加速度 [ax, ay] 映射到 4 維狀態）：
#
#      G = ┌ ½dt²   0   ┐
#          │  0    ½dt² │          Q = G · (σ_a²·I) · Gᵀ
#          │  dt    0   │
#          └  0     dt  ┘
#
#  展開（x 軸、y 軸各自一樣、互不相關）：
#
#      Q = σ_a² · ┌ dt⁴/4    0     dt³/2    0   ┐
#                 │   0    dt⁴/4    0     dt³/2 │
#                 │ dt³/2    0      dt²     0   │
#                 └   0    dt³/2    0      dt²  ┘
#
#  非對角的 dt³/2：同一個加速度「同時」推動位置與速度，所以兩者的誤差是相關的。
#
#  σ_a 的物理意義：「目標在一個 dt 內，可能出現多猛的、模型沒預料到的加速度」，單位 m/s²。
#  經驗法則：σ_a ≈ 目標最大加速度的 0.5 ～ 1 倍。
#      人走路 ~0.5、一般車輛 ~1～3、無人機 ~2～5、戰鬥機 30+
#  → Q 是從「目標的物理能力」推出來的；調 σ_a 一個數字，比亂調 Q 的 16 個元素有意義得多。
def make_Q(dt: float, sigma_a: float) -> np.ndarray:
    """STEP 4：DWNA 過程雜訊 Q = G·σ_a²·Gᵀ（4×4）。"""
    half_dt2 = 0.5 * dt * dt
    G = np.array([[half_dt2, 0.0],
                  [0.0, half_dt2],
                  [dt, 0.0],
                  [0.0, dt]])
    return (sigma_a ** 2) * (G @ G.T)


def cv_model_2d(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    """STEP 2 + STEP 4 打包：回傳 (F, Q)。模型換掉時（例如改等加速 CA 模型），只要換這個函式。"""
    return make_F(dt), make_Q(dt, sigma_a)


# =====================================================================
#  STEP 5　量測雜訊 R：感測器有多不準？—— 直接從規格書來
# =====================================================================
#      R = σ_z² · I（2×2）
#  例：GPS 水平精度 3 m（1σ）→ σ_z = 3 → R = diag(9, 9) m²
#  小心規格書寫的常常不是 1σ（以下假設 x、y 兩軸 σ 相同）：
#      CEP（50% 圓）≈ 1.18σ      95% 圓 ≈ 2.45σ      2DRMS ≈ 2.83σ
#  → 先換算回 1σ 再填進來。若感測器每筆會回報精度（例如 GPS 的 HDOP），
#    可以在 update(z, R=...) 每筆給不同的 R。
def make_R(sigma_z: float) -> np.ndarray:
    """STEP 5：量測雜訊協方差 R = σ_z²·I。"""
    return (sigma_z ** 2) * np.eye(MEAS_DIM)


# =====================================================================
#  STEP 6　初始化：第一筆量測進來，什麼都不知道時怎麼開始？
# =====================================================================
#  最誠實的做法：
#      位置 = 量到的位置，     不確定度 = 感測器雜訊 σ_z²
#      速度 = 0（反正不知道），不確定度 = v_max²（「我不知道」就用大 P 表示）
#
#      x₀ = [zx, zy, 0, 0]ᵀ        P₀ = diag(σ_z², σ_z², v_max², v_max²)
#
#  為什麼不給 P₀ 很小？那等於宣稱「我非常確定速度是 0」→ 之後的量測都被當成雜訊忽略，
#  濾波器要很久才「醒過來」。大 P 讓前幾筆量測有很大的話語權，速度兩三筆就推出來了。
#  （另一種常見做法：兩點初始化，等第二筆量測，用差分當初始速度。）
def initial_state(z: Sequence[float], sigma_z: float, v_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """STEP 6：由第一筆量測建立 (x₀, P₀)。"""
    z = np.asarray(z, dtype=float).reshape(MEAS_DIM)
    x0 = np.array([z[0], z[1], 0.0, 0.0])
    P0 = np.diag([sigma_z ** 2, sigma_z ** 2, v_max ** 2, v_max ** 2])
    return x0, P0


# =====================================================================
#  STEP 7　預測 predict(dt)：時間往前走 dt，狀態與不確定度怎麼變？
# =====================================================================
#      x⁻ = F·x
#      P⁻ = F·P·Fᵀ + Q
#  F·P·Fᵀ：把舊的不確定度「搬」到新時刻（速度的不確定，經過 dt 會變成位置的不確定）
#  + Q    ：再加上這段時間模型可能出錯的量 → P 一定變大（沒有新資訊，只會更不確定）
#  可變 dt：每次都用「這一筆與上一筆的實際時間差」重算 F(dt)、Q(dt)，不假設固定取樣率。
#  dt < 0 代表資料亂序 → 直接報錯（工程上應該先排序，或丟棄過期資料）。
def predict_state(x: np.ndarray, F: np.ndarray) -> np.ndarray:
    """STEP 7a：x⁻ = F·x"""
    return F @ x


def predict_covariance(P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """STEP 7b：P⁻ = F·P·Fᵀ + Q"""
    return _symmetrize(F @ P @ F.T + Q)


# =====================================================================
#  STEP 8　創新 innovation：量測跟預測差多少？「應該」差多少？
# =====================================================================
#      y = z − H·x⁻           「量到的」減「預測會量到的」= 這次的落差（驚訝程度）
#      S = H·P⁻·Hᵀ + R        「落差應該有多大」= 預測的不確定（投影到量測空間）+ 感測器雜訊
#  y 與 S 一起看才有意義：落差 10 m 算大嗎？要看 S —— 若 S 說「正常就會差 ±20 m」，那 10 m 很正常。
def innovation(z: np.ndarray, x_prior: np.ndarray, H: np.ndarray) -> np.ndarray:
    """STEP 8a：y = z − H·x⁻"""
    return z - H @ x_prior


def innovation_covariance(P_prior: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """STEP 8b：S = H·P⁻·Hᵀ + R"""
    return _symmetrize(H @ P_prior @ H.T + R)


# =====================================================================
#  STEP 9　閘門 gating：這筆量測可信嗎？
# =====================================================================
#  NIS（Normalized Innovation Squared，正規化創新平方）＝ 馬氏距離的平方：
#      NIS = yᵀ·S⁻¹·y
#  直覺：把落差 y 用「預期的大小 S」量化成「幾個標準差的平方」。
#  模型正確時，NIS 服從「自由度 = 量測維度 = 2」的卡方分佈。
#  2 自由度卡方剛好有封閉解：P(NIS > g) = exp(−g/2)  →  g = −2·ln(1 − p)
#      p = 95%  → g = 5.99        p = 99%  → g = 9.21        p = 99.9% → g = 13.82
#  取 99%：正常量測只有 1% 機率被誤殺；超過門檻的判為離群值（例如 GPS 多路徑跳點），
#  不拿來更新（狀態維持預測值）。
#
#  閘門的副作用：如果是濾波器自己跟丟了（σ_a 太小、轉彎跟不上），好的量測也會被一直拒絕，
#  然後就永遠回不來了。保險絲：連續被拒 N 次，判定「錯的是濾波器、不是量測」→ 重新初始化。
def chi2_threshold_2dof(prob: float) -> float:
    """2 自由度卡方分佈的 prob 分位數（封閉解）：g = −2·ln(1 − prob)。"""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob 必須介於 0 與 1 之間")
    return -2.0 * math.log(1.0 - prob)


GATE_99_2DOF = chi2_threshold_2dof(0.99)      # ≈ 9.21


def nis_value(y: np.ndarray, S: np.ndarray) -> float:
    """STEP 9：NIS = yᵀ·S⁻¹·y（用 solve，不求逆，理由見 STEP 10）。"""
    return float(y @ np.linalg.solve(S, y))


# =====================================================================
#  STEP 10　增益 K：這次要「聽量測幾成」？
# =====================================================================
#      K = P⁻·Hᵀ·S⁻¹         （4×2：每個狀態分量，對 x、y 兩個方向的落差各修正多少）
#  直覺（1D 版本）：K = P⁻ / (P⁻ + R)
#      預測很不準（P⁻ 大）→ K → 1：幾乎全聽量測
#      量測很不準（R 大） → K → 0：幾乎全聽預測
#  K 的第 3、4 列（速度那兩列）就是「位置落差 → 速度修正」的比例 —— 速度就是這樣被推出來的。
#
#  為什麼用 np.linalg.solve，而不是 np.linalg.inv(S)？
#   1. 數值精度：solve 用 LU 分解直接解方程；「先求逆再相乘」多一次捨入，S 接近奇異時誤差放大；
#   2. 速度：少一次矩陣乘法（量測維度大時差很多）；
#   3. 習慣：數值計算的通則是「需要 A⁻¹·b 時，解 A·x = b」。
#  推導：K·S = P⁻·Hᵀ → 兩邊轉置（S、P⁻ 都對稱）→ S·Kᵀ = H·P⁻ → Kᵀ = solve(S, H·P⁻)
def kalman_gain(P_prior: np.ndarray, H: np.ndarray, S: np.ndarray) -> np.ndarray:
    """STEP 10：K = P⁻·Hᵀ·S⁻¹（以 solve 實作）。"""
    return np.linalg.solve(S, H @ P_prior).T


# =====================================================================
#  STEP 11　更新 update：把量測「融合」進來
# =====================================================================
#      x = x⁻ + K·y
#      P = (I − K·H)·P⁻·(I − K·H)ᵀ + K·R·Kᵀ          ← Joseph form
#  教科書常寫 P = (I − K·H)·P⁻，數學上等價，但「只有在 K 剛好是最佳增益時」才成立；
#  浮點誤差（或 K 被人手動改過、被限幅）會讓它算出不對稱、甚至非正定（負的變異數！）的 P，
#  濾波器就慢慢發散。Joseph form 是「兩個半正定矩陣相加」，結構上保證對稱半正定，
#  對 K 的小誤差也不敏感。最後再 P = (P + Pᵀ)/2 強制對稱，抹掉殘留的浮點誤差。
#  多花幾次 4×4 乘法，換來長時間運作不發散 —— 工程上非常划算。
def update_state(x_prior: np.ndarray, K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """STEP 11a：x = x⁻ + K·y"""
    return x_prior + K @ y


def update_covariance_joseph(P_prior: np.ndarray, K: np.ndarray,
                             H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """STEP 11b：Joseph form  P = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ，並強制對稱。"""
    I_KH = np.eye(P_prior.shape[0]) - K @ H
    return _symmetrize(I_KH @ P_prior @ I_KH.T + K @ R @ K.T)


# =====================================================================
#  每一步的「中間量」都打包回傳：方便除錯、記錄 log、也讓 GUI 能一行一行展示
# =====================================================================
@dataclass
class PredictResult:
    """predict(dt) 的完整紀錄。"""
    dt: float
    F: np.ndarray
    Q: np.ndarray
    x_prev: np.ndarray          # 預測前（上一時刻的後驗）
    P_prev: np.ndarray
    x_prior: np.ndarray         # x⁻
    P_prior: np.ndarray         # P⁻


@dataclass
class UpdateResult:
    """
    update(z) 的完整紀錄。status 可能是：
      'init'     第一筆量測，拿來初始化（STEP 6）
      'accepted' 通過閘門，正常更新（STEP 8～11）
      'rejected' NIS 超過門檻，判為離群值，不更新（STEP 9）
      'reinit'   連續被拒太多次，保險絲啟動，重新初始化（STEP 9）
      'missing'  這個時刻沒有量測，只預測（STEP 12）
      'waiting'  還沒初始化、也沒收到量測（什麼都不做）
    """
    status: str
    z: Optional[np.ndarray] = None
    x_prior: Optional[np.ndarray] = None
    P_prior: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    nis: Optional[float] = None
    gate_threshold: float = GATE_99_2DOF
    K: Optional[np.ndarray] = None
    x_post: Optional[np.ndarray] = None
    P_post: Optional[np.ndarray] = None
    consecutive_rejects: int = 0

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def missing(self) -> bool:
        return self.status == "missing"

    @property
    def gated_out(self) -> bool:
        """NIS 是否超過門檻（不論閘門有沒有開）。"""
        return self.nis is not None and self.nis > self.gate_threshold


@dataclass
class StepResult:
    """step(z, dt) = predict + update 的打包。初始化那一步 predict 為 None。"""
    predict: Optional[PredictResult]
    update: UpdateResult


# =====================================================================
#  把 STEP 2～12 組起來：工程用的 2D 卡爾曼濾波器
# =====================================================================
class KalmanFilter2D:
    """
    2D 等速（CV）模型卡爾曼濾波器（工程用）。

    參數
      sigma_a      過程雜訊：目標可能的（未建模）加速度 1σ，m/s²（STEP 4）
      sigma_z      量測雜訊：感測器位置誤差 1σ，m（STEP 5）
      gate_threshold  閘門門檻（NIS 上限），預設 2 自由度卡方 99% = 9.21（STEP 9）
      use_gating   是否啟用閘門
      v_max        初始化時的速度不確定度 1σ，m/s（STEP 6）
      max_consecutive_rejects  連續被拒幾次就重新初始化；None = 不啟用保險絲

    用法
      kf = KalmanFilter2D(sigma_a=0.8, sigma_z=5.0)
      res = kf.step(z, dt)          # z = (x, y) 或 None
      kf.position, kf.velocity, kf.P
    """

    def __init__(self, sigma_a: float, sigma_z: float,
                 gate_threshold: float = GATE_99_2DOF, use_gating: bool = True,
                 v_max: float = 20.0, max_consecutive_rejects: Optional[int] = 3):
        if sigma_a <= 0 or sigma_z <= 0 or v_max <= 0:
            raise ValueError("sigma_a、sigma_z、v_max 都必須 > 0")
        self.sigma_a = float(sigma_a)
        self.sigma_z = float(sigma_z)
        self.gate_threshold = float(gate_threshold)
        self.use_gating = bool(use_gating)
        self.v_max = float(v_max)
        self.max_consecutive_rejects = max_consecutive_rejects
        self.H = make_H()
        self.R = make_R(self.sigma_z)
        self.reset()

    # ---------- 狀態 ----------
    def reset(self) -> None:
        """回到「還沒初始化」。"""
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self.consecutive_rejects = 0
        self.nis_history: List[float] = []      # 只記「被接受」的 NIS（給 STEP 13 健康監控用）
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_missing = 0
        self.n_reinit = 0

    @property
    def initialized(self) -> bool:
        return self._x is not None

    @property
    def x(self) -> np.ndarray:
        self._require_init()
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        self._require_init()
        return self._P.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    @property
    def position_cov(self) -> np.ndarray:
        """位置的 2×2 協方差（畫不確定度橢圓用）。"""
        return self.P[:2, :2]

    def _require_init(self) -> None:
        if self._x is None:
            raise RuntimeError("濾波器尚未初始化：請先 initialize(z) 或 step(z, dt)")

    # ---------- STEP 6 ----------
    def initialize(self, z: Sequence[float]) -> UpdateResult:
        """STEP 6：用第一筆量測初始化。"""
        z = np.asarray(z, dtype=float).reshape(MEAS_DIM)
        self._x, self._P = initial_state(z, self.sigma_z, self.v_max)
        self.consecutive_rejects = 0
        return UpdateResult(status="init", z=z, R=self.R.copy(),
                            gate_threshold=self.gate_threshold,
                            x_post=self._x.copy(), P_post=self._P.copy())

    # ---------- STEP 7 ----------
    def predict(self, dt: float) -> PredictResult:
        """STEP 7：時間往前推 dt 秒（dt 可以每次不同）。"""
        self._require_init()
        dt = float(dt)
        if dt < 0:
            raise ValueError(f"dt = {dt} < 0：量測時間倒流（亂序），請先排序或丟棄過期資料")
        F, Q = cv_model_2d(dt, self.sigma_a)
        x_prev, P_prev = self._x.copy(), self._P.copy()
        self._x = predict_state(self._x, F)
        self._P = predict_covariance(self._P, F, Q)
        return PredictResult(dt=dt, F=F, Q=Q, x_prev=x_prev, P_prev=P_prev,
                             x_prior=self._x.copy(), P_prior=self._P.copy())

    # ---------- STEP 8～12 ----------
    def update(self, z: Optional[Sequence[float]], R: Optional[np.ndarray] = None) -> UpdateResult:
        """
        STEP 8～12：用量測 z 修正目前（預測後的）狀態。
        z 為 None → 量測遺失，只保留預測（STEP 12）。
        R 可每筆不同（例如依 GPS 回報的精度）；不給就用建構時的 σ_z²·I。
        """
        self._require_init()
        x_prior, P_prior = self._x.copy(), self._P.copy()

        # STEP 12：量測遺失 → 不更新，P 維持預測後（變大）的值
        if z is None:
            self.n_missing += 1
            return UpdateResult(status="missing", x_prior=x_prior, P_prior=P_prior,
                                gate_threshold=self.gate_threshold,
                                x_post=x_prior.copy(), P_post=P_prior.copy(),
                                consecutive_rejects=self.consecutive_rejects)

        z = np.asarray(z, dtype=float).reshape(MEAS_DIM)
        R = self.R if R is None else np.asarray(R, dtype=float)

        # STEP 8：創新與其協方差
        y = innovation(z, x_prior, self.H)
        S = innovation_covariance(P_prior, self.H, R)

        # STEP 9：閘門
        d2 = nis_value(y, S)
        base = dict(z=z, x_prior=x_prior, P_prior=P_prior, R=R.copy(), y=y, S=S,
                    nis=d2, gate_threshold=self.gate_threshold)
        if self.use_gating and d2 > self.gate_threshold:
            self.consecutive_rejects += 1
            limit = self.max_consecutive_rejects
            if limit is not None and self.consecutive_rejects >= limit:
                # 保險絲：連續被拒太多次 → 比較可能是濾波器跟丟，用這筆量測重新開始。
                # 位置信這筆量測、速度沿用目前估計，但不確定度全部回到初始的「我不知道」。
                n_rej = self.consecutive_rejects
                _, P0 = initial_state(z, self.sigma_z, self.v_max)
                self._x = np.array([z[0], z[1], x_prior[2], x_prior[3]])
                self._P = P0
                self.consecutive_rejects = 0
                self.n_reinit += 1
                return UpdateResult(status="reinit", x_post=self._x.copy(), P_post=self._P.copy(),
                                    consecutive_rejects=n_rej, **base)
            self.n_rejected += 1
            return UpdateResult(status="rejected", x_post=x_prior.copy(), P_post=P_prior.copy(),
                                consecutive_rejects=self.consecutive_rejects, **base)

        # STEP 10：增益
        K = kalman_gain(P_prior, self.H, S)
        # STEP 11：更新狀態與協方差（Joseph form）
        self._x = update_state(x_prior, K, y)
        self._P = update_covariance_joseph(P_prior, K, self.H, R)

        self.consecutive_rejects = 0
        self.n_accepted += 1
        self.nis_history.append(d2)
        return UpdateResult(status="accepted", K=K, x_post=self._x.copy(), P_post=self._P.copy(),
                            consecutive_rejects=0, **base)

    def step(self, z: Optional[Sequence[float]], dt: float,
             R: Optional[np.ndarray] = None) -> StepResult:
        """
        便利函式：一個時刻要做的全部事情。
          還沒初始化 → 有量測就初始化（STEP 6），沒有就等待
          已初始化   → predict(dt)（STEP 7）+ update(z)（STEP 8～12）
        """
        if not self.initialized:
            if z is None:
                return StepResult(None, UpdateResult(status="waiting",
                                                     gate_threshold=self.gate_threshold))
            return StepResult(None, self.initialize(z))
        pred = self.predict(dt)
        return StepResult(pred, self.update(z, R))


# =====================================================================
#  STEP 13　健康監控：現場沒有真值，怎麼知道 Q、R 設得對不對？
# =====================================================================
#  看 NIS 的長期平均（不需要真值！這是現場唯一拿得到的「考卷」）：
#      模型與 Q、R 都正確 → E[NIS] = 量測維度 = 2
#      長期明顯 > 2 → 實際落差比濾波器以為的大 → 濾波器「過度自信」→ Q 或 R 設太小
#      長期明顯 < 2 → 實際落差比以為的小     → 濾波器「過度保守」→ Q 或 R 設太大
#  N 筆平均的 95% 區間 ≈ m ± 1.96·√(2m/N)（m = 量測維度；卡方(m) 的變異數 = 2m）。
#  註：開閘門時，被拒的大 NIS 不會進統計，平均會略低一點點（99% 門檻下約 1.9），可忽略。
@dataclass
class NisReport:
    n: int
    mean: float
    expected: float
    lower: float
    upper: float
    verdict: str            # 'ok' | 'too_large' | 'too_small' | 'insufficient'
    message: str


def nis_health(nis_values: Sequence[float], meas_dim: int = MEAS_DIM,
               z_score: float = 1.96, min_samples: int = 10) -> NisReport:
    """STEP 13：由一串 NIS 判斷濾波器是否「誠實」（Q、R 是否合理）。"""
    vals = np.asarray([v for v in nis_values if v is not None and np.isfinite(v)], dtype=float)
    n = int(vals.size)
    m = float(meas_dim)
    if n < min_samples:
        return NisReport(n, float(vals.mean()) if n else float("nan"), m,
                         float("nan"), float("nan"), "insufficient",
                         f"樣本太少（{n} 筆 < {min_samples}），先別下結論")
    mean = float(vals.mean())
    half = z_score * math.sqrt(2.0 * m / n)
    lo, hi = m - half, m + half
    if mean > hi:
        verdict = "too_large"
        msg = (f"平均 NIS = {mean:.2f} > {hi:.2f}：濾波器過度自信（落差比它以為的大）"
               f"→ Q（σ_a）或 R（σ_z）設太小")
    elif mean < lo:
        verdict = "too_small"
        msg = (f"平均 NIS = {mean:.2f} < {lo:.2f}：濾波器過度保守（落差比它以為的小）"
               f"→ Q（σ_a）或 R（σ_z）設太大")
    else:
        verdict = "ok"
        msg = f"平均 NIS = {mean:.2f}，落在 [{lo:.2f}, {hi:.2f}] 內：Q、R 與實際相符"
    return NisReport(n, mean, m, lo, hi, verdict, msg)


def steady_state_gain(dt: float, sigma_a: float, sigma_z: float,
                      iters: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """
    工程小工具：固定 dt、一直有量測時，K 會收斂到一個定值（穩態增益）。
    做法：反覆跑「預測 + 更新」的協方差遞迴直到 P 不再變。
    穩態 K 的位置/速度那兩個數，就是經典 α-β 濾波器的 α 與 β/dt。
    """
    F, Q = cv_model_2d(dt, sigma_a)
    H, R = make_H(), make_R(sigma_z)
    P = np.diag([sigma_z ** 2, sigma_z ** 2, 1e4, 1e4])
    K = np.zeros((STATE_DIM, MEAS_DIM))
    for _ in range(iters):
        P_prior = predict_covariance(P, F, Q)
        S = innovation_covariance(P_prior, H, R)
        K = kalman_gain(P_prior, H, S)
        P_new = update_covariance_joseph(P_prior, K, H, R)
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    return K


# =====================================================================
#  模擬器：產生「真值軌跡 + 帶雜訊的量測」來測試濾波器
# =====================================================================
#  真值：直線（等速）→ 轉彎（等速率圓弧）→ 加速（直線等加速）
#  量測：高斯雜訊 σ_z；少數大跳點（多路徑離群值）；一段遺失（進隧道）
#  所有亂數「一次全部抽好」，開關只決定要不要套用 → 同一個 seed 下，不同設定之間比較才公平。
@dataclass
class SimData:
    t: np.ndarray                       # (n,) 時間戳記 s
    dt: np.ndarray                      # (n,) 與上一筆的時間差；dt[0] = 0
    truth: np.ndarray                   # (n, 4) 真值 [px, py, vx, vy]
    z: List[Optional[np.ndarray]]       # 長度 n；遺失的時刻為 None
    z_raw: np.ndarray                   # (n, 2) 含離群值、不管遺失的原始量測（畫圖算範圍用）
    is_outlier: np.ndarray              # (n,) 這筆是不是注入的離群值
    is_dropout: np.ndarray              # (n,) 這個時刻是不是在遺失區段
    sigma_z: float                      # 模擬感測器的「真實」雜訊 1σ

    @property
    def n(self) -> int:
        return len(self.t)


def _maneuver_truth(t: float, v0: float, heading0: float, turn_start: float,
                    turn_dur: float, turn_rate: float, accel: float) -> np.ndarray:
    """解析解：給時間 t，回傳真值 [px, py, vx, vy]（直線 → 圓弧 → 等加速直線）。"""
    c0, s0 = math.cos(heading0), math.sin(heading0)
    if t <= turn_start:                                         # 段 1：等速直線
        return np.array([v0 * t * c0, v0 * t * s0, v0 * c0, v0 * s0])
    p1 = np.array([v0 * turn_start * c0, v0 * turn_start * s0])
    if t <= turn_start + turn_dur:                              # 段 2：等速率轉彎
        tau = t - turn_start
        h = heading0 + turn_rate * tau
        r = v0 / turn_rate
        p = p1 + r * np.array([math.sin(h) - s0, -math.cos(h) + c0])
        return np.array([p[0], p[1], v0 * math.cos(h), v0 * math.sin(h)])
    h2 = heading0 + turn_rate * turn_dur                        # 段 3：直線等加速
    r = v0 / turn_rate
    p2 = p1 + r * np.array([math.sin(h2) - s0, -math.cos(h2) + c0])
    tau = t - turn_start - turn_dur
    dist = v0 * tau + 0.5 * accel * tau * tau
    v = v0 + accel * tau
    c2, s2 = math.cos(h2), math.sin(h2)
    return np.array([p2[0] + dist * c2, p2[1] + dist * s2, v * c2, v * s2])


def simulate_2d_track(n: int = 60, dt: float = 1.0, seed: int = 7, sigma_z: float = 5.0,
                      outliers: bool = True, dropout: bool = True,
                      outlier_steps: Sequence[int] = (12, 26, 50), outlier_mag: float = 40.0,
                      dropout_range: Tuple[int, int] = (38, 44), dt_jitter: float = 0.0,
                      v0: float = 6.0, heading0_deg: float = 45.0,
                      turn_start: float = 20.0, turn_dur: float = 12.0, turn_deg: float = -90.0,
                      accel: float = 0.3) -> SimData:
    """
    產生一段 2D 追蹤測資。
      outliers / dropout   開關：注入離群值 / 訊號遺失段
      outlier_steps        哪幾個時刻放離群值（跳 outlier_mag 公尺左右，方向隨機）
      dropout_range        遺失區段 [start, end)（時刻索引）
      dt_jitter            取樣間隔抖動比例（0.3 = 每筆間隔在 dt×(1±0.3) 之間）→ 測試可變 dt
      其餘為真值機動參數（初速、初始航向、轉彎開始時間與長度、轉彎角度、之後的加速度）
    """
    rng = np.random.default_rng(seed)
    # 亂數一次抽齊（順序固定），開關不影響抽到的值
    jit = rng.uniform(-1.0, 1.0, n)
    noise = rng.standard_normal((n, MEAS_DIM))
    out_dir = rng.uniform(0.0, 2.0 * math.pi, n)
    out_scale = rng.uniform(0.8, 1.2, n)

    dts = dt * (1.0 + dt_jitter * jit)
    dts[0] = 0.0
    t = np.cumsum(dts)

    heading0 = math.radians(heading0_deg)
    turn_rate = math.radians(turn_deg) / turn_dur
    truth = np.array([_maneuver_truth(tk, v0, heading0, turn_start, turn_dur, turn_rate, accel)
                      for tk in t])

    z_raw = truth[:, :2] + sigma_z * noise
    is_outlier = np.zeros(n, dtype=bool)
    is_dropout = np.zeros(n, dtype=bool)
    if dropout:
        a, b = dropout_range
        is_dropout[max(1, a):min(n, b)] = True
    for k in outlier_steps:
        if 1 <= k < n:
            mag = outlier_mag * out_scale[k]
            jump = mag * np.array([math.cos(out_dir[k]), math.sin(out_dir[k])])
            z_raw[k] = z_raw[k] + jump               # z_raw 永遠含離群值（畫圖範圍穩定）
            if outliers and not is_dropout[k]:
                is_outlier[k] = True

    z: List[Optional[np.ndarray]] = []
    for k in range(n):
        if is_dropout[k]:
            z.append(None)
        elif is_outlier[k]:
            z.append(z_raw[k].copy())
        else:
            z.append(truth[k, :2] + sigma_z * noise[k])
    return SimData(t=t, dt=dts, truth=truth, z=z, z_raw=z_raw,
                   is_outlier=is_outlier, is_dropout=is_dropout, sigma_z=float(sigma_z))


# =====================================================================
#  跑一整段、算指標（__main__ 與 GUI 都可以用）
# =====================================================================
@dataclass
class RunResult:
    est: np.ndarray                 # (n, 4) 每個時刻「處理完」後的狀態估計
    P: np.ndarray                   # (n, 4, 4)
    status: List[str]
    nis: np.ndarray                 # (n,) 該時刻算出的 NIS（無量測 = nan）
    K00: np.ndarray                 # (n,) 該時刻的 K[0,0]（沒更新 = nan）
    steps: List[StepResult]
    kf: KalmanFilter2D


def run_filter(sim: SimData, sigma_a: float, sigma_z: float, use_gating: bool = True,
               gate_threshold: float = GATE_99_2DOF, v_max: float = 20.0,
               max_consecutive_rejects: Optional[int] = 3) -> RunResult:
    kf = KalmanFilter2D(sigma_a, sigma_z, gate_threshold=gate_threshold, use_gating=use_gating,
                        v_max=v_max, max_consecutive_rejects=max_consecutive_rejects)
    n = sim.n
    est = np.full((n, STATE_DIM), np.nan)
    Ps = np.full((n, STATE_DIM, STATE_DIM), np.nan)
    nis = np.full(n, np.nan)
    K00 = np.full(n, np.nan)
    status: List[str] = []
    steps: List[StepResult] = []
    for k in range(n):
        res = kf.step(sim.z[k], sim.dt[k])
        steps.append(res)
        u = res.update
        status.append(u.status)
        if kf.initialized:
            est[k] = kf.x
            Ps[k] = kf.P
        if u.nis is not None:
            nis[k] = u.nis
        if u.K is not None:
            K00[k] = u.K[0, 0]
    return RunResult(est=est, P=Ps, status=status, nis=nis, K00=K00, steps=steps, kf=kf)


def rmse(err_vectors: np.ndarray) -> float:
    """err_vectors: (m, d) 的誤差向量 → sqrt(mean(‖e‖²))，自動略過 nan。"""
    e = np.asarray(err_vectors, dtype=float)
    ok = np.all(np.isfinite(e), axis=1)
    if not np.any(ok):
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(e[ok] ** 2, axis=1))))


# =====================================================================
#  命令列示範
# =====================================================================
def _disp_width(s: str) -> int:
    """終端機顯示寬度（中文字算 2 格），讓表格對齊。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _cell(s: str, width: int, right: bool = False) -> str:
    pad = " " * max(0, width - _disp_width(s))
    return pad + s if right else s + pad


def _fmt(v: float, nd: int = 2) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _plot(sim: SimData, raw: RunResult, gated: RunResult, gate: float,
          save_path: Optional[str] = None) -> None:
    try:
        import matplotlib
        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("（沒有安裝 matplotlib，略過畫圖：pip install matplotlib）")
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC",
                                       "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    k = np.arange(sim.n)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axs[0, 0]
    ax.plot(sim.truth[:, 0], sim.truth[:, 1], "--", color="#94A3B8", label="真值")
    zs = np.array([z if z is not None else [np.nan, np.nan] for z in sim.z])
    ax.plot(zs[:, 0], zs[:, 1], ".", color="#EF4444", label="量測")
    ax.plot(zs[sim.is_outlier, 0], zs[sim.is_outlier, 1], "x", color="#B91C1C", ms=10,
            label="注入的離群值")
    ax.plot(raw.est[:, 0], raw.est[:, 1], "-", color="#F59E0B", lw=1.2, label="KF 無閘門")
    ax.plot(gated.est[:, 0], gated.est[:, 1], "-", color="#2563EB", lw=2, label="KF 有閘門")
    ax.set_aspect("equal")
    ax.set_title("軌跡")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[0, 1]
    meas_err = np.linalg.norm(zs - sim.truth[:, :2], axis=1)
    ax.plot(k, meas_err, ".-", color="#EF4444", lw=0.8, label="量測誤差")
    ax.plot(k, np.linalg.norm(raw.est[:, :2] - sim.truth[:, :2], axis=1), color="#F59E0B",
            label="KF 無閘門")
    ax.plot(k, np.linalg.norm(gated.est[:, :2] - sim.truth[:, :2], axis=1), color="#2563EB",
            lw=2, label="KF 有閘門")
    ax.set_ylim(0, max(30.0, 3 * sim.sigma_z * 2))
    ax.set_title("位置誤差 (m)")
    ax.set_xlabel("時刻 k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[1, 0]
    ax.plot(k, gated.K00, "o-", color="#2563EB", ms=3, label="K[0,0]（位置）")
    ax.set_ylim(0, 1.05)
    ax.set_title("增益 K[0,0] 收斂")
    ax.set_xlabel("時刻 k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[1, 1]
    ax.semilogy(k, gated.nis, "o", color="#8B5CF6", ms=4, label="NIS")
    from matplotlib.ticker import FuncFormatter          # 刻度用 0.1、1、10，不用 10⁻¹ 的數學字
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axhline(gate, color="#EF4444", ls="--", label=f"閘門 {gate:.2f}")
    ax.axhline(MEAS_DIM, color="#22C55E", ls=":", label="理想平均 2")
    ax.set_title("NIS（對數座標）")
    ax.set_xlabel("時刻 k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"圖已存到 {save_path}")
    else:
        plt.show()


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="二維卡爾曼濾波器（工程用）示範")
    ap.add_argument("--n", type=int, default=60, help="時刻數（預設 60）")
    ap.add_argument("--dt", type=float, default=1.0, help="名目取樣間隔 s（預設 1.0）")
    ap.add_argument("--jitter", type=float, default=0.0, help="取樣間隔抖動比例（預設 0）")
    ap.add_argument("--seed", type=int, default=7, help="亂數種子（預設 7）")
    ap.add_argument("--sigma-a", type=float, default=0.8, help="濾波器 σ_a m/s²（預設 0.8）")
    ap.add_argument("--sigma-z", type=float, default=5.0, help="濾波器 σ_z m（預設 5.0）")
    ap.add_argument("--sensor-sigma", type=float, default=5.0, help="模擬感測器真實 σ m（預設 5.0）")
    ap.add_argument("--no-outliers", action="store_true", help="不注入離群值")
    ap.add_argument("--no-dropout", action="store_true", help="不模擬訊號遺失段")
    ap.add_argument("--plot", action="store_true", help="用 matplotlib 畫圖")
    ap.add_argument("--save-plot", metavar="PNG", help="把圖存成檔案（不開視窗）")
    args = ap.parse_args(argv)

    sim = simulate_2d_track(n=args.n, dt=args.dt, seed=args.seed, sigma_z=args.sensor_sigma,
                            outliers=not args.no_outliers, dropout=not args.no_dropout,
                            dt_jitter=args.jitter)
    gate = GATE_99_2DOF
    raw = run_filter(sim, args.sigma_a, args.sigma_z, use_gating=False)
    gated = run_filter(sim, args.sigma_a, args.sigma_z, use_gating=True)

    truth_p, truth_v = sim.truth[:, :2], sim.truth[:, 2:]
    have = np.array([z is not None for z in sim.z])
    zs = np.array([z if z is not None else [np.nan, np.nan] for z in sim.z])
    meas_rmse = rmse(zs - truth_p)
    meas_rmse_clean = rmse((zs - truth_p)[have & ~sim.is_outlier])
    # 差分速度：(z_k − z_{k−1}) / dt —— 「沒有濾波器時」唯一拿得到速度的方法
    diff_v = np.full((sim.n, 2), np.nan)
    for k in range(1, sim.n):
        if have[k] and have[k - 1] and sim.dt[k] > 0:
            diff_v[k] = (zs[k] - zs[k - 1]) / sim.dt[k]
    diff_v_rmse = rmse(diff_v[1:] - truth_v[1:])

    def kf_row(r: RunResult) -> Tuple[float, float, float]:
        return (rmse(r.est[:, :2] - truth_p), rmse(r.est[1:, 2:] - truth_v[1:]),
                float(np.nanmean(r.nis[np.array(r.status) == "accepted"])))

    raw_p, raw_v, raw_nis = kf_row(raw)
    g_p, g_v, g_nis = kf_row(gated)

    print("=" * 74)
    print(" 二維卡爾曼濾波器（工程用）—— 模擬測試")
    print("=" * 74)
    print(f" 時刻數 n={sim.n}、名目 dt={args.dt}s（抖動 ±{args.jitter * 100:.0f}%）、seed={args.seed}")
    print(f" 感測器真實 σ={args.sensor_sigma} m；濾波器設定 σ_a={args.sigma_a} m/s²、"
          f"σ_z={args.sigma_z} m；閘門門檻={gate:.2f}（2 自由度 99%）")
    print(f" 離群值：{int(sim.is_outlier.sum())} 筆；遺失：{int(sim.is_dropout.sum())} 個時刻")
    print("-" * 74)
    W = (34, 14, 15, 10)
    print(_cell("方法", W[0]) + _cell("位置RMSE(m)", W[1], True) + _cell("速度RMSE(m/s)", W[2], True)
          + _cell("平均NIS", W[3], True))
    print("-" * 74)
    rows = [
        ("原始量測（含離群值）", meas_rmse, float("nan"), float("nan")),
        ("原始量測（不含離群值，參考）", meas_rmse_clean, float("nan"), float("nan")),
        ("相鄰量測差分速度 Δz/dt", float("nan"), diff_v_rmse, float("nan")),
        ("KF 無閘門", raw_p, raw_v, raw_nis),
        ("KF 有閘門（工程版）", g_p, g_v, g_nis),
    ]
    for name, a, b, c in rows:
        print(_cell(name, W[0]) + _cell(_fmt(a), W[1], True) + _cell(_fmt(b), W[2], True)
              + _cell(_fmt(c), W[3], True))
    print("-" * 74)
    print(f" 有閘門：接受 {gated.kf.n_accepted}、拒絕 {gated.kf.n_rejected}、"
          f"遺失 {gated.kf.n_missing}、重新初始化 {gated.kf.n_reinit}")
    rej = [k for k, s in enumerate(gated.status) if s in ("rejected", "reinit")]
    inj = [int(k) for k in np.nonzero(sim.is_outlier)[0]]
    print(f" 被閘門拒絕的時刻：{rej}　（注入離群值的時刻：{inj}）")
    report = nis_health(gated.kf.nis_history)
    print(f" 健康監控（STEP 13）：{report.message}")
    Kss = steady_state_gain(args.dt, args.sigma_a, args.sigma_z)
    print(f" 穩態增益：K[0,0] ≈ {Kss[0, 0]:.3f}（位置修正比例 α）、"
          f"K[2,0] ≈ {Kss[2, 0]:.3f} 1/s（β/dt）")

    # 可變 dt 也要能用：同樣的機動，但取樣間隔隨機抖動 ±30%
    sim_j = simulate_2d_track(n=args.n, dt=args.dt, seed=args.seed + 1, sigma_z=args.sensor_sigma,
                              outliers=not args.no_outliers, dropout=not args.no_dropout,
                              dt_jitter=0.3)
    gj = run_filter(sim_j, args.sigma_a, args.sigma_z, use_gating=True)
    zj = np.array([z if z is not None else [np.nan, np.nan] for z in sim_j.z])
    mj, kj = rmse(zj - sim_j.truth[:, :2]), rmse(gj.est[:, :2] - sim_j.truth[:, :2])
    print(f" 可變 dt 測試（間隔 {sim_j.dt[1:].min():.2f}～{sim_j.dt[1:].max():.2f} s）："
          f"量測 RMSE {mj:.2f} m → KF {kj:.2f} m")
    print("-" * 74)

    # ---------- 自我檢查 ----------
    # 預設參數 = 回歸測試，沒過就 assert 失敗；自訂參數只印警告（某些極端設定本來就會輸）
    sym_pd = all(np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > 0)
                 for r in (raw, gated) for P in r.P)
    checks = [
        (g_p < meas_rmse, f"有閘門 KF 位置 RMSE ({g_p:.2f}) 應小於原始量測 ({meas_rmse:.2f})"),
        (not sim.is_outlier.any() or g_p <= raw_p + 1e-9,
         f"有離群值時，有閘門 ({g_p:.2f}) 應 ≤ 無閘門 ({raw_p:.2f})"),
        (g_v < diff_v_rmse, f"KF 速度 RMSE ({g_v:.2f}) 應優於相鄰量測差分 ({diff_v_rmse:.2f})"),
        (set(inj).issubset(rej), "注入的離群值應全部被閘門擋下"),
        (sym_pd, "P 必須一直保持對稱正定（Joseph form + 對稱化）"),
        (kj < mj, f"可變 dt 下 KF ({kj:.2f}) 也應優於原始量測 ({mj:.2f})"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    defaults = ap.parse_args([])
    is_default = all(getattr(args, k) == getattr(defaults, k)
                     for k in vars(args) if k not in ("plot", "save_plot"))
    if is_default:
        for ok, msg in checks:
            assert ok, msg
        print(" ✓ 自我檢查全部通過")
    elif failed:
        for msg in failed:
            print(f" ⚠ 這組參數下未通過：{msg}")
    else:
        print(" ✓ 這組參數下，自我檢查項目也都成立")

    if args.plot or args.save_plot:
        _plot(sim, raw, gated, gate, save_path=args.save_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
