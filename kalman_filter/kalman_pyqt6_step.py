# -*- coding: utf-8 -*-
"""
二維卡爾曼濾波器 逐步動畫（解說版）
=========================================
這是什麼：
  用 PyQt6 把「2D 目標追蹤卡爾曼濾波器」（kalman_2d.py 的工程版）一個時刻一個時刻畫出來，
  而且把每個時刻再拆成 7 個子步驟 —— 一次只走「一行公式」，並用白話＋當下的真實數字
  告訴你「這一行算了什麼、為什麼要算」。

核心觀念（畫面上全部看得到）：
  1. 預測（橘）：只靠等速模型往前推 → 橘點；不確定度變大 → 橘色虛線橢圓變大
  2. 量測（紅）：感測器給的位置，有雜訊、偶爾亂跳（多路徑）、有時收不到（隧道）
  3. 創新（紫）：量測 − 預測 = 這次的落差；S = 這個落差「應該」多大
  4. 閘門：NIS = 落差的「標準差數」平方，超過 9.21 就判為離群值、拒收（紅 ✕）
  5. 增益 K：誰比較準就聽誰的；K[0,0] = 0.43 → 往量測方向修正 43%
  6. 更新（藍＋綠箭頭）：估計被拉向量測，藍色實線橢圓縮小；速度也順便被「推」出來

畫面：
  上方流程條 = 公式走到第幾行；左 = 全景；中 = 放大鏡（目前時刻的細節）；
  右 = 矩陣面板（x、P、y、S、NIS、K 的實際數值，剛算出的會高亮）；
  下方三張小圖 = 位置誤差、增益 K 收斂、NIS 與閘門門檻。

操作：
  ⏮ 上一步 / ⏭ 單步（子步驟）/ ⏩ 下一時刻 / ▶ 自動播放 / ↺ 從頭重來 / ⏭| 跑到最後
  σ_a、σ_z 滑桿與三個勾選框：一改就用同一組亂數從頭重算，畫面停在同一時刻方便比較。

需求： pip install PyQt6 numpy
執行： python kalman_pyqt6_step.py   （需與 kalman_2d.py 放在同一個資料夾）
"""

import math
import os
import sys
from bisect import bisect_right

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kalman_2d import (KalmanFilter2D, chi2_threshold_2dof, nis_health,  # noqa: E402
                       rmse, simulate_2d_track, steady_state_gain)

from PyQt6.QtWidgets import (                                               # noqa: E402
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QSlider, QCheckBox,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF                         # noqa: E402
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF, QFontMetricsF  # noqa: E402


# ---------- 模擬情境（固定 seed：上一步、改參數都能完全重現） ----------
N_STEPS = 60                 # 時刻數
DT = 1.0                     # 取樣間隔 s
SEED = 7
SENSOR_SIGMA = 5.0           # 感測器「真實」雜訊 1σ（m）—— 濾波器並不知道這個值
V_MAX = 20.0                 # 初始化時的速度不確定度 1σ（m/s）
GATE = chi2_threshold_2dof(0.99)          # 9.21
SIGMA_A_DEFAULT = 0.8
SIGMA_Z_DEFAULT = 5.0

# ---------- 版面 ----------
OVER_W, ZOOM_W, MAT_W, ROW_H = 480, 300, 310, 420
GAP = 8
CONTENT_W = OVER_W + ZOOM_W + MAT_W + 2 * GAP          # 1106
CHART_H = 170

# ---------- 配色（與 Dijkstra 動畫同一套） ----------
C_PANEL_BG = QColor("#FFFFFF")
C_TEXT     = QColor("#0F172A")
C_MUTED    = QColor("#64748B")
C_GRID     = QColor("#E2E8F0")
C_BORDER   = QColor("#CBD5E1")
C_TRUTH    = QColor("#94A3B8")   # 真值（灰）
C_MEAS     = QColor("#EF4444")   # 量測（紅）
C_PRED     = QColor("#F59E0B")   # 預測（琥珀）
C_PRED_DK  = QColor("#B45309")
C_EST      = QColor("#2563EB")   # 估計（藍）
C_INNOV    = QColor("#8B5CF6")   # 創新 y（紫）
C_CORR     = QColor("#16A34A")   # 修正 K·y（綠）
C_VEL      = QColor("#0891B2")   # 速度（青）
C_TUNNEL   = QColor("#CBD5E1")   # 隧道（訊號遺失）
C_CURRENT  = QColor("#FCD34D")   # 流程條：目前這一行
C_HILITE   = QColor("#FEF3C7")   # 矩陣面板：剛算出來的
C_DONE     = QColor("#DBEAFE")   # 流程條：本時刻已做完

FONT_UI = "Microsoft JhengHei"
FONT_MONO = "Consolas"

# 子步驟：(名稱, 公式)
SUB_INFO = {
    0: ("⓪ 初始化", "x₀ = [z₀, 0]ᵀ\nP₀ = 大"),
    1: ("① 預測狀態", "x⁻ = F·x"),
    2: ("② 預測協方差", "P⁻ = F·P·Fᵀ + Q"),
    3: ("③ 創新", "y = z − H·x⁻\nS = H·P⁻·Hᵀ + R"),
    4: ("④ 閘門", f"NIS = yᵀ·S⁻¹·y\n≤ {GATE:.2f} ?"),
    5: ("⑤ 增益", "K = P⁻·Hᵀ·S⁻¹"),
    6: ("⑥ 更新狀態", "x = x⁻ + K·y"),
    7: ("⑦ 更新協方差", "P = (I−KH)P⁻(I−KH)ᵀ\n+ K·R·Kᵀ"),
}


# =========================================================
#  小工具
# =========================================================
def fmt_xy(v, nd=1, signed=False):
    f = "{:+.%df}" % nd if signed else "{:.%df}" % nd
    return "(" + f.format(v[0]) + ", " + f.format(v[1]) + ")"


def pos_sigma(P):
    """位置 1σ（x、y 兩軸變異數取平均再開根號），解說用的一個代表數字。"""
    return math.sqrt(max(0.0, 0.5 * (P[0, 0] + P[1, 1])))


def nice_step(span, n=6):
    raw = max(span, 1e-9) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def fmt_num(v, nd=2):
    """矩陣面板用：依大小自動決定小數位，避免欄位爆掉。"""
    a = abs(v)
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.{nd}f}"


def ellipse_params(cov2, nsig):
    """2×2 協方差 → (長半軸, 短半軸, 長軸角度°)。特徵值 = 主軸方向的變異數。"""
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.clip(vals, 0.0, None)
    a = nsig * math.sqrt(vals[1])
    b = nsig * math.sqrt(vals[0])
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    return a, b, ang


class WorldView:
    """世界座標（公尺，y 向上）↔ 像素（y 向下）。"""
    def __init__(self, rect, cx, cy, scale):
        self.rect, self.cx, self.cy, self.s = rect, cx, cy, scale

    def pt(self, x, y):
        c = self.rect.center()
        return QPointF(c.x() + (x - self.cx) * self.s, c.y() - (y - self.cy) * self.s)

    def bounds(self):
        hw, hh = self.rect.width() / 2 / self.s, self.rect.height() / 2 / self.s
        return self.cx - hw, self.cx + hw, self.cy - hh, self.cy + hh


def draw_grid(p, v, step, labels=True):
    x0, x1, y0, y1 = v.bounds()
    r = v.rect
    p.setPen(QPen(C_GRID, 1))
    p.setFont(QFont(FONT_UI, 7))
    gx = math.ceil(x0 / step) * step
    while gx <= x1:
        q = v.pt(gx, 0)
        p.setPen(QPen(C_GRID, 1))
        p.drawLine(QPointF(q.x(), r.top()), QPointF(q.x(), r.bottom()))
        if labels:
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(q.x() - 24, r.bottom() + 2, 48, 14),
                       Qt.AlignmentFlag.AlignCenter, f"{gx:g}")
        gx += step
    gy = math.ceil(y0 / step) * step
    while gy <= y1:
        q = v.pt(0, gy)
        p.setPen(QPen(C_GRID, 1))
        p.drawLine(QPointF(r.left(), q.y()), QPointF(r.right(), q.y()))
        if labels:
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(r.left() - 40, q.y() - 7, 36, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{gy:g}")
        gy += step


def draw_arrow(p, a, b, color, width=2.0, head=9.0, dashed=False):
    dx, dy = b.x() - a.x(), b.y() - a.y()
    L = math.hypot(dx, dy)
    if L < 1.0:
        return
    ux, uy = dx / L, dy / L
    head = min(head, L * 0.6)
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(a, QPointF(b.x() - ux * head * 0.7, b.y() - uy * head * 0.7))
    left = QPointF(b.x() - ux * head - uy * head * 0.45, b.y() - uy * head + ux * head * 0.45)
    right = QPointF(b.x() - ux * head + uy * head * 0.45, b.y() - uy * head - ux * head * 0.45)
    p.setPen(QPen(color, 1))
    p.setBrush(QBrush(color))
    p.drawPolygon(QPolygonF([b, left, right]))


def draw_xmark(p, c, r, color, width=2.5):
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(c.x() - r, c.y() - r), QPointF(c.x() + r, c.y() + r))
    p.drawLine(QPointF(c.x() - r, c.y() + r), QPointF(c.x() + r, c.y() - r))


def draw_dot(p, c, r, fill, edge=None, width=1.2):
    p.setPen(QPen(edge or fill.darker(140), width))
    p.setBrush(QBrush(fill))
    p.drawEllipse(c, r, r)


def draw_diamond(p, c, r, color):
    p.setPen(QPen(color.darker(130), 1.5))
    p.setBrush(QBrush(QColor(255, 255, 255, 200)))
    p.drawPolygon(QPolygonF([QPointF(c.x(), c.y() - r), QPointF(c.x() + r, c.y()),
                             QPointF(c.x(), c.y() + r), QPointF(c.x() - r, c.y())]))


def draw_cov_ellipse(p, v, mean, cov2, nsig, color, dashed=False, fill_alpha=28,
                     width=2.0, min_px=3.0, dotted=False):
    """畫 nsig·σ 的不確定度橢圓。太小時用最小可視半徑 min_px 畫，回傳 True 代表被放大過。"""
    a, b, ang = ellipse_params(cov2, nsig)
    ap, bp = a * v.s, b * v.s
    clamped = ap < min_px
    ap, bp = max(ap, min_px), max(bp, min_px)
    c = v.pt(mean[0], mean[1])
    pen = QPen(color, width)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    if dotted:
        pen.setStyle(Qt.PenStyle.DotLine)
    fill = QColor(color)
    fill.setAlpha(fill_alpha)
    p.save()
    p.translate(c)
    p.rotate(-ang)
    p.setPen(pen)
    p.setBrush(QBrush(fill) if fill_alpha > 0 else Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(0, 0), ap, bp)
    p.restore()
    return clamped


def draw_tag(p, pt, text, color, dx=7, dy=-7, size=8, bold=True, bg_alpha=215, center=False):
    """在點旁邊寫小標籤（白底，避免被線條蓋住看不清楚）；center=True 時以 pt 為中心。"""
    font = QFont(FONT_UI, size)
    if bold:
        font.setWeight(QFont.Weight.Bold)
    p.setFont(font)
    fm = QFontMetricsF(font)
    w, h = fm.horizontalAdvance(text) + 6, fm.height() + 1
    if center:
        x, y = pt.x() - w / 2, pt.y() - h / 2
    else:
        x = pt.x() + dx if dx >= 0 else pt.x() + dx - w
        y = pt.y() + dy - h / 2
    rect = QRectF(x, y, w, h)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(255, 255, 255, bg_alpha)))
    p.drawRoundedRect(rect, 3, 3)
    p.setPen(QPen(color))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def frame_panel(p, w, h, title=None):
    p.setPen(QPen(C_BORDER, 1))
    p.setBrush(QBrush(C_PANEL_BG))
    p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 10, 10)
    if title:
        f = QFont(FONT_UI, 10)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        p.setPen(QPen(QColor("#1D4ED8")))
        p.drawText(QRectF(10, 4, w - 20, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   title)


# =========================================================
#  卡爾曼「逐步揭露」引擎（含「為何走這一步」的解說）
# =========================================================
class KalmanStepper:
    """
    整段 60 個時刻先用 kalman_2d.KalmanFilter2D 真的算一遍（每一步的中間量都在 StepResult 裡），
    再依「子步驟」一格一格揭露。frame = 已揭露的子步驟數；
    上一步 = 從頭重播 frame−1 步（固定 seed，完全可重現）。
    """

    def __init__(self):
        self.sigma_a = SIGMA_A_DEFAULT
        self.sigma_z = SIGMA_Z_DEFAULT
        self.use_gating = True
        self.outliers = True
        self.dropout = True
        self.recompute()
        self.reset()

    # ---------- 用函式庫把整段算好 ----------
    def recompute(self):
        sim = simulate_2d_track(n=N_STEPS, dt=DT, seed=SEED, sigma_z=SENSOR_SIGMA,
                                outliers=self.outliers, dropout=self.dropout)
        kf = KalmanFilter2D(self.sigma_a, self.sigma_z, gate_threshold=GATE,
                            use_gating=self.use_gating, v_max=V_MAX)
        self.sim = sim
        self.n = sim.n
        self.recs = [kf.step(sim.z[k], sim.dt[k]) for k in range(sim.n)]
        if self.recs[0].update.status != "init":
            raise RuntimeError("模擬的第一筆必須有量測（用來初始化）")
        self.kf = kf
        self.subs = [[0]] + [self._subs_of(k) for k in range(1, self.n)]
        self.schedule = [(k, s) for k in range(self.n) for s in self.subs[k]]
        self.K_ss = steady_state_gain(DT, self.sigma_a, self.sigma_z)

        truth = sim.truth
        self.meas_err = [None if z is None else float(np.linalg.norm(z - truth[k, :2]))
                         for k, z in enumerate(sim.z)]
        self.est_err = [float(np.linalg.norm(r.update.x_post[:2] - truth[k, :2]))
                        for k, r in enumerate(self.recs)]
        self.prior_err = [None if r.predict is None else
                          float(np.linalg.norm(r.predict.x_prior[:2] - truth[k, :2]))
                          for k, r in enumerate(self.recs)]
        self.zoom = [self._zoom_for(k) for k in range(self.n)]
        pts = np.vstack([truth[:, :2], sim.z_raw])       # 不受濾波參數影響 → 改參數畫面不跳
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 12.0
        self.bbox = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)

    def _subs_of(self, k):
        st = self.recs[k].update.status
        subs = [1, 2, 3]
        if st == "missing":
            return subs
        subs.append(4)
        if st == "accepted":
            subs += [5, 6, 7]
        return subs

    def _zoom_for(self, k):
        """放大鏡的中心與半寬：把本時刻會畫的東西（橢圓、量測、真值、速度箭頭）全部框進來。"""
        r = self.recs[k]
        u = r.update
        truth = self.sim.truth[k]

        def rad(P2, nsig=2.3):
            return nsig * math.sqrt(max(0.0, np.linalg.eigvalsh(P2).max()))

        items = [(truth[:2], 4.0), (truth[:2] + truth[2:], 2.0)]       # (中心, 半徑)
        if r.predict is None:
            items += [(u.x_post[:2], rad(u.P_post[:2, :2]))]
        else:
            pr = r.predict
            items += [(pr.x_prior[:2], rad(pr.P_prior[:2, :2])), (pr.x_prev[:2], rad(pr.P_prev[:2, :2])),
                      (pr.x_prior[:2] + pr.x_prior[2:], 2.0),
                      (u.x_post[:2], rad(u.P_post[:2, :2])), (u.x_post[:2] + u.x_post[2:], 2.0)]
            if u.S is not None:
                # 量測與 S 的 2σ 橢圓要看得到；閘門橢圓（約 3σ）比較大，不硬塞進畫面，超出就讓它被裁切
                items += [(u.z, 4.0), (pr.x_prior[:2], 2.0 * math.sqrt(np.linalg.eigvalsh(u.S).max()))]
        lo = np.min([c - r_ for c, r_ in items], axis=0)
        hi = np.max([c + r_ for c, r_ in items], axis=0)
        center = (lo + hi) / 2
        aspect = ZoomWidget.PLOT.width() / ZoomWidget.PLOT.height()
        half = max((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2 * aspect) * 1.06
        return center, float(np.clip(half, 10.0, 400.0))

    # ---------- 進度 ----------
    def reset(self):
        self.frame = 0
        self.explain = ("還沒開始。灰色虛線是目標的真實路徑（濾波器看不到，只是給我們對答案），"
                        "目標會：直線 → 右轉 90° → 加速，途中有 3 筆多路徑跳點、還會進隧道收不到訊號。"
                        "濾波器手上只有紅色的雜訊量測。按「⏭ 單步」開始：先用第一筆量測初始化。")

    @property
    def total(self):
        return len(self.schedule)

    @property
    def done(self):
        return self.frame >= self.total

    def cur(self):
        return self.schedule[self.frame - 1] if self.frame > 0 else (-1, -1)

    def last_sub(self, k):
        return self.subs[k][-1]

    def seen(self, k, need):
        ck, cs = self.cur()
        return k < ck or (k == ck and cs >= need)

    def complete(self, k):
        return self.seen(k, self.last_sub(k))

    def outcome_seen(self, k):
        """這個時刻「接受 / 拒絕 / 遺失」的結果揭曉了沒。"""
        if k == 0:
            return self.seen(0, 0)
        return self.seen(k, 3 if self.recs[k].update.status == "missing" else 4)

    def step(self, explain=True):
        if self.done:
            return False
        self.frame += 1
        if explain:
            self.refresh_explain()
        return True

    def refresh_explain(self):
        if self.frame == 0:
            self.reset()
            return
        k, sub = self.cur()
        self.explain = self.explain_for(k, sub)
        if self.done:
            self.explain += "　" + self.final_summary()

    # ---------- 目前要顯示的 x、P ----------
    def shown_state(self):
        """回傳 (x, P, x 的標題, P 的標題)，依子步驟決定顯示預測前 / 預測後 / 更新後。"""
        k, sub = self.cur()
        r = self.recs[k]
        u = r.update
        if sub == 0:
            return u.x_post, u.P_post, "x₀ 初始狀態（⓪）", "P₀ 初始協方差（⓪：位置 σ_z²、速度 v_max²）"
        pr = r.predict
        if u.status == "reinit" and sub >= 4:
            return u.x_post, u.P_post, "x 重新初始化後", "P 重新初始化（回到大不確定）"
        if sub >= 6:
            x, xl = u.x_post, "x 更新後狀態（⑥ x = x⁻ + K·y）"
        else:
            x, xl = pr.x_prior, "x⁻ 預測狀態（① x⁻ = F·x）"
        if sub >= 7:
            P, Pl = u.P_post, "P 更新後協方差（⑦ Joseph）"
        elif sub >= 2:
            P, Pl = pr.P_prior, "P⁻ 預測協方差（② F·P·Fᵀ + Q）"
        else:
            P, Pl = pr.P_prev, "P 上一時刻的（還沒做 ②）"
        return x, P, xl, Pl

    def highlight(self):
        k, sub = self.cur()
        if k < 0:
            return set()
        st = self.recs[k].update.status
        if sub == 0:
            return {"x", "P"}
        if sub == 4 and st == "reinit":
            return {"nis", "x", "P"}
        return {1: {"x"}, 2: {"P"}, 3: {"yS"} if st != "missing" else set(),
                4: {"nis"}, 5: {"K"}, 6: {"x"}, 7: {"P"}}[sub]

    # ---------- 解說（帶入真實數字） ----------
    def _missing_streak(self, k):
        n = 0
        while k >= 1 and self.recs[k].update.status == "missing":
            n += 1
            k -= 1
        return n

    def explain_for(self, k, sub):
        sim = self.sim
        r = self.recs[k]
        u = r.update
        R = self.sigma_z ** 2
        truth = sim.truth[k]
        if sub == 0:
            z = u.z
            return (f"收到第一筆量測 z₀ = {fmt_xy(z)} m，拿來初始化（STEP 6）：位置先相信它，"
                    f"不確定度 = 感測器雜訊 σ_z² = {R:.1f} m²（±{self.sigma_z:.1f} m）；"
                    f"速度根本沒量到 → 先填 0，但給超大的不確定度 v_max² = {V_MAX ** 2:.0f}"
                    f"（＝「速度在 ±{V_MAX:.0f} m/s 之間都有可能」，大 P 就是「我不知道」）。"
                    f"藍色實線橢圓 = 2σ 範圍。之後每個時刻都照上方流程條 ①～⑦ 一行一行走。")
        pr = r.predict
        dt = pr.dt
        if sub == 1:
            x0, x1 = pr.x_prev, pr.x_prior
            return (f"時刻 k={k}（距上一筆 dt = {dt:.1f} s）。先不看量測，純用等速模型往前推："
                    f"位置 + 速度×dt = {fmt_xy(x0[:2])} + {fmt_xy(x0[2:], 2)}×{dt:.1f} = "
                    f"{fmt_xy(x1[:2])}（橘點 x⁻），速度 {fmt_xy(x1[2:], 2)} m/s 維持不變。"
                    f"x⁻ = F·x 只是把「位置 + 速度×dt」寫成矩陣。"
                    f"（對答案：這個預測離真值 {self.prior_err[k]:.1f} m。）")
        if sub == 2:
            s0, s1 = pos_sigma(pr.P_prev), pos_sigma(pr.P_prior)
            sv = math.sqrt(max(0.0, 0.5 * (pr.P_prev[2, 2] + pr.P_prev[3, 3])))
            streak = self._missing_streak(k - 1)
            extra = (f"已經連續 {streak} 個時刻沒量測，橢圓會一直長大。" if streak else
                     "沒有新資訊，就只會更不確定。")
            return (f"預測也讓不確定度變大：P⁻ = F·P·Fᵀ + Q。F·P·Fᵀ 把「速度的不確定（±{sv:.2f} m/s）」"
                    f"經過 dt 灌進位置；+Q 再加上「這 {dt:.1f} 秒內目標可能偷偷加速」"
                    f"（σ_a = {self.sigma_a:.2f} m/s²）。位置 1σ：{s0:.2f} m → {s1:.2f} m，"
                    f"橘色虛線橢圓比上一刻的大一圈。{extra}")
        if sub == 3:
            if u.status == "missing":
                s1 = pos_sigma(pr.P_prior)
                return (f"時刻 k={k} 沒有收到量測（目標在隧道裡，訊號遺失）→ STEP 12：沒資料就沒得修正，"
                        f"只做預測：x = x⁻、P = P⁻。已連續 {self._missing_streak(k)} 個時刻沒量測，"
                        f"位置 1σ 長到 {s1:.1f} m —— 橢圓越來越大，就是濾波器誠實地說"
                        f"「我越來越不確定」。本時刻結束（④～⑦ 略過）。")
            y, S = u.y, u.S
            p_avg = 0.5 * (pr.P_prior[0, 0] + pr.P_prior[1, 1])
            s_avg = 0.5 * (S[0, 0] + S[1, 1])
            return (f"收到量測 z = {fmt_xy(u.z)}（紅點）。創新 y = z − H·x⁻ = {fmt_xy(y, 1, True)} m"
                    f"（紫箭頭，長 {np.linalg.norm(y):.1f} m）＝ 量測比預測偏了多少。"
                    f"S = H·P⁻·Hᵀ + R ＝ 這個落差「應該」多大：預測的不確定 {p_avg:.1f} + "
                    f"感測器 R {R:.1f} ≈ {s_avg:.1f} m²（約 ±{math.sqrt(s_avg):.1f} m，紫色虛線 = 2σ）。"
                    f"落差 {np.linalg.norm(y):.1f} m 算不算大？下一步用 S 來量。")
        if sub == 4:
            nis = u.nis
            base = (f"閘門檢查：NIS = yᵀ·S⁻¹·y = {nis:.2f}（落差約是 {math.sqrt(nis):.1f} 個標準差，"
                    f"平方後就是 NIS）。門檻 {GATE:.2f} = 2 自由度卡方的 99% 分位："
                    f"正常量測只有 1% 機率超過。")
            outlier = bool(sim.is_outlier[k])
            if u.status == "accepted" and self.use_gating:
                return base + f"{nis:.2f} ≤ {GATE:.2f} → 合理，接受這筆量測（紫色點線 = 閘門範圍，量測在圈內）。"
            if u.status == "accepted":
                if nis > GATE:
                    return base + (f"{nis:.2f} > {GATE:.2f}！但閘門已關閉 → 照單全收。"
                                   + ("這筆正是注入的多路徑跳點，" if outlier else "")
                                   + "估計會被硬拉過去 —— 看接下來的綠箭頭。")
                return base + f"{nis:.2f} ≤ {GATE:.2f}，就算閘門開著也會接受（閘門目前關閉）。"
            if u.status == "reinit":
                return base + (f"{nis:.2f} > {GATE:.2f}，而且已是連續第 {u.consecutive_rejects} 次被拒 → "
                               f"比較可能是濾波器自己跟丟了，不是量測全錯。保險絲啟動：用這筆量測重新初始化"
                               f"（位置 = z、P 回到初始的大不確定）。本時刻結束。")
            text = base + f"{nis:.2f} > {GATE:.2f} → 判為離群值，丟掉不用：x = x⁻、P = P⁻（紅 ✕）。"
            if outlier:
                text += "（對答案：這筆的確是注入的多路徑跳點 ✓）"
            else:
                text += ("（對答案：這筆其實是正常量測，被誤殺了 —— 通常代表 σ_a 太小，"
                         "濾波器在轉彎時跟不上、預測偏太遠。）")
            if u.consecutive_rejects >= 2:
                text += f"已連續被拒 {u.consecutive_rejects} 次。"
            return text + "本時刻結束（⑤～⑦ 略過）。"
        if sub == 5:
            K = u.K
            p00 = pr.P_prior[0, 0]
            better = "預測比較不準 → 多聽量測" if K[0, 0] > 0.5 else "量測比較不準 → 多聽預測"
            return (f"算卡爾曼增益 K = P⁻·Hᵀ·S⁻¹（4×2 矩陣）。K[0,0] = {K[0, 0]:.2f} 代表這次 x 位置會往"
                    f"量測方向修正 {K[0, 0] * 100:.0f}%：預測 ±{math.sqrt(p00):.1f} m vs 量測 "
                    f"±{self.sigma_z:.1f} m，{better}（1D 直覺：K = P⁻/(P⁻+R) = {p00:.1f}/({p00:.1f}+{R:.1f})"
                    f" = {p00 / (p00 + R):.2f}）。K[2,0] = {K[2, 0]:.3f}：x 方向每偏 1 m，x 速度就修正 "
                    f"{K[2, 0]:.3f} m/s —— 量不到的速度就是這樣被「推」出來的。")
        if sub == 6:
            d = u.x_post - pr.x_prior
            return (f"更新狀態 x = x⁻ + K·y：位置修正 {fmt_xy(d[:2], 2, True)} m（綠箭頭），"
                    f"速度修正 {fmt_xy(d[2:], 2, True)} m/s。新的最佳估計（藍點）= {fmt_xy(u.x_post[:2])}，"
                    f"速度 {fmt_xy(u.x_post[2:], 2)} m/s（真值 {fmt_xy(truth[2:], 2)}）。"
                    f"對答案：位置誤差 預測 {self.prior_err[k]:.1f} m → 更新後 {self.est_err[k]:.1f} m"
                    f"（量測本身誤差 {self.meas_err[k]:.1f} m）。")
        # sub == 7
        s1, s2 = pos_sigma(pr.P_prior), pos_sigma(u.P_post)
        tail = f"時刻 k={k} 完成，下一步進入 k={k + 1}。" if k + 1 < self.n else f"時刻 k={k} 完成。"
        return (f"更新協方差（Joseph 形式）P = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ：吸收了量測的資訊，"
                f"不確定度變小 —— 位置 1σ {s1:.2f} → {s2:.2f} m（藍實線橢圓比橘虛線小）。"
                f"用 Joseph 形式而不用 (I−KH)·P⁻，是為了保證 P 永遠對稱、正定，長時間跑也不會發散。{tail}")

    def summary_for(self, k):
        """「⏩ 下一時刻」一次走完時的摘要。"""
        if k == 0:
            return self.explain_for(0, 0)
        r = self.recs[k]
        u, pr = r.update, r.predict
        head = f"時刻 k={k} 一次走完：預測到 {fmt_xy(pr.x_prior[:2])}（1σ {pos_sigma(pr.P_prior):.1f} m）；"
        if u.status == "missing":
            body = "沒有量測（隧道）→ 只預測，不確定度一路變大。"
        elif u.status == "rejected":
            body = (f"量測 NIS = {u.nis:.1f} > {GATE:.2f} 被閘門拒絕 → 維持預測"
                    + ("（的確是離群值 ✓）。" if self.sim.is_outlier[k] else "（其實是好量測，被誤殺）。"))
        elif u.status == "reinit":
            body = f"量測連續第 {u.consecutive_rejects} 次被拒 → 保險絲啟動，用這筆量測重新初始化。"
        else:
            body = (f"量測 NIS = {u.nis:.2f}" + ("（閘門關閉，照收）" if u.nis > GATE else " 通過閘門")
                    + f"，K[0,0] = {u.K[0, 0]:.2f} → 往量測修正 {u.K[0, 0] * 100:.0f}%；"
                    f"位置誤差 量測 {self.meas_err[k]:.1f} m → 估計 {self.est_err[k]:.1f} m，"
                    f"1σ 縮到 {pos_sigma(u.P_post):.1f} m。")
        return head + body + "（想看每一行公式，用「⏭ 單步」。）"

    def run_metrics(self):
        sim = self.sim
        tp, tv = sim.truth[:, :2], sim.truth[:, 2:]
        zs = np.array([z if z is not None else [np.nan, np.nan] for z in sim.z])
        est = np.array([r.update.x_post for r in self.recs])
        diff_v = np.full((self.n, 2), np.nan)
        for k in range(1, self.n):
            if sim.z[k] is not None and sim.z[k - 1] is not None:
                diff_v[k] = (zs[k] - zs[k - 1]) / sim.dt[k]
        return (rmse(zs - tp), rmse(est[:, :2] - tp), rmse(diff_v[1:] - tv[1:]),
                rmse(est[1:, 2:] - tv[1:]))

    def final_summary(self):
        m, g, dv, kv = self.run_metrics()
        rep = nis_health(self.kf.nis_history)
        text = (f"🏁 全部 {self.n} 個時刻跑完！位置 RMSE：原始量測 {m:.2f} m → 卡爾曼 {g:.2f} m；"
                f"速度 RMSE：相鄰量測差分 {dv:.2f} m/s → 卡爾曼 {kv:.2f} m/s（速度從頭到尾沒有感測器在量）。"
                f"健康監控：{rep.message}。")
        if g >= m:
            text += ("⚠ 濾波後反而比原始量測差 —— 這組參數沒調好：σ_a 太小會跟不上轉彎、被閘門誤殺；"
                     "σ_z 太小會太相信雜訊、也一直被閘門拒絕。按「↺ 預設參數」比較看看。")
        return text


# =========================================================
#  流程條：公式走到第幾行
# =========================================================
class FlowBar(QWidget):
    H = 66
    K_W = 84
    INIT_W = 96
    SP = 8

    def __init__(self, eng):
        super().__init__()
        self.eng = eng
        self.setFixedSize(CONTENT_W, self.H)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        eng = self.eng
        k, sub = eng.cur()

        # 最左：目前時刻
        box = QRectF(1, 1, self.K_W, self.H - 2)
        p.setPen(QPen(QColor("#1D4ED8"), 1.5))
        p.setBrush(QBrush(QColor("#EFF6FF")))
        p.drawRoundedRect(box, 8, 8)
        f = QFont(FONT_UI, 15)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        p.setPen(QPen(QColor("#1D4ED8")))
        p.drawText(QRectF(box.x(), box.y() + 4, box.width(), 30), Qt.AlignmentFlag.AlignCenter,
                   "k = —" if k < 0 else f"k = {k}")
        p.setFont(QFont(FONT_UI, 8))
        p.setPen(QPen(C_MUTED))
        t_txt = "尚未開始" if k < 0 else f"t = {eng.sim.t[k]:.1f} s"
        p.drawText(QRectF(box.x(), box.y() + 34, box.width(), 22), Qt.AlignmentFlag.AlignCenter, t_txt)

        x = box.right() + self.SP
        box_w = (CONTENT_W - x - self.INIT_W - 7 * self.SP) / 7.0
        st = eng.recs[k].update.status if k >= 1 else None
        skip_reason = {"missing": "無量測 → 略過", "rejected": "被拒 → 略過",
                       "reinit": "重新初始化 → 略過"}.get(st, "")
        for s in range(8):
            w = self.INIT_W if s == 0 else box_w
            rect = QRectF(x, 1, w, self.H - 2)
            # 狀態：current / done / skipped / pending
            if s == 0:
                state = "current" if k == 0 else ("done" if k > 0 else "pending")
            elif k <= 0:
                state = "pending"
            elif s not in eng.subs[k]:
                state = "skipped"
            elif s == sub:
                state = "current"
            elif s < sub:
                state = "done"
            else:
                state = "pending"
            fill, edge, tcol = {
                "current": (C_CURRENT, QColor("#F59E0B"), C_TEXT),
                "done":    (C_DONE, QColor("#93C5FD"), QColor("#1E3A8A")),
                "skipped": (QColor("#F1F5F9"), QColor("#E2E8F0"), QColor("#94A3B8")),
                "pending": (QColor("#FFFFFF"), C_BORDER, QColor("#334155")),
            }[state]
            p.setPen(QPen(edge, 2.5 if state == "current" else 1))
            p.setBrush(QBrush(fill))
            p.drawRoundedRect(rect, 8, 8)
            name, formula = SUB_INFO[s]
            fn = QFont(FONT_UI, 9)
            fn.setWeight(QFont.Weight.Bold)
            p.setFont(fn)
            p.setPen(QPen(tcol))
            p.drawText(QRectF(rect.x() + 4, rect.y() + 3, rect.width() - 8, 18),
                       Qt.AlignmentFlag.AlignCenter, name)
            if state == "skipped":
                p.setFont(QFont(FONT_UI, 8))
                p.setPen(QPen(QColor("#94A3B8")))
                p.drawText(QRectF(rect.x() + 2, rect.y() + 22, rect.width() - 4, 38),
                           Qt.AlignmentFlag.AlignCenter, skip_reason)
            else:
                p.setFont(QFont(FONT_MONO, 8))
                p.setPen(QPen(tcol))
                p.drawText(QRectF(rect.x() + 2, rect.y() + 21, rect.width() - 4, 40),
                           Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, formula)
            # 方塊之間的小箭頭
            if s < 7:
                p.setPen(QPen(QColor("#94A3B8"), 1.5))
                mx, my = rect.right() + self.SP / 2, rect.center().y()
                p.drawLine(QPointF(mx - 2, my - 4), QPointF(mx + 2, my))
                p.drawLine(QPointF(mx + 2, my), QPointF(mx - 2, my + 4))
            x += w + self.SP
        p.end()


# =========================================================
#  全景：整條軌跡
# =========================================================
class OverviewWidget(QWidget):
    def __init__(self, eng):
        super().__init__()
        self.eng = eng
        self.setFixedSize(OVER_W, ROW_H)
        self.plot = QRectF(44, 30, OVER_W - 44 - 12, ROW_H - 30 - 24)

    def view(self):
        x0, x1, y0, y1 = self.eng.bbox
        s = min(self.plot.width() / (x1 - x0), self.plot.height() / (y1 - y0))
        return WorldView(self.plot, (x0 + x1) / 2, (y0 + y1) / 2, s)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        eng, sim = self.eng, self.eng.sim
        v = self.view()
        step = nice_step(v.bounds()[1] - v.bounds()[0], 7)
        frame_panel(p, OVER_W, ROW_H, f"🗺 全景（格線 {step:g} m）")
        draw_grid(p, v, step)
        p.setClipRect(self.plot)
        ck, cs = eng.cur()

        # 隧道（訊號遺失段）：沿真值路徑的灰色粗帶
        if sim.is_dropout.any():
            ks = np.nonzero(sim.is_dropout)[0]
            a, b = max(0, ks[0] - 1), min(eng.n - 1, ks[-1] + 1)
            pen = QPen(QColor(203, 213, 225, 170), 16)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([v.pt(*sim.truth[i, :2]) for i in range(a, b + 1)]))
            mid = v.pt(*sim.truth[(a + b) // 2, :2])
            draw_tag(p, mid, f"隧道：k={ks[0]}–{ks[-1]} 收不到訊號", QColor("#475569"), dx=14, dy=-14,
                     size=8)

        # 真值軌跡（灰虛線）
        pen = QPen(C_TRUTH, 1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(QPolygonF([v.pt(*sim.truth[i, :2]) for i in range(eng.n)]))
        draw_tag(p, v.pt(*sim.truth[0, :2]), "起點", C_MUTED, dx=8, dy=12, size=8)

        # 已揭露的量測
        for k in range(eng.n):
            z = sim.z[k]
            if z is None or not eng.seen(k, 0 if k == 0 else 3):
                continue
            c = v.pt(*z)
            st = eng.recs[k].update.status
            if st in ("rejected", "reinit") and eng.seen(k, 4):
                draw_xmark(p, c, 5, C_MEAS, 2.4)
            else:
                draw_dot(p, c, 3.2, C_MEAS)
                if sim.is_outlier[k] and st == "accepted" and eng.seen(k, 4):
                    p.setPen(QPen(C_MEAS, 1.5))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(c, 7, 7)

        # 估計軌跡（藍線）
        pts = [v.pt(*eng.recs[k].update.x_post[:2]) for k in range(eng.n)
               if eng.complete(k) or (k == ck and cs >= 6)]
        if len(pts) >= 2:
            pen = QPen(C_EST, 2.2)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QPolygonF(pts))

        if ck >= 0:
            self._current(p, v, ck, cs)
            # 放大鏡範圍
            c, half = eng.zoom[ck]
            zv = ZoomWidget.make_view(c, half)
            zx0, zx1, zy0, zy1 = zv.bounds()
            a, b = v.pt(zx0, zy1), v.pt(zx1, zy0)
            pen = QPen(QColor("#7C3AED"), 1.2)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(a, b))
            draw_tag(p, QPointF(b.x(), a.y()), "🔍", QColor("#7C3AED"), dx=2, dy=-2, size=8,
                     bold=False)
        p.setClipping(False)
        p.end()

    def _current(self, p, v, k, sub):
        eng = self.eng
        r = eng.recs[k]
        u = r.update
        draw_diamond(p, v.pt(*eng.sim.truth[k, :2]), 5, C_TRUTH)
        if k == 0:
            draw_cov_ellipse(p, v, u.x_post[:2], u.P_post[:2, :2], 2, C_EST)
            draw_dot(p, v.pt(*u.x_post[:2]), 4.5, C_EST)
            return
        pr = r.predict
        prior = v.pt(*pr.x_prior[:2])
        if sub >= 2 and not (u.status == "reinit" and sub >= 4):
            draw_cov_ellipse(p, v, pr.x_prior[:2], pr.P_prior[:2, :2], 2, C_PRED, dashed=True)
        if sub >= 3 and u.z is not None:
            draw_arrow(p, prior, v.pt(*u.z), C_INNOV, 1.8, 7)
        if sub >= 6:
            draw_arrow(p, prior, v.pt(*u.x_post[:2]), C_CORR, 2.2, 7)
        draw_dot(p, prior, 4.2, C_PRED, C_PRED_DK)
        if sub >= 7 or (u.status == "reinit" and sub >= 4):
            draw_cov_ellipse(p, v, u.x_post[:2], u.P_post[:2, :2], 2, C_EST)
        if sub >= 6 or (u.status == "reinit" and sub >= 4):
            q = v.pt(*u.x_post[:2])
            draw_dot(p, q, 4.5, C_EST)
            vel = u.x_post[2:]
        else:
            q, vel = prior, pr.x_prior[2:]
        tip = QPointF(q.x() + vel[0] * 5 * v.s, q.y() - vel[1] * 5 * v.s)   # 畫「5 秒位移」才看得見
        draw_arrow(p, q, tip, C_VEL, 2.0, 7)


# =========================================================
#  放大鏡：目前時刻的細節（橢圓、創新、閘門、修正）
# =========================================================
class ZoomWidget(QWidget):
    PLOT = QRectF(8, 30, ZOOM_W - 16, ROW_H - 30 - 48)

    def __init__(self, eng):
        super().__init__()
        self.eng = eng
        self.setFixedSize(ZOOM_W, ROW_H)

    @classmethod
    def make_view(cls, center, half):
        s = (cls.PLOT.width() / 2) / half
        return WorldView(cls.PLOT, center[0], center[1], s)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        eng = self.eng
        k, sub = eng.cur()
        frame_panel(p, ZOOM_W, ROW_H, "🔍 放大鏡" if k < 0 else f"🔍 放大鏡：時刻 k = {k} 的細節")
        if k < 0:
            p.setFont(QFont(FONT_UI, 10))
            p.setPen(QPen(C_MUTED))
            p.drawText(self.PLOT, Qt.AlignmentFlag.AlignCenter,
                       "按「⏭ 單步」開始\n\n這裡會放大顯示\n目前時刻的每一個細節")
            p.end()
            return
        c, half = eng.zoom[k]
        v = self.make_view(c, half)
        step = nice_step(2 * half, 5)
        p.save()
        p.setClipRect(self.PLOT)
        p.setPen(QPen(C_BORDER, 1))
        p.setBrush(QBrush(QColor("#FCFCFD")))
        p.drawRect(self.PLOT)
        draw_grid(p, v, step, labels=False)
        clamped = self._draw(p, v, k, sub)
        self._info_box(p, k, sub)
        p.restore()

        # 底部第 1 行：比例尺（長度 ≤ 70 px）+ 註記
        y = self.PLOT.bottom() + 6
        bar = step
        while bar * v.s > 70:
            bar = nice_step(bar * 0.99, 2)
        L = bar * v.s
        p.setPen(QPen(C_TEXT, 2))
        p.drawLine(QPointF(14, y + 6), QPointF(14 + L, y + 6))
        p.drawLine(QPointF(14, y + 1), QPointF(14, y + 11))
        p.drawLine(QPointF(14 + L, y + 1), QPointF(14 + L, y + 11))
        p.setFont(QFont(FONT_UI, 8))
        p.drawText(QRectF(20 + L, y - 1, 50, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{bar:g} m")
        p.setPen(QPen(C_MUTED))
        note = "橢圓 2σ" + ("（過小，以最小尺寸畫）" if clamped else "；青箭頭 = 1 秒位移")
        room = ZOOM_W - 12 - (66 + L)
        note = QFontMetricsF(p.font()).elidedText(note, Qt.TextElideMode.ElideRight, room)
        p.drawText(QRectF(66 + L, y - 1, room, 14),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, note)
        # 底部第 2 行：小圖例
        y2 = y + 23
        x = 14
        p.setFont(QFont(FONT_UI, 8))
        items = [("diamond", C_TRUTH, "真值"), ("hollow", C_EST, "上一刻"),
                 ("dot", C_PRED, "x⁻ 預測"), ("dot", C_MEAS, "z 量測"), ("dot", C_EST, "x 更新")]
        fm = QFontMetricsF(p.font())
        for kind, color, text in items:
            cpt = QPointF(x + 5, y2)
            if kind == "diamond":
                draw_diamond(p, cpt, 5, color)
            elif kind == "hollow":
                p.setPen(QPen(color, 1.6))
                p.setBrush(QBrush(QColor("#FFFFFF")))
                p.drawEllipse(cpt, 4, 4)
            else:
                draw_dot(p, cpt, 4.2, color)
            p.setPen(QPen(QColor("#334155")))
            tw = fm.horizontalAdvance(text)
            p.drawText(QRectF(x + 12, y2 - 8, tw + 4, 16),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            x += 12 + tw + 10
        p.end()

    def _info_box(self, p, k, sub):
        """左上角資訊框：這一行公式算出的關鍵數字。"""
        eng = self.eng
        r = eng.recs[k]
        u = r.update
        pr = r.predict
        col = C_EST
        if sub == 0:
            txt = f"x₀ = z₀；速度未知：0 ± {V_MAX:.0f} m/s"
        elif sub == 1:
            dist = np.linalg.norm(pr.x_prior[:2] - pr.x_prev[:2])
            txt, col = f"① x⁻ = F·x：照速度往前推 {dist:.1f} m", C_PRED_DK
        elif sub == 2:
            txt, col = (f"② 位置 1σ：{pos_sigma(pr.P_prev):.1f} → {pos_sigma(pr.P_prior):.1f} m（變大）",
                        C_PRED_DK)
        elif sub == 3:
            if u.status == "missing":
                txt, col = "③ 沒有量測（隧道）→ 只預測", C_MEAS
            else:
                txt, col = f"③ y = {fmt_xy(u.y, 1, True)}，|y| = {np.linalg.norm(u.y):.1f} m", C_INNOV
        elif sub == 4:
            ok = u.nis <= GATE
            verdict = {"accepted": "在閘門內，接受" if ok else "閘門關 → 照收", "rejected": "拒絕",
                       "reinit": "拒絕 → 重新初始化"}[u.status]
            txt = f"④ NIS = {u.nis:.2f} {'≤' if ok else '>'} {GATE:.2f} → {verdict}"
            col = C_MEAS if u.status in ("rejected", "reinit") else (C_INNOV if ok else C_PRED_DK)
        elif sub == 5:
            txt, col = f"⑤ K[0,0] = {u.K[0, 0]:.2f} → 往量測走 {u.K[0, 0] * 100:.0f}%", C_CORR
        elif sub == 6:
            d = u.x_post[:2] - pr.x_prior[:2]
            txt, col = f"⑥ K·y = {fmt_xy(d, 1, True)} m", C_CORR
        else:
            txt = f"⑦ 位置 1σ：{pos_sigma(pr.P_prior):.1f} → {pos_sigma(u.P_post):.1f} m（變小）"
        font = QFont(FONT_UI, 8)
        font.setWeight(QFont.Weight.Bold)
        p.setFont(font)
        w = QFontMetricsF(font).horizontalAdvance(txt) + 14
        rect = QRectF(self.PLOT.left() + 5, self.PLOT.top() + 5, min(w, self.PLOT.width() - 10), 20)
        p.setPen(QPen(col, 1.4))
        p.setBrush(QBrush(QColor(255, 255, 255, 235)))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QPen(col))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)

    def _draw(self, p, v, k, sub):
        eng = self.eng
        r = eng.recs[k]
        u = r.update
        truth = eng.sim.truth[k]
        clamped = False
        tq = v.pt(*truth[:2])
        if k == 0:
            clamped |= draw_cov_ellipse(p, v, u.x_post[:2], u.P_post[:2, :2], 2, C_EST)
            draw_diamond(p, tq, 6, C_TRUTH)
            q = v.pt(*u.x_post[:2])
            draw_dot(p, q, 5.5, C_EST)
            draw_dot(p, q, 2.5, C_MEAS)
            return clamped

        pr = r.predict
        reinit = u.status == "reinit" and sub >= 4
        prev_q, prior_q = v.pt(*pr.x_prev[:2]), v.pt(*pr.x_prior[:2])

        # ① 上一時刻的估計（空心藍點，① 時連同它的橢圓）→ 預測位置（橘點）
        if sub == 1:
            clamped |= draw_cov_ellipse(p, v, pr.x_prev[:2], pr.P_prev[:2, :2], 2, C_EST,
                                        fill_alpha=18, width=1.4)
        p.setPen(QPen(C_EST, 1.8))
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawEllipse(prev_q, 4.5, 4.5)
        draw_arrow(p, prev_q, prior_q, C_PRED_DK, 1.6, 8, dashed=True)

        # ② 預測協方差（橘虛線橢圓）
        if sub >= 2 and not reinit:
            clamped |= draw_cov_ellipse(p, v, pr.x_prior[:2], pr.P_prior[:2, :2], 2, C_PRED,
                                        dashed=True, fill_alpha=30)

        # 真值（灰菱形 + 真實速度灰虛線）
        draw_diamond(p, tq, 6, C_TRUTH)
        draw_arrow(p, tq, QPointF(tq.x() + truth[2] * v.s, tq.y() - truth[3] * v.s), C_TRUTH, 1.5, 7,
                   dashed=True)

        # ③ 量測、創新 y、S（2σ 紫虛線）；④ 閘門（紫點線，√9.21 σ）
        if sub >= 3 and u.z is not None:
            zq = v.pt(*u.z)
            hz = pr.x_prior[:2]
            if sub == 3:
                clamped |= draw_cov_ellipse(p, v, hz, u.S, 2, C_INNOV, dashed=True, fill_alpha=0,
                                            width=1.3)
            if sub >= 4:
                gate_sig = math.sqrt(GATE)
                draw_cov_ellipse(p, v, hz, u.S, gate_sig, C_INNOV, dotted=True, fill_alpha=10,
                                 width=1.6)
                d = u.y / max(np.linalg.norm(u.y), 1e-9)
                edge = hz + d * math.sqrt(GATE / float(d @ np.linalg.solve(u.S, d)))
                eq = v.pt(*edge)
                nx, ny = -d[1], -d[0]                      # 螢幕座標下，垂直於 y 的左手邊
                tag_pt = QPointF(eq.x() + nx * 16, eq.y() + ny * 16)
                if self.PLOT.adjusted(24, 16, -24, -16).contains(tag_pt):
                    draw_tag(p, tag_pt, "閘門", C_INNOV, bold=False, center=True)
            draw_arrow(p, prior_q, zq, C_INNOV, 2.2, 10)
            L = math.hypot(zq.x() - prior_q.x(), zq.y() - prior_q.y())
            if L > 36:                                  # 箭頭夠長才標，標在箭頭右手邊
                nx, ny = -(zq.y() - prior_q.y()) / L, (zq.x() - prior_q.x()) / L
                draw_tag(p, QPointF((prior_q.x() + zq.x()) / 2 + nx * 12, (prior_q.y() + zq.y()) / 2 + ny * 12),
                         "y", C_INNOV, center=True)
            if u.status in ("rejected", "reinit") and sub >= 4:
                draw_xmark(p, zq, 8, C_MEAS, 3.2)
                draw_tag(p, zq, "拒絕", C_MEAS, dx=10, dy=-12)
            else:
                draw_dot(p, zq, 5, C_MEAS)

        # ⑤ 增益：在創新箭頭上標出「會走到幾 %」的位置
        if sub == 5:
            k00 = u.K[0, 0]
            zq = v.pt(*u.z)
            mark = QPointF(prior_q.x() + (zq.x() - prior_q.x()) * k00,
                           prior_q.y() + (zq.y() - prior_q.y()) * k00)
            p.setPen(QPen(C_CORR, 2.2))
            p.setBrush(QBrush(QColor("#FFFFFF")))
            p.drawEllipse(mark, 4.5, 4.5)
            L = max(math.hypot(zq.x() - prior_q.x(), zq.y() - prior_q.y()), 1e-9)
            nx, ny = -(zq.y() - prior_q.y()) / L, (zq.x() - prior_q.x()) / L
            draw_tag(p, QPointF(mark.x() + nx * 18, mark.y() + ny * 18), f"{k00 * 100:.0f}%", C_CORR,
                     center=True)

        # ⑥ 修正 K·y（綠）＋ 更新後估計（藍）；⑦ 更新後橢圓（藍實線）
        if sub >= 7 or reinit:
            clamped |= draw_cov_ellipse(p, v, u.x_post[:2], u.P_post[:2, :2], 2, C_EST, fill_alpha=40)
        if sub >= 6 and not reinit:
            draw_arrow(p, prior_q, v.pt(*u.x_post[:2]), C_CORR, 2.8, 10)
        draw_dot(p, prior_q, 5, C_PRED, C_PRED_DK)
        if sub >= 6 or reinit:
            q = v.pt(*u.x_post[:2])
            draw_dot(p, q, 5.5, C_EST)
            vel = u.x_post[2:]
        else:
            q, vel = prior_q, pr.x_prior[2:]
        draw_arrow(p, q, QPointF(q.x() + vel[0] * v.s, q.y() - vel[1] * v.s), C_VEL, 2.2, 8)
        return clamped


# =========================================================
#  矩陣面板：x、P、y、S、NIS、K 的實際數值
# =========================================================
class MatrixPanel(QWidget):
    HEAD = 17
    ROW = 14
    COLHEAD = 13
    PAD = 6
    LABEL_W = 30

    def __init__(self, eng):
        super().__init__()
        self.eng = eng
        self.setFixedSize(MAT_W, ROW_H)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        eng = self.eng
        frame_panel(p, MAT_W, ROW_H, "🧮 矩陣面板（剛算出的會高亮）")
        k, sub = eng.cur()
        y = 30
        names = ["px", "py", "vx", "vy"]
        if k < 0:
            p.setFont(QFont(FONT_UI, 10))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(10, 40, MAT_W - 20, ROW_H - 60), Qt.AlignmentFlag.AlignCenter,
                       "x = [px, py, vx, vy]ᵀ\n位置 m、速度 m/s\n\n開始後這裡會顯示\nx、P、y、S、NIS、K\n的實際數值")
            p.end()
            return
        hi = eng.highlight()
        r = eng.recs[k]
        u = r.update
        x, P, xl, Pl = eng.shown_state()
        st = u.status

        y = self._block(p, y, xl, [("", list(x))], "x" in hi, col_head=names)
        y = self._block(p, y, Pl, [(names[i], list(P[i])) for i in range(4)], "P" in hi,
                        col_head=names)

        # ③ y 與 S
        head = "③ 創新 y 與其協方差 S"
        if k == 0:
            rows, note = None, "（初始化不需要）"
        elif not eng.seen(k, 3):
            rows, note = None, "—（本時刻還沒算到）"
        elif st == "missing":
            rows, note = None, "本時刻沒有量測 → 不計算"
        else:
            rows = [("y", list(u.y)), ("S", list(u.S[0])), ("", list(u.S[1]))]
            note = None
        y = self._block(p, y, head, rows, "yS" in hi, col_head=["x", "y"], note=note)

        # ④ NIS
        head = "④ NIS = yᵀ·S⁻¹·y（閘門）"
        col = C_MUTED
        if k == 0:
            txt = "（初始化不需要）"
        elif st == "missing" and eng.seen(k, 3):
            txt = "本時刻沒有量測 → 不計算"
        elif not eng.seen(k, 4):
            txt = "—（本時刻還沒算到）"
        else:
            if u.nis <= GATE:
                txt, col = f"NIS = {u.nis:.2f} ≤ {GATE:.2f}  ✓ 接受", QColor("#15803D")
            elif st == "accepted":
                txt, col = f"NIS = {u.nis:.2f} > {GATE:.2f}  閘門關→照收", QColor("#B45309")
            else:
                txt, col = f"NIS = {u.nis:.2f} > {GATE:.2f}  ✕ 拒絕", C_MEAS
        y = self._block(p, y, head, None, "nis" in hi, note=txt, note_color=col)

        # ⑤ K
        head = "⑤ 增益 K = P⁻·Hᵀ·S⁻¹（4×2）"
        if k > 0 and eng.seen(k, 5) and u.K is not None:
            rows, note = [(names[i], list(u.K[i])) for i in range(4)], None
        else:
            rows = None
            note = "—（本時刻還沒算到）" if k > 0 and 5 in eng.subs[k] else "本時刻不計算"
            if k == 0:
                note = "（初始化不需要）"
        self._block(p, y, head, rows, "K" in hi, col_head=["zx", "zy"], note=note, nd=3)
        p.end()

    def _block(self, p, y, head, rows, hi, col_head=None, note=None, note_color=None, nd=2):
        n_rows = len(rows) if rows else 1
        h = self.HEAD + (self.COLHEAD if rows and col_head else 0) + n_rows * self.ROW + self.PAD
        rect = QRectF(6, y, MAT_W - 12, h)
        if hi:
            p.setPen(QPen(QColor("#F59E0B"), 1.8))
            p.setBrush(QBrush(C_HILITE))
        else:
            p.setPen(QPen(C_GRID, 1))
            p.setBrush(QBrush(QColor("#F8FAFC")))
        p.drawRoundedRect(rect, 6, 6)
        fh = QFont(FONT_UI, 8)
        fh.setWeight(QFont.Weight.Bold)
        p.setFont(fh)
        p.setPen(QPen(QColor("#92400E") if hi else QColor("#334155")))
        p.drawText(QRectF(rect.x() + 6, y + 1, rect.width() - 12, self.HEAD),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, head)
        cy = y + self.HEAD
        if not rows:
            p.setFont(QFont(FONT_UI, 8))
            p.setPen(QPen(note_color or C_MUTED))
            p.drawText(QRectF(rect.x() + 10, cy, rect.width() - 16, self.ROW),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, note or "—")
            return y + h + 4
        ncol = len(rows[0][1])
        x0 = rect.x() + self.LABEL_W
        col_w = (rect.right() - 8 - x0) / ncol
        if col_head:
            p.setFont(QFont(FONT_MONO, 7))
            p.setPen(QPen(QColor("#94A3B8")))
            for j, name in enumerate(col_head):
                p.drawText(QRectF(x0 + j * col_w, cy, col_w, self.COLHEAD),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
            cy += self.COLHEAD
        fm = QFont(FONT_MONO, 9)
        for label, vals in rows:
            p.setFont(QFont(FONT_MONO, 8))
            p.setPen(QPen(QColor("#94A3B8")))
            p.drawText(QRectF(rect.x() + 6, cy, self.LABEL_W - 6, self.ROW),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            p.setFont(fm)
            p.setPen(QPen(C_TEXT))
            for j, val in enumerate(vals):
                p.drawText(QRectF(x0 + j * col_w, cy, col_w, self.ROW),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fmt_num(val, nd))
            cy += self.ROW
        return y + h + 4


# =========================================================
#  下方小圖：(a) 位置誤差 (b) 增益 K[0,0] (c) NIS
# =========================================================
class ChartWidget(QWidget):
    def __init__(self, eng, kind, width):
        super().__init__()
        self.eng = eng
        self.kind = kind
        self.setFixedSize(int(width), CHART_H)
        self.plot = QRectF(34, 26, width - 34 - 10, CHART_H - 26 - 20)

    def xk(self, k):
        return self.plot.left() + k / (self.eng.n - 1) * self.plot.width()

    def yv(self, val):
        y0, y1 = self.yr
        t = (val - y0) / (y1 - y0)
        return self.plot.bottom() - t * self.plot.height()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        eng = self.eng
        ck, cs = eng.cur()
        if self.kind == "err":
            self.yr, ticks = (0.0, 25.0), [0, 5, 10, 15, 20, 25]
            title = "(a) 位置誤差 m：紅 = 量測、藍 = 估計"
        elif self.kind == "gain":
            self.yr, ticks = (0.0, 1.0), [0, 0.25, 0.5, 0.75, 1.0]
            title = "(b) 增益 K[0,0]：往量測修正的比例"
        else:
            self.yr, ticks = (0.0, 18.0), [0, 2, 5, 9.21, 15]
            acc = [eng.recs[k].update.nis for k in range(eng.n)
                   if eng.recs[k].update.status == "accepted" and eng.outcome_seen(k)]
            mean_txt = f"，平均 {np.mean(acc):.2f}" if acc else ""
            title = f"(c) NIS 與閘門（理想平均 ≈ 2{mean_txt}）"
        frame_panel(p, self.width(), CHART_H)
        f = QFont(FONT_UI, 9)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        p.setPen(QPen(QColor("#1D4ED8")))
        p.drawText(QRectF(10, 4, self.width() - 20, 20),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        pr = self.plot
        # 隧道區段
        sim = eng.sim
        if sim.is_dropout.any():
            ks = np.nonzero(sim.is_dropout)[0]
            x0, x1 = self.xk(ks[0] - 0.5), self.xk(ks[-1] + 0.5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(203, 213, 225, 110)))
            p.drawRect(QRectF(x0, pr.top(), x1 - x0, pr.height()))
        # 格線與刻度
        p.setFont(QFont(FONT_UI, 7))
        for t in ticks:
            yy = self.yv(t)
            p.setPen(QPen(C_GRID, 1))
            p.drawLine(QPointF(pr.left(), yy), QPointF(pr.right(), yy))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(2, yy - 7, pr.left() - 5, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{t:g}")
        for kk in range(0, eng.n, 10):
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(self.xk(kk) - 15, pr.bottom() + 2, 30, 14), Qt.AlignmentFlag.AlignCenter,
                       f"{kk}")
        p.drawText(QRectF(pr.right() - 40, pr.bottom() + 2, 40, 14), Qt.AlignmentFlag.AlignRight, "k")
        p.setPen(QPen(C_BORDER, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(pr)

        p.save()
        p.setClipRect(pr.adjusted(-6, -6, 6, 2))
        self._last_label_x = -1e9
        getattr(self, "_draw_" + self.kind)(p)
        p.restore()

        # 目前時刻游標
        if ck >= 0:
            pen = QPen(QColor("#F59E0B"), 1.5)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(self.xk(ck), pr.top()), QPointF(self.xk(ck), pr.bottom()))
        p.end()

    def _series(self, p, pts, color, width, dot_r, markers=True):
        """pts: [(k, value or None)]；None 斷線。超出上限：線交給裁切，點畫成頂端的 ▲。"""
        top = self.yr[1]
        seg = []
        segs = []
        for k, val in pts:
            if val is None:
                if seg:
                    segs.append(seg)
                seg = []
                continue
            seg.append((k, val))
        if seg:
            segs.append(seg)
        pen = QPen(color, width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        for s in segs:
            if len(s) >= 2:
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolyline(QPolygonF([QPointF(self.xk(k), self.yv(val)) for k, val in s]))
            for k, val in s:
                if val > top:
                    if markers:
                        self._clip_marker(p, k, val, color)
                elif dot_r > 0:
                    draw_dot(p, QPointF(self.xk(k), self.yv(val)), dot_r, color)

    def _clip_marker(self, p, k, val, color):
        """超出上限的點：頂端畫 ▲，旁邊標實際值（太擠就不標，避免數字疊在一起）。"""
        x, y = self.xk(k), self.plot.top()
        p.setPen(QPen(color.darker(120), 1))
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([QPointF(x, y - 1), QPointF(x - 4, y + 6), QPointF(x + 4, y + 6)]))
        if x - self._last_label_x < 26:
            return
        self._last_label_x = x
        p.setFont(QFont(FONT_UI, 7))
        p.setPen(QPen(color.darker(130)))
        p.drawText(QRectF(x + 4, y + 1, 40, 12), Qt.AlignmentFlag.AlignLeft, f"{val:.0f}")

    def _draw_err(self, p):
        eng = self.eng
        meas = [(k, eng.meas_err[k] if eng.seen(k, 0 if k == 0 else 3) else None) for k in range(eng.n)]
        est = [(k, eng.est_err[k] if eng.complete(k) else None) for k in range(eng.n)]
        self._series(p, meas, C_MEAS, 1.0, 2.0)
        self._series(p, est, C_EST, 2.2, 0, markers=False)

    def _draw_gain(self, p):
        eng = self.eng
        yss = self.yv(eng.K_ss[0, 0])
        pen = QPen(QColor("#64748B"), 1.2)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(self.plot.left(), yss), QPointF(self.plot.right(), yss))
        draw_tag(p, QPointF(self.plot.left(), yss), f"穩態 {eng.K_ss[0, 0]:.2f}", C_MUTED,
                 dx=4, dy=9, size=7, bold=False)
        pts = []
        for k in range(eng.n):
            u = eng.recs[k].update
            pts.append((k, float(u.K[0, 0]) if u.K is not None and eng.seen(k, 5) else None))
        self._series(p, pts, C_EST, 1.6, 2.2)

    def _draw_nis(self, p):
        eng = self.eng
        for val, color, txt, style in ((GATE, C_MEAS, f"閘門 {GATE:.2f}", Qt.PenStyle.DashLine),
                                       (2.0, QColor("#22C55E"), "期望 2", Qt.PenStyle.DotLine)):
            pen = QPen(color, 1.4)
            pen.setStyle(style)
            p.setPen(pen)
            yy = self.yv(val)
            p.drawLine(QPointF(self.plot.left(), yy), QPointF(self.plot.right(), yy))
            draw_tag(p, QPointF(self.plot.left(), yy), txt, color.darker(120), dx=4, dy=-9, size=7,
                     bold=False)
        for k in range(1, eng.n):
            u = eng.recs[k].update
            if u.nis is None or not eng.seen(k, 4):
                continue
            if u.nis > self.yr[1]:
                self._clip_marker(p, k, u.nis, C_MEAS if u.status != "accepted" else C_INNOV)
                continue
            c = QPointF(self.xk(k), self.yv(u.nis))
            if u.status in ("rejected", "reinit"):
                draw_xmark(p, c, 4, C_MEAS, 2)
            else:
                draw_dot(p, c, 2.6, C_INNOV)


# =========================================================
#  主視窗
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("二維卡爾曼濾波器 逐步動畫（解說版）")
        self.eng = KalmanStepper()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self._build_ui()
        self.refresh()

    # ---------- 介面組裝 ----------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())
        root.addLayout(self._build_param_bar())

        hint = QLabel(
            "💡 σ_a 大 → 更相信量測：跟得快但較抖；σ_a 小 → 更相信模型：平滑但轉彎會落後（太小還會被閘門誤殺而跟丟）。"
            f"　σ_z 是濾波器「以為」的量測雜訊（感測器實際 = {SENSOR_SIGMA:.0f} m）：設太小 → 太相信量測、"
            "NIS 偏大；設太大 → 反應遲鈍、NIS 偏小。")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:12px;color:#475569;padding:0 2px;")
        root.addWidget(hint)

        title = QLabel("二維卡爾曼濾波器　—　「預測」與「量測」都不準，依「誰比較準」加權平均")
        title.setStyleSheet("font-size:17px;font-weight:bold;color:#1D4ED8;")
        root.addWidget(title)

        self.flow = FlowBar(self.eng)
        root.addWidget(self.flow)

        self.explain = QLabel()
        self.explain.setWordWrap(True)
        self.explain.setMinimumHeight(96)
        self.explain.setFixedWidth(CONTENT_W)
        self.explain.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.explain.setStyleSheet(
            "font-size:13px;color:#0F172A;background:#FFFFFF;"
            "border:1px solid #E2E8F0;border-left:4px solid #2563EB;"
            "border-radius:6px;padding:8px;")
        root.addWidget(self.explain)

        row = QHBoxLayout()
        row.setSpacing(GAP)
        self.overview = OverviewWidget(self.eng)
        self.zoom = ZoomWidget(self.eng)
        self.matrix = MatrixPanel(self.eng)
        row.addWidget(self.overview)
        row.addWidget(self.zoom)
        row.addWidget(self.matrix)
        row.addStretch(1)
        root.addLayout(row)

        root.addWidget(self._build_legend())

        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:13px;color:#0F172A;padding:2px;")
        root.addWidget(self.stats)

        charts = QHBoxLayout()
        charts.setSpacing(GAP)
        cw = (CONTENT_W - 2 * GAP) / 3
        self.charts = [ChartWidget(self.eng, kind, cw) for kind in ("err", "gain", "nis")]
        for c in self.charts:
            charts.addWidget(c)
        charts.addStretch(1)
        root.addLayout(charts)

        self.health = QLabel()
        self.health.setWordWrap(True)
        self.health.setStyleSheet("font-size:12px;color:#475569;padding:2px;")
        root.addWidget(self.health)
        root.addStretch(1)

        container = QWidget()
        container.setObjectName("root")
        container.setLayout(root)
        # 只設定 root 自己的底色；不寫選擇器的話會往下蓋掉 QPushButton:checked 的樣式
        container.setStyleSheet("QWidget#root { background:#F8FAFC; }")

        # 包進可捲動區：1366×768 筆電也能捲著看
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        self.container = container
        self.setStyleSheet("""
            QPushButton {
                background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
                padding:7px 12px; font-size:13px; color:#0F172A;
            }
            QPushButton:hover { background:#EFF6FF; border-color:#60A5FA; }
            QPushButton:checked { background:#2563EB; color:white; border-color:#2563EB; }
            QLabel { color:#0F172A; }
            QCheckBox { font-size:13px; color:#0F172A; spacing:5px; }
        """)
        scr = QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.resize(min(CONTENT_W + 50, g.width()), min(980, g.height() - 40))
        else:
            self.resize(CONTENT_W + 50, 900)

    def _build_toolbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        for text, slot in (("⏮ 上一步", self.step_back), ("⏭ 單步（子步驟）", self.step_once),
                           ("⏩ 下一時刻", self.next_time)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        self.btn_play = QPushButton("▶ 自動播放")
        self.btn_play.clicked.connect(self.toggle_play)
        bar.addWidget(self.btn_play)
        for text, slot in (("↺ 從頭重來", self.reset_run), ("⏭| 跑到最後", self.run_to_end)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(QLabel("速度（慢→快）："))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(40)
        self.speed.setFixedWidth(150)
        self.speed.valueChanged.connect(self.update_speed)
        bar.addWidget(self.speed)
        return bar

    def _build_param_bar(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("σ_a 過程雜訊："))
        self.sa = QSlider(Qt.Orientation.Horizontal)
        self.sa.setRange(1, 60)                       # × 0.05 → 0.05 ～ 3.00 m/s²
        self.sa.setValue(int(round(SIGMA_A_DEFAULT / 0.05)))
        self.sa.setFixedWidth(130)
        bar.addWidget(self.sa)
        self.sa_val = QLabel()
        self.sa_val.setFixedWidth(82)
        bar.addWidget(self.sa_val)
        bar.addSpacing(6)
        bar.addWidget(QLabel("σ_z 量測雜訊："))
        self.sz = QSlider(Qt.Orientation.Horizontal)
        self.sz.setRange(1, 40)                       # × 0.5 → 0.5 ～ 20 m
        self.sz.setValue(int(round(SIGMA_Z_DEFAULT / 0.5)))
        self.sz.setFixedWidth(130)
        bar.addWidget(self.sz)
        self.sz_val = QLabel()
        self.sz_val.setFixedWidth(56)
        bar.addWidget(self.sz_val)
        bar.addSpacing(10)
        self.cb_gate = QCheckBox("閘門 gating")
        self.cb_out = QCheckBox("注入離群值")
        self.cb_drop = QCheckBox("訊號遺失段（隧道）")
        for cb in (self.cb_gate, self.cb_out, self.cb_drop):
            cb.setChecked(True)
            bar.addWidget(cb)
        btn_def = QPushButton("↺ 預設參數")
        btn_def.clicked.connect(self.restore_params)
        bar.addWidget(btn_def)
        bar.addStretch(1)
        self.sa.valueChanged.connect(self.on_params_changed)
        self.sz.valueChanged.connect(self.on_params_changed)
        for cb in (self.cb_gate, self.cb_out, self.cb_drop):
            cb.toggled.connect(self.on_params_changed)
        self._update_param_labels()
        return bar

    def _build_legend(self):
        frame = QFrame()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 4, 0)
        items = [
            (C_TRUTH, "真值（◇）"), (C_MEAS, "量測 z（✕ = 被拒）"), (C_PRED, "預測 x⁻ + 橢圓"),
            (C_EST, "估計 x + 橢圓"), (C_INNOV, "創新 y / 閘門"), (C_CORR, "修正 K·y"),
            (C_VEL, "速度 v"), (C_TUNNEL, "隧道（無訊號）"),
        ]
        for color, text in items:
            sw = QLabel()
            sw.setFixedSize(16, 16)
            sw.setStyleSheet(
                f"background:{color.name()};border:1px solid #94A3B8;border-radius:4px;")
            lay.addWidget(sw)
            lab = QLabel(text)
            lab.setStyleSheet("font-size:12px;color:#334155;")
            lay.addWidget(lab)
            lay.addSpacing(6)
        note = QLabel("橢圓 = 2σ（2D 約含 86% 機率）")
        note.setStyleSheet("font-size:12px;color:#64748B;")
        lay.addWidget(note)
        lay.addStretch(1)
        return frame

    # ---------- 參數 ----------
    def _update_param_labels(self):
        self.sa_val.setText(f"{self.sa.value() * 0.05:.2f} m/s²")
        self.sz_val.setText(f"{self.sz.value() * 0.5:.1f} m")

    def restore_params(self):
        for w in (self.sa, self.sz, self.cb_gate, self.cb_out, self.cb_drop):
            w.blockSignals(True)
        self.sa.setValue(int(round(SIGMA_A_DEFAULT / 0.05)))
        self.sz.setValue(int(round(SIGMA_Z_DEFAULT / 0.5)))
        for cb in (self.cb_gate, self.cb_out, self.cb_drop):
            cb.setChecked(True)
        for w in (self.sa, self.sz, self.cb_gate, self.cb_out, self.cb_drop):
            w.blockSignals(False)
        self.on_params_changed()

    def on_params_changed(self, *_):
        """改參數 → 同一個 seed 從頭重算整段，再快轉回「同一時刻、同一子步驟」方便比較。"""
        self._update_param_labels()
        eng = self.eng
        k, sub = eng.cur()
        eng.sigma_a = self.sa.value() * 0.05
        eng.sigma_z = self.sz.value() * 0.5
        eng.use_gating = self.cb_gate.isChecked()
        eng.outliers = self.cb_out.isChecked()
        eng.dropout = self.cb_drop.isChecked()
        eng.recompute()
        target = bisect_right(eng.schedule, (k, sub)) if k >= 0 else 0
        self.rebuild_to(target)
        on = lambda b: "開" if b else "關"           # noqa: E731
        eng.explain = (f"⚙ 參數已改（σ_a = {eng.sigma_a:.2f} m/s²、σ_z = {eng.sigma_z:.1f} m、"
                       f"閘門{on(eng.use_gating)}、離群值{on(eng.outliers)}、隧道{on(eng.dropout)}）"
                       f"→ 用同一組亂數（seed = {SEED}）從頭重算整段，停在同一時刻方便比較。　"
                       + eng.explain)
        self.refresh()

    # ---------- 播放 / 速度 ----------
    def toggle_play(self):
        if self.timer.isActive():
            self.stop_play()
        else:
            if self.eng.done:
                self.reset_run()
            self.update_speed()
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def stop_play(self, text="▶ 自動播放"):
        self.timer.stop()
        self.btn_play.setText(text)

    def update_speed(self):
        self.timer.setInterval(int(620 - 6 * self.speed.value()))

    # ---------- 前進 / 倒退 ----------
    def tick(self):
        if not self.eng.step():
            self.stop_play("▶ 重新播放")
        elif self.eng.done:
            self.stop_play("▶ 重新播放")
        self.refresh()

    def step_once(self):
        self.stop_play()
        self.eng.step()
        self.refresh()

    def next_time(self):
        """把目前時刻剩下的子步驟一次走完；若目前時刻已走完，就走完下一個時刻。"""
        self.stop_play()
        eng = self.eng
        if eng.done:
            return
        k, sub = eng.cur()
        if k < 0:
            target = 0
        elif sub == eng.last_sub(k):
            target = k + 1
        else:
            target = k
        while not eng.done:
            eng.step(explain=False)
            ck, cs = eng.cur()
            if ck == target and cs == eng.last_sub(target):
                break
        eng.explain = eng.summary_for(eng.cur()[0])
        if eng.done:
            eng.explain += "　" + eng.final_summary()
        self.refresh()

    def step_back(self):
        self.stop_play()
        if self.eng.frame > 0:
            self.rebuild_to(self.eng.frame - 1)
            self.refresh()

    def rebuild_to(self, n):
        """倒退做法：整個重設後，再快轉前進 n 步（整段結果固定 seed 算好，完全可重現）。"""
        self.eng.reset()
        for _ in range(n):
            if not self.eng.step(explain=False):
                break
        self.eng.refresh_explain()

    def run_to_end(self):
        self.stop_play("▶ 重新播放")
        self.rebuild_to(self.eng.total)
        self.eng.explain = self.eng.final_summary()
        self.refresh()

    def reset_run(self):
        self.stop_play()
        self.eng.reset()
        self.refresh()

    # ---------- 更新顯示 ----------
    def refresh(self):
        for w in (self.flow, self.overview, self.zoom, self.matrix, *self.charts):
            w.update()
        self.explain.setText("為何走這一步：" + self.eng.explain)
        self.stats.setText(self._stats_text())
        self.health.setText(self._health_text())

    def _stats_text(self):
        eng = self.eng
        k, sub = eng.cur()
        if k < 0:
            return (f"尚未開始　|　共 {eng.n} 個時刻、{eng.total} 個子步驟　|　"
                    f"感測器實際 σ = {SENSOR_SIGMA:.0f} m　|　閘門門檻 {GATE:.2f}")
        cnt = {"accepted": 0, "rejected": 0, "missing": 0, "reinit": 0}
        for kk in range(1, eng.n):
            if eng.outcome_seen(kk):
                cnt[eng.recs[kk].update.status] += 1
        em = eng.meas_err[k]
        if em is not None and eng.seen(k, 0 if k == 0 else 3):
            em_txt = f"{em:.1f} m"
        else:
            em_txt = "—（沒收到）" if eng.sim.z[k] is None and eng.seen(k, 3) else "—（還沒到）"
        ee_txt = f"{eng.est_err[k]:.1f} m" if eng.complete(k) else "（本時刻還沒走完）"
        extra = f"、重新初始化 {cnt['reinit']}" if cnt["reinit"] else ""
        return (f"子步驟 {eng.frame}/{eng.total}　|　目前：{SUB_INFO[sub][0]}　|　"
                f"接受 {cnt['accepted']}、拒絕 {cnt['rejected']}、遺失 {cnt['missing']}{extra}　|　"
                f"本時刻位置誤差：量測 {em_txt}、估計 {ee_txt}")

    def _health_text(self):
        eng = self.eng
        acc = [eng.recs[k].update.nis for k in range(1, eng.n)
               if eng.recs[k].update.status == "accepted" and eng.outcome_seen(k)]
        rep = nis_health(acc)
        return (f"🩺 健康監控（STEP 13，不需要真值）：{rep.message}。"
                f"　穩態增益 K[0,0] ≈ {eng.K_ss[0, 0]:.2f}、K[2,0] ≈ {eng.K_ss[2, 0]:.3f}（固定 dt 下 K 會收斂成定值）。")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setFont(QFont(FONT_UI, 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
