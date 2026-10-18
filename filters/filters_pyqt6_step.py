# -*- coding: utf-8 -*-
"""
工程常用濾波器　逐點動畫比較器（解說版）
=========================================
這支程式用 PyQt6 把 9 種工程常用濾波器「一個取樣、一個取樣」地跑給你看：
每吃進一個新量測，畫面就更新輸出曲線、畫出濾波器的「內部狀態」，
並且用白話＋實際數字告訴你「為什麼是這個輸出」。
（濾波器本體在同資料夾的 filters.py；卡爾曼濾波另見 kalman_filter/ 資料夾。）

核心觀念（畫面上全部看得到）：
  1. 延遲 vs 平滑：藍線越平滑，通常就越落後綠線 —— 拖動參數滑桿就能親眼看到這個取捨。
  2. 雜訊種類決定濾波器：白噪 → 平均類（MA / EMA / Biquad）；尖峰 → 排序類（中位數 / Hampel）；
     漂移 → 互補融合；要少延遲 → 模型類（Alpha-Beta / One Euro）；限制變化率 → 斜率限制。
  3. 串流式：每個濾波器只用「現在與過去」的取樣，內部只需要很小的記憶
     （一個窗口、一兩個狀態），所以下方的「內部狀態」面板能把它整個畫出來。

操作：
  ⏮ 上一步 / ⏭ 單步（吃一個取樣）/ ▶ 自動播放 / ⏩ 直接跑完 / ↺ 從頭重來 + 速度滑桿
  左側選濾波器、選訊號、拖參數；改任何設定都會「從頭重算到目前的播放頭位置」，
  所以可以停在某一點、拖滑桿，直接比較參數的影響。
  互補濾波需要「陀螺儀 + 加速度計」兩路輸入，選它會自動切到 IMU 訊號；
  其他濾波器在 IMU 訊號上吃的是加速度計算出的角度。

需求： pip install PyQt6 numpy
執行： python filters_pyqt6_step.py   （同資料夾需有 filters.py）
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QSlider,
    QHBoxLayout, QVBoxLayout, QGridLayout, QButtonGroup, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetricsF, QPolygonF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filters as F  # noqa: E402


# ---------- 訊號與版面 ----------
N_SAMPLES = 200            # 每段訊號的取樣數
DT = 0.05                  # 取樣間隔 [s] → fs = 20 Hz，整段 10 秒
FS = 1.0 / DT
SEED = 7                   # 固定亂數種子：每次開啟、每次「上一步」重播都一模一樣
FONT = "Microsoft JhengHei"

PLOT_W, PLOT_H = 800, 330
STATE_H = 250
LEFT_W = 340

C_BG       = "#F8FAFC"
C_TEXT     = "#0F172A"
C_MUTED    = "#64748B"
C_BORDER   = "#E2E8F0"
C_BLUE     = "#2563EB"
C_BLUE_L   = "#DBEAFE"
C_BLUE_M   = "#93C5FD"
C_BLUE_D   = "#3B82F6"     # 「上一步的值」這類次要藍色標籤（淺藍字在白底上太淡）
C_AMBER    = "#F59E0B"
C_AMBER_L  = "#FCD34D"
C_GREEN    = "#22C55E"
C_GREEN_L  = "#BBF7D0"
C_RED      = "#EF4444"
C_RED_L    = "#FECACA"
C_GRAY     = "#94A3B8"
C_GRAY_L   = "#CBD5E1"
C_ORANGE   = "#EA580C"

LEFT_VC = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
RIGHT_VC = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
CENTER = Qt.AlignmentFlag.AlignCenter


# =========================================================
#  小工具：數字格式、畫圖 helper
# =========================================================
def num(v: float, d: int) -> str:
    """固定小數位數；去掉「-0.000」的負號，負號改用較清楚的 U+2212（−）。"""
    s = f"{v:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s.replace("-", "−")


def qf(size: float, bold: bool = False) -> QFont:
    f = QFont(FONT)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def text_w(text: str, size: float, bold: bool = False) -> float:
    return QFontMetricsF(qf(size, bold)).horizontalAdvance(text)


def draw_text(p, x, y, w, h, text, size=10.0, color=C_TEXT, bold=False, align=LEFT_VC):
    p.setFont(qf(size, bold))
    p.setPen(QPen(QColor(color)))
    p.drawText(QRectF(x, y, w, h), align, text)


def draw_rbox(p, rect: QRectF, fill, border, width=1.2, radius=6, dash=False):
    pen = QPen(QColor(border), width)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.setBrush(QBrush(QColor(fill)) if fill else Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(rect, radius, radius)


def draw_arrow(p, x1, y1, x2, y2, color, width=2.0, head=7.0, dash=False):
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    ang = math.atan2(y2 - y1, x2 - x1)
    pen.setStyle(Qt.PenStyle.SolidLine)
    p.setPen(pen)
    p.setBrush(QBrush(QColor(color)))
    tip = QPointF(x2, y2)
    a1 = QPointF(x2 - head * math.cos(ang - 0.42), y2 - head * math.sin(ang - 0.42))
    a2 = QPointF(x2 - head * math.cos(ang + 0.42), y2 - head * math.sin(ang + 0.42))
    p.drawPolygon(QPolygonF([tip, a1, a2]))


def draw_marker(p, x, y, kind, color, r=6.0):
    c = QColor(color)
    if kind == "dot":
        p.setPen(QPen(QColor("white"), 1.5))
        p.setBrush(QBrush(c))
        p.drawEllipse(QPointF(x, y), r, r)
    elif kind == "ring":
        p.setPen(QPen(c, 2.5))
        p.setBrush(QBrush(QColor("white")))
        p.drawEllipse(QPointF(x, y), r - 1, r - 1)
    elif kind == "diamond":
        p.setPen(QPen(QColor("white"), 1.2))
        p.setBrush(QBrush(c))
        p.drawPolygon(QPolygonF([QPointF(x, y - r - 1), QPointF(x + r + 1, y),
                                 QPointF(x, y + r + 1), QPointF(x - r - 1, y)]))
    elif kind == "x":
        pen = QPen(c, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(x - r, y - r), QPointF(x + r, y + r))
        p.drawLine(QPointF(x - r, y + r), QPointF(x + r, y - r))
    elif kind == "tick":
        p.setPen(QPen(c, 3))
        p.drawLine(QPointF(x, y - 9), QPointF(x, y + 9))


def span_of(values, min_span):
    """把一串數值換成一條數線的顯示範圍（留邊、且不小於 min_span）。"""
    lo, hi = min(values), max(values)
    span = max(hi - lo, min_span)
    mid = 0.5 * (lo + hi)
    lo, hi = mid - span / 2, mid + span / 2
    pad = span * 0.08
    return lo - pad, hi + pad


def draw_numline(p, x0, x1, y, lo, hi, marks, band=None, arrows=(), dots=()):
    """
    水平數線：
      band   = (a, b, 顏色, 標籤)          畫一段半透明區間（例如「正常範圍」「允許範圍」）
      dots   = [v, ...]                     灰色小點（例如窗口內的值）
      arrows = [(v0, v1, 顏色, 標籤, 層, 虛線?)]  數線上方的箭頭（第 0 層最靠近數線）
      marks  = [(v, 標籤, 顏色, 形狀)]       數線上的標記，標籤自動排在下方兩列、避開重疊
    """
    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    p.setPen(QPen(QColor(C_GRAY), 1.6))
    p.drawLine(QPointF(x0, y), QPointF(x1, y))
    for xx in (x0, x1):
        p.drawLine(QPointF(xx, y - 4), QPointF(xx, y + 4))

    if band is not None:
        a, b, color, label = band
        xa, xb = X(min(a, b)), X(max(a, b))
        if xb - xa < 6:
            m = 0.5 * (xa + xb)
            xa, xb = m - 3, m + 3
        col = QColor(color)
        fill = QColor(col)
        fill.setAlpha(55)
        p.setPen(QPen(col, 1.4))
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(QRectF(xa, y - 10, xb - xa, 20), 4, 4)
        if label:
            tw = text_w(label, 8.5) + 6
            lx = min(max(0.5 * (xa + xb) - tw / 2, x0 - 40), x1 + 40 - tw)
            draw_text(p, lx, y - 27, tw, 15, label, 8.5, color, True, CENTER)

    for v in dots:
        p.setPen(QPen(QColor(C_GRAY), 1))
        p.setBrush(QBrush(QColor(C_GRAY_L)))
        p.drawEllipse(QPointF(X(v), y), 3.5, 3.5)

    for (v0, v1, color, label, level, dashed) in arrows:
        yy = y - 22 - 21 * level
        xa, xb = X(v0), X(v1)
        p.setPen(QPen(QColor(C_GRAY_L), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(xa, yy), QPointF(xa, y))
        p.drawLine(QPointF(xb, yy), QPointF(xb, y))
        if abs(xb - xa) >= 5:
            draw_arrow(p, xa, yy, xb, yy, color, 2.2, 7, dashed)
        else:
            p.setPen(QPen(QColor(color), 2))
            p.setBrush(QBrush(QColor(color)))
            p.drawEllipse(QPointF(xb, yy), 2.5, 2.5)
        if label:
            tw = text_w(label, 8.5) + 6
            lx = min(max(0.5 * (xa + xb) - tw / 2, x0 - 50), x1 + 50 - tw)
            draw_text(p, lx, yy - 16, tw, 14, label, 8.5, color, True, CENTER)

    # 標籤排在數線下方最多三列：先試自然位置（置中對齊標記）；三列都撞到時，
    # 放進最空的那一列並往右挪，再用虛線拉回標記。
    rows_right = [-1e9, -1e9, -1e9]
    ordered = sorted(marks, key=lambda m: X(m[0]))
    placed = []
    lim_lo, lim_hi = x0 - 55, x1 + 55
    for (v, label, color, kind) in ordered:
        x = X(v)
        tw = text_w(label, 9) + 8
        lx = min(max(x - tw / 2, lim_lo), lim_hi - tw)
        row = next((rr for rr in range(3) if lx >= rows_right[rr] + 4), None)
        if row is None:
            row = min(range(3), key=lambda rr: rows_right[rr])
            lx = min(rows_right[row] + 4, lim_hi - tw)
        rows_right[row] = lx + tw
        placed.append((x, lx, tw, row, label, color, kind))
    for (x, lx, tw, row, label, color, kind) in placed:
        ly = y + 11 + 17 * row
        if row >= 1 or not (lx <= x <= lx + tw):
            p.setPen(QPen(QColor(color), 1, Qt.PenStyle.DotLine))
            tx = min(max(x, lx + 6), lx + tw - 6)
            p.drawLine(QPointF(x, y + 6), QPointF(tx, ly + 2))
        draw_text(p, lx, ly, tw, 16, label, 9, color, True, CENTER)
    for (x, lx, tw, row, label, color, kind) in placed:
        draw_marker(p, x, y, kind, color)


def draw_value_boxes(p, x, y, w_avail, h, values, c: "Ctx", fills, borders,
                     tags=None, tag_colors=None, empties=0, max_bw=58.0, border_w=None):
    """一排數值方塊（窗口內容）。太窄放不下數字時，改畫方塊內的小長條。回傳每格的矩形。"""
    n = len(values) + empties
    gap = 4.0 if n <= 16 else 2.0
    bw = min(max_bw, (w_avail - gap * (n - 1)) / max(1, n))
    rects = []
    vals = [v for v in values]
    vmax = max([abs(v) for v in vals] + [1e-9])
    for i in range(n):
        r = QRectF(x + i * (bw + gap), y, bw, h)
        rects.append(r)
        if i >= len(values):
            draw_rbox(p, r, None, C_GRAY_L, 1.2, 5, dash=True)
            draw_text(p, r.x(), r.y(), r.width(), r.height(), "空", 8, C_GRAY, False, CENTER)
            continue
        bwid = border_w[i] if border_w else 1.2
        draw_rbox(p, r, fills[i], borders[i], bwid, 5)
        txt = c.short(values[i])
        drawn = False
        for size in (9.0, 8.0, 7.0):
            if text_w(txt, size, True) <= bw - 3:
                draw_text(p, r.x(), r.y(), r.width(), r.height(), txt, size, C_TEXT, True, CENTER)
                drawn = True
                break
        if not drawn:                                # 太窄：畫小長條代表大小與正負
            mid = r.y() + r.height() / 2
            bh = (abs(values[i]) / vmax) * (r.height() / 2 - 4)
            col = QColor(C_BLUE if values[i] >= 0 else C_RED)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            top = mid - bh if values[i] >= 0 else mid
            p.drawRect(QRectF(r.x() + bw * 0.25, top, bw * 0.5, max(1.0, bh)))
        if tags and tags[i]:
            col = tag_colors[i] if tag_colors else C_MUTED
            tw = text_w(tags[i], 8, True) + 4
            draw_text(p, r.center().x() - tw / 2, y - 16, tw, 14, tags[i], 8, col, True, CENTER)
    return rects


def draw_ratio_bar(p, x, y, w, h, frac, left_fill, right_fill, left_label, right_label,
                   left_color, right_color):
    """兩段式比例條（例如 EMA 的 α 與 1−α），標籤寫在條的下方左右兩端。"""
    frac = min(max(frac, 0.0), 1.0)
    wl = max(3.0, w * frac)
    p.setPen(QPen(QColor(C_GRAY), 1))
    p.setBrush(QBrush(QColor(left_fill)))
    p.drawRoundedRect(QRectF(x, y, wl, h), 4, 4)
    p.setBrush(QBrush(QColor(right_fill)))
    p.drawRoundedRect(QRectF(x + wl, y, w - wl, h), 4, 4)
    pl = f"{frac * 100:.1f}%"
    pr = f"{(1 - frac) * 100:.1f}%"
    if text_w(pl, 9, True) < wl - 6:
        draw_text(p, x, y, wl, h, pl, 9, C_TEXT, True, CENTER)
    if text_w(pr, 9, True) < (w - wl) - 6:
        draw_text(p, x + wl, y, w - wl, h, pr, 9, C_TEXT, True, CENTER)
    draw_text(p, x, y + h + 3, w / 2, 16, left_label, 9, left_color, True, LEFT_VC)
    draw_text(p, x + w / 2, y + h + 3, w / 2, 16, right_label, 9, right_color, True, RIGHT_VC)


def draw_gauge(p, x, y, w, h, label, value, vmax, text, color, ticks=()):
    """水平量表：label | ███████░░░ | 數值。ticks = [(數值, 標籤)] 會畫成直線刻度。"""
    lw = 150
    draw_text(p, x, y, lw, h, label, 9.5, C_TEXT, True, LEFT_VC)
    bx, bw = x + lw, w - lw - 170
    draw_rbox(p, QRectF(bx, y + 3, bw, h - 6), "#F1F5F9", C_GRAY_L, 1, 4)
    frac = 0.0 if vmax <= 0 else min(1.0, max(0.0, value / vmax))
    if frac > 0:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(color)))
        p.drawRoundedRect(QRectF(bx, y + 3, max(3.0, bw * frac), h - 6), 4, 4)
    for (tv, tl) in ticks:
        tx = bx + bw * min(1.0, max(0.0, tv / vmax))
        p.setPen(QPen(QColor(C_TEXT), 2))
        p.drawLine(QPointF(tx, y), QPointF(tx, y + h))
        if tl:
            draw_text(p, tx - 60, y + h - 2, 120, 13, tl, 7.5, C_MUTED, False, CENTER)
    draw_text(p, bx + bw + 10, y, 160, h, text, 9.5, color, True, LEFT_VC)


def nice_ticks(lo, hi, n=6):
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks, step


# =========================================================
#  參數規格（滑桿 ↔ 實際數值）
# =========================================================
class ParamSpec:
    """
    一個滑桿參數。scaling 處理「單位跟著訊號變」的參數：
      'mul'：數值 × 訊號尺度（例如斜率限制的 max_rate：IMU 用度，要大 30 倍）
      'div'：數值 ÷ 訊號尺度（例如 One Euro 的 β：單位是 1/(訊號單位/秒)）
    """

    def __init__(self, key, label, lo, hi, step, default, unit="", decimals=2, scaling="none"):
        self.key, self.label = key, label
        self.lo, self.hi, self.step, self.default = lo, hi, step, default
        self.unit, self.decimals, self.scaling = unit, decimals, scaling

    def count(self) -> int:
        return int(round((self.hi - self.lo) / self.step))

    def pos_of(self, v) -> int:
        return int(round((v - self.lo) / self.step))

    def value(self, pos, scale) -> float:
        v = self.lo + pos * self.step
        if self.scaling == "mul":
            v *= scale
        elif self.scaling == "div":
            v /= scale
        return int(round(v)) if self.decimals == 0 and self.scaling == "none" else v

    def text(self, v, sig) -> str:
        d = self.decimals
        if self.scaling == "div" and sig["scale"] != 1.0:
            d += 2
        if self.scaling == "mul" and sig["scale"] != 1.0:
            d = max(0, d - 1)
        unit = self.unit.replace("{u}", sig["unit"])
        return f"{num(v, d)} {unit}".rstrip()


# =========================================================
#  Ctx：解說/畫圖時共用的格式化工具
# =========================================================
class Ctx:
    def __init__(self, sig, pv, frame, infos):
        self.sig = sig
        self.unit = sig["unit"]
        self.scale = sig["scale"]
        self.pv = pv
        self.frame = frame
        self.infos = infos
        self.d = 3 if self.scale == 1.0 else 2

    def f(self, v, d=None):
        return num(v, self.d if d is None else d)

    def fu(self, v, d=None):
        return self.f(v, d) + self.unit

    def fp(self, v, d=None):
        """負數加括號，給「0.24×(−0.35)」這種式子用。"""
        s = self.f(v, d)
        return f"({s})" if s.startswith("−") else s

    def pm(self, v, d=None):
        """接在式子後面的「 + 0.10」或「 − 0.10」。"""
        s = self.f(abs(v), d)
        return (" − " if (v < 0 and float(s) != 0.0) else " + ") + s

    def short(self, v):
        return num(v, 2 if self.scale == 1.0 else 1)

    def min_span(self):
        return 0.06 * self.scale


# =========================================================
#  濾波器 adapter：GUI 對每個濾波器的「知識」
#  （怎麼建、怎麼餵、怎麼解說、內部狀態怎麼畫），讀的都是 filters.py 的公開屬性
# =========================================================
class Adapter:
    key = ""
    button = ""
    title = ""
    state_title = ""
    one_liner = ""
    pros = ""
    cons = ""
    when = ""
    needs_imu = False
    params: List[ParamSpec] = []

    def build(self, pv, dt):
        raise NotImplementedError

    def feed(self, f, s) -> dict:
        raise NotImplementedError

    def param_note(self, pv, c: Ctx) -> str:
        return ""

    def idle_lines(self, f, c: Ctx) -> List[str]:
        return []

    def explain(self, info, c: Ctx) -> str:
        return ""

    def paint(self, p, r: QRectF, info, c: Ctx):
        pass


# ---------- 1. 移動平均 ----------
class MAAdapter(Adapter):
    key, button, title = "ma", "移動平均 MA", "移動平均（Moving Average）"
    state_title = "環形緩衝裡的 N 個取樣 + 累加和 S"
    one_liner = "最近 N 點取平均：白噪一正一負，平均起來互相抵消。"
    pros = "最直覺；環形緩衝 + 累加和 → 每步 O(1)；線性相位（波形不變形）；N·dt 取干擾週期可完全消掉該頻率。"
    cons = "延遲 (N−1)/2 點；尖峰會被攤成一個小方塊；步階被拉成斜坡；要存 N 個數。"
    when = "感測器讀值顯示、資料記錄平滑、消除固定週期干擾（例如 60 Hz 市電）。"
    params = [ParamSpec("n", "窗口長度 N", 2, 30, 1, 8, "點", 0)]

    def build(self, pv, dt):
        return F.MovingAverage(int(pv["n"]))

    def feed(self, f, s):
        s_prev = f.total
        x = s["meas"]
        y = f.update(x)
        return dict(x=x, y=y, window=f.window(), dropped=f.last_out, s_prev=s_prev,
                    s=f.total, count=f.count, n=f.n)

    def param_note(self, pv, c):
        n = int(pv["n"])
        return (f"窗長 N·dt = {n * DT:.2f} s；延遲 ≈ (N−1)/2 = {(n - 1) / 2:.1f} 點；"
                f"白噪剩 1/√N = {1 / math.sqrt(n):.2f} 倍")

    def idle_lines(self, f, c):
        return [f"環形緩衝有 {f.n} 格，目前全空；累加和 S = 0。",
                "暖機期（還沒滿 N 格）會「有幾點就除以幾」，避免開頭被 0 拖低。"]

    def explain(self, i, c):
        x, y, n = i["x"], i["y"], i["n"]
        if i["dropped"] is None:
            if i["count"] == 1:
                return (f"第 1 筆：窗口裡只有它自己，y = x = {c.f(x)}。"
                        f"（暖機期不硬除以 N={n}，否則開頭會從 0 慢慢爬上來，像一個假的步階。）")
            return (f"暖機中：窗口還沒滿（{i['count']}/{n}），只加不減："
                    f"S = {c.f(i['s_prev'])}{c.pm(x)} = {c.f(i['s'])}，"
                    f"y = S ÷ {i['count']} = {c.f(y)}（目前有幾點就除以幾）。")
        d = i["dropped"]
        return (f"新量測 x = {c.f(x)} 進窗口、最舊的 {c.f(d)} 被擠出。總和不必重加 {n} 個數："
                f"S = {c.f(i['s_prev'])}{c.pm(x)}{c.pm(-d)} = {c.f(i['s'])}，"
                f"y = S ÷ {n} = {c.f(y)}。每一點的權重都是 1/{n}，"
                f"所以輸出大約落後 (N−1)/2 = {(n - 1) / 2:.1f} 點。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        n = i["n"]
        win = i["window"]
        x0 = L
        if i["dropped"] is not None:
            bw = 58
            rr = QRectF(L, T + 22, bw, 40)
            draw_rbox(p, rr, "#F1F5F9", C_RED, 1.2, 5, dash=True)
            draw_text(p, rr.x(), rr.y(), bw, 40, c.short(i["dropped"]), 9, C_MUTED, True, CENTER)
            p.setPen(QPen(QColor(C_RED), 1.6))
            p.drawLine(QPointF(rr.left() + 6, rr.center().y()), QPointF(rr.right() - 6, rr.center().y()))
            draw_text(p, L, T + 4, bw, 16, "擠出", 8.5, C_RED, True, CENTER)
            draw_arrow(p, L + bw + 22, T + 42, L + bw + 6, T + 42, C_RED, 1.6, 6)
            x0 = L + bw + 28
        fills = [C_BLUE_L] * len(win)
        borders = [C_BLUE_M] * len(win)
        tags = [""] * len(win)
        tcol = [C_MUTED] * len(win)
        if win:
            fills[-1], borders[-1], tags[-1], tcol[-1] = C_AMBER_L, C_AMBER, "新", "#B45309"
            if len(win) > 1:
                tags[0] = "最舊"
        draw_value_boxes(p, x0, T + 22, L + W - x0, 40, win, c, fills, borders, tags, tcol,
                         empties=n - len(win))
        y = T + 80
        if i["dropped"] is None:
            l1 = f"累加和 S = S_prev + x_new = {c.f(i['s_prev'])}{c.pm(i['x'])} = {c.f(i['s'])}　（暖機：只加不減）"
        else:
            l1 = (f"累加和 S = S_prev + x_new − x_old = {c.f(i['s_prev'])}{c.pm(i['x'])}"
                  f"{c.pm(-i['dropped'])} = {c.f(i['s'])}")
        draw_text(p, L, y, W, 24, l1, 10.5, C_TEXT, True)
        draw_text(p, L, y + 28, W, 24,
                  f"輸出 y = S ÷ 目前點數 = {c.f(i['s'])} ÷ {i['count']} = {c.f(i['y'])}",
                  10.5, C_BLUE, True)
        draw_text(p, L, y + 60, W, 22,
                  f"每一步只做 1 加、1 減、1 除 → O(1)，N 再大也一樣快；"
                  f"代價是延遲 ≈ (N−1)/2 = {(n - 1) / 2:.1f} 點、要存 N = {n} 個數。",
                  9.5, C_MUTED)
        draw_text(p, L, y + 84, W, 22,
                  "工程細節：S 一直加加減減會累積浮點誤差 → filters.py 每 1000 步用 math.fsum 重算一次。",
                  9.5, C_MUTED)


# ---------- 2. EMA ----------
class EMAAdapter(Adapter):
    key, button, title = "ema", "EMA 一階低通", "EMA／一階低通（RC 電路的數位版）"
    state_title = "新量測 x 與舊輸出 y_prev 的加權（α : 1−α）"
    one_liner = "每一步，輸出從上一個值往新量測「走 α 的比例」——RC 低通電路的數位版。"
    pros = "只存 1 個數、1 乘 1 加，MCU 上最便宜；fc ↔ α 一條公式互換。"
    cons = "一階衰減很鈍（−20 dB/dec）；越平滑越延遲；尖峰會拖出一條指數尾巴。"
    when = "「先濾一下再說」的萬用款：控制迴路的量測、ADC 讀值、UI 顯示數值。"
    params = [ParamSpec("fc", "截止頻率 fc", 0.05, 5.0, 0.05, 1.0, "Hz", 2)]

    def build(self, pv, dt):
        return F.EMA.from_cutoff(pv["fc"], dt)

    def feed(self, f, s):
        y_prev = f.y
        x = s["meas"]
        y = f.update(x)
        return dict(x=x, y=y, y_prev=y_prev, alpha=f.alpha)

    def param_note(self, pv, c):
        a = F.alpha_from_cutoff(pv["fc"], DT)
        rc = 1 / (2 * math.pi * pv["fc"])
        return (f"RC = 1/(2π·fc) = {rc:.3f} s → α = dt/(RC+dt) = {a:.3f}；"
                f"平滑度約等於 {2 / a - 1:.0f} 點的 MA；延遲 ≈ (1−α)/α = {(1 - a) / a:.1f} 點")

    def idle_lines(self, f, c):
        return [f"α = {f.alpha:.3f}：每步只相信新量測 {f.alpha * 100:.1f}%，"
                f"其餘 {(1 - f.alpha) * 100:.1f}% 沿用上一個輸出。",
                "內部只存一個數：上一次的輸出 y_prev。"]

    def explain(self, i, c):
        x, y, a = i["x"], i["y"], i["alpha"]
        if i["y_prev"] is None:
            return (f"第 1 筆：還沒有上一個輸出可以混合，直接令 y = x = {c.f(x)}"
                    f"（暖機；如果從 y=0 開始，開頭會慢慢爬上來，像假的步階）。")
        yp = i["y_prev"]
        fc = c.pv["fc"]
        rc = 1 / (2 * math.pi * fc)
        return (f"EMA：y = α·x + (1−α)·y_prev = {a:.3f}×{c.fp(x)} + {1 - a:.3f}×{c.fp(yp)} = {c.f(y)}。"
                f"換句話說，從上一個輸出 {c.f(yp)} 往新量測 {c.f(x)} 走 {a * 100:.1f}% 的路"
                f"（差 {c.f(x - yp)}，只走 {c.f(a * (x - yp))}）。α 由 fc = {fc:.2f} Hz 換算："
                f"RC = 1/(2π·fc) = {rc:.3f} s，α = dt/(RC+dt) = {DT}/{rc + DT:.3f}。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        a = i["alpha"]
        draw_text(p, L, T, W, 18, "權重分配（兩段加起來 = 1）", 9.5, C_MUTED, True)
        draw_ratio_bar(p, L, T + 20, W, 26, a, C_GRAY_L, C_BLUE_L,
                       f"α = {a:.3f} × 新量測 x", f"(1−α) = {1 - a:.3f} × 舊輸出 y_prev",
                       "#475569", C_BLUE)
        if i["y_prev"] is None:
            draw_text(p, L, T + 90, W, 24, f"第 1 筆：y = x = {c.f(i['x'])}（暖機）", 11, C_BLUE, True)
            return
        x, yp, y = i["x"], i["y_prev"], i["y"]
        lo, hi = span_of([x, yp, y], c.min_span())
        draw_numline(p, L + 70, L + W - 70, T + 130, lo, hi,
                     marks=[(yp, f"y_prev = {c.f(yp)}", C_BLUE_D, "ring"),
                            (x, f"x = {c.f(x)}", "#475569", "dot"),
                            (y, f"y = {c.f(y)}", C_BLUE, "dot")],
                     arrows=[(yp, x, C_GRAY, "全部的差 x − y_prev", 1, True),
                             (yp, y, C_BLUE, f"只走 α = {a * 100:.1f}%", 0, False)])
        draw_text(p, L, T + 188, W, 22,
                  f"y = α·x + (1−α)·y_prev = {a:.3f}×{c.fp(x)} + {1 - a:.3f}×{c.fp(yp)} = {c.f(y)}",
                  10.5, C_BLUE, True)


# ---------- 3. 中位數 ----------
class MedianAdapter(Adapter):
    key, button, title = "median", "中位數濾波", "中位數濾波（Median Filter）"
    state_title = "窗口（時間順序）→ 排序 → 取正中間"
    one_liner = "把窗口排序、取正中間那個：尖峰被擠到兩端，碰不到中位數。"
    pros = "去尖峰最強；步階邊緣保持陡峭；不需要任何雜訊模型或統計。"
    cons = "對白噪的效率比 MA 差；每一點都被換成中位數（細節變階梯）；每步要維護排序 O(N)。"
    when = "偶發錯讀、脈衝雜訊、超音波 / 雷射測距的離譜值、要保留邊緣的訊號。"
    params = [ParamSpec("n", "窗口長度 N（奇數）", 3, 21, 2, 5, "點", 0)]

    def build(self, pv, dt):
        return F.MedianFilter(int(pv["n"]))

    def feed(self, f, s):
        x = s["meas"]
        y = f.update(x)
        return dict(x=x, y=y, window=f.window(), sorted=list(f.sorted), dropped=f.last_out, n=f.n)

    def param_note(self, pv, c):
        n = int(pv["n"])
        return f"最多能去掉連續 (N−1)/2 = {(n - 1) // 2} 點的尖峰；邊緣延遲約 {(n - 1) // 2} 點"

    def idle_lines(self, f, c):
        return [f"窗口長 {f.n}：每一步把窗口排序，取正中間那個當輸出。",
                "新值用二分搜尋插入排序串列、最舊的值用二分搜尋刪除。"]

    def _list(self, s, c):
        if len(s) <= 9:
            return "[" + ", ".join(c.short(v) for v in s) + "]"
        return ("[" + ", ".join(c.short(v) for v in s[:3]) + ", …, "
                + ", ".join(c.short(v) for v in s[-3:]) + "]")

    def explain(self, i, c):
        s, x, y = i["sorted"], i["x"], i["y"]
        m = len(s)
        if m % 2:
            head = (f"窗口 {m} 筆排序後 = {self._list(s, c)}，取正中間第 {m // 2 + 1} 個 → y = {c.f(y)}。")
        else:
            head = (f"暖機中、窗口 {m} 筆（偶數）排序後 = {self._list(s, c)}，"
                    f"取中間兩個 {c.f(s[m // 2 - 1])}、{c.f(s[m // 2])} 的平均 → y = {c.f(y)}。")
        pos = s.index(x) + 1
        spread = (s[-1] - s[0]) if m > 1 else 0.0
        if m >= 3 and (pos == 1 or pos == m) and abs(x - y) > 0.5 * spread and abs(x - y) > 0.3 * c.scale:
            side = "最左" if pos == 1 else "最右"
            mid = (f"新量測 {c.f(x)} 排序後落在{side}端（第 {pos}/{m} 個），是極端值 → "
                   f"完全碰不到中位數，這就是尖峰被濾掉的原因。")
        else:
            mid = f"新量測 {c.f(x)} 插入後排在第 {pos}/{m} 個。"
        tail = f"最舊的 {c.f(i['dropped'])} 已移出窗口。" if i["dropped"] is not None else ""
        return head + mid + tail

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        lw = 84
        win, srt, n = i["window"], i["sorted"], i["n"]
        draw_text(p, L, T + 18, lw, 38, "時間順序\n舊 → 新", 9, C_MUTED, True, LEFT_VC)
        fills = [C_BLUE_L] * len(win)
        borders = [C_BLUE_M] * len(win)
        tags = [""] * len(win)
        tcol = [C_MUTED] * len(win)
        fills[-1], borders[-1], tags[-1], tcol[-1] = C_AMBER_L, C_AMBER, "新", "#B45309"
        draw_value_boxes(p, L + lw, T + 18, W - lw, 36, win, c, fills, borders, tags, tcol,
                         empties=n - len(win))
        draw_text(p, L, T + 84, lw, 38, "排序後\n小 → 大", 9, C_MUTED, True, LEFT_VC)
        m = len(srt)
        mids = {m // 2} if m % 2 else {m // 2 - 1, m // 2}
        new_idx = srt.index(i["x"])
        fills, borders, tags, tcol, bws = [], [], [], [], []
        for k in range(m):
            if k in mids:
                fills.append(C_GREEN_L)
                borders.append(C_GREEN)
                tags.append("中位數" if m % 2 else "取平均")
                tcol.append("#15803D")
                bws.append(2.6)
            else:
                fills.append("#F1F5F9")
                borders.append(C_GRAY_L)
                tags.append("")
                tcol.append(C_MUTED)
                bws.append(1.2)
        if new_idx not in mids:
            borders[new_idx] = C_AMBER
            bws[new_idx] = 2.6
            tags[new_idx] = "新"
            tcol[new_idx] = "#B45309"
        draw_value_boxes(p, L + lw, T + 84, W - lw, 36, srt, c, fills, borders, tags, tcol,
                         empties=n - m, border_w=bws + [1.2] * (n - m))
        y = T + 136
        if m % 2:
            l1 = f"y = 排序後正中間（第 {m // 2 + 1}/{m} 個）= {c.f(i['y'])}"
        else:
            l1 = f"y = 中間兩個的平均 = ({c.f(srt[m // 2 - 1])} + {c.f(srt[m // 2])}) ÷ 2 = {c.f(i['y'])}"
        draw_text(p, L, y, W, 22, l1, 10.5, C_BLUE, True)
        note = (f"新量測 {c.f(i['x'])} 經二分搜尋插在排序後第 {new_idx + 1} 格"
                + (f"；最舊的 {c.f(i['dropped'])} 已刪除" if i["dropped"] is not None else "")
                + "。極端值（尖峰）永遠被擠到兩端。")
        draw_text(p, L, y + 26, W, 22, note, 9.5, C_MUTED)


# ---------- 4. Hampel ----------
class HampelAdapter(Adapter):
    key, button, title = "hampel", "Hampel 離群", "Hampel 離群值濾波（中位數 + MAD）"
    state_title = "窗口 → 中位數 med 與 MAD → 正常範圍 med ± k·σ"
    one_liner = "先用「中位數 + MAD」判斷這點是不是離群；是才換成中位數，不是就原值照用。"
    pros = "只動壞點、好點原封不動；MAD 對尖峰免疫（不會被尖峰撐大）；因果版零延遲。"
    cons = "不平滑白噪；真實步階的前幾點會被誤判成離群；MAD = 0 時要給 σ 下限。"
    when = "資料清洗、感測器前處理，常接在 EMA / 卡爾曼之前先把壞點剔掉。"
    params = [ParamSpec("n", "窗口長度 N（奇數）", 5, 21, 2, 9, "點", 0),
              ParamSpec("k", "門檻倍數 k", 1.0, 5.0, 0.5, 3.0, "σ", 1)]

    def build(self, pv, dt):
        return F.HampelFilter(int(pv["n"]), pv["k"])

    def feed(self, f, s):
        x = s["meas"]
        y = f.update(x)
        return dict(x=x, y=y, window=f.window(), med=f.med, mad=f.mad, sigma=f.sigma,
                    thr=f.threshold, dev=f.deviation, outlier=f.is_outlier, count=len(f.buf),
                    k=f.k, n=f.n)

    def param_note(self, pv, c):
        return (f"σ ≈ 1.4826 × MAD；k = {pv['k']:.1f} → 正常點被誤判的機率約 "
                f"{math.erfc(pv['k'] / math.sqrt(2)) * 100:.2f}%（常態假設）")

    def idle_lines(self, f, c):
        return [f"窗口 N = {f.n}、門檻 k = {f.k:.1f}：|x − med| > k·σ 才算離群。",
                "σ 不用標準差（會被尖峰撐大），改用穩健的 1.4826 × MAD。"]

    def explain(self, i, c):
        x = i["x"]
        if i["count"] < 3:
            return (f"暖機：窗口只有 {i['count']} 點，MAD 還沒有意義，一律當正常點 y = x = {c.f(x)}。")
        base = (f"窗口中位數 med = {c.f(i['med'])}，MAD = median(|w − med|) = {c.f(i['mad'])} → "
                f"σ ≈ 1.4826×MAD = {c.f(i['sigma'])}，門檻 = k·σ = {i['k']:.1f}×{c.f(i['sigma'])} = "
                f"{c.f(i['thr'])}。")
        if i["outlier"]:
            return base + (f"新量測 x = {c.f(x)}，|x − med| = {c.f(i['dev'])} > {c.f(i['thr'])} → "
                           f"判定離群，輸出改用中位數 y = {c.f(i['y'])}。")
        return base + (f"新量測 x = {c.f(x)}，|x − med| = {c.f(i['dev'])} ≤ {c.f(i['thr'])} → "
                       f"正常點，原值照用 y = x = {c.f(x)}（Hampel 只換壞點，好點一個都不碰，"
                       f"所以細節和白噪都原封保留）。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        win, n = i["window"], i["n"]
        out = i["outlier"]
        lw = 84
        draw_text(p, L, T + 16, lw, 32, "窗口\n舊 → 新", 9, C_MUTED, True, LEFT_VC)
        fills = [C_BLUE_L] * len(win)
        borders = [C_BLUE_M] * len(win)
        tags = [""] * len(win)
        tcol = [C_MUTED] * len(win)
        if out:
            fills[-1], borders[-1], tags[-1], tcol[-1] = C_RED_L, C_RED, "新：離群", C_RED
        else:
            fills[-1], borders[-1], tags[-1], tcol[-1] = C_AMBER_L, C_AMBER, "新", "#B45309"
        draw_value_boxes(p, L + lw, T + 16, W - lw, 32, win, c, fills, borders, tags, tcol,
                         empties=n - len(win), max_bw=52)
        if i["count"] < 3:
            draw_text(p, L, T + 90, W, 24, f"暖機：窗口只有 {i['count']} 點，先一律當正常點。",
                      11, C_BLUE, True)
            return
        med, thr, x = i["med"], i["thr"], i["x"]
        lo, hi = span_of(list(win) + [med - thr, med + thr], c.min_span())
        xcol = C_RED if out else C_BLUE
        marks = [(med, f"med = {c.f(med)}" + ("（輸出）" if out else ""), "#15803D", "tick"),
                 (x, f"x = {c.f(x)}" + ("　離群 ✕" if out else "　正常 → 輸出"), xcol,
                  "x" if out else "dot")]
        draw_numline(p, L + 70, L + W - 70, T + 112, lo, hi, marks,
                     band=(med - thr, med + thr, C_GREEN, f"正常範圍 med ± k·σ（±{c.f(thr)}）"),
                     arrows=[(med, x, xcol, f"|x − med| = {c.f(i['dev'])}", 1, False)],
                     dots=win[:-1])
        y = T + 162
        draw_text(p, L, y, W, 22,
                  f"MAD = {c.f(i['mad'])} → σ = 1.4826×MAD = {c.f(i['sigma'])} → "
                  f"門檻 k·σ = {i['k']:.1f}×{c.f(i['sigma'])} = {c.f(thr)}", 10, C_TEXT, True)
        if out:
            l2 = f"|x − med| = {c.f(i['dev'])} > {c.f(thr)} → 離群：輸出換成 med = {c.f(i['y'])}"
        else:
            l2 = f"|x − med| = {c.f(i['dev'])} ≤ {c.f(thr)} → 正常：輸出 = 原值 x = {c.f(i['y'])}"
        draw_text(p, L, y + 24, W, 22, l2, 10.5, xcol, True)


# ---------- 5. Alpha-Beta ----------
class AlphaBetaAdapter(Adapter):
    key, button, title = "ab", "α-β 追蹤", "Alpha-Beta 追蹤濾波（穩態卡爾曼的簡化版）"
    state_title = "預測 → 殘差 → 修正（位置 x 與速度 v 兩個狀態）"
    one_liner = "多估一個「速度」：先用速度預測下一點，再拿殘差（量測 − 預測）修正位置與速度。"
    pros = "斜坡（等速）上沒有穩態延遲；順便得到速度估計；只要幾個乘加，就是卡爾曼的穩態增益。"
    cons = "要調 α、β 兩個參數；急停急轉會過衝；尖峰會把速度帶歪（先過 Hampel）。"
    when = "目標追蹤（雷達、影像）、編碼器位置 → 速度、任何近似等速移動的量測。"
    params = [ParamSpec("alpha", "位置增益 α", 0.02, 0.98, 0.02, 0.30, "", 2),
              ParamSpec("beta", "速度增益 β", 0.002, 0.4, 0.002, 0.05, "", 3)]

    def build(self, pv, dt):
        return F.AlphaBetaFilter(pv["alpha"], pv["beta"], dt)

    def feed(self, f, s):
        x_prev, v_prev = f.x, f.v
        z = s["meas"]
        y = f.update(z)
        return dict(z=z, y=y, x_prev=x_prev, v_prev=v_prev, x_pred=f.x_pred, r=f.residual,
                    x=f.x, v=f.v, alpha=f.alpha, beta=f.beta, first=x_prev is None)

    def param_note(self, pv, c):
        a, b = pv["alpha"], pv["beta"]
        ok = 0 < a < 2 and 0 < b < 4 - 2 * a
        return (f"經驗值 β = α²/(2−α) = {F.benedict_bordner_beta(a):.3f}（Benedict–Bordner）；"
                f"穩定條件 0<α<2、0<β<4−2α：{'✔ 穩定' if ok else '✘ 不穩定'}")

    def idle_lines(self, f, c):
        return [f"α = {f.alpha:.2f}（位置多相信量測幾成）、β = {f.beta:.3f}（速度被殘差修正多快）。",
                "內部只存兩個數：位置估計 x 與速度估計 v（量測叫 z）。"]

    def explain(self, i, c):
        if i["first"]:
            return (f"第 1 筆：位置直接設成量測 x = z = {c.f(i['z'])}，速度先假設 v = 0"
                    f"（之後由殘差一步步修正出來）。")
        u = c.unit
        r = i["r"]
        return (f"① 預測：x_pred = x + v·dt = {c.f(i['x_prev'])} + {c.fp(i['v_prev'])}×{DT} = {c.f(i['x_pred'])}　"
                f"② 殘差：r = z − x_pred = {c.f(i['z'])}{c.pm(-i['x_pred'])} = {c.f(r)}　"
                f"③ 修正位置：x = x_pred + α·r = {c.f(i['x_pred'])} + {i['alpha']:.2f}×{c.fp(r)} = {c.f(i['x'])}　"
                f"④ 修正速度：v = v + (β/dt)·r = {c.f(i['v_prev'])} + {i['beta'] / DT:.3f}×{c.fp(r)} = "
                f"{c.f(i['v'])}{u}/s。殘差{'為正 → 目標比預測的更前面' if r > 0 else '為負 → 目標比預測的更後面'}，"
                f"位置與速度都往量測那邊修一點。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        if i["first"]:
            draw_text(p, L, T + 70, W, 24,
                      f"第 1 筆：x = z = {c.f(i['z'])}，v = 0（暖機）", 11, C_BLUE, True)
            return
        xp, xq, z, xn = i["x_prev"], i["x_pred"], i["z"], i["x"]
        lo, hi = span_of([xp, xq, z, xn], c.min_span())
        draw_numline(p, L + 70, L + W - 70, T + 84, lo, hi,
                     marks=[(xp, f"上一步 x = {c.f(xp)}", C_BLUE_D, "ring"),
                            (xq, f"預測 x_pred = {c.f(xq)}", "#B45309", "diamond"),
                            (z, f"量測 z = {c.f(z)}", "#475569", "dot"),
                            (xn, f"修正後 x = {c.f(xn)}", C_BLUE, "dot")],
                     arrows=[(xp, xq, C_AMBER, "① 預測 v·dt", 0, False),
                             (xq, z, C_RED, f"② 殘差 r = {c.f(i['r'])}", 1, True),
                             (xq, xn, C_BLUE, f"③ 修正 α·r = {c.f(i['alpha'] * i['r'])}", 2, False)])
        y = T + 148
        draw_text(p, L, y, W, 20,
                  f"① x_pred = x + v·dt = {c.f(xp)} + {c.fp(i['v_prev'])}×{DT} = {c.f(xq)}　　"
                  f"② r = z − x_pred = {c.f(z)}{c.pm(-xq)} = {c.f(i['r'])}", 10, C_TEXT, True)
        draw_text(p, L, y + 20, W, 20,
                  f"③ x = x_pred + α·r = {c.f(xq)} + {i['alpha']:.2f}×{c.fp(i['r'])} = {c.f(xn)}",
                  10, C_BLUE, True)
        draw_text(p, L, y + 40, W, 20,
                  f"④ 速度 v = v + (β/dt)·r = {c.f(i['v_prev'])} + {i['beta'] / DT:.3f}×{c.fp(i['r'])}"
                  f" = {c.f(i['v'])} {c.unit}/s", 10, "#B45309", True)


# ---------- 6. 互補濾波 ----------
class ComplementaryAdapter(Adapter):
    key, button, title = "comp", "互補濾波 IMU", "互補濾波（陀螺儀 + 加速度計 → 姿態角）"
    state_title = "陀螺儀預測（α）與加速度計（1−α）各佔多少"
    one_liner = "陀螺儀管短期（快、平滑但會漂）、加速度計管長期（準但很吵），兩個權重加起來 = 1。"
    pros = "幾乎零延遲又不會漂；只有一行公式；不需要雜訊統計。"
    cons = "穩態誤差 ≈ 陀螺偏差 × τ；只適合單軸 / 小角度；載具加速時加速度計長期也不準。"
    when = "IMU 姿態角：平衡車、無人機、手勢 / 穿戴裝置。"
    needs_imu = True
    params = [ParamSpec("tau", "時間常數 τ", 0.1, 5.0, 0.1, 1.0, "s", 1)]

    def build(self, pv, dt):
        return F.ComplementaryFilter(dt, tau=pv["tau"])

    def feed(self, f, s):
        prev = f.angle
        g, a = s["gyro"], s["accel"]
        y = f.update(g, a)
        return dict(prev=prev, gyro=g, accel=a, pred=f.angle_pred, gt=f.gyro_term, at=f.accel_term,
                    y=y, x=a, alpha=f.alpha, first=prev is None, truth=s["truth"], gint=s["gyro_int"])

    def param_note(self, pv, c):
        a = pv["tau"] / (pv["tau"] + DT)
        return (f"α = τ/(τ+dt) = {a:.4f}：比 {pv['tau']:.1f} s 快的變化信陀螺儀、比它慢的信加速度計；"
                f"陀螺偏差 2°/s 會留下約 {2 * pv['tau']:.1f}° 的穩態誤差")

    def idle_lines(self, f, c):
        return [f"α = {f.alpha:.4f}：每步 {f.alpha * 100:.1f}% 信陀螺儀的預測、"
                f"{(1 - f.alpha) * 100:.1f}% 信加速度計。",
                "兩路輸入：陀螺儀角速度 ω（°/s）、加速度計算出的角度 a（°）。"]

    def explain(self, i, c):
        if i["first"]:
            return (f"第 1 筆：還沒有角度可以讓陀螺儀往前推，只能先相信加速度計：θ = a = {c.f(i['accel'])}°。")
        a = i["alpha"]
        return (f"陀螺儀把上一個角度往前推一步：θ_pred = θ_prev + ω·dt = {c.f(i['prev'])}° + "
                f"{c.fp(i['gyro'])}°/s×{DT} = {c.f(i['pred'])}°；加速度計說 {c.f(i['accel'])}°。"
                f"互補混合：θ = α·θ_pred + (1−α)·a = {a:.3f}×{c.fp(i['pred'])} + {1 - a:.3f}×{c.fp(i['accel'])}"
                f" = {c.f(i['gt'])} + {c.fp(i['at'])} = {c.f(i['y'])}°。陀螺儀負責短期（快、平滑但會漂），"
                f"加速度計每步只拉 {(1 - a) * 100:.1f}%，把漂移慢慢拉回來；"
                f"此刻真值 {c.f(i['truth'])}°，純陀螺積分已漂到 {c.f(i['gint'])}°。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        a = i["alpha"]
        draw_text(p, L, T, W, 18, "權重分配（兩段加起來 = 1；陀螺儀走高通、加速度計走低通）", 9.5, C_MUTED, True)
        draw_ratio_bar(p, L, T + 18, W, 26, a, "#FDE68A", C_GRAY_L,
                       f"α = {a:.3f} × 陀螺儀預測（短期）", f"(1−α) = {1 - a:.3f} × 加速度計（長期）",
                       "#B45309", "#475569")
        if i["first"]:
            draw_text(p, L, T + 90, W, 24, f"第 1 筆：θ = a = {c.f(i['accel'])}°（暖機）", 11, C_BLUE, True)
            return
        pv_, pr, ac, y = i["prev"], i["pred"], i["accel"], i["y"]
        lo, hi = span_of([pv_, pr, ac, y], c.min_span())
        draw_numline(p, L + 70, L + W - 70, T + 126, lo, hi,
                     marks=[(pv_, f"θ_prev = {c.f(pv_)}°", C_BLUE_D, "ring"),
                            (pr, f"陀螺預測 θ_pred = {c.f(pr)}°", "#B45309", "diamond"),
                            (ac, f"加速度計 a = {c.f(ac)}°", "#475569", "dot"),
                            (y, f"θ = {c.f(y)}°", C_BLUE, "dot")],
                     arrows=[(pv_, pr, C_AMBER, "ω·dt（陀螺儀往前推）", 1, False),
                             (pr, y, C_BLUE, f"往加速度計拉 {(1 - a) * 100:.1f}%", 0, False)])
        draw_text(p, L, T + 188, W, 22,
                  f"陀螺項 {a:.3f}×{c.fp(pr)} = {c.f(i['gt'])}°　＋　加速度計項 {1 - a:.3f}×{c.fp(ac)} = "
                  f"{c.f(i['at'])}°　＝　θ = {c.f(y)}°（真值 {c.f(i['truth'])}°）", 10, C_BLUE, True)


# ---------- 7. Biquad ----------
class BiquadAdapter(Adapter):
    key, button, title = "biquad", "Biquad 二階低通", "Biquad 二階低通（DF2T；Q=0.707 即 Butterworth）"
    state_title = "Direct Form II Transposed：兩個延遲暫存器 z1、z2"
    one_liner = "二階 IIR：只用兩格記憶 z1、z2，衰減 −40 dB/dec，比 EMA 陡一倍。"
    pros = "規格明確（fc、Q）；比一階陡很多；可串接成 4、6 階（SOS）；DF2T 浮點數值特性好。"
    cons = "非線性相位；Q 大會過衝、振鈴；fc 接近 fs/2 時係數很敏感；定點數實作要防溢位。"
    when = "需要明確截止頻率的低通 / 高通：音訊、振動、抗混疊後處理、控制迴路量測。"
    params = [ParamSpec("fc", "截止頻率 fc", 0.1, 5.0, 0.05, 1.5, "Hz", 2),
              ParamSpec("q", "品質因數 Q", 0.5, 3.0, 0.001, 0.707, "", 3)]

    def build(self, pv, dt):
        return F.rbj_lowpass(pv["fc"], 1.0 / dt, pv["q"], steady_init=True)

    def feed(self, f, s):
        x = s["meas"]
        first = f.y is None
        if first:                                     # 與 Biquad.steady_init 同一套穩態公式
            y0 = f.dc_gain() * x
            z1b, z2b = y0 - f.b0 * x, f.b2 * x - f.a2 * y0
        else:
            z1b, z2b = f.z1, f.z2
        y = f.update(x)
        return dict(x=x, y=y, z1b=z1b, z2b=z2b, z1=f.z1, z2=f.z2, first=first, **f.coefficients())

    def param_note(self, pv, c):
        gd = 1.0 / (pv["q"] * 2 * math.pi * pv["fc"]) / DT
        kind = "Butterworth（最平坦、不過衝的極限）" if abs(pv["q"] - 0.707) < 0.01 else (
            "會過衝、振鈴" if pv["q"] > 0.707 else "較鈍、不過衝")
        return f"fs = {FS:.0f} Hz（fc 必須 < fs/2 = {FS / 2:.0f} Hz）；Q = {pv['q']:.3f}：{kind}；低頻延遲 ≈ {gd:.1f} 點"

    def idle_lines(self, f, c):
        k = f.coefficients()
        return ["係數（RBJ Audio EQ Cookbook，已除以 a0）：" + "　".join(f"{n} = {num(v, 4)}" for n, v in k.items()),
                "z1、z2 兩個延遲暫存器裝著「過去輸入與輸出的加權和」。"]

    def explain(self, i, c):
        if i["first"]:
            return (f"第 1 筆：把 z1、z2 預設成「輸入一直都是 x = {c.f(i['x'])}」的穩態"
                    f"（z1 = {c.f(i['z1b'])}、z2 = {c.f(i['z2b'])}），輸出直接從 y = {c.f(i['y'])} 起跑，"
                    f"不會從 0 暴衝上來。（scipy 的 lfilter 預設 z=0，就會看到開頭那段暫態。）")
        return (f"DF2T 三行：① y = b0·x + z1 = {i['b0']:.4f}×{c.fp(i['x'])} + {c.fp(i['z1b'])} = {c.f(i['y'])}　"
                f"② z1 ← b1·x − a1·y + z2 = {i['b1']:.4f}×{c.fp(i['x'])} − {c.fp(i['a1'], 4)}×{c.fp(i['y'])}"
                f" + {c.fp(i['z2b'])} = {c.f(i['z1'])}　"
                f"③ z2 ← b2·x − a2·y = {i['b2']:.4f}×{c.fp(i['x'])} − {c.fp(i['a2'], 4)}×{c.fp(i['y'])}"
                f" = {c.f(i['z2'])}。z1、z2 是兩格「記憶」，裝著過去輸入與輸出的加權和；下一步就靠它們產生輸出。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        yt, ya, yb = T + 14, T + 84, T + 160            # 上軌（x）、加法器、下軌（y）
        A = [L + 120, L + 290, L + 460]
        rad = 14
        # 上軌：x 送往三個乘法
        draw_rbox(p, QRectF(L, yt - 13, 92, 26), C_GRAY_L, C_GRAY, 1.2, 6)
        draw_text(p, L, yt - 13, 92, 26, f"x = {c.short(i['x'])}", 9.5, C_TEXT, True, CENTER)
        p.setPen(QPen(QColor(C_GRAY), 2))
        p.drawLine(QPointF(L + 92, yt), QPointF(A[2], yt))
        gains_top = [("b0", i["b0"]), ("b1", i["b1"]), ("b2", i["b2"])]
        for ax, (nm, gv) in zip(A, gains_top):
            draw_arrow(p, ax, yt, ax, ya - rad - 1, C_GRAY, 1.8, 7)
            gr = QRectF(ax + 6, (yt + ya) / 2 - 10, 92, 20)
            draw_rbox(p, gr, "#F8FAFC", C_GRAY_L, 1, 4)
            draw_text(p, gr.x(), gr.y(), gr.width(), gr.height(), f"×{nm} {gv:.4f}", 8.5, "#475569", True, CENTER)
        # 加法器
        for k, ax in enumerate(A):
            p.setPen(QPen(QColor(C_BLUE), 2))
            p.setBrush(QBrush(QColor("white")))
            p.drawEllipse(QPointF(ax, ya), rad, rad)
            draw_text(p, ax - rad, ya - rad, 2 * rad, 2 * rad, "+", 13, C_BLUE, True, CENTER)
        # 延遲暫存器 z1（在 A0 與 A1 之間）、z2（在 A1 與 A2 之間）
        regs = [("z1", i["z1b"], i["z1"], A[0], A[1]), ("z2", i["z2b"], i["z2"], A[1], A[2])]
        for nm, old, new, xl, xr in regs:
            bx = QRectF((xl + xr) / 2 - 56, ya - 22, 112, 44)
            draw_rbox(p, bx, C_AMBER_L, C_AMBER, 1.8, 7)
            draw_text(p, bx.x(), bx.y() + 2, bx.width(), 18, f"z⁻¹ 暫存器 {nm}", 8.5, "#92400E", True, CENTER)
            draw_text(p, bx.x(), bx.y() + 21, bx.width(), 18, f"{c.short(old)} → {c.short(new)}",
                      9.5, C_TEXT, True, CENTER)
            draw_arrow(p, xr - rad, ya, bx.right() + 1, ya, C_AMBER, 1.8, 7)
            draw_arrow(p, bx.left() - 1, ya, xl + rad + 1, ya, C_AMBER, 1.8, 7)
        # 輸出與回授：A0 → y，y 經 −a1、−a2 回到 A1、A2
        p.setPen(QPen(QColor(C_BLUE), 2))
        p.drawLine(QPointF(A[0], ya + rad), QPointF(A[0], yb))
        p.drawLine(QPointF(A[0], yb), QPointF(A[2], yb))
        for ax, (nm, gv) in zip(A[1:], [("a1", i["a1"]), ("a2", i["a2"])]):
            draw_arrow(p, ax, yb, ax, ya + rad + 1, C_BLUE, 1.8, 7)
            gr = QRectF(ax + 6, (ya + yb) / 2 - 10, 100, 20)
            draw_rbox(p, gr, "#EFF6FF", C_BLUE_M, 1, 4)
            draw_text(p, gr.x(), gr.y(), gr.width(), gr.height(), f"×(−{nm}) {num(-gv, 4)}", 8.5, C_BLUE, True, CENTER)
        yr = QRectF(L, yb - 13, 92, 26)
        draw_rbox(p, yr, C_BLUE, C_BLUE, 1.2, 6)
        draw_text(p, yr.x(), yr.y(), yr.width(), yr.height(), f"y = {c.short(i['y'])}", 9.5, "white", True, CENTER)
        draw_arrow(p, A[0], yb, L + 94, yb, C_BLUE, 2, 7)
        # 右側：三行公式的結果
        tx = L + 590
        tw = W - 590
        lines = [("① y = b0·x + z1", c.f(i["y"]), C_BLUE),
                 ("② z1 ← b1·x − a1·y + z2", c.f(i["z1"]), "#B45309"),
                 ("③ z2 ← b2·x − a2·y", c.f(i["z2"]), "#B45309")]
        draw_text(p, tx, T, tw, 18, "每一步就這三行：", 9.5, C_MUTED, True)
        for k, (lab, val, col) in enumerate(lines):
            draw_text(p, tx, T + 24 + 44 * k, tw, 18, lab, 9.5, C_TEXT, True)
            draw_text(p, tx + 14, T + 42 + 44 * k, tw, 18, f"= {val}", 10, col, True)
        draw_text(p, tx, T + 160, tw, 36,
                  "藍 = 輸出回授（−a1, −a2）\n琥珀 = 延遲暫存器（下一步才用）", 8.5, C_MUTED)


# ---------- 8. One Euro ----------
class OneEuroAdapter(Adapter):
    key, button, title = "oneeuro", "One Euro", "One Euro Filter（速度自適應低通）"
    state_title = "速度 |dx_hat| → 自適應截止頻率 fc → 新量測權重 α"
    one_liner = "會看速度調整截止頻率的 EMA：慢的時候很平滑、快的時候延遲很小。"
    pros = "同時解決「靜止抖動」與「移動拖延」；調參有固定流程（先 min_cutoff、再 β）；很便宜。"
    cons = "β 跟訊號單位有關（度、公尺、像素差好幾個數量級）；β 太大靜止也會抖；尖峰會被當成快速移動放行。"
    when = "滑鼠 / 觸控 / 手勢、VR 頭部追蹤、即時顯示的感測值。"
    params = [ParamSpec("mincut", "最低截止 min_cutoff", 0.05, 3.0, 0.05, 0.3, "Hz", 2),
              ParamSpec("beta", "速度係數 β", 0.0, 2.0, 0.01, 0.5, "", 2, scaling="div")]

    def build(self, pv, dt):
        return F.OneEuroFilter(dt, pv["mincut"], pv["beta"], d_cutoff=1.0)

    def feed(self, f, s):
        y_prev = f.y
        x = s["meas"]
        y = f.update(x)
        return dict(x=x, y=y, y_prev=y_prev, dx=f.dx, edx=f.dx_hat, fc=f.cutoff, alpha=f.alpha,
                    first=y_prev is None, mincut=f.min_cutoff, beta=f.beta)

    def param_note(self, pv, c):
        return ("調法：β=0 先把 min_cutoff 調到靜止不抖，再加大 β 直到移動時不拖；"
                f"d_cutoff 固定 1 Hz{'；IMU 用「度」，β 自動縮小 30 倍' if c.scale != 1.0 else ''}")

    def idle_lines(self, f, c):
        return [f"min_cutoff = {f.min_cutoff:.2f} Hz、β = {f.beta:.4g}、d_cutoff = {f.d_cutoff:.1f} Hz。",
                "fc = min_cutoff + β·|dx_hat|：動得越快，截止頻率越高、跟得越緊。"]

    def explain(self, i, c):
        if i["first"]:
            return f"第 1 筆：直接輸出 y = x = {c.f(i['x'])}，速度估計 dx_hat 先設 0。"
        a = i["alpha"]
        moving = i["fc"] > 1.6 * i["mincut"]
        return (f"① 速度：dx = (x − y_prev)/dt = ({c.f(i['x'])}{c.pm(-i['y_prev'])})/{DT} = {c.f(i['dx'])}{c.unit}/s，"
                f"再用 1 Hz 的 EMA 濾成 dx_hat = {c.f(i['edx'])}　"
                f"② 自適應截止：fc = min_cutoff + β·|dx_hat| = {i['mincut']:.2f} + {i['beta']:.4g}×{c.f(abs(i['edx']))}"
                f" = {i['fc']:.2f} Hz　"
                f"③ α = dt/(RC+dt) = {a:.3f}，y = α·x + (1−α)·y_prev = {a:.3f}×{c.fp(i['x'])} + "
                f"{1 - a:.3f}×{c.fp(i['y_prev'])} = {c.f(i['y'])}。"
                + ("訊號正在動 → fc 被拉高 → 跟得緊、延遲小。" if moving else
                   "訊號幾乎沒動 → fc 貼近 min_cutoff → 很平滑，把抖動壓掉。"))

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        if i["first"]:
            draw_text(p, L, T + 70, W, 24, f"第 1 筆：y = x = {c.f(i['x'])}，dx_hat = 0（暖機）", 11, C_BLUE, True)
            return
        seen = [inf for inf in c.infos if not inf["first"]]
        vmax = max([abs(inf["edx"]) for inf in seen] + [c.scale]) * 1.1
        fmax = max([inf["fc"] for inf in seen] + [2 * i["mincut"], 1.0]) * 1.1
        draw_gauge(p, L, T + 6, W, 30, "① 速度 |dx_hat|", abs(i["edx"]), vmax,
                   f"{c.f(abs(i['edx']))} {c.unit}/s", C_AMBER)
        draw_gauge(p, L, T + 52, W, 30, "② 截止頻率 fc", i["fc"], fmax, f"{i['fc']:.2f} Hz", C_GREEN,
                   ticks=[(i["mincut"], "min_cutoff")])
        draw_gauge(p, L, T + 98, W, 30, "③ 新量測權重 α", i["alpha"], 1.0, f"{i['alpha']:.3f}", C_BLUE)
        draw_text(p, L, T + 146, W, 22,
                  f"fc = {i['mincut']:.2f} + {i['beta']:.4g}×{c.f(abs(i['edx']))} = {i['fc']:.2f} Hz　→　"
                  f"y = {i['alpha']:.3f}×{c.fp(i['x'])} + {1 - i['alpha']:.3f}×{c.fp(i['y_prev'])} = {c.f(i['y'])}",
                  10, C_BLUE, True)
        draw_text(p, L, T + 172, W, 22,
                  "靜止 → |dx_hat| 小 → fc ≈ min_cutoff → 很平滑；移動 → fc 拉高 → α 變大 → 延遲變小。",
                  9.5, C_MUTED)


# ---------- 9. 斜率限制 ----------
class RateLimiterAdapter(Adapter):
    key, button, title = "rate", "斜率限制", "斜率限制（Rate / Slew Limiter）"
    state_title = "想走多少 Δ vs 允許多少 ±max_rate·dt"
    one_liner = "每一步最多只准變 max_rate·dt：小變化照走，大跳變被拉成斜坡。"
    pros = "小變化零延遲、零失真；保證輸出變化率有上限；最簡單（1 減 1 比較）。"
    cons = "白噪照樣通過（它不管小抖動）；大步階要 |Δ|/max_rate 秒才走到；尖峰後要花時間走回來。"
    when = "致動器命令（馬達、閥門）、setpoint 斜坡化、保護機構、UI 數字平順變化。"
    params = [ParamSpec("rate", "最大斜率 max_rate", 0.1, 5.0, 0.1, 1.5, "{u}/s", 1, scaling="mul")]

    def build(self, pv, dt):
        return F.RateLimiter(pv["rate"], dt)

    def feed(self, f, s):
        x = s["meas"]
        first = f.y is None
        y = f.update(x)
        return dict(x=x, y=y, y_prev=f.y_prev, wanted=f.wanted, up=f.up, down=f.down,
                    step=f.step, limited=f.limited, first=first)

    def param_note(self, pv, c):
        span = f"{c.scale:g}{c.unit}"
        return (f"每步最多變 max_rate·dt = {c.f(pv['rate'] * DT)}{c.unit}；"
                f"走完 {span} 的大跳變要 {c.scale / pv['rate']:.1f} s")

    def idle_lines(self, f, c):
        return [f"每步允許的變化量：±max_rate·dt = ±{c.f(f.up)}{c.unit}。",
                "內部只存一個數：上一次的輸出。"]

    def explain(self, i, c):
        if i["first"]:
            return f"第 1 筆：直接採用 y = x = {c.f(i['x'])}（不然會從 0 慢慢爬上來）。"
        rate = c.pv["rate"]
        if i["limited"]:
            sign = "+" if i["wanted"] > 0 else "−"
            lim = i["up"] if i["wanted"] > 0 else i["down"]
            return (f"想從 {c.f(i['y_prev'])} 走到 {c.f(i['x'])}（Δ = {c.f(i['wanted'])}），"
                    f"但每步最多只能動 max_rate·dt = {c.f(rate, 2)}×{DT} = {c.f(lim)} → 被夾住，"
                    f"y = {c.f(i['y_prev'])} {sign} {c.f(lim)} = {c.f(i['y'])}。大跳變被拉成斜坡，"
                    f"這就是它保護致動器的方式。")
        return (f"想走 Δ = {c.f(i['wanted'])}，在允許範圍 ±{c.f(i['up'])} 之內 → 照走，y = x = {c.f(i['y'])}"
                f"（小變化完全不受影響——所以白噪也會原封不動地通過）。")

    def paint(self, p, r, i, c):
        L, T, W = r.left(), r.top(), r.width()
        if i["first"]:
            draw_text(p, L, T + 70, W, 24, f"第 1 筆：y = x = {c.f(i['x'])}（暖機）", 11, C_BLUE, True)
            return
        yp, x, y = i["y_prev"], i["x"], i["y"]
        lo, hi = span_of([yp, x, y, yp - i["down"], yp + i["up"]], c.min_span())
        col = C_RED if i["limited"] else C_BLUE
        draw_numline(p, L + 70, L + W - 70, T + 70, lo, hi,
                     marks=[(yp, f"y_prev = {c.f(yp)}", C_BLUE_D, "ring"),
                            (x, f"想去 x = {c.f(x)}", "#475569", "dot"),
                            (y, f"y = {c.f(y)}", col, "dot")],
                     band=(yp - i["down"], yp + i["up"], C_GREEN, "允許範圍"),
                     arrows=[(yp, x, C_GRAY, f"想走 Δ = {c.f(i['wanted'])}", 1, True)])
        big = max(abs(i["wanted"]), i["up"], 1e-12)
        bx, bw = L + 150, W - 330
        for k, (lab, v, colr) in enumerate([("想走 |Δ|", abs(i["wanted"]), C_GRAY),
                                            ("允許 max_rate·dt", i["up"], C_GREEN)]):
            yy = T + 134 + 26 * k
            draw_text(p, L, yy, 150, 22, lab, 9.5, C_TEXT, True)
            draw_rbox(p, QRectF(bx, yy + 4, bw, 14), "#F1F5F9", C_GRAY_L, 1, 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(colr)))
            p.drawRoundedRect(QRectF(bx, yy + 4, max(3.0, bw * v / big), 14), 4, 4)
            draw_text(p, bx + bw + 10, yy, 170, 22, c.f(v), 9.5, colr, True)
        msg = (f"|Δ| > 允許 → 被夾住：y = y_prev ± {c.f(i['up'])} = {c.f(y)}" if i["limited"]
               else f"|Δ| ≤ 允許 → 照走：y = x = {c.f(y)}")
        draw_text(p, L, T + 188, W, 22, msg, 10.5, col, True)


ADAPTERS: List[Adapter] = [MAAdapter(), EMAAdapter(), MedianAdapter(), HampelAdapter(),
                           AlphaBetaAdapter(), ComplementaryAdapter(), BiquadAdapter(),
                           OneEuroAdapter(), RateLimiterAdapter()]

SIGNAL_ORDER = [("sine", "正弦+雜訊"), ("step", "步階+雜訊"), ("spike", "尖峰脈衝"),
                ("ramp", "斜坡+雜訊"), ("imu", "IMU 姿態角")]


# =========================================================
#  主圖：真值 / 原始量測 / 濾波輸出，只畫到播放頭
# =========================================================
class PlotWidget(QWidget):
    ML, MR, MT, MB = 62, 16, 26, 40

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setFixedSize(PLOT_W, PLOT_H)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.main
        sig = m.sig
        k = m.frame
        draw_rbox(p, QRectF(1, 1, self.width() - 2, self.height() - 2), "#FFFFFF", C_BORDER, 1, 10)
        pr = QRectF(self.ML, self.MT, self.width() - self.ML - self.MR,
                    self.height() - self.MT - self.MB)
        lo, hi = m.y_range

        def X(idx):
            return pr.left() + idx / (N_SAMPLES - 1) * pr.width()

        def Y(v):
            return pr.bottom() - (v - lo) / (hi - lo) * pr.height()

        # 軸標題（y 軸標題直排在左邊，讓上方留給播放頭標籤）
        ylab = "角度（°）" if sig["unit"] == "°" else "數值"
        p.save()
        p.translate(12, pr.center().y())
        p.rotate(-90)
        draw_text(p, -80, -9, 160, 18, ylab, 9, C_MUTED, True, CENTER)
        p.restore()

        # 未來區域（還沒播放到）淡灰
        xh = X(k - 1) if k > 0 else pr.left()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#F1F5F9")))
        p.drawRect(QRectF(xh, pr.top(), pr.right() - xh, pr.height()))

        # y 刻度與格線
        ticks, step = nice_ticks(lo, hi, 6)
        dec = 0 if step >= 1 else (1 if step >= 0.1 else 2)
        for tv in ticks:
            yy = Y(tv)
            p.setPen(QPen(QColor("#E2E8F0") if abs(tv) > 1e-12 else QColor(C_GRAY_L), 1))
            p.drawLine(QPointF(pr.left(), yy), QPointF(pr.right(), yy))
            draw_text(p, 4, yy - 8, self.ML - 10, 16, num(tv, dec), 8.5, C_MUTED, False, RIGHT_VC)
        # x 刻度
        for xi in range(0, N_SAMPLES + 1, 20):
            idx = min(xi, N_SAMPLES - 1)
            xx = X(idx)
            p.setPen(QPen(QColor("#E2E8F0"), 1))
            p.drawLine(QPointF(xx, pr.top()), QPointF(xx, pr.bottom()))
            draw_text(p, xx - 30, pr.bottom() + 3, 60, 14, str(xi), 8.5, C_MUTED, False, CENTER)
        draw_text(p, pr.left(), pr.bottom() + 18, pr.width(), 16,
                  f"取樣序號 k（dt = {DT} s，fs = {FS:.0f} Hz，整段 {N_SAMPLES * DT:.0f} 秒）",
                  8.5, C_MUTED, False, LEFT_VC)
        draw_text(p, pr.left(), pr.bottom() + 18, pr.width(), 16,
                  f"{sig['name']}　×　{m.adapter.button}", 9, "#1D4ED8", True, RIGHT_VC)
        p.setPen(QPen(QColor(C_GRAY), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(pr)

        p.save()
        p.setClipRect(pr.adjusted(-6, -6, 6, 6))
        if k > 0:
            idxs = range(k)
            truth, meas = sig["truth"], sig["meas"]
            # 原始量測：灰細線 + 灰點
            p.setPen(QPen(QColor(C_GRAY_L), 1))
            p.drawPolyline(QPolygonF([QPointF(X(j), Y(meas[j])) for j in idxs]))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(C_GRAY)))
            for j in idxs:
                p.drawEllipse(QPointF(X(j), Y(meas[j])), 2.2, 2.2)
            # IMU：純陀螺積分（橘虛線）
            if "gyro_int" in sig:
                pen = QPen(QColor(C_ORANGE), 2, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolyline(QPolygonF([QPointF(X(j), Y(sig["gyro_int"][j])) for j in idxs]))
            # 真值：綠虛線
            pen = QPen(QColor(C_GREEN), 2.2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(X(j), Y(truth[j])) for j in idxs]))
            # Hampel：被判離群的點畫紅圈
            if m.adapter.key == "hampel":
                p.setPen(QPen(QColor(C_RED), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                for j, inf in enumerate(m.infos):
                    if inf["outlier"]:
                        p.drawEllipse(QPointF(X(j), Y(meas[j])), 6, 6)
            # 濾波輸出：藍粗線
            pen = QPen(QColor(C_BLUE), 2.8)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QPolygonF([QPointF(X(j), Y(m.ys[j])) for j in idxs]))
        p.restore()

        # 播放頭
        p.setPen(QPen(QColor(C_AMBER), 2))
        p.drawLine(QPointF(xh, pr.top() - 4), QPointF(xh, pr.bottom()))
        lab = f"k = {k}" if k > 0 else "k = 0（還沒開始）"
        tw = text_w(lab, 8.5, True) + 10
        lx = min(max(xh - tw / 2, pr.left()), pr.right() - tw)
        draw_rbox(p, QRectF(lx, 4, tw, 17), C_AMBER_L, C_AMBER, 1, 5)
        draw_text(p, lx, 4, tw, 17, lab, 8.5, "#78350F", True, CENTER)

        # 目前這一點：加大標記
        if k > 0:
            j = k - 1
            xm, ym, yy = X(j), Y(sig["meas"][j]), Y(m.ys[j])
            p.setPen(QPen(QColor(C_AMBER), 1.5, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(xm, ym), QPointF(xm, yy))
            p.setPen(QPen(QColor("white"), 1.5))
            p.setBrush(QBrush(QColor("#64748B")))
            p.drawEllipse(QPointF(xm, ym), 5.5, 5.5)
            p.setPen(QPen(QColor("white"), 2))
            p.setBrush(QBrush(QColor(C_BLUE)))
            p.drawEllipse(QPointF(xm, yy), 7, 7)
        p.end()


# =========================================================
#  內部狀態面板
# =========================================================
class StateWidget(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setFixedSize(PLOT_W, STATE_H)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.main
        draw_rbox(p, QRectF(1, 1, self.width() - 2, self.height() - 2), "#FFFFFF", C_BORDER, 1, 10)
        draw_text(p, 14, 6, self.width() - 28, 22, f"🔍 內部狀態：{m.adapter.state_title}",
                  10.5, "#1D4ED8", True)
        r = QRectF(16, 34, self.width() - 32, self.height() - 40)
        c = m.ctx()
        if m.frame == 0:
            lines = ["還沒吃進任何取樣。按「⏭ 單步」吃進第 1 個，這裡會畫出濾波器內部的每個數字。"]
            lines += m.adapter.idle_lines(m.filt, c)
            for k, ln in enumerate(lines):
                draw_text(p, r.left(), r.top() + 20 + 30 * k, r.width(), 24, ln,
                          10.5 if k == 0 else 10, C_TEXT if k == 0 else C_MUTED, k == 0)
        else:
            m.adapter.paint(p, r, m.infos[-1], c)
        p.end()


# =========================================================
#  圖例小樣本
# =========================================================
class LegendSwatch(QWidget):
    def __init__(self, kind, color):
        super().__init__()
        self.kind, self.color = kind, color
        self.setFixedSize(30, 16)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self.color)
        y = self.height() / 2
        if self.kind == "dash":
            p.setPen(QPen(c, 2.2, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(2, y), QPointF(28, y))
        elif self.kind == "dots":
            p.setPen(QPen(QColor(C_GRAY_L), 1))
            p.drawLine(QPointF(2, y), QPointF(28, y))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(c))
            for x in (6, 15, 24):
                p.drawEllipse(QPointF(x, y), 2.4, 2.4)
        elif self.kind == "thick":
            p.setPen(QPen(c, 3))
            p.drawLine(QPointF(2, y), QPointF(28, y))
        elif self.kind == "vline":
            p.setPen(QPen(c, 2))
            p.drawLine(QPointF(15, 1), QPointF(15, 15))
        elif self.kind == "ring":
            p.setPen(QPen(c, 2))
            p.drawEllipse(QPointF(15, y), 5.5, 5.5)
        p.end()


# =========================================================
#  主視窗
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工程常用濾波器　逐點動畫比較器（解說版）")
        self.signals = F.make_demo_signals(N_SAMPLES, DT, SEED)
        self.adapter: Adapter = ADAPTERS[1]          # 預設 EMA
        self.sig_key = "sine"
        self.param_pos: Dict[str, List[int]] = {
            a.key: [s.pos_of(s.default) for s in a.params] for a in ADAPTERS}
        self.prev_sig_key = "sine"                    # 從互補濾波切走時，回到原本的訊號
        self.frame = 0
        self.filt = None
        self.full_out: Optional[np.ndarray] = None
        self.ys: List[float] = []
        self.infos: List[dict] = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self._build_ui()
        self._sync_controls()
        self.rebuild_to(0)

    # ---------- 屬性 ----------
    @property
    def sig(self):
        return self.signals[self.sig_key]

    @property
    def y_range(self):
        """y 軸範圍：訊號本身 + 這組參數跑完整段的輸出（例如 Q 很大的過衝），整段固定不跳動。"""
        s = self.sig
        arrs = [s["truth"], s["meas"]] + ([s["gyro_int"]] if "gyro_int" in s else [])
        lo = min(float(np.min(a)) for a in arrs)
        hi = max(float(np.max(a)) for a in arrs)
        span = hi - lo
        if self.full_out is not None:
            lo = min(lo, max(float(np.min(self.full_out)), lo - 0.5 * span))
            hi = max(hi, min(float(np.max(self.full_out)), hi + 0.5 * span))
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    def pv(self) -> Dict[str, float]:
        pos = self.param_pos[self.adapter.key]
        return {s.key: s.value(pos[k], self.sig["scale"]) for k, s in enumerate(self.adapter.params)}

    def ctx(self) -> Ctx:
        return Ctx(self.sig, self.pv(), self.frame, self.infos)

    # ---------- 介面組裝 ----------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())

        title = QLabel("工程常用濾波器　—　逐點看：每吃進一個取樣，濾波器心裡算了什麼")
        title.setStyleSheet("font-size:17px;font-weight:bold;color:#1D4ED8;")
        root.addWidget(title)

        why = QLabel(
            "🔑 所有濾波都在做同一個取捨：越平滑 ↔ 越延遲。先看雜訊長什麼樣子（白噪？尖峰？漂移？）再挑濾波器；"
            "停在任一點拖動參數滑桿，藍線會立刻從頭重算，親眼看它在「抖」和「慢」之間移動。")
        why.setWordWrap(True)
        why.setStyleSheet(
            "font-size:12px;color:#1E3A8A;background:#EFF6FF;"
            "border:1px solid #BFDBFE;border-radius:6px;padding:7px;")
        root.addWidget(why)

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self._build_left(), 0, Qt.AlignmentFlag.AlignTop)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.plot = PlotWidget(self)
        right.addWidget(self.plot)
        right.addWidget(self._build_legend())
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setFixedWidth(PLOT_W)
        self.stats.setStyleSheet("font-size:13px;color:#0F172A;padding:2px;")
        right.addWidget(self.stats)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setFixedWidth(PLOT_W)
        self.metrics.setStyleSheet(
            "font-size:13px;color:#0F172A;background:#FFFFFF;border:1px solid #E2E8F0;"
            "border-radius:6px;padding:5px 8px;")
        right.addWidget(self.metrics)
        self.state = StateWidget(self)
        right.addWidget(self.state)
        self.explain = QLabel()
        self.explain.setWordWrap(True)
        self.explain.setFixedWidth(PLOT_W)
        self.explain.setMinimumHeight(96)
        self.explain.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.explain.setStyleSheet(
            "font-size:13px;color:#0F172A;background:#FFFFFF;"
            "border:1px solid #E2E8F0;border-left:4px solid #2563EB;"
            "border-radius:6px;padding:8px;")
        right.addWidget(self.explain)
        right.addStretch(1)
        body.addLayout(right)
        body.addStretch(1)
        root.addLayout(body)

        container = QWidget()
        container.setLayout(root)
        # 注意：背景色一定要用 #root 限定範圍；沒有選擇器的 "background:..." 會一路蔓延到子按鈕，
        # 蓋掉 QPushButton:checked 的藍底，選中的按鈕會變成白底白字（看不見）。
        container.setObjectName("root")
        container.setStyleSheet("QWidget#root { background:#F8FAFC; }")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        self.resize(1210, 960)
        self.setStyleSheet("""
            QPushButton {
                background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
                padding:7px 12px; font-size:13px; color:#0F172A;
            }
            QPushButton:hover { background:#EFF6FF; border-color:#60A5FA; }
            QPushButton:checked { background:#2563EB; color:white; border-color:#2563EB; }
            QPushButton:disabled { background:#F1F5F9; color:#CBD5E1; border-color:#E2E8F0; }
            QLabel { color:#0F172A; }
        """)

    def _build_toolbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        btn_back = QPushButton("⏮ 上一步")
        btn_back.clicked.connect(self.step_back)
        bar.addWidget(btn_back)
        btn_step = QPushButton("⏭ 單步（吃一個取樣）")
        btn_step.clicked.connect(self.step_once)
        bar.addWidget(btn_step)
        self.btn_play = QPushButton("▶ 自動播放")
        self.btn_play.clicked.connect(self.toggle_play)
        bar.addWidget(self.btn_play)
        btn_end = QPushButton("⏩ 直接跑完")
        btn_end.clicked.connect(self.run_to_end)
        bar.addWidget(btn_end)
        btn_reset = QPushButton("↺ 從頭重來")
        btn_reset.clicked.connect(self.reset_run)
        bar.addWidget(btn_reset)
        bar.addStretch(1)
        bar.addWidget(QLabel("速度（慢→快）："))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(60)
        self.speed.setFixedWidth(150)
        self.speed.valueChanged.connect(self.update_speed)
        bar.addWidget(self.speed)
        return bar

    def _section(self, text):
        lab = QLabel(text)
        lab.setStyleSheet("font-size:13px;font-weight:bold;color:#334155;padding-top:4px;")
        return lab

    def _build_left(self):
        panel = QWidget()
        panel.setFixedWidth(LEFT_W)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(self._section("① 濾波器"))
        grid = QGridLayout()
        grid.setSpacing(6)
        self.filter_group = QButtonGroup(self)
        self.filter_btns: Dict[str, QPushButton] = {}
        for idx, a in enumerate(ADAPTERS):
            b = QPushButton(a.button)
            b.setCheckable(True)
            b.clicked.connect(lambda _, key=a.key: self.set_filter(key))
            self.filter_group.addButton(b)
            self.filter_btns[a.key] = b
            grid.addWidget(b, idx // 2, idx % 2)
        lay.addLayout(grid)

        lay.addWidget(self._section("② 訊號（固定亂數種子，可重現）"))
        grid = QGridLayout()
        grid.setSpacing(6)
        self.signal_group = QButtonGroup(self)
        self.signal_btns: Dict[str, QPushButton] = {}
        for idx, (key, text) in enumerate(SIGNAL_ORDER):
            b = QPushButton(text)
            b.setCheckable(True)
            b.clicked.connect(lambda _, k=key: self.set_signal(k))
            self.signal_group.addButton(b)
            self.signal_btns[key] = b
            grid.addWidget(b, idx // 2, idx % 2)
        lay.addLayout(grid)
        self.imu_hint = QLabel("互補濾波需要「陀螺儀 + 加速度計」兩路輸入，只能用 IMU 訊號。")
        self.imu_hint.setWordWrap(True)
        self.imu_hint.setStyleSheet("font-size:11px;color:#B45309;")
        lay.addWidget(self.imu_hint)

        lay.addWidget(self._section("③ 參數（改了會從頭重算到目前位置）"))
        self.param_labels: List[QLabel] = []
        self.param_sliders: List[QSlider] = []
        for k in range(2):
            lab = QLabel()
            lab.setStyleSheet("font-size:12px;color:#0F172A;")
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.valueChanged.connect(lambda v, kk=k: self.on_param(kk, v))
            lay.addWidget(lab)
            lay.addWidget(sl)
            self.param_labels.append(lab)
            self.param_sliders.append(sl)
        self.param_note = QLabel()
        self.param_note.setWordWrap(True)
        self.param_note.setStyleSheet("font-size:11px;color:#64748B;")
        lay.addWidget(self.param_note)

        self.card = QLabel()
        self.card.setWordWrap(True)
        self.card.setTextFormat(Qt.TextFormat.RichText)
        self.card.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.card.setStyleSheet(
            "font-size:12px;color:#0F172A;background:#FFFFFF;border:1px solid #E2E8F0;"
            "border-top:4px solid #F59E0B;border-radius:6px;padding:8px;")
        lay.addSpacing(4)
        lay.addWidget(self.card)
        lay.addStretch(1)
        return panel

    def _build_legend(self):
        frame = QFrame()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(4)
        self.legend_items = {}
        items = [("truth", "dash", C_GREEN, "真值"),
                 ("raw", "dots", C_GRAY, "原始量測"),
                 ("out", "thick", C_BLUE, "濾波輸出"),
                 ("head", "vline", C_AMBER, "播放頭"),
                 ("gyro", "dash", C_ORANGE, "純陀螺積分（會漂）"),
                 ("outlier", "ring", C_RED, "Hampel 判為離群")]
        for key, kind, color, text in items:
            sw = LegendSwatch(kind, color)
            lab = QLabel(text)
            lab.setStyleSheet("font-size:12px;color:#334155;")
            lay.addWidget(sw)
            lay.addWidget(lab)
            lay.addSpacing(10)
            self.legend_items[key] = (sw, lab)
        lay.addStretch(1)
        frame.setFixedWidth(PLOT_W)
        return frame

    # ---------- 控制項同步 ----------
    def _sync_controls(self):
        self.filter_btns[self.adapter.key].setChecked(True)
        self.signal_btns[self.sig_key].setChecked(True)
        imu_only = self.adapter.needs_imu
        for key, b in self.signal_btns.items():
            b.setEnabled(key == "imu" or not imu_only)
        self.imu_hint.setVisible(imu_only)
        pos = self.param_pos[self.adapter.key]
        for k in range(2):
            sl, lab = self.param_sliders[k], self.param_labels[k]
            if k < len(self.adapter.params):
                spec = self.adapter.params[k]
                sl.blockSignals(True)
                sl.setRange(0, spec.count())
                sl.setValue(pos[k])
                sl.blockSignals(False)
                sl.setVisible(True)
                lab.setVisible(True)
            else:
                sl.setVisible(False)
                lab.setVisible(False)
        is_imu = "gyro_int" in self.sig
        for w in self.legend_items["gyro"]:
            w.setVisible(is_imu)
        for w in self.legend_items["outlier"]:
            w.setVisible(self.adapter.key == "hampel")

    def _update_param_texts(self):
        pv = self.pv()
        for k, spec in enumerate(self.adapter.params):
            self.param_labels[k].setText(f"<b>{spec.label}</b> = {spec.text(pv[spec.key], self.sig)}")
        self.param_note.setText(self.adapter.param_note(pv, self.ctx()))
        a = self.adapter
        self.card.setText(
            f"<div style='font-size:13px;font-weight:bold;color:#92400E;'>🧠 設計思路：{a.title}</div>"
            f"<p style='margin:6px 0 4px 0;'>{a.one_liner}</p>"
            f"<p style='margin:4px 0;'><b style='color:#15803D;'>✅ 優點：</b>{a.pros}</p>"
            f"<p style='margin:4px 0;'><b style='color:#B91C1C;'>⚠️ 缺點：</b>{a.cons}</p>"
            f"<p style='margin:4px 0 0 0;'><b style='color:#1D4ED8;'>🎯 何時用：</b>{a.when}</p>")

    # ---------- 使用者操作 ----------
    def set_filter(self, key):
        new = next(a for a in ADAPTERS if a.key == key)
        if new is self.adapter:
            return
        if new.needs_imu and self.sig_key != "imu":
            self.prev_sig_key = self.sig_key
            self.sig_key = "imu"                      # 互補濾波需要兩路輸入 → 自動切到 IMU
        self.adapter = new
        self._sync_controls()
        self.rebuild_to(self.frame)

    def set_signal(self, key):
        if key == self.sig_key:
            return
        self.sig_key = key
        self._sync_controls()
        self.rebuild_to(self.frame)

    def on_param(self, k, v):
        self.param_pos[self.adapter.key][k] = v
        self.rebuild_to(self.frame)

    # ---------- 播放 / 速度 ----------
    def toggle_play(self):
        if self.timer.isActive():
            self._stop()
        else:
            if self.frame >= N_SAMPLES:
                self.rebuild_to(0)
            self.update_speed()
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def _stop(self, text="▶ 自動播放"):
        self.timer.stop()
        self.btn_play.setText(text)

    def update_speed(self):
        self.timer.setInterval(int(620 - 6 * self.speed.value()))

    # ---------- 前進 / 倒退 ----------
    def _sample(self, i):
        s = self.sig
        d = dict(i=i, t=float(s["t"][i]), truth=float(s["truth"][i]), meas=float(s["meas"][i]))
        if "gyro" in s:
            d.update(gyro=float(s["gyro"][i]), accel=float(s["accel"][i]), gyro_int=float(s["gyro_int"][i]))
        return d

    def advance_one(self):
        if self.frame >= N_SAMPLES:
            return False
        info = self.adapter.feed(self.filt, self._sample(self.frame))
        self.ys.append(info["y"])
        self.infos.append(info)
        self.frame += 1
        return True

    def tick(self):
        if not self.advance_one():
            self._stop("▶ 重新播放")
        elif self.frame >= N_SAMPLES:
            self._stop("▶ 重新播放")
        self.refresh()

    def step_once(self):
        if self.timer.isActive():
            self._stop()
        self.advance_one()
        self.refresh()

    def step_back(self):
        if self.timer.isActive():
            self._stop()
        if self.frame > 0:
            self.rebuild_to(self.frame - 1)

    def run_to_end(self):
        self._stop("▶ 重新播放")
        self.rebuild_to(N_SAMPLES)

    def reset_run(self):
        self._stop()
        self.rebuild_to(0)

    def rebuild_to(self, n):
        """倒退 / 改設定的做法：重建濾波器，再從頭快轉 n 步（訊號固定 seed，結果完全可重現）。"""
        pv = self.pv()
        probe = self.adapter.build(pv, DT)          # 先整段跑一次，只為了決定 y 軸範圍
        self.full_out = np.array([self.adapter.feed(probe, self._sample(i))["y"]
                                  for i in range(N_SAMPLES)])
        self.filt = self.adapter.build(pv, DT)
        self.frame = 0
        self.ys, self.infos = [], []
        for _ in range(min(n, N_SAMPLES)):
            self.advance_one()
        self.refresh()

    # ---------- 更新顯示 ----------
    def refresh(self):
        if not self.timer.isActive():
            self.btn_play.setText("▶ 重新播放" if self.frame >= N_SAMPLES else "▶ 自動播放")
        self._update_param_texts()
        self.plot.update()
        self.state.update()
        self.stats.setText(self._stats_text())
        self.metrics.setText(self._metrics_text())
        c = self.ctx()
        if self.frame == 0:
            txt = (f"還沒開始：{self.adapter.title} 已就緒，訊號是「{self.sig['name']}」。"
                   f"按「⏭ 單步」吃進第 1 個取樣，或按「▶ 自動播放」。")
            if self.adapter.key != "comp" and self.sig_key == "imu":
                txt += "（在 IMU 訊號上，這個濾波器吃的是加速度計算出的角度。）"
            self.explain.setText(txt)
        else:
            self.explain.setText(f"<b>為何是這個輸出（第 {self.frame} 個取樣）：</b>"
                                 + self.adapter.explain(self.infos[-1], c))

    def _stats_text(self):
        k = self.frame
        if k == 0:
            return f"第 0 / {N_SAMPLES} 個取樣　|　狀態：等待開始　|　濾波器：{self.adapter.button}"
        s = self._sample(k - 1)
        c = self.ctx()
        u = self.sig["unit"]
        status = "✅ 跑完了" if k >= N_SAMPLES else "播放中…" if self.timer.isActive() else "暫停"
        if "gyro" in s:
            meas = f"加速度計 a = {c.f(s['accel'])}°　|　陀螺 ω = {c.f(s['gyro'])}°/s"
        else:
            meas = f"量測 x = {c.f(s['meas'])}{u}"
        return (f"第 {k} / {N_SAMPLES} 個取樣　|　t = {s['t']:.2f} s　|　{status}　|　"
                f"{meas}　|　輸出 y = {c.f(self.ys[-1])}{u}　|　真值 = {c.f(s['truth'])}{u}")

    def _metrics_text(self):
        k = self.frame
        if k == 0:
            return "📏 指標：還沒有資料（RMSE 與延遲會隨著播放即時更新）"
        s = self.sig
        u = s["unit"]
        c = self.ctx()
        truth = s["truth"][:k]
        ys = np.asarray(self.ys)
        r_f = F.rmse(ys, truth)
        r_r = F.rmse(s["meas"][:k], truth)
        gain = (1 - r_f / r_r) * 100 if r_r > 0 else 0.0
        verdict = (f"改善 {gain:.0f}%" if gain >= 0
                   else f"<span style='color:#B91C1C;'>變差 {-gain:.0f}%</span>")
        parts = [f"📏 RMSE（對真值）：濾波 <b style='color:#2563EB;'>{c.f(r_f)}{u}</b>"
                 f"　原始 {c.f(r_r)}{u}（{verdict}）"]
        if k >= 30 and float(np.std(truth)) > 0.05 * s["scale"]:
            lag = F.align_delay(ys, truth, max_lag=min(25, k // 3))
            parts.append(f"估計延遲 ≈ <b>{lag} 點</b>（{lag * DT:.2f} s）")
        else:
            parts.append("估計延遲：資料還不夠（真值還沒動）")
        if "gyro_int" in s:
            parts.append(f"純陀螺積分 RMSE = <span style='color:#EA580C;'>"
                         f"{c.f(F.rmse(s['gyro_int'][:k], truth))}°</span>")
        return "　|　".join(parts)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setFont(QFont(FONT, 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
