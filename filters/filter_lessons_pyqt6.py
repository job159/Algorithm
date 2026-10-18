# -*- coding: utf-8 -*-
"""
工程常用濾波器：六堂互動課（PyQt6）
=========================================
先看「雜訊長什麼樣」，再看「哪個濾波器對症」—— 一章只解決一種問題：

  第 1 章  濾波在做什麼？用過去幾個點「投票」      移動平均：越平滑 ⇄ 越落後（最根本的取捨）
  第 2 章  只記一個數也能平滑：EMA（一階低通）      權重指數衰減；省記憶體，取捨一樣
  第 3 章  遇到尖峰：平均會被拖累，中位數不會        排隊取中間，瘋狂的點碰不到中間
  第 4 章  訊號在移動：平均一定落後，α-β 會預測      多記一個「速度」→ 預測 + 修正（卡爾曼的雛形）
  第 5 章  兩個感測器互補：陀螺儀 + 加速度計        快的聽陀螺儀、慢的聽加速度計
  第 6 章  怎麼選？                                  一張決策圖 + 一張總表

畫面的顏色固定：灰點 = 原始量測、深灰虛線 = 真值、藍 = 主角濾波器、橘 = 對照組、琥珀 = 窗口 / 權重
濾波器本身全部來自 filters.py（工程用、串流式），這支只負責「教」。

操作：每章都能用 ⏮ ⏭ ▶ 一點一點走，也可以直接用滑鼠在圖上拖動時間游標。

需求： pip install PyQt6 numpy
執行： python filter_lessons_pyqt6.py
"""

import sys
import math

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF, QFontMetricsF

import filters as F


# =========================================================
#  配色：整套課程固定意義
# =========================================================
C_BG     = QColor("#F8FAFC")
C_INK    = QColor("#0F172A")
C_MUTED  = QColor("#64748B")
C_GRID   = QColor("#EEF2F7")
C_FRAME  = QColor("#CBD5E1")
C_RAW    = QColor("#94A3B8")     # 灰點：原始量測
C_TRUE   = QColor("#334155")     # 深灰虛線：真值
C_MAIN   = QColor("#2563EB")     # 藍：這一章的主角濾波器
C_OTHER  = QColor("#EA580C")     # 橘：對照組
C_WIN    = QColor("#F59E0B")     # 琥珀：窗口 / 權重
C_GOOD   = QColor("#16A34A")
C_BAD    = QColor("#DC2626")

HEX_MAIN, HEX_OTHER, HEX_WIN, HEX_RAW = "#2563EB", "#C2410C", "#B45309", "#64748B"


def alpha(c, a):
    q = QColor(c)
    q.setAlpha(a)
    return q


def ui_font(pt, bold=False):
    f = QFont("Microsoft JhengHei")
    f.setPointSizeF(pt)
    f.setBold(bold)
    return f


def span(s, color, bold=True):
    w = "font-weight:700;" if bold else ""
    return f"<span style='color:{color};{w}'>{s}</span>"


def blue(s):
    return span(s, HEX_MAIN)


def orange(s):
    return span(s, HEX_OTHER)


def amber(s):
    return span(s, HEX_WIN)


def gray(s):
    return span(s, HEX_RAW, bold=False)


# =========================================================
#  畫圖小工具
# =========================================================
def draw_text(p, x, y, s, color=C_INK, pt=9.5, bold=False, ha="l", va="b"):
    f = ui_font(pt, bold)
    p.setFont(f)
    fm = QFontMetricsF(f)
    w = fm.horizontalAdvance(s)
    asc, desc = fm.ascent(), fm.descent()
    if ha == "c":
        x -= w / 2
    elif ha == "r":
        x -= w
    if va == "t":
        y += asc
    elif va == "m":
        y += (asc - desc) / 2
    p.setPen(QPen(color))
    p.drawText(QPointF(x, y), s)
    return QRectF(x, y - asc, w, asc + desc)


def text_size(s, pt=9.5, bold=False):
    fm = QFontMetricsF(ui_font(pt, bold))
    return fm.horizontalAdvance(s), fm.height()


def draw_pill(p, x, y, s, fg, bg, border=None, pt=9.5, bold=True, ha="c", pad=7):
    w, h = text_size(s, pt, bold)
    w += 2 * pad
    h += 6
    x0 = x - w / 2 if ha == "c" else (x - w if ha == "r" else x)
    rect = QRectF(x0, y - h / 2, w, h)
    p.setPen(QPen(border, 1.4) if border is not None else Qt.PenStyle.NoPen)
    p.setBrush(QBrush(bg))
    p.drawRoundedRect(rect, h / 2, h / 2)
    draw_text(p, rect.center().x(), rect.center().y(), s, fg, pt, bold, ha="c", va="m")
    return rect


def draw_arrow(p, a, b, color, width=2.0, head=8.0, dash=False):
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(a, b)
    ang = math.atan2(b.y() - a.y(), b.x() - a.x())
    p.setPen(QPen(color, width))
    p.setBrush(QBrush(color))
    p.drawPolygon(QPolygonF([
        b,
        QPointF(b.x() - head * math.cos(ang - 0.42), b.y() - head * math.sin(ang - 0.42)),
        QPointF(b.x() - head * math.cos(ang + 0.42), b.y() - head * math.sin(ang + 0.42))]))


def nice_ticks(lo, hi, n=5):
    span_ = hi - lo
    if span_ <= 0:
        return [lo]
    raw = span_ / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span_ / step <= n:
            break
    v = math.ceil(lo / step) * step
    ticks = []
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


class Plot:
    """把「世界座標」畫進畫布上的某個矩形。"""

    def __init__(self, rect, x0, x1, y0, y1):
        self.r = QRectF(rect)
        self.x0, self.x1, self.y0, self.y1 = float(x0), float(x1), float(y0), float(y1)

    def px(self, x):
        return self.r.left() + (x - self.x0) / (self.x1 - self.x0) * self.r.width()

    def py(self, y):
        return self.r.bottom() - (y - self.y0) / (self.y1 - self.y0) * self.r.height()

    def pt(self, x, y):
        return QPointF(self.px(x), self.py(y))

    def wx(self, px):
        return self.x0 + (px - self.r.left()) / self.r.width() * (self.x1 - self.x0)

    def frame(self, p, xticks=(), yticks=(), xfmt=None, yfmt=None, title="", xlabel=""):
        xfmt = xfmt or (lambda v: f"{v:g}")
        yfmt = yfmt or (lambda v: f"{v:g}")
        r = self.r
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawRect(r)
        p.setPen(QPen(C_GRID, 1))
        for x in xticks:
            if self.x0 <= x <= self.x1:
                p.drawLine(QPointF(self.px(x), r.top()), QPointF(self.px(x), r.bottom()))
        for y in yticks:
            if self.y0 <= y <= self.y1:
                p.drawLine(QPointF(r.left(), self.py(y)), QPointF(r.right(), self.py(y)))
        p.setPen(QPen(C_FRAME, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        for x in xticks:
            if self.x0 <= x <= self.x1:
                draw_text(p, self.px(x), r.bottom() + 3, xfmt(x), C_MUTED, 8.5, ha="c", va="t")
        for y in yticks:
            if self.y0 <= y <= self.y1:
                draw_text(p, r.left() - 6, self.py(y), yfmt(y), C_MUTED, 8.5, ha="r", va="m")
        if title:
            draw_text(p, r.left(), r.top() - 7, title, C_INK, 10, bold=True)
        if xlabel:
            draw_text(p, r.right(), r.bottom() + 18, xlabel, C_MUTED, 8.5, ha="r", va="t")
        p.restore()

    def polyline(self, p, xs, ys, color, width=2.0, dash=False):
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        seg = []
        for x, y in zip(xs, ys):
            if y is None or not np.isfinite(y):
                if len(seg) >= 2:
                    p.drawPolyline(QPolygonF(seg))
                seg = []
                continue
            seg.append(self.pt(x, y))
        if len(seg) >= 2:
            p.drawPolyline(QPolygonF(seg))

    def dots(self, p, xs, ys, color, r=3.0, outline=None):
        p.setPen(QPen(outline, 1.2) if outline is not None else Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                p.drawEllipse(self.pt(x, y), r, r)


# =========================================================
#  共用小元件（跟 kalman_filter/kalman_lessons_pyqt6.py 同一套）
# =========================================================
class LabeledSlider(QWidget):
    def __init__(self, title, lo, hi, value, fmt, on_change, log=False, steps=200,
                 color="#0F172A", title_w=150):
        super().__init__()
        self.lo, self.hi, self.log, self.steps, self.fmt = lo, hi, log, steps, fmt
        self.on_change = on_change
        self._quiet = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        t = QLabel(title)
        t.setStyleSheet(f"QLabel {{ font-size:14px; font-weight:600; color:{color}; }}")
        t.setFixedWidth(title_w)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, steps)
        self.slider.setMinimumWidth(160)
        self.val = QLabel()
        self.val.setMinimumWidth(150)
        self.val.setStyleSheet(f"QLabel {{ font-size:14px; font-weight:700; color:{color}; }}")
        lay.addWidget(t)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val)
        self.set_value(value)
        self.slider.valueChanged.connect(self._changed)

    def _to_val(self, i):
        t = i / self.steps
        if self.log:
            return self.lo * (self.hi / self.lo) ** t
        return self.lo + (self.hi - self.lo) * t

    def _to_int(self, v):
        v = min(max(v, self.lo), self.hi)
        if self.log:
            t = math.log(v / self.lo) / math.log(self.hi / self.lo)
        else:
            t = (v - self.lo) / (self.hi - self.lo)
        return int(round(t * self.steps))

    def value(self):
        return self._to_val(self.slider.value())

    def set_value(self, v):
        self._quiet = True
        self.slider.setValue(self._to_int(v))
        self._quiet = False
        self._show()

    def _show(self):
        v = self.value()
        self.val.setText(self.fmt(v) if callable(self.fmt) else self.fmt.format(v))

    def _changed(self, _):
        self._show()
        if not self._quiet:
            self.on_change()


def button(text, fn, primary=False):
    b = QPushButton(text)
    if primary:
        b.setObjectName("primary")
    b.clicked.connect(fn)
    return b


class Chapter(QWidget):
    """標題 → 💡 一句話重點 → 主畫面 → 下方左「現在發生什麼」、右「🧪 試試看」"""
    NUM = 0
    TITLE = ""
    IDEA = ""
    TRIES = ()

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        head = QLabel(f"第 {self.NUM} 章　{self.TITLE}")
        head.setStyleSheet("QLabel { font-size:22px; font-weight:800; color:#1E3A8A; }")
        outer.addWidget(head)
        idea = QLabel("💡 " + self.IDEA)
        idea.setWordWrap(True)
        idea.setStyleSheet(
            "QLabel { font-size:15px; color:#1E3A8A; background:#EFF6FF; border:1px solid #BFDBFE;"
            " border-left:5px solid #2563EB; border-radius:8px; padding:10px 12px; }")
        outer.addWidget(idea)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.now = QLabel()
        self.now.setWordWrap(True)
        self.now.setTextFormat(Qt.TextFormat.RichText)
        self.now.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.now.setMinimumHeight(110)
        self.now.setStyleSheet(
            "QLabel { font-size:14px; color:#0F172A; background:#FFFFFF; border:1px solid #E2E8F0;"
            " border-left:5px solid #2563EB; border-radius:8px; padding:10px 12px; }")
        bottom.addWidget(self.now, 3)
        if self.TRIES:
            tries = QLabel("<b>🧪 試試看</b><br>" + "<br>".join(f"▸ {t}" for t in self.TRIES))
            tries.setWordWrap(True)
            tries.setTextFormat(Qt.TextFormat.RichText)
            tries.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            tries.setStyleSheet(
                "QLabel { font-size:13px; color:#78350F; background:#FFFBEB; border:1px solid #FDE68A;"
                " border-radius:8px; padding:10px 12px; }")
            bottom.addWidget(tries, 2)
        outer.addLayout(bottom)
        self.build()

    def build(self):
        pass

    def enter(self):
        pass

    def leave(self):
        pass


class Stepper:
    """⏮ 上一點 / ⏭ 下一點 / ▶ 播放 / ↺ 從頭 —— 控制時間游標 ch.s。"""

    def __init__(self, chapter, total_fn, interval=60):
        self.ch = chapter
        self.total_fn = total_fn
        self.timer = QTimer(chapter)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self._tick)
        self.bar = QHBoxLayout()
        self.bar.setSpacing(8)
        self.bar.addWidget(button("⏮ 上一點", self.back))
        self.bar.addWidget(button("⏭ 下一點", self.forward, primary=True))
        self.btn_play = button("▶ 播放", self.toggle)
        self.bar.addWidget(self.btn_play)
        self.bar.addWidget(button("↺ 從頭", self.reset))
        hint = QLabel("也可以用滑鼠在圖上拖動時間游標")
        hint.setStyleSheet("QLabel { color:#64748B; font-size:12px; }")
        self.bar.addSpacing(8)
        self.bar.addWidget(hint)
        self.bar.addStretch(1)

    def forward(self):
        self.stop()
        self._advance()

    def _advance(self):
        if self.ch.s < self.total_fn() - 1:
            self.ch.s += 1
            self.ch.on_step()
            return True
        return False

    def back(self):
        self.stop()
        if self.ch.s > 0:
            self.ch.s -= 1
            self.ch.on_step()

    def reset(self):
        self.stop()
        self.ch.s = 0
        self.ch.on_step()

    def toggle(self):
        if self.timer.isActive():
            self.stop()
        else:
            if self.ch.s >= self.total_fn() - 1:
                self.ch.s = 0
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def stop(self):
        self.timer.stop()
        self.btn_play.setText("▶ 播放")

    def _tick(self):
        if not self._advance():
            self.stop()


class ScrubCanvas(QWidget):
    """可以用滑鼠拖動時間游標的畫布：子類別在 paintEvent 裡把主圖存到 self.main_plot。"""

    def __init__(self, ch, height):
        super().__init__()
        self.ch = ch
        self.main_plot = None
        self.setMinimumHeight(height)

    def _scrub(self, ev):
        pl = self.main_plot
        if pl is None:
            return
        pos = ev.position()
        if pl.r.adjusted(0, -10, 0, 10).contains(pos):
            k = int(round(pl.wx(pos.x())))
            k = min(max(k, 0), self.ch.total() - 1)
            if k != self.ch.s:
                self.ch.stepper.stop()
                self.ch.s = k
                self.ch.on_step()

    def mousePressEvent(self, ev):
        self._scrub(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._scrub(ev)


def draw_signal(p, pl, truth, meas, k, out=None, out_color=C_MAIN, other=None, other_color=C_OTHER,
                cursor=True):
    """通用的訊號圖：真值虛線、量測灰點（未來的點很淡）、濾波輸出畫到游標為止。"""
    n = len(meas)
    ks = np.arange(n)
    pl.polyline(p, ks, truth, C_TRUE, 1.4, dash=True)
    pl.dots(p, ks[k + 1:], meas[k + 1:], alpha(C_RAW, 55), 2.4)
    pl.dots(p, ks[:k + 1], meas[:k + 1], alpha(C_RAW, 200), 2.8)
    if other is not None:
        pl.polyline(p, ks[:k + 1], other[:k + 1], other_color, 2.2, dash=True)
    if out is not None:
        pl.polyline(p, ks[:k + 1], out[:k + 1], out_color, 2.8)
    if cursor:
        pen = QPen(alpha(C_INK, 120), 1.2)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(pl.px(k), pl.r.top()), QPointF(pl.px(k), pl.r.bottom()))


# =========================================================
#  示範訊號
# =========================================================
DT = 0.05     # 20 Hz


def make_smooth_signal():
    """第 1、2 章：慢慢起伏 + 中途一個步階 + 白噪。"""
    n = 160
    k = np.arange(n)
    t = k * DT
    truth = 0.6 * np.sin(2 * math.pi * 0.35 * t) + np.where(k >= 90, 1.2, 0.0)
    meas = truth + np.random.default_rng(3).normal(0, 0.25, n)
    return truth, meas


def make_moving_signal():
    """第 4 章：靜止 → 等速移動 → 停下來（例如滑台的位置）。"""
    n = 200
    t = np.arange(n) * DT
    truth = np.clip(t - 2.0, 0.0, 5.0) * 0.5
    meas = truth + np.random.default_rng(4).normal(0, 0.15, n)
    return truth, meas


# =========================================================
#  取捨曲線（第 1、2 章共用）
# =========================================================
def draw_tradeoff(p, rect, delay, jitter_pct, label):
    """
    橫軸 = 落後幾點，縱軸 = 剩下多少抖動。
    對「白噪 + 平均類濾波器」：抖動² ≈ 1 / (2·落後 + 1) —— MA 和 EMA 落在同一條曲線上。
    """
    pl = Plot(rect, 0, 20, 0, 100)
    pl.frame(p, xticks=range(0, 21, 5), yticks=[0, 25, 50, 75, 100], yfmt=lambda v: f"{v:g}%",
             title="取捨曲線：想更平滑（往下）就得更落後（往右）", xlabel="落後幾點")
    # 曲線左下方「到不了」的區域
    D = np.linspace(0, 20, 120)
    J = 100 / np.sqrt(2 * D + 1)
    poly = [pl.pt(0, 0)] + [pl.pt(d, j) for d, j in zip(D, J)] + [pl.pt(20, 0)]
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(241, 245, 249)))
    p.drawPolygon(QPolygonF(poly))
    draw_text(p, pl.px(2.0), pl.py(12), "這一區到不了（沒有白吃的午餐）", C_MUTED, 8.5)
    pl.polyline(p, D, J, C_FRAME, 2.4)
    x = min(delay, 20)
    pl.dots(p, [x], [jitter_pct], C_MAIN, 7, outline=QColor("#FFFFFF"))
    draw_pill(p, pl.px(x) + 10, pl.py(jitter_pct) - 16, label, QColor("#FFFFFF"), C_MAIN, ha="l", pt=9)


def draw_meter(p, x, y, w, title, frac, text, color):
    draw_text(p, x, y, title, C_INK, 10, bold=True, va="t")
    bar = QRectF(x, y + 24, w, 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#E2E8F0")))
    p.drawRoundedRect(bar, 8, 8)
    p.setBrush(QBrush(color))
    p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(16.0, w * min(frac, 1.0)), 16), 8, 8)
    draw_text(p, x, y + 46, text, color, 10, bold=True, va="t")


# =========================================================
#  第 1 章　移動平均：用過去幾個點投票
# =========================================================
class MACanvas(ScrubCanvas):
    def __init__(self, ch):
        super().__init__(ch, 600)

    def paintEvent(self, _):
        ch = self.ch
        k, N = ch.s, ch.N()
        truth, meas, out = ch.truth, ch.meas, ch.out
        n = len(meas)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        lo, hi = min(meas.min(), truth.min()) - 0.2, max(meas.max(), truth.max()) + 0.3
        pl = Plot(QRectF(56, 30, W - 84, 230), -0.5, n - 0.5, lo, hi)
        self.main_plot = pl
        pl.frame(p, xticks=range(0, n, 20), yticks=nice_ticks(lo, hi, 5),
                 title=f"訊號（灰點 = 量測，虛線 = 真值，藍 = 移動平均 N = {N}）", xlabel="取樣點")
        # 窗口
        a = max(0, k - N + 1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(alpha(C_WIN, 45)))
        p.drawRect(QRectF(pl.px(a - 0.5), pl.r.top(), pl.px(k + 0.5) - pl.px(a - 0.5), pl.r.height()))
        draw_signal(p, pl, truth, meas, k, out)
        pl.dots(p, range(a, k + 1), meas[a:k + 1], C_WIN, 3.8)
        # 落後指示：平均值其實代表「窗口中間」那個時間
        if N > 1:
            c = k - (N - 1) / 2
            yv = out[k]
            y_arrow = pl.py(yv) - 22
            pen = QPen(C_MAIN, 1.6)
            p.setPen(pen)
            p.drawLine(QPointF(pl.px(c), y_arrow - 6), QPointF(pl.px(c), y_arrow + 6))
            draw_arrow(p, QPointF(pl.px(c), y_arrow), QPointF(pl.px(k) - 7, y_arrow), C_MAIN, 1.6, 7)
            draw_text(p, pl.px(c), y_arrow - 8, f"落後 {(N - 1) / 2:g} 點", C_MAIN, 9, bold=True, ha="r")
        pl.dots(p, [k], [out[k]], C_MAIN, 6.5, outline=QColor("#FFFFFF"))

        # 權重條：跟上圖共用同一條時間軸
        wts = np.zeros(n)
        wts[a:k + 1] = 1.0 / (k + 1 - a)
        pw = Plot(QRectF(56, 300, W - 84, 70), -0.5, n - 0.5, 0, wts.max() * 1.25)
        pw.frame(p, xticks=range(0, n, 20), title="每個點的「發言權」（權重）：窗口內人人平等，窗口外 0")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_WIN))
        bw = max(1.5, pw.px(1) - pw.px(0) - 1.0)
        for j in range(a, k + 1):
            p.drawRect(QRectF(pw.px(j) - bw / 2, pw.py(wts[j]), bw, pw.py(0) - pw.py(wts[j])))
        draw_text(p, pw.px(k) + 8, pw.py(wts[k]), f"每點 1/{k + 1 - a} = {wts[k]:.3f}", C_WIN, 9,
                  bold=True, va="m")

        # 取捨曲線 + 兩個儀表
        delay = (N - 1) / 2
        jit = 100 / math.sqrt(N)
        draw_tradeoff(p, QRectF(56, 420, 420, 150), delay, jit, f"N = {N}")
        mx = 540
        mw = W - mx - 40
        draw_meter(p, mx, 420, mw, "✅ 剩下的抖動（越短越好）", jit / 100, f"約 1/√N = {jit:.0f}%", C_GOOD)
        draw_meter(p, mx, 500, mw, "⚠ 落後（越短越好）", delay / 20,
                   f"(N−1)/2 = {delay:g} 點 = {delay * DT:.2f} 秒", C_BAD)
        p.end()


class Ch1MovingAverage(Chapter):
    NUM = 1
    TITLE = "濾波在做什麼？用過去幾個點「投票」"
    IDEA = ("量測一下高、一下低，是隨機雜訊。把最近 N 個點平均，正負誤差會互相抵消 → 變平滑。"
            "但平均值代表的是窗口「中間」的時間 → 一定會落後。越平滑就越落後，這是所有濾波器的根本取捨。")
    TRIES = ("把 N 拉到 1：等於沒濾波，藍線就是灰點",
             "把 N 拉到 31：很平滑，但看第 90 點的步階，藍線要過好久才爬上去",
             "拖動游標到步階附近：窗口一半是舊值、一半是新值時，輸出剛好在中間",
             "看下面的取捨曲線：不管 N 是多少，藍點永遠在那條線上 —— 平滑和即時不能兼得")

    def build(self):
        self.truth, self.meas = make_smooth_signal()
        self.s = 78
        self.slider = LabeledSlider("窗口 N（幾個點投票）", 1, 41, 15, "{:.0f} 點", self.refresh,
                                    steps=40, color=HEX_WIN, title_w=200)
        self.stepper = Stepper(self, self.total)
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.slider)
        self.canvas = MACanvas(self)
        self.body.addWidget(self.canvas)
        self.refresh()

    def total(self):
        return len(self.meas)

    def N(self):
        return int(round(self.slider.value()))

    def leave(self):
        self.stepper.stop()

    def refresh(self):
        self.out = F.apply(F.MovingAverage(self.N()), self.meas)
        self.on_step()

    def on_step(self):
        k, N = self.s, self.N()
        a = max(0, k - N + 1)
        win = self.meas[a:k + 1]
        shown = " + ".join(f"{v:.2f}" for v in win[-3:])
        more = f"{win[0]:.2f} + … + " if len(win) > 3 else ""
        self.now.setText(
            f"<b>第 {k} 點</b>：輸出 = 最近 {len(win)} 點的平均 = ({more}{shown}) ÷ {len(win)} = "
            f"{blue(f'{self.out[k]:.2f}')}　{gray(f'（真值 {self.truth[k]:.2f}）')}<br>"
            f"✅ 好處：N 個隨機誤差一正一負互相抵消 → 抖動剩下約 1/√N = <b>{100 / math.sqrt(N):.0f}%</b><br>"
            f"⚠ 代價：平均值代表的是窗口「中間」那個時間 → 落後 (N−1)/2 = <b>{(N - 1) / 2:g} 點</b>"
            f"（{(N - 1) / 2 * DT:.2f} 秒）。訊號一變化，藍線就慢半拍。")
        self.canvas.update()


# =========================================================
#  第 2 章　EMA：只記一個數也能平滑
# =========================================================
class EMACanvas(ScrubCanvas):
    def __init__(self, ch):
        super().__init__(ch, 545)

    def paintEvent(self, _):
        ch = self.ch
        k, a_ = ch.s, ch.alpha_()
        truth, meas, out = ch.truth, ch.meas, ch.out
        n = len(meas)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        lo, hi = min(meas.min(), truth.min()) - 0.2, max(meas.max(), truth.max()) + 0.3
        pl = Plot(QRectF(56, 30, W - 84, 230), -0.5, n - 0.5, lo, hi)
        self.main_plot = pl
        pl.frame(p, xticks=range(0, n, 20), yticks=nice_ticks(lo, hi, 5),
                 title=f"訊號（藍 = EMA，α = {a_:.2f}）", xlabel="取樣點")
        draw_signal(p, pl, truth, meas, k, out)
        if k > 0:
            pl.dots(p, [k - 1], [out[k - 1]], alpha(C_MAIN, 160), 4.5)
        pl.dots(p, [k], [meas[k]], C_WIN, 5)
        pl.dots(p, [k], [out[k]], C_MAIN, 6.5, outline=QColor("#FFFFFF"))

        # 權重：越舊越小（指數衰減），對照同等平滑度的 MA（虛線方框）
        js = np.arange(0, k + 1)
        wts = a_ * (1 - a_) ** js
        wts[-1] = (1 - a_) ** k                     # 第一筆（暖機）吃掉剩下的權重
        n_eq = max(1, int(round(2 / a_ - 1)))
        top = max(wts.max(), 1.0 / n_eq) * 1.25
        pw = Plot(QRectF(56, 300, W - 84, 70), -0.5, n - 0.5, 0, top)
        pw.frame(p, xticks=range(0, n, 20), title="每個點的「發言權」：越舊越小（指數衰減）；虛線 = 同樣平滑度的 MA")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_WIN))
        bw = max(1.5, pw.px(1) - pw.px(0) - 1.0)
        for j, w in zip(js, wts):
            if w < top * 0.004:
                continue
            x = k - j
            p.drawRect(QRectF(pw.px(x) - bw / 2, pw.py(w), bw, pw.py(0) - pw.py(w)))
        pen = QPen(C_MUTED, 1.4)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        a0 = max(0, k - n_eq + 1)
        p.drawRect(QRectF(pw.px(a0 - 0.5), pw.py(1.0 / n_eq), pw.px(k + 0.5) - pw.px(a0 - 0.5),
                          pw.py(0) - pw.py(1.0 / n_eq)))

        # 混合條：y = α·x + (1−α)·y前
        y0 = 410
        draw_text(p, 56, y0, "這一點怎麼算出來？把「新量測」和「上一次的輸出」照比例混在一起",
                  C_INK, 10, bold=True, va="t")
        bar = QRectF(56, y0 + 28, W - 112, 30)
        wa = bar.width() * a_
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_WIN))
        p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(wa, 4), bar.height()), 6, 6)
        p.setBrush(QBrush(alpha(C_MAIN, 200)))
        p.drawRoundedRect(QRectF(bar.left() + wa, bar.top(), bar.width() - wa, bar.height()), 6, 6)
        y_prev = out[k - 1] if k > 0 else meas[0]
        if wa > 120:
            draw_text(p, bar.left() + wa / 2, bar.center().y(), f"新量測 {a_:.0%}", QColor("#FFFFFF"), 10,
                      bold=True, ha="c", va="m")
        draw_text(p, bar.left() + wa + (bar.width() - wa) / 2, bar.center().y(),
                  f"上一次的輸出 {1 - a_:.0%}", QColor("#FFFFFF"), 10, bold=True, ha="c", va="m")
        draw_text(p, W / 2, y0 + 72,
                  f"y = α·x + (1−α)·y前 = {a_:.2f} × {meas[k]:.2f} + {1 - a_:.2f} × {y_prev:.2f} = {out[k]:.2f}",
                  C_INK, 12, bold=True, ha="c", va="t")

        delay = (1 - a_) / a_
        jit = 100 * math.sqrt(a_ / (2 - a_))
        draw_text(p, 56, y0 + 110,
                  f"取捨：落後約 (1−α)/α = {delay:.1f} 點、抖動剩 {jit:.0f}% —— "
                  f"跟 N = {n_eq} 的移動平均幾乎一樣（第 1 章那條曲線）",
                  C_MUTED, 10, va="t")
        p.end()


class Ch2EMA(Chapter):
    NUM = 2
    TITLE = "只記一個數也能平滑：EMA（一階低通）"
    IDEA = ("MA 要存 N 個點。EMA 只記「上一次的輸出」，每次把新量測混一點進去：y = α·x + (1−α)·y前。"
            "越舊的點權重越小。記憶體只要 1 個數，是 MCU 上最常用的濾波器（就是 RC 低通電路的數位版）。")
    TRIES = ("α = 1：完全相信新量測，等於沒濾波",
             "α = 0.05：非常平滑，但步階要好久才爬上去",
             "看權重條：EMA 的「發言權」是一條滑下去的指數，虛線方框是同樣平滑的 MA",
             "比較第 1 章：同樣的落後，換到幾乎一樣的平滑 —— EMA 贏在省記憶體，不是更聰明")

    def build(self):
        self.truth, self.meas = make_smooth_signal()
        self.s = 95
        self.slider = LabeledSlider("α（新量測佔多少）", 0.02, 1.0, 0.15, self._fmt, self.refresh,
                                    steps=98, color=HEX_MAIN, title_w=200)
        self.stepper = Stepper(self, self.total)
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.slider)
        self.canvas = EMACanvas(self)
        self.body.addWidget(self.canvas)
        self.refresh()

    @staticmethod
    def _fmt(a_):
        if a_ >= 0.999:
            return "α = 1.00（不濾）"
        return f"α = {a_:.2f}（fc ≈ {F.cutoff_from_alpha(a_, DT):.2f} Hz）"

    def total(self):
        return len(self.meas)

    def alpha_(self):
        return min(1.0, self.slider.value())

    def leave(self):
        self.stepper.stop()

    def refresh(self):
        self.out = F.apply(F.EMA(self.alpha_()), self.meas)
        self.on_step()

    def on_step(self):
        k, a_ = self.s, self.alpha_()
        y_prev = self.out[k - 1] if k > 0 else self.meas[0]
        self.now.setText(
            f"<b>第 {k} 點</b>：y = α × {amber('新量測')} + (1 − α) × {blue('上一次輸出')} = "
            f"{a_:.2f} × {self.meas[k]:.2f} + {1 - a_:.2f} × {y_prev:.2f} = {blue(f'{self.out[k]:.2f}')}"
            f"　{gray(f'（真值 {self.truth[k]:.2f}）')}<br>"
            f"✅ 記憶體只要 <b>1 個數</b>（上一次的輸出），計算只要 1 乘 1 加 —— MA 要記 N 個<br>"
            f"⚠ 一樣會落後：約 (1−α)/α = <b>{(1 - a_) / a_:.1f} 點</b>；α 越小越平滑、也越慢")
        self.canvas.update()


# =========================================================
#  第 3 章　尖峰：平均會被拖累，中位數不會
# =========================================================
class SpikeCanvas(ScrubCanvas):
    def __init__(self, ch):
        super().__init__(ch, 600)

    def paintEvent(self, _):
        ch = self.ch
        k, N = ch.s, ch.N()
        sig = ch.sig
        truth, meas, mask = sig["truth"], sig["meas"], sig["spike_mask"]
        n = len(meas)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        lo, hi = meas.min() - 0.2, meas.max() + 0.2
        pl = Plot(QRectF(56, 30, W - 84, 240), -0.5, n - 0.5, lo, hi)
        self.main_plot = pl
        pl.frame(p, xticks=range(0, n, 20), yticks=nice_ticks(lo, hi, 5),
                 title=f"有尖峰的訊號（紅點 = 尖峰；橘虛線 = 移動平均；藍 = 中位數，N = {N}）",
                 xlabel="取樣點")
        a = max(0, k - N + 1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(alpha(C_WIN, 45)))
        p.drawRect(QRectF(pl.px(a - 0.5), pl.r.top(), pl.px(k + 0.5) - pl.px(a - 0.5), pl.r.height()))
        draw_signal(p, pl, truth, meas, k, ch.med, other=ch.ma)
        sk = np.where(mask[:k + 1])[0]
        pl.dots(p, sk, meas[sk], C_BAD, 4)
        pl.dots(p, [k], [ch.ma[k]], C_OTHER, 5.5, outline=QColor("#FFFFFF"))
        pl.dots(p, [k], [ch.med[k]], C_MAIN, 6.5, outline=QColor("#FFFFFF"))

        win = list(meas[a:k + 1])
        spikes = list(mask[a:k + 1])
        srt = sorted(zip(win, spikes))
        mid = len(srt) // 2
        y0 = 310
        draw_text(p, 56, y0, "窗口裡的值", C_INK, 10.5, bold=True, va="t")
        bw = min(62.0, (W - 230) / max(1, len(win)))
        x0 = 170

        def box(x, y, v, spike, mark=None):
            r = QRectF(x, y, bw - 6, 34)
            if mark == "median":
                p.setPen(QPen(C_GOOD, 2.6))
                p.setBrush(QBrush(QColor("#DCFCE7")))
            elif spike:
                p.setPen(QPen(C_BAD, 1.8))
                p.setBrush(QBrush(QColor("#FEE2E2")))
            else:
                p.setPen(QPen(C_FRAME, 1))
                p.setBrush(QBrush(QColor("#FFFFFF")))
            p.drawRoundedRect(r, 6, 6)
            draw_text(p, r.center().x(), r.center().y(), f"{v:.2f}", C_BAD if spike else C_INK,
                      9 if bw > 48 else 8, bold=True, ha="c", va="m")

        draw_text(p, x0 - 12, y0 + 34, "時間順序 →", C_MUTED, 9, ha="r", va="m")
        for i, (v, s) in enumerate(zip(win, spikes)):
            box(x0 + i * bw, y0 + 17, v, s)
        draw_text(p, x0 - 12, y0 + 86, "排隊（小 → 大）", C_MUTED, 9, ha="r", va="m")
        for i, (v, s) in enumerate(srt):
            box(x0 + i * bw, y0 + 69, v, s, "median" if i == mid and len(srt) % 2 else None)
        if len(srt) % 2:
            draw_text(p, x0 + mid * bw + (bw - 6) / 2, y0 + 109, "▲ 中間 = 中位數", C_GOOD, 9, bold=True,
                      ha="c", va="t")

        # 數線：平均被尖峰拉走，中位數留在人群裡
        y1 = y0 + 160
        vals = np.array(win)
        lo_v, hi_v = min(vals.min(), lo), max(vals.max(), hi)
        nl = Plot(QRectF(170, y1 + 20, W - 250, 30), lo_v - 0.1, hi_v + 0.1, 0, 1)
        draw_text(p, 56, y1, "放到數線上看", C_INK, 10.5, bold=True, va="t")
        p.setPen(QPen(C_FRAME, 2))
        p.drawLine(nl.pt(nl.x0, 0.5), nl.pt(nl.x1, 0.5))
        for v, s in zip(win, spikes):
            nl.dots(p, [v], [0.5], C_BAD if s else alpha(C_RAW, 230), 5)
        mean, med = float(vals.mean()), ch.med[k]
        for v, c, s_, up in ((mean, C_OTHER, f"平均 {mean:.2f}", True), (med, C_MAIN, f"中位數 {med:.2f}", False)):
            x = nl.px(v)
            y = nl.py(0.5)
            tri = [QPointF(x, y - 8), QPointF(x - 7, y - 20), QPointF(x + 7, y - 20)] if up else \
                  [QPointF(x, y + 8), QPointF(x - 7, y + 20), QPointF(x + 7, y + 20)]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(c))
            p.drawPolygon(QPolygonF(tri))
            draw_text(p, x, y - 24 if up else y + 24, s_, c, 9.5, bold=True, ha="c", va="b" if up else "t")
        p.end()


class Ch3Median(Chapter):
    NUM = 3
    TITLE = "遇到尖峰：平均會被拖累，中位數不會"
    IDEA = ("尖峰 = 偶爾一個瘋狂的值（ADC 讀錯、超音波打到雜物）。平均是「大家一起扛」，一個瘋狂的點就能把結果拉走；"
            "中位數是「排隊取中間」，瘋狂的點會被擠到隊伍兩端，根本碰不到中間。")
    TRIES = ("按「跳到下一個尖峰」：橘線（平均）被拉出一個凸塊，藍線（中位數）不為所動",
             "看數線：橘色三角形被紅點拉過去，藍色三角形留在人群裡",
             "走到第 100 點附近的步階：中位數的邊緣仍然陡峭，平均變成斜坡",
             "N = 3 時遇到連續兩個尖峰會怎樣？（尖峰超過半個窗口，中位數也擋不住）")

    def build(self):
        self.sig = F.make_demo_signals()["spike"]
        self.s = 0
        self.slider = LabeledSlider("窗口 N（奇數）", 0, 6, 2, lambda v: f"{3 + 2 * int(round(v))} 點",
                                    self.refresh, steps=6, color=HEX_WIN, title_w=200)
        self.stepper = Stepper(self, self.total)
        self.stepper.bar.insertWidget(4, button("⏩ 跳到下一個尖峰", self.next_spike))
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.slider)
        self.canvas = SpikeCanvas(self)
        self.body.addWidget(self.canvas)
        self.refresh()
        self.s = 30
        self.next_spike()          # 從第 30 點之後的第一個尖峰開始（避開暖機期）

    def total(self):
        return len(self.sig["meas"])

    def N(self):
        return 3 + 2 * int(round(self.slider.value()))

    def leave(self):
        self.stepper.stop()

    def next_spike(self):
        self.stepper.stop()
        idx = np.where(self.sig["spike_mask"])[0]
        nxt = idx[idx > self.s]
        self.s = int(nxt[0]) if len(nxt) else int(idx[0])
        self.on_step()

    def refresh(self):
        meas = self.sig["meas"]
        self.ma = F.apply(F.MovingAverage(self.N()), meas)
        self.med = F.apply(F.MedianFilter(self.N()), meas)
        self.on_step()

    def on_step(self):
        k, N = self.s, self.N()
        meas, mask = self.sig["meas"], self.sig["spike_mask"]
        tk = self.sig["truth"][k]
        a = max(0, k - N + 1)
        n_sp = int(mask[a:k + 1].sum())
        if n_sp:
            sp_vals = "、".join(f"{v:.2f}" for v in meas[a:k + 1][mask[a:k + 1]])
            pull = self.ma[k] - self.med[k]
            line2 = (f"這個窗口裡有 <b>{n_sp} 個尖峰</b>（{sp_vals}）：{orange('平均')}被拉偏了 "
                     f"<b>{pull:+.2f}</b>，{blue('中位數')}只看「排在中間的那一個」，完全不受影響。")
        else:
            line2 = "這個窗口裡沒有尖峰：平均和中位數差不多（平均其實更平滑一點）。"
        self.now.setText(
            f"<b>第 {k} 點</b>：窗口 {k + 1 - a} 個值排隊取中間 → {blue(f'中位數 {self.med[k]:.2f}')}；"
            f"同一個窗口的 {orange(f'平均 {self.ma[k]:.2f}')}　{gray(f'（真值 {tk:.2f}）')}<br>"
            f"{line2}<br>"
            f"💡 想「只換掉壞點、其他原封不動」？用 <b>Hampel</b>（filters.py）：先用中位數判斷是不是離群，是才換。")
        self.canvas.update()


# =========================================================
#  第 4 章　訊號在移動：平均一定落後，α-β 會預測
# =========================================================
def run_alpha_beta(meas, a_, dt):
    """跑 α-β，把每一步的內部量（預測、殘差、速度）記下來給畫面用。"""
    f = F.AlphaBetaFilter(a_, F.benedict_bordner_beta(a_), dt)
    n = len(meas)
    rec = {key: np.full(n, np.nan) for key in ("x", "v", "x_pred", "res", "v_prev", "x_prev")}
    for i, z in enumerate(meas):
        x_prev, v_prev = f.x, f.v
        f.update(z)
        rec["x"][i], rec["v"][i] = f.x, f.v
        rec["x_pred"][i], rec["res"][i] = f.x_pred, f.residual
        rec["x_prev"][i] = np.nan if x_prev is None else x_prev
        rec["v_prev"][i] = v_prev
    rec["beta"] = f.beta
    return rec


class MovingCanvas(ScrubCanvas):
    def __init__(self, ch):
        super().__init__(ch, 560)

    def paintEvent(self, _):
        ch = self.ch
        k = ch.s
        truth, meas, ab, ema = ch.truth, ch.meas, ch.ab, ch.ema
        n = len(meas)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        split = int(W * 0.64)
        lo, hi = meas.min() - 0.2, meas.max() + 0.3
        pl = Plot(QRectF(56, 30, split - 70, 270), -0.5, n - 0.5, lo, hi)
        self.main_plot = pl
        pl.frame(p, xticks=range(0, n, 40), yticks=nice_ticks(lo, hi, 5),
                 title="滑台位置（橘虛線 = EMA，藍 = α-β）", xlabel="取樣點")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(254, 243, 199, 110)))
        p.drawRect(QRectF(pl.px(40), pl.r.top(), pl.px(140) - pl.px(40), pl.r.height()))
        draw_text(p, (pl.px(40) + pl.px(140)) / 2, pl.r.top() + 6, "等速移動中", QColor("#92400E"), 9,
                  ha="c", va="t")
        draw_signal(p, pl, truth, meas, k, ab["x"], other=ema)
        pl.dots(p, [k], [ema[k]], C_OTHER, 5.5, outline=QColor("#FFFFFF"))
        pl.dots(p, [k], [ab["x"][k]], C_MAIN, 6.5, outline=QColor("#FFFFFF"))
        # 放大鏡框：縱軸只框住 α-β 與量測（EMA 落後太多就跑出框外，用箭頭指出來）
        z0 = max(0, k - 7)
        cen = ab["x"][k]
        dev = np.abs(np.concatenate([meas[z0:k + 1] - cen, ab["x"][z0:k + 1] - cen]))
        half = max(0.2, float(np.nanmax(dev)) * 1.2)
        zlo, zhi = cen - half, cen + half
        p.setPen(QPen(C_INK, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(pl.pt(z0 - 0.5, zhi), pl.pt(k + 0.5, zlo)))

        # 右側：放大看 α-β 這一步怎麼算
        pz = Plot(QRectF(split + 40, 30, W - split - 64, 270), z0 - 0.7, k + 2.6, zlo, zhi)
        pz.frame(p, title="放大：α-β 這一步怎麼算")
        idx = np.arange(z0, k + 1)
        p.save()
        p.setClipRect(pz.r)
        pz.dots(p, idx, meas[z0:k + 1], alpha(C_RAW, 200), 3.5)
        pz.polyline(p, idx, ema[z0:k + 1], C_OTHER, 2, dash=True)
        pz.polyline(p, idx[:-1], ab["x"][z0:k], C_MAIN, 2.4)
        pz.dots(p, idx[:-1], ab["x"][z0:k], C_MAIN, 3.5)
        labels = []
        if k >= 1 and np.isfinite(ab["x_prev"][k]):
            xp, xpred, z, x = ab["x_prev"][k], ab["x_pred"][k], meas[k], ab["x"][k]
            draw_arrow(p, pz.pt(k - 1, xp), pz.pt(k, xpred), C_MAIN, 1.8, 7, dash=True)
            pen = QPen(C_BAD, 2)
            pen.setStyle(Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawLine(pz.pt(k, xpred), pz.pt(k, z))
            p.setPen(QPen(C_MAIN, 2))
            p.setBrush(QBrush(QColor("#FFFFFF")))
            p.drawEllipse(pz.pt(k, xpred), 6, 6)
            pz.dots(p, [k], [z], C_RAW, 6)
            pz.dots(p, [k], [x], C_MAIN, 6.5, outline=QColor("#FFFFFF"))
            labels = [(pz.py(xpred), "① 預測", C_MAIN), (pz.py(z), "量測", C_MUTED),
                      (pz.py(x), "③ 修正後", C_MAIN)]
            draw_text(p, pz.px(k) - 9, (pz.py(z) + pz.py(xpred)) / 2, f"② 殘差 {z - xpred:+.3f}", C_BAD, 9,
                      bold=True, ha="r", va="m")
        p.restore()
        # 右邊的標籤：依高度排好、至少隔 17 px，再用細線連回自己的點
        labels.sort(key=lambda t: t[0])
        y_prev = -1e9
        for y, s, c in labels:
            y_lab = max(y, y_prev + 17)
            y_prev = y_lab
            p.setPen(QPen(alpha(c, 140), 1))
            p.drawLine(QPointF(pz.px(k) + 7, y), QPointF(pz.px(k) + 22, y_lab))
            draw_text(p, pz.px(k) + 25, y_lab, s, c, 9, bold=True, va="m")
        if not zlo <= ema[k] <= zhi:
            below = ema[k] < zlo
            draw_pill(p, pz.r.center().x(), pz.r.bottom() - 14 if below else pz.r.top() + 14,
                      f"{'↓' if below else '↑'} EMA 在框外：差了 {ema[k] - ab['x'][k]:+.2f}",
                      QColor("#FFFFFF"), C_OTHER, pt=9)
        # 下方：誤差
        pe = Plot(QRectF(56, 350, W - 84, 150), -0.5, n - 0.5, -0.45, 0.45)
        pe.frame(p, xticks=range(0, n, 20), yticks=[-0.4, -0.2, 0, 0.2, 0.4], yfmt=lambda v: f"{v:+g}" if v else "0",
                 title="誤差 = 輸出 − 真值：EMA 在移動時一直落後（整段偏一邊），α-β 貼著 0", xlabel="取樣點")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(254, 243, 199, 110)))
        p.drawRect(QRectF(pe.px(40), pe.r.top(), pe.px(140) - pe.px(40), pe.r.height()))
        p.setPen(QPen(C_TRUE, 1))
        p.drawLine(pe.pt(-0.5, 0), pe.pt(n - 0.5, 0))
        p.save()
        p.setClipRect(pe.r)
        ks = np.arange(k + 1)
        pe.polyline(p, ks, ema[:k + 1] - truth[:k + 1], C_OTHER, 2.2)
        pe.polyline(p, ks, ab["x"][:k + 1] - truth[:k + 1], C_MAIN, 2.2)
        p.restore()
        p.end()


class Ch4AlphaBeta(Chapter):
    NUM = 4
    TITLE = "訊號在移動：平均一定落後，α-β 會預測"
    IDEA = ("目標在移動時，只會「平均過去」的濾波器一定落後一截（落後量 = 速度 × 延遲）。"
            "α-β 多記一個「速度」：先預測「現在應該走到哪」，再用量測修正 —— 這就是卡爾曼濾波的雛形。")
    TRIES = ("拖到「等速移動中」那段：橘線一直在藍線下面，誤差圖整段偏一邊",
             "看右邊放大圖：① 用速度先往前推一步 → ② 跟量測比 → 只修正一部分（α）",
             "把 α 調小：兩條線都更平滑，但 EMA 落後得更多，α-β 仍然貼著真值",
             "看第 140 點（突然停下）：α-β 會稍微衝過頭 —— 它以為還在動。這就是「模型」的代價")

    def build(self):
        self.truth, self.meas = make_moving_signal()
        self.s = 100
        self.slider = LabeledSlider("α（兩個濾波器共用）", 0.03, 0.6, 0.12, "α = {:.2f}", self.refresh,
                                    steps=114, color=HEX_MAIN, title_w=200)
        self.stepper = Stepper(self, self.total)
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.slider)
        self.canvas = MovingCanvas(self)
        self.body.addWidget(self.canvas)
        self.refresh()
        # 一打開就停在「移動中、殘差最大」的那一點：放大圖裡最看得清楚預測 → 修正
        self.s = 60 + int(np.argmax(np.abs(self.ab["res"][60:130])))
        self.on_step()

    def total(self):
        return len(self.meas)

    def leave(self):
        self.stepper.stop()

    def refresh(self):
        a_ = self.slider.value()
        self.ema = F.apply(F.EMA(a_), self.meas)
        self.ab = run_alpha_beta(self.meas, a_, DT)
        seg = slice(60, 140)
        self.err_ema = float(np.mean(self.ema[seg] - self.truth[seg]))
        self.err_ab = float(np.mean(self.ab["x"][seg] - self.truth[seg]))
        self.on_step()

    def on_step(self):
        k = self.s
        ab, a_ = self.ab, self.slider.value()
        if k == 0:
            calc = "第 0 點：直接用第一筆量測當位置，速度先猜 0。"
        else:
            x_prev, v_prev, x_pred = ab["x_prev"][k], ab["v_prev"][k], ab["x_pred"][k]
            res, x_new, v_new = ab["res"][k], ab["x"][k], ab["v"][k]
            calc = (f"① 預測 = 上次位置 + 速度 × dt = {x_prev:.3f} + {v_prev:.2f} × {DT} = "
                    f"{blue(f'{x_pred:.3f}')}<br>"
                    f"② 殘差 = 量測 − 預測 = {self.meas[k]:.3f} − {x_pred:.3f} = {res:+.3f}<br>"
                    f"③ 位置 = 預測 + α × 殘差 = {blue(f'{x_new:.3f}')}；"
                    f"速度 += (β/dt) × 殘差 → {v_new:.2f} /秒")
        self.now.setText(
            f"<b>α-β 第 {k} 點</b>（α = {a_:.2f}，β = α²/(2−α) = {ab['beta']:.3f}）<br>{calc}<br>"
            f"等速段的平均誤差：{orange(f'EMA {self.err_ema:+.3f}')}（一直落後）vs "
            f"{blue(f'α-β {self.err_ab:+.3f}')}")
        self.canvas.update()


# =========================================================
#  第 5 章　互補濾波：陀螺儀 + 加速度計
# =========================================================
class IMUCanvas(ScrubCanvas):
    def __init__(self, ch):
        super().__init__(ch, 560)

    def paintEvent(self, _):
        ch = self.ch
        k = ch.s
        sig, out = ch.sig, ch.out
        n = len(out)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        lo = min(sig["accel"].min(), sig["gyro_int"].min()) - 3
        hi = max(sig["accel"].max(), sig["gyro_int"].max()) + 3
        pl = Plot(QRectF(56, 30, W - 84, 280), -0.5, n - 0.5, lo, hi)
        self.main_plot = pl
        pl.frame(p, xticks=range(0, n, 20), yticks=nice_ticks(lo, hi, 6), yfmt=lambda v: f"{v:g}°",
                 title="傾斜角（灰點 = 加速度計：很吵；橘虛線 = 陀螺儀積分：會漂；藍 = 互補濾波）",
                 xlabel="取樣點")
        draw_signal(p, pl, sig["truth"], sig["accel"], k, out, other=sig["gyro_int"])
        pl.dots(p, [k], [out[k]], C_MAIN, 6.5, outline=QColor("#FFFFFF"))
        drift = sig["gyro_int"][k] - sig["truth"][k]
        draw_pill(p, pl.px(k) + 8, pl.py(sig["gyro_int"][k]) - 14, f"陀螺儀已漂了 {drift:+.1f}°",
                  QColor("#FFFFFF"), C_OTHER, ha="l", pt=9)

        # 方塊圖：這一點怎麼算
        y0 = 360
        draw_text(p, 56, y0, "這一點怎麼算出來？", C_INK, 10.5, bold=True, va="t")
        a_ = ch.alpha_()
        gyro, acc = sig["gyro"][k], sig["accel"][k]
        prev = out[k - 1] if k > 0 else acc
        pred = prev + gyro * DT if k > 0 else acc

        def blk(x, y, w, text, color, bg):
            r = QRectF(x, y, w, 40)
            p.setPen(QPen(color, 1.6))
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(r, 8, 8)
            draw_text(p, r.center().x(), r.center().y(), text, color, 9.5, bold=True, ha="c", va="m")
            return r

        bw = (W - 150) / 4
        r1 = blk(56, y0 + 30, bw, f"陀螺儀 ω = {gyro:.1f} °/s", C_OTHER, QColor("#FFF7ED"))
        r2 = blk(56 + bw + 20, y0 + 30, bw, f"往前推：{prev:.1f} + ω·dt = {pred:.1f}°", C_OTHER,
                 QColor("#FFF7ED"))
        r3 = blk(56 + 2 * (bw + 20), y0 + 30, bw, f"× α = {a_:.3f} → {a_ * pred:.2f}", C_OTHER,
                 QColor("#FFF7ED"))
        r4 = blk(56, y0 + 100, bw, f"加速度計 a = {acc:.1f}°", C_MUTED, QColor("#F1F5F9"))
        r5 = blk(56 + 2 * (bw + 20), y0 + 100, bw, f"× (1−α) = {1 - a_:.3f} → {(1 - a_) * acc:.2f}",
                 C_MUTED, QColor("#F1F5F9"))
        r6 = blk(56 + 3 * (bw + 20) + 10, y0 + 65, bw - 10, f"θ = {out[k]:.2f}°", C_MAIN, QColor("#EFF6FF"))
        for a, b in ((r1, r2), (r2, r3), (r4, r5)):
            draw_arrow(p, QPointF(a.right() + 2, a.center().y()), QPointF(b.left() - 2, b.center().y()),
                       C_MUTED, 1.6, 7)
        for a in (r3, r5):
            draw_arrow(p, QPointF(a.right() + 2, a.center().y()), QPointF(r6.left() - 2, r6.center().y()),
                       C_MAIN, 1.6, 7)
        draw_text(p, (r3.right() + r6.left()) / 2 - 4, r6.center().y(), "＋", C_MAIN, 13, bold=True, ha="c",
                  va="m")
        draw_text(p, 56, y0 + 160,
                  f"陀螺儀管「快的變化」（佔 {a_:.1%}）、加速度計管「長期的位置」（每一點只拉 {1 - a_:.1%}）。"
                  f"分界線就是 τ = {ch.tau():.2f} 秒。",
                  C_MUTED, 10, va="t")
        p.end()


class Ch5Complementary(Chapter):
    NUM = 5
    TITLE = "兩個感測器互補：陀螺儀 + 加速度計"
    IDEA = ("陀螺儀：短時間很準、很平滑，但積分久了會漂走。加速度計：長時間平均是準的，但每一刻都很吵。"
            "互補濾波各取所長 —— 快的變化聽陀螺儀，慢的趨勢聽加速度計。一行公式、零延遲。")
    TRIES = ("把 τ 拉到最大：幾乎只信陀螺儀 → 藍線跟著橘線慢慢漂走",
             "把 τ 拉到最小：幾乎只信加速度計 → 藍線跟灰點一樣抖",
             "τ ≈ 1 秒：又平滑、又不漂，而且沒有延遲（比 EMA 濾加速度計好得多）",
             "看方塊圖：每一點，加速度計只能「拉」一點點，但長期下來足以把漂移拉回來")

    def build(self):
        self.sig = F.make_demo_signals()["imu"]
        self.s = 150
        self.slider = LabeledSlider("時間常數 τ", 0.05, 8.0, 1.0, self._fmt, self.refresh, log=True,
                                    steps=120, color=HEX_MAIN, title_w=200)
        self.stepper = Stepper(self, self.total)
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.slider)
        self.canvas = IMUCanvas(self)
        self.body.addWidget(self.canvas)
        self.refresh()

    @staticmethod
    def _fmt(tau):
        return f"τ = {tau:.2f} 秒（α = {tau / (tau + DT):.3f}）"

    def tau(self):
        return self.slider.value()

    def alpha_(self):
        return self.tau() / (self.tau() + DT)

    def total(self):
        return len(self.sig["meas"])

    def leave(self):
        self.stepper.stop()

    def refresh(self):
        self.out = F.apply(F.ComplementaryFilter(DT, tau=self.tau()), self.sig["gyro"], self.sig["accel"])
        t = self.sig["truth"]
        self.err = {"互補濾波": F.rmse(self.out, t), "加速度計": F.rmse(self.sig["accel"], t),
                    "陀螺儀積分": F.rmse(self.sig["gyro_int"], t)}
        self.on_step()

    def on_step(self):
        k = self.s
        tk = self.sig["truth"][k]
        e_acc, e_gyro, e_comp = self.err["加速度計"], self.err["陀螺儀積分"], self.err["互補濾波"]
        self.now.setText(
            f"<b>第 {k} 點</b>：θ = α × (θ前 + ω·dt) + (1 − α) × a = {blue(f'{self.out[k]:.2f}°')}"
            f"　{gray(f'（真值 {tk:.2f}°）')}<br>"
            f"整段誤差：{gray(f'加速度計 {e_acc:.1f}°')}、{orange(f'陀螺儀積分 {e_gyro:.1f}°')} → "
            f"{blue(f'互補濾波 {e_comp:.1f}°')}<br>"
            f"為什麼叫「互補」？陀螺儀那條是高通、加速度計那條是低通，兩者加起來剛好 = 1，真實的角度不會被扭曲。")
        self.canvas.update()


# =========================================================
#  第 6 章　怎麼選？
# =========================================================
class DecisionCanvas(QWidget):
    ROWS = [
        ("一直小小地抖（白噪）", "移動平均 / EMA", "第 1、2 章", "#2563EB"),
        ("偶爾一個瘋狂的值（尖峰）", "中位數 / Hampel", "第 3 章", "#DC2626"),
        ("訊號在移動、不能落後", "α-β → 卡爾曼", "第 4 章", "#7C3AED"),
        ("兩個感測器：一個漂、一個吵", "互補濾波", "第 5 章", "#EA580C"),
    ]
    MORE = [
        ("要精準的頻率規格（例如濾掉 50 Hz 以上）", "Biquad 巴特沃斯"),
        ("靜止時要很穩、動起來要跟得上（滑鼠、手勢）", "One Euro"),
        ("不怕雜訊、只是不准變化太快（馬達命令）", "斜率限制"),
    ]

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(450)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        root = QRectF(20, 150, 190, 64)
        p.setPen(QPen(C_INK, 2))
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawRoundedRect(root, 10, 10)
        draw_text(p, root.center().x(), root.center().y() - 10, "先看你的雜訊", C_INK, 11, bold=True, ha="c",
                  va="m")
        draw_text(p, root.center().x(), root.center().y() + 12, "長什麼樣？", C_INK, 11, bold=True, ha="c",
                  va="m")
        col1, col2 = 270, 270 + (W - 300) * 0.5
        w1, w2 = (W - 300) * 0.45, (W - 300) * 0.42
        for i, (prob, fix, where, color) in enumerate(self.ROWS):
            y = 20 + i * 82
            c = QColor(color)
            a = QRectF(col1, y, w1, 58)
            b = QRectF(col2, y, w2, 58)
            draw_arrow(p, QPointF(root.right() + 2, root.center().y()), QPointF(a.left() - 2, a.center().y()),
                       C_FRAME, 1.6, 7)
            p.setPen(QPen(c, 1.6))
            p.setBrush(QBrush(alpha(c, 22)))
            p.drawRoundedRect(a, 10, 10)
            draw_text(p, a.left() + 14, a.center().y(), prob, C_INK, 10.5, bold=True, va="m")
            draw_arrow(p, QPointF(a.right() + 2, a.center().y()), QPointF(b.left() - 2, b.center().y()),
                       c, 2, 8)
            p.setBrush(QBrush(c))
            p.drawRoundedRect(b, 10, 10)
            draw_text(p, b.left() + 14, b.center().y() - 9, fix, QColor("#FFFFFF"), 11.5, bold=True, va="m")
            draw_text(p, b.left() + 14, b.center().y() + 12, where, QColor("#FFFFFF"), 9, va="m")
        y = 352
        draw_text(p, 20, y, "進階（filters.py 裡也有，這裡不展開）：", C_MUTED, 10, bold=True, va="t")
        for i, (prob, fix) in enumerate(self.MORE):
            draw_text(p, 40, y + 24 + i * 20, f"▸ {prob}　→　{fix}", C_INK, 10, va="t")
        p.end()


SUMMARY_HTML = """
<table border='1' cellspacing='0' cellpadding='7' style='border-color:#CBD5E1; font-size:13px'>
<tr style='background:#F1F5F9'><th>濾波器</th><th>一句話</th><th>落後</th><th>去尖峰</th><th>記憶體</th><th>最常用在</th></tr>
<tr><td><b>移動平均</b></td><td>最近 N 點平均</td><td>(N−1)/2 點</td><td>✗ 會被拉走</td><td>N 個數</td><td>顯示讀值、資料記錄</td></tr>
<tr><td><b>EMA</b></td><td>新舊照比例混合</td><td>(1−α)/α 點</td><td>✗ 拖尾巴</td><td>1 個數</td><td>MCU 感測器平滑</td></tr>
<tr><td><b>中位數</b></td><td>排隊取中間</td><td>(N−1)/2 點</td><td>✓ 最強</td><td>N 個數</td><td>ADC 錯讀、測距跳點</td></tr>
<tr><td><b>Hampel</b></td><td>判斷是壞點才換</td><td>0</td><td>✓</td><td>N 個數</td><td>資料清洗、前處理</td></tr>
<tr><td><b>α-β</b></td><td>預測 + 修正（固定增益）</td><td>等速時 0</td><td>✗ 先過 Hampel</td><td>2 個數</td><td>位置 → 速度、目標追蹤</td></tr>
<tr><td><b>互補濾波</b></td><td>快的信 A、慢的信 B</td><td>≈ 0</td><td>✗</td><td>1 個數</td><td>IMU 姿態角</td></tr>
<tr><td><b>卡爾曼</b></td><td>增益會自己算的 α-β</td><td>依模型</td><td>✓ 加閘門</td><td>矩陣</td><td>多感測器融合、導航</td></tr>
</table>
<p style='font-size:13px;color:#475569'>工程實務順序：<b>先去尖峰（中位數 / Hampel）→ 再平滑（EMA / MA）→ 要估速度或融合多個感測器才上 α-β / 卡爾曼</b>。
每個濾波器的實作與參數怎麼選，都寫在 <code>filters.py</code> 的 docstring 裡（STEP 1、STEP 2…）。</p>
"""


class Ch6Choose(Chapter):
    NUM = 6
    TITLE = "怎麼選？"
    IDEA = "沒有萬用的濾波器。先看雜訊長什麼樣子，再對症下藥；而且永遠記得：越平滑就越落後。"
    TRIES = ()

    def build(self):
        self.body.addWidget(DecisionCanvas())
        lab = QLabel(SUMMARY_HTML)
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setStyleSheet("QLabel { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px;"
                          " padding:12px; color:#0F172A; }")
        self.body.addWidget(lab)
        self.now.setText("<b>三句話總結</b><br>"
                         "① 濾波 = 用過去的點投票；投票的人越多越平滑，但結果代表的是「過去」→ 一定落後。<br>"
                         "② 雜訊不只一種：白噪用平均、尖峰用中位數、漂移靠另一個感測器互補。<br>"
                         "③ 想要又平滑又不落後，就得加入「模型」（α-β、卡爾曼）—— 用預測補上延遲，"
                         "代價是模型錯的時候會衝過頭。")


# =========================================================
#  主視窗
# =========================================================
CHAPTER_SHORT = ["用過去幾點投票（MA）", "只記一個數（EMA）", "遇到尖峰（中位數）",
                 "訊號在移動（α-β）", "兩個感測器（互補）", "怎麼選？"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工程常用濾波器：六堂互動課")
        self.chapters = [Ch1MovingAverage(), Ch2EMA(), Ch3Median(), Ch4AlphaBeta(), Ch5Complementary(),
                         Ch6Choose()]
        self.cur = 0
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet("QWidget#root { background:#F8FAFC; }")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._sidebar())
        content = QWidget()
        content.setObjectName("content")
        content.setStyleSheet("QWidget#content { background:#F8FAFC; }")
        v = QVBoxLayout(content)
        v.setContentsMargins(22, 16, 22, 16)
        v.setSpacing(12)
        for ch in self.chapters:
            v.addWidget(ch)
            ch.hide()
        nav = QHBoxLayout()
        self.btn_prev = button("← 上一章", lambda: self.go(self.cur - 1))
        self.btn_next = button("下一章 →", lambda: self.go(self.cur + 1), primary=True)
        self.lbl_prog = QLabel()
        self.lbl_prog.setStyleSheet("QLabel { color:#64748B; font-size:13px; }")
        nav.addWidget(self.btn_prev)
        nav.addStretch(1)
        nav.addWidget(self.lbl_prog)
        nav.addStretch(1)
        nav.addWidget(self.btn_next)
        v.addLayout(nav)
        v.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidget(content)
        h.addWidget(self.scroll, 1)
        self.setCentralWidget(root)
        self.setStyleSheet(STYLE)
        self.resize(1240, 900)
        self.go(0)

    def _sidebar(self):
        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(230)
        side.setStyleSheet("QFrame#side { background:#FFFFFF; border-right:1px solid #E2E8F0; }")
        v = QVBoxLayout(side)
        v.setContentsMargins(14, 18, 14, 18)
        v.setSpacing(6)
        t = QLabel("工程常用濾波器")
        t.setStyleSheet("QLabel { font-size:20px; font-weight:800; color:#0F172A; }")
        v.addWidget(t)
        st = QLabel("六堂互動課・先看問題再看解法")
        st.setStyleSheet("QLabel { font-size:12px; color:#64748B; }")
        v.addWidget(st)
        v.addSpacing(10)
        self.chap_group = QButtonGroup(self)
        self.chap_buttons = []
        for i, name in enumerate(CHAPTER_SHORT):
            b = QPushButton(f"{i + 1}　{name}")
            b.setObjectName("chap")
            b.setCheckable(True)
            b.clicked.connect(lambda _, j=i: self.go(j))
            self.chap_group.addButton(b)
            self.chap_buttons.append(b)
            v.addWidget(b)
        v.addStretch(1)
        legend = QLabel(
            "<b>顏色固定代表</b><br>"
            "<span style='color:#94A3B8'>●</span> 灰點　原始量測<br>"
            "<span style='color:#334155'>┅</span> 虛線　真值（教學用）<br>"
            f"<span style='color:{HEX_MAIN}'>■</span> 藍　這一章的主角<br>"
            f"<span style='color:{HEX_OTHER}'>■</span> 橘　對照組<br>"
            f"<span style='color:{HEX_WIN}'>■</span> 琥珀　窗口 / 權重")
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setStyleSheet("QLabel { font-size:12px; color:#334155; background:#F8FAFC;"
                             " border:1px solid #E2E8F0; border-radius:8px; padding:10px; }")
        v.addWidget(legend)
        return side

    def go(self, i):
        if not 0 <= i < len(self.chapters):
            return
        self.chapters[self.cur].leave()
        self.chapters[self.cur].hide()
        self.cur = i
        self.chapters[i].show()
        self.chapters[i].enter()
        self.chap_buttons[i].setChecked(True)
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(i < len(self.chapters) - 1)
        self.lbl_prog.setText(f"第 {i + 1} / {len(self.chapters)} 章")
        self.scroll.verticalScrollBar().setValue(0)


STYLE = """
QPushButton {
    background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
    padding:7px 14px; font-size:13px; color:#0F172A;
}
QPushButton:hover { background:#EFF6FF; border-color:#60A5FA; }
QPushButton:checked { background:#2563EB; color:white; border-color:#2563EB; }
QPushButton:disabled { color:#CBD5E1; border-color:#E2E8F0; }
QPushButton#primary { background:#2563EB; color:white; border-color:#2563EB; font-weight:600; }
QPushButton#primary:hover { background:#1D4ED8; }
QPushButton#primary:disabled { background:#BFDBFE; border-color:#BFDBFE; }
QPushButton#chap {
    text-align:left; border:none; border-radius:8px; padding:10px 12px;
    font-size:14px; background:transparent; color:#334155;
}
QPushButton#chap:hover { background:#EFF6FF; }
QPushButton#chap:checked { background:#2563EB; color:white; font-weight:700; }
QLabel { color:#0F172A; font-size:13px; }
"""


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setFont(ui_font(10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
