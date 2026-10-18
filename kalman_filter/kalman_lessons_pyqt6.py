# -*- coding: utf-8 -*-
"""
卡爾曼濾波器：六堂互動課（PyQt6）
=========================================
一章只講一個觀念，先看懂直覺、最後才翻成公式：

  第 1 章  兩個都不準的資訊，怎麼合併？        1D：兩個鐘形 → 一個更窄的鐘形（核心中的核心）
  第 2 章  讓時間動起來：預測 → 量測 → 融合     1D 追車：不確定度「預測變胖、融合變瘦」
  第 3 章  Q 與 R：相信模型，還是相信感測器？   兩個旋鈕，看「抖」和「慢」的取捨
  第 4 章  量不到的速度，怎麼估出來？            位置–速度平面上的「斜橢圓」
  第 5 章  工程版：2D 追蹤 + 閘門 + 訊號遺失     用 kalman_2d.py 跑真正的工程版本
  第 6 章  把直覺翻成公式                        1D 寫法 ↔ 矩陣寫法 ↔ kalman_2d.py 的 STEP

整套課程的顏色意義固定不變：
  藍 = 預測（模型說的）　紅 = 量測（感測器說的）　綠 = 融合後的估計（卡爾曼的答案）　灰 = 真值

需求： pip install PyQt6 numpy
執行： python kalman_lessons_pyqt6.py
"""

import sys
import math

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QCheckBox,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF, QFontMetricsF

import kalman_2d as k2d


# =========================================================
#  配色：整套課程固定意義
# =========================================================
C_BG    = QColor("#F8FAFC")
C_INK   = QColor("#0F172A")
C_MUTED = QColor("#64748B")
C_GRID  = QColor("#EEF2F7")
C_FRAME = QColor("#CBD5E1")
C_PRED  = QColor("#2563EB")     # 藍：預測
C_MEAS  = QColor("#DC2626")     # 紅：量測
C_FUSE  = QColor("#16A34A")     # 綠：融合後的估計
C_TRUE  = QColor("#475569")     # 灰：真值
C_AMBER = QColor("#F59E0B")

HEX_PRED, HEX_MEAS, HEX_FUSE, HEX_TRUE = "#2563EB", "#DC2626", "#15803D", "#475569"


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
    return span(s, HEX_PRED)


def red(s):
    return span(s, HEX_MEAS)


def green(s):
    return span(s, HEX_FUSE)


def gray(s):
    return span(s, HEX_TRUE, bold=False)


# =========================================================
#  畫圖小工具
# =========================================================
def draw_text(p, x, y, s, color=C_INK, pt=9.5, bold=False, ha="l", va="b"):
    """在 (x, y) 畫字；ha = l/c/r 水平對齊，va = t/m/b 垂直對齊。回傳文字外框。"""
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
    """圓角小標籤；(x, y) 是標籤的「水平對齊點、垂直中心」。"""
    w, h = text_size(s, pt, bold)
    w += 2 * pad
    h += 6
    if ha == "c":
        x0 = x - w / 2
    elif ha == "r":
        x0 = x - w
    else:
        x0 = x
    rect = QRectF(x0, y - h / 2, w, h)
    p.setPen(QPen(border, 1.4) if border is not None else Qt.PenStyle.NoPen)
    p.setBrush(QBrush(bg))
    p.drawRoundedRect(rect, h / 2, h / 2)
    draw_text(p, rect.center().x(), rect.center().y(), s, fg, pt, bold, ha="c", va="m")
    return rect


def place_labels(p, items, pt=10, bold=True, bounds=None, floor=None):
    """
    items: [(x_center, y_bottom, text, color)]
    由下往上擺標籤，若跟已擺好的重疊就往上移 —— 避免三個鐘形的標籤疊在一起。
    floor(左, 右) 回傳這段水平範圍內「標籤底部最低能到哪」（例如：所有鐘形的最高點之上），
    避免標籤蓋住別的鐘形的頂端。
    """
    placed = []
    for x, y, s, color in sorted(items, key=lambda t: t[1], reverse=True):
        w, h = text_size(s, pt, bold)
        rect = QRectF(x - w / 2, y - h, w, h)
        if bounds is not None:
            if rect.left() < bounds.left():
                rect.moveLeft(bounds.left())
            if rect.right() > bounds.right():
                rect.moveRight(bounds.right())
        if floor is not None:
            rect.moveBottom(min(rect.bottom(), floor(rect.left() - 4, rect.right() + 4)))
        moved = True
        while moved:
            moved = False
            for r in placed:
                if rect.adjusted(-4, -2, 4, 2).intersects(r):
                    rect.moveBottom(r.top() - 3)
                    moved = True
        if bounds is not None and rect.top() < bounds.top():
            rect.moveTop(bounds.top())
        placed.append(rect)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(alpha(QColor("#FFFFFF"), 215)))
        p.drawRoundedRect(rect.adjusted(-3, 0, 3, 0), 4, 4)
        draw_text(p, rect.left(), rect.bottom() - QFontMetricsF(ui_font(pt, bold)).descent(),
                  s, color, pt, bold)


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
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


class Plot:
    """把「世界座標」畫進畫布上的某個矩形（每次 paintEvent 臨時建立一個）。"""

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
        p.setPen(QPen(outline, 1) if outline is not None else Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        for x, y in zip(xs, ys):
            if y is not None and np.isfinite(y):
                p.drawEllipse(self.pt(x, y), r, r)

    def band(self, p, xs, lo, hi, color):
        xs, lo, hi = list(xs), list(lo), list(hi)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        seg = []
        for i in range(len(xs) + 1):
            ok = i < len(xs) and np.isfinite(lo[i]) and np.isfinite(hi[i])
            if ok:
                seg.append(i)
                continue
            if len(seg) >= 2:
                pts = [self.pt(xs[j], hi[j]) for j in seg] + [self.pt(xs[j], lo[j]) for j in reversed(seg)]
                p.drawPolygon(QPolygonF(pts))
            seg = []


def gauss(xs, mu, sig):
    return np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))


def bell_floor(plot, bells, gap=6):
    """給 place_labels 用：一段水平範圍內，所有鐘形的最高點（像素 y）再往上 gap。"""
    def floor(x_left, x_right):
        xs = np.linspace(plot.wx(x_left), plot.wx(x_right), 40)
        top = max(gauss(xs, mu, sig).max() for mu, sig in bells)
        return plot.py(top) - gap
    return floor


def draw_bell(p, plot, mu, sig, color, fill=55, width=2.6, dash=False):
    """畫一個常態分布的鐘形：越窄 = 越確定 = 越高。"""
    xs = np.linspace(plot.x0, plot.x1, 420)
    ys = gauss(xs, mu, sig)
    pts = [plot.pt(x, y) for x, y in zip(xs, ys)]
    p.save()
    p.setClipRect(plot.r)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(alpha(color, fill)))
    p.drawPolygon(QPolygonF([plot.pt(plot.x0, 0)] + pts + [plot.pt(plot.x1, 0)]))
    pen = QPen(color, width)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF(pts))
    p.restore()
    return plot.py(gauss(np.array([mu]), mu, sig)[0])


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
    pts = [b,
           QPointF(b.x() - head * math.cos(ang - 0.42), b.y() - head * math.sin(ang - 0.42)),
           QPointF(b.x() - head * math.cos(ang + 0.42), b.y() - head * math.sin(ang + 0.42))]
    p.drawPolygon(QPolygonF(pts))


def draw_car(p, cx, base_y, color=C_TRUE):
    """一台小車（灰色 = 真實位置）。cx = 車子中心，base_y = 輪子底部。"""
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))
    p.drawRoundedRect(QRectF(cx - 17, base_y - 15, 34, 10), 3, 3)
    p.drawRoundedRect(QRectF(cx - 9, base_y - 22, 18, 9), 3, 3)
    p.setBrush(QBrush(QColor("#1E293B")))
    p.drawEllipse(QPointF(cx - 10, base_y - 4), 4, 4)
    p.drawEllipse(QPointF(cx + 10, base_y - 4), 4, 4)
    p.restore()


def ellipse_world(mean, cov, nsig=2.0, n=96):
    """2×2 協方差 → 橢圓上的世界座標點（nsig 個標準差）。"""
    vals, vecs = np.linalg.eigh(np.asarray(cov, float))
    vals = np.clip(vals, 1e-12, None)
    t = np.linspace(0, 2 * math.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    pts = np.asarray(mean, float).reshape(2, 1) + vecs @ (nsig * np.sqrt(vals).reshape(2, 1) * circle)
    return pts.T


def draw_ellipse(p, plot, mean, cov, color, width=2.2, dash=False, fill=None, nsig=2.0):
    pts = [plot.pt(x, y) for x, y in ellipse_world(mean, cov, nsig)]
    pen = QPen(color, width)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF(pts))


def phase_pills(p, x, y, names, current, done_color=C_FUSE):
    """畫一排步驟標籤（① 預測 ② 量測 ③ 融合），目前這一步高亮。"""
    colors = [C_PRED, C_MEAS, C_FUSE]
    for i, s in enumerate(names):
        if i == current:
            r = draw_pill(p, x, y, s, QColor("#FFFFFF"), colors[i % 3], ha="l", pt=10.5)
        else:
            r = draw_pill(p, x, y, s, C_MUTED, QColor("#F1F5F9"), border=C_FRAME, ha="l",
                          pt=10.5, bold=False)
        x = r.right() + 28
        if i < len(names) - 1:
            draw_text(p, r.right() + 14, y, "→", C_MUTED, 11, ha="c", va="m")


# =========================================================
#  共用小元件
# =========================================================
class LabeledSlider(QWidget):
    """標題 + 滑桿 + 數值。可用線性或對數刻度。"""

    def __init__(self, title, lo, hi, value, fmt, on_change, log=False, steps=200,
                 color="#0F172A", title_w=170):
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
        self.slider.setMinimumWidth(140)
        self.val = QLabel()
        self.val.setFixedWidth(84)
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
        self.val.setText(self.fmt.format(self.value()))

    def _changed(self, _):
        self.val.setText(self.fmt.format(self.value()))
        if not self._quiet:
            self.on_change()


def button(text, fn, primary=False):
    b = QPushButton(text)
    if primary:
        b.setObjectName("primary")
    b.clicked.connect(fn)
    return b


class Chapter(QWidget):
    """
    每一章的固定版型：
      標題 → 💡 一句話重點 → 主畫面（子類別放）→ 下方左「現在發生什麼」、右「🧪 試試看」
    """
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
        self.now.setMinimumHeight(120)
        self.now.setStyleSheet(
            "QLabel { font-size:14px; color:#0F172A; background:#FFFFFF; border:1px solid #E2E8F0;"
            " border-left:5px solid #16A34A; border-radius:8px; padding:10px 12px; }")
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
    """⏮ 上一步 / ⏭ 下一步 / ▶ 播放 / ↺ 重來 —— 每章共用的播放控制。"""

    def __init__(self, chapter, total_fn, interval=800, next_text="⏭ 下一步"):
        self.ch = chapter
        self.total_fn = total_fn
        self.timer = QTimer(chapter)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self._tick)
        self.bar = QHBoxLayout()
        self.bar.setSpacing(8)
        self.bar.addWidget(button("⏮ 上一步", self.back))
        self.bar.addWidget(button(next_text, self.forward, primary=True))
        self.btn_play = button("▶ 自動播放", self.toggle)
        self.bar.addWidget(self.btn_play)
        self.bar.addWidget(button("↺ 從頭開始", self.reset))
        self.bar.addStretch(1)

    def forward(self):
        self.stop()
        self._advance()

    def _advance(self):
        if self.ch.s < self.total_fn() - 1:
            self.ch.s += 1
            self.ch.on_step(forward=True)
            return True
        return False

    def back(self):
        self.stop()
        if self.ch.s > 0:
            self.ch.s -= 1
            self.ch.on_step(forward=False)

    def reset(self):
        self.stop()
        self.ch.s = 0
        self.ch.on_step(forward=False)

    def toggle(self):
        if self.timer.isActive():
            self.stop()
        else:
            if self.ch.s >= self.total_fn() - 1:
                self.ch.s = 0
                self.ch.on_step(forward=False)
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def stop(self):
        self.timer.stop()
        self.btn_play.setText("▶ 自動播放")

    def _tick(self):
        if not self._advance():
            self.stop()


# =========================================================
#  第 1 章　兩個都不準的資訊，怎麼合併？
# =========================================================
def fuse_1d(mu_p, sig_p, z, sig_z):
    """一維融合：回傳 (K, 融合平均, 融合標準差)。"""
    K = sig_p ** 2 / (sig_p ** 2 + sig_z ** 2)
    return K, mu_p + K * (z - mu_p), math.sqrt(1 - K) * sig_p


class FusionCanvas(QWidget):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.drag = None
        self.setMinimumHeight(380)
        self.setMouseTracking(True)

    def plot(self):
        ch = self.ch
        _, _, s_f = fuse_1d(ch.mu_p, ch.sig_p.value(), ch.z, ch.sig_z.value())
        ymax = gauss(np.array([0.0]), 0.0, s_f)[0] * 1.45
        return Plot(QRectF(56, 34, self.width() - 84, self.height() - 128), 0, 25, 0, ymax)

    def paintEvent(self, _):
        ch = self.ch
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        pl = self.plot()
        pl.frame(p, xticks=range(0, 26, 5), xfmt=lambda v: f"{v:g} m",
                 title="車子在哪裡？（橫軸 = 位置，鐘形越窄越高 = 越確定）")
        draw_text(p, pl.r.right(), pl.r.top() - 7, "可以用滑鼠拖動藍色、紅色鐘形 ↔", C_MUTED, 9, ha="r")

        sp, sz = ch.sig_p.value(), ch.sig_z.value()
        K, m, sf = fuse_1d(ch.mu_p, sp, ch.z, sz)
        top_p = draw_bell(p, pl, ch.mu_p, sp, C_PRED)
        top_z = draw_bell(p, pl, ch.z, sz, C_MEAS)
        top_f = draw_bell(p, pl, m, sf, C_FUSE, fill=80, width=3.2)

        # 三條平均值虛線
        for mu, top, c in ((ch.mu_p, top_p, C_PRED), (ch.z, top_z, C_MEAS), (m, top_f, C_FUSE)):
            pen = QPen(c, 1.3)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(pl.px(mu), pl.r.bottom()), QPointF(pl.px(mu), top))

        place_labels(p, [
            (pl.px(ch.mu_p), top_p - 6, f"預測 {ch.mu_p:.1f} m（±{sp:.1f}）", C_PRED),
            (pl.px(ch.z), top_z - 6, f"GPS {ch.z:.1f} m（±{sz:.1f}）", C_MEAS),
            (pl.px(m), top_f - 6, f"融合 {m:.1f} m（±{sf:.1f}）", C_FUSE),
        ], pt=10.5, bounds=pl.r.adjusted(4, 4, -4, 0),
            floor=bell_floor(pl, [(ch.mu_p, sp), (ch.z, sz), (m, sf)]))

        # 下方的「拔河尺」：融合結果在預測與量測之間的哪裡
        y = pl.r.bottom() + 52
        xa, xb, xm = pl.px(ch.mu_p), pl.px(ch.z), pl.px(m)
        for c, x_end in ((QColor("#CBD5E1"), xb), (C_FUSE, xm)):
            pen = QPen(c, 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(xa, y), QPointF(x_end, y))
        p.setPen(Qt.PenStyle.NoPen)
        for x, c, r in ((xa, C_PRED, 7), (xb, C_MEAS, 7), (xm, C_FUSE, 8)):
            p.setBrush(QBrush(c))
            p.drawEllipse(QPointF(x, y), r, r)
        draw_text(p, (xa + xb) / 2, y + 14,
                  f"K = {K:.2f}：從預測往 GPS 走了 {K * 100:.0f}%", C_INK, 11, bold=True, ha="c", va="t")
        p.end()

    def _hit(self, x):
        pl = self.plot()
        cands = [(abs(pl.px(self.ch.mu_p) - x), "p"), (abs(pl.px(self.ch.z) - x), "z")]
        d, which = min(cands)
        return which if d < 30 else None

    def mousePressEvent(self, ev):
        self.drag = self._hit(ev.position().x())

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        if self.drag:
            v = min(max(self.plot().wx(x), 1.0), 24.0)
            self.ch.move_mean(self.drag, v)
        else:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self._hit(x) else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, _):
        self.drag = None


class Ch1Fusion(Chapter):
    NUM = 1
    TITLE = "兩個都不準的資訊，怎麼合併？"
    IDEA = ("「預測」和「GPS」都有誤差。照『誰比較準』來加權平均 —— 越準的權重越大。"
            "這一步就是卡爾曼濾波的核心，後面每一章都只是在重複它。")
    TRIES = ("把「GPS 的不確定」拉到最小：綠色幾乎貼到紅色上（K → 1，幾乎全信 GPS）",
             "按「兩個一樣準」：K = 0.5，綠色剛好在正中間",
             "不管怎麼調，綠色永遠比藍、紅都更窄更高 —— 多一份資訊，一定更確定",
             "拖動紅色鐘形到很遠：結果被拉過去多少只看 K，不管離多遠（第 5 章會講怎麼擋掉離譜的量測）")

    def build(self):
        self.mu_p, self.z = 10.0, 14.0
        self.canvas = FusionCanvas(self)
        self.body.addWidget(self.canvas)
        self.sig_p = LabeledSlider("預測的不確定 σ", 0.5, 6.0, 2.0, "± {:.1f} m", self.refresh,
                                   color=HEX_PRED)
        self.sig_z = LabeledSlider("GPS 的不確定 σ", 0.5, 6.0, 4.0, "± {:.1f} m", self.refresh,
                                   color=HEX_MEAS)
        row = QHBoxLayout()
        row.addWidget(self.sig_p, 1)
        row.addSpacing(16)
        row.addWidget(self.sig_z, 1)
        self.body.addLayout(row)
        presets = QHBoxLayout()
        presets.addWidget(QLabel("快速設定："))
        for text, sp, sz in (("兩個一樣準", 3.0, 3.0), ("GPS 很準", 3.0, 0.8),
                             ("GPS 很爛", 1.5, 6.0), ("還原", 2.0, 4.0)):
            presets.addWidget(button(text, lambda _, a=sp, b=sz: self.preset(a, b)))
        presets.addStretch(1)
        self.body.addLayout(presets)
        self.refresh()

    def preset(self, sp, sz):
        self.sig_p.set_value(sp)
        self.sig_z.set_value(sz)
        if sp == 2.0 and sz == 4.0:
            self.mu_p, self.z = 10.0, 14.0
        self.refresh()

    def move_mean(self, which, v):
        if which == "p":
            self.mu_p = v
        else:
            self.z = v
        self.refresh()

    def refresh(self):
        sp, sz = self.sig_p.value(), self.sig_z.value()
        K, m, sf = fuse_1d(self.mu_p, sp, self.z, sz)
        if K < 0.4:
            verdict = "預測比較準 → 結果比較靠近" + blue("預測")
        elif K > 0.6:
            verdict = "GPS 比較準 → 結果比較靠近" + red("GPS")
        else:
            verdict = "兩個差不多準 → 結果在兩者中間附近"
        self.now.setText(
            "<b>現在發生什麼</b><br>"
            f"① 權重 <b>K</b> = {blue('σ預測²')} ÷ ({blue('σ預測²')} + {red('σGPS²')}) = "
            f"{sp:.1f}² ÷ ({sp:.1f}² + {sz:.1f}²) = <b>{K:.2f}</b><br>"
            f"② 融合位置 = {blue('預測')} + K × ({red('GPS')} − {blue('預測')}) = "
            f"{self.mu_p:.1f} + {K:.2f} × ({self.z:.1f} − {self.mu_p:.1f}) = {green(f'{m:.1f} m')}<br>"
            f"③ 融合後的不確定 = √(1 − K) × σ預測 = {green(f'± {sf:.2f} m')} "
            f"—— 比 {sp:.1f} 和 {sz:.1f} 都小<br>"
            f"<span style='color:#475569'>→ {verdict}</span>")
        self.canvas.update()


# =========================================================
#  一維卡爾曼（第 2、3 章共用）：模型 = 「車速大約 v m/s」
# =========================================================
def kf_1d(z, v_model, q, r, dt=1.0):
    """
    最簡單的一維卡爾曼：
      預測  x⁻ = x + v·dt        P⁻ = P + Q        （變胖）
      融合  K = P⁻/(P⁻+R)        x = x⁻ + K(z − x⁻)  P = (1−K)P⁻   （變瘦，就是第 1 章）
    第 0 秒直接用第一筆量測當起點，P = R。
    """
    n = len(z)
    out = {k: np.full(n, np.nan) for k in ("x_pr", "P_pr", "x_po", "P_po", "K")}
    out["x_po"][0], out["P_po"][0] = z[0], r
    for k in range(1, n):
        out["x_pr"][k] = out["x_po"][k - 1] + v_model * dt
        out["P_pr"][k] = out["P_po"][k - 1] + q
        K = out["P_pr"][k] / (out["P_pr"][k] + r)
        out["K"][k] = K
        out["x_po"][k] = out["x_pr"][k] + K * (z[k] - out["x_pr"][k])
        out["P_po"][k] = (1 - K) * out["P_pr"][k]
    return out


def make_ch2_data():
    rng = np.random.default_rng(252)      # 挑一組「GPS 誤差典型、前幾秒沒有離譜值、沒有偏一邊」的亂數
    n, v, q, sz = 21, 1.0, 0.5, 2.5
    truth = np.empty(n)
    truth[0] = 0.5
    for k in range(1, n):
        truth[k] = truth[k - 1] + v + rng.normal(0, math.sqrt(q))
    z = truth + rng.normal(0, sz, n)
    kf = kf_1d(z, v, q, sz ** 2)
    return dict(n=n, v=v, q=q, sz=sz, truth=truth, z=z, kf=kf)


# =========================================================
#  第 2 章　讓時間動起來：預測 → 量測 → 融合
# =========================================================
class CycleCanvas(QWidget):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.setMinimumHeight(560)

    def paintEvent(self, _):
        ch, d = self.ch, self.ch.d
        kf = d["kf"]
        k, ph = ch.kp()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()

        draw_text(p, 56, 18, f"第 {k} 秒", C_INK, 13, bold=True, va="m")
        if k == 0:
            draw_pill(p, 140, 18, "起點：第一筆 GPS", QColor("#FFFFFF"), C_FUSE, ha="l", pt=10.5)
        else:
            phase_pills(p, 140, 18, ["① 預測", "② 量測", "③ 融合"], ph)

        # ---- 面板 A：這一秒放大看 ----
        center = d["z"][0] if k == 0 else kf["x_pr"][k]
        road = QRectF(56, 60, W - 84, 30)
        pa = Plot(QRectF(56, 92, W - 84, 128), center - 9, center + 9, 0, 0.62)
        draw_text(p, 56, road.top() - 7, "這一秒，放大看（橫軸 = 位置；上方灰色道路上的車 = 真實位置）",
                  C_INK, 10, bold=True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#E2E8F0")))
        p.drawRect(road)
        pen = QPen(QColor("#FFFFFF"), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(road.left(), road.center().y()), QPointF(road.right(), road.center().y()))
        pa.frame(p, xticks=nice_ticks(pa.x0, pa.x1, 9), xfmt=lambda v: f"{v:g} m")
        xt = d["truth"][k]
        if pa.x0 < xt < pa.x1:
            pen = QPen(C_TRUE, 1.2)
            pen.setStyle(Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawLine(QPointF(pa.px(xt), road.bottom()), QPointF(pa.px(xt), pa.r.bottom()))
            draw_car(p, pa.px(xt), road.bottom() - 3)

        labels, bells = [], []
        sd = d["sz"]
        if k == 0:
            top = draw_bell(p, pa, d["z"][0], sd, C_FUSE, fill=80, width=3)
            bells.append((d["z"][0], sd))
            labels.append((pa.px(d["z"][0]), top - 6, f"起點 {d['z'][0]:.1f} m（±{sd:.1f}）", C_FUSE))
        else:
            xp, sp = kf["x_pr"][k], math.sqrt(kf["P_pr"][k])
            bells.append((xp, sp))
            if ph == 0:
                xo, so = kf["x_po"][k - 1], math.sqrt(kf["P_po"][k - 1])
                bells.append((xo, so))
                top_o = draw_bell(p, pa, xo, so, alpha(C_FUSE, 150), fill=18, width=1.6, dash=True)
                top = draw_bell(p, pa, xp, sp, C_PRED, fill=60, width=3)
                ya = (top_o + pa.r.bottom()) / 2 + 10
                draw_arrow(p, QPointF(pa.px(xo), ya), QPointF(pa.px(xp), ya), C_PRED, 2.2)
                draw_text(p, (pa.px(xo) + pa.px(xp)) / 2, ya + 6, f"+{d['v']:.1f} m（車速 × 1 秒）",
                          C_PRED, 9, bold=True, ha="c", va="t")
                labels.append((pa.px(xo), top_o - 6, f"上一秒 {xo:.1f}（±{so:.2f}）", alpha(C_FUSE, 200)))
                labels.append((pa.px(xp), top - 6, f"預測 {xp:.1f}（±{sp:.2f}）", C_PRED))
            else:
                zk = d["z"][k]
                bells.append((zk, sd))
                fade = 255 if ph == 1 else 140
                top = draw_bell(p, pa, xp, sp, alpha(C_PRED, fade), fill=50 if ph == 1 else 20)
                top_z = draw_bell(p, pa, zk, sd, alpha(C_MEAS, fade), fill=50 if ph == 1 else 20)
                labels.append((pa.px(xp), top - 6, f"預測 {xp:.1f}（±{sp:.2f}）", C_PRED))
                labels.append((pa.px(zk), top_z - 6, f"GPS {zk:.1f}（±{sd:.1f}）", C_MEAS))
                if ph == 2:
                    xf, sf = kf["x_po"][k], math.sqrt(kf["P_po"][k])
                    bells.append((xf, sf))
                    top_f = draw_bell(p, pa, xf, sf, C_FUSE, fill=85, width=3.2)
                    labels.append((pa.px(xf), top_f - 6, f"融合 {xf:.1f}（±{sf:.2f}）", C_FUSE))
        place_labels(p, labels, pt=10, bounds=pa.r.adjusted(4, 4, -4, 0), floor=bell_floor(pa, bells))

        # ---- 面板 B：一路走來（位置 vs 時間）----
        n = d["n"]
        lo = min(d["truth"].min(), d["z"].min()) - 2
        hi = max(d["truth"].max(), d["z"].max()) + 2
        pb = Plot(QRectF(56, 262, W - 84, 160), -0.5, n - 0.5, lo, hi)
        pb.frame(p, xticks=range(0, n, 2), yticks=nice_ticks(lo, hi, 5),
                 title="一路走來（橫軸 = 時間 秒，縱軸 = 位置 m）", xlabel="秒")
        ks = np.arange(n)
        done = ks < k                                   # 已經完整處理完的秒數
        cur_fused = (k == 0) or ph == 2
        upto = k + 1 if cur_fused else k
        sd_po = np.sqrt(kf["P_po"])
        pb.band(p, ks[:upto], kf["x_po"][:upto] - 2 * sd_po[:upto], kf["x_po"][:upto] + 2 * sd_po[:upto],
                alpha(C_FUSE, 40))
        pb.polyline(p, ks[:k + 1], d["truth"][:k + 1], C_TRUE, 1.6, dash=True)
        pb.polyline(p, ks[:upto], kf["x_po"][:upto], C_FUSE, 2.6)
        zshow = k + 1 if (k == 0 or ph >= 1) else k
        pb.dots(p, ks[:zshow], d["z"][:zshow], alpha(C_MEAS, 200), 3.6)
        if k >= 1 and ph < 2:
            xp, sp = kf["x_pr"][k], math.sqrt(kf["P_pr"][k])
            p.setPen(QPen(C_PRED, 2))
            p.drawLine(pb.pt(k, xp - 2 * sp), pb.pt(k, xp + 2 * sp))
            pb.dots(p, [k], [xp], C_PRED, 5)
        pen = QPen(C_AMBER, 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(pb.px(k), pb.r.top()), QPointF(pb.px(k), pb.r.bottom()))
        del done

        # ---- 面板 C：不確定度的「呼吸」----
        pc = Plot(QRectF(56, 470, W - 84, 70), -0.5, n - 0.5, 0, d["sz"] * 1.1)
        pc.frame(p, xticks=range(0, n, 2), yticks=[0, 1, 2],
                 title="不確定度 σ（m）：藍 = 預測時變胖 ↑　綠 = 融合時變瘦 ↓")
        prev = (0, sd_po[0])
        for kk in range(1, k + 1):
            a = pc.pt(*prev)
            b = pc.pt(kk, math.sqrt(kf["P_pr"][kk]))
            p.setPen(QPen(C_PRED, 2.4))
            p.drawLine(a, b)
            if kk < k or ph == 2:
                c = pc.pt(kk, sd_po[kk])
                p.setPen(QPen(C_FUSE, 2.4))
                p.drawLine(b, c)
                prev = (kk, sd_po[kk])
            else:
                prev = (kk, math.sqrt(kf["P_pr"][kk]))
        pc.dots(p, [prev[0]], [prev[1]], C_INK, 3.5)
        p.end()


class Ch2Cycle(Chapter):
    NUM = 2
    TITLE = "讓時間動起來：預測 → 量測 → 融合"
    IDEA = ("每一秒都做同一件事：用模型「預測」車子走到哪（不確定變胖）→ 收到 GPS → "
            "用第 1 章的方法「融合」（不確定變瘦）。卡爾曼濾波就是這三步一直循環。")
    TRIES = ("連按「下一步」：藍色鐘形總是比上一秒的綠色胖一點，融合後又瘦回來",
             "看最下面：前幾秒不確定度掉很快，之後每秒「漲一點、掉一點」，穩定下來",
             "注意 K：第 1 秒比較大（還不太信自己的預測），幾秒後就固定在約 0.25",
             "比較中間那張圖：綠線比紅點平滑很多，而且一直貼著灰色的真實位置")

    def build(self):
        self.d = make_ch2_data()
        self.s = 0
        self.canvas = CycleCanvas(self)
        self.stepper = Stepper(self, self.total, interval=900)
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.canvas)
        self.on_step()

    def total(self):
        return 1 + 3 * (self.d["n"] - 1)

    def kp(self):
        if self.s == 0:
            return 0, 0
        return 1 + (self.s - 1) // 3, (self.s - 1) % 3

    def leave(self):
        self.stepper.stop()

    def on_step(self, forward=True):
        d, kf = self.d, self.d["kf"]
        k, ph = self.kp()
        truth = gray(f"（真實位置 {d['truth'][k]:.1f} m —— 現實中看不到，這裡只是讓你對答案）")
        z0, sz = d["z"][0], d["sz"]
        if k == 0:
            html = (f"<b>第 0 秒・起點</b>：第一筆 GPS = {red(f'{z0:.1f} m')}，先直接當作起點；"
                    f"不確定度就是 GPS 的 ± {sz:.1f} m。<br>{truth}")
        else:
            xo, Po = kf["x_po"][k - 1], kf["P_po"][k - 1]
            xp, Pp = kf["x_pr"][k], kf["P_pr"][k]
            zk, K = d["z"][k], kf["K"][k]
            xf, Pf = kf["x_po"][k], kf["P_po"][k]
            if ph == 0:
                html = (f"<b>① 預測</b>：模型說車速約 {d['v']:.1f} m/s，所以 "
                        f"x⁻ = {xo:.1f} + {d['v']:.1f} × 1 = {blue(f'{xp:.1f} m')}。<br>"
                        f"模型不完美（油門會有點鬆緊）→ 不確定度變胖：± {math.sqrt(Po):.2f} → "
                        f"{blue(f'± {math.sqrt(Pp):.2f} m')}　（P⁻ = P + Q = {Po:.2f} + {d['q']:.2f}）")
            elif ph == 1:
                html = (f"<b>② 量測</b>：GPS 說 {red(f'{zk:.1f} m')}（GPS 本身 ± {d['sz']:.1f} m）。<br>"
                        f"現在手上有兩個都不準的答案：{blue(f'預測 {xp:.1f}（±{math.sqrt(Pp):.2f}）')} 和 "
                        f"{red(f'GPS {zk:.1f}（±{sz:.1f}）')} —— 跟第 1 章一模一樣的局面。")
            else:
                html = (f"<b>③ 融合</b>（就是第 1 章）：K = {math.sqrt(Pp):.2f}² ÷ ({math.sqrt(Pp):.2f}² + "
                        f"{d['sz']:.1f}²) = <b>{K:.2f}</b><br>"
                        f"x = {xp:.1f} + {K:.2f} × ({zk:.1f} − {xp:.1f}) = {green(f'{xf:.1f} m')}；"
                        f"不確定度變瘦：± {math.sqrt(Pp):.2f} → {green(f'± {math.sqrt(Pf):.2f} m')}")
            html += f"<br>{truth}"
        self.now.setText(html)
        self.canvas.update()


# =========================================================
#  第 3 章　Q 與 R
# =========================================================
def ch3_speed(t):
    """真實車速：模型以為一直是 1 m/s，但司機會加速、會放慢。"""
    if t < 10:
        return 1.0
    if t < 14:
        return 1.0 + 1.2 * (t - 10) / 4
    if t < 24:
        return 2.2
    if t < 28:
        return 2.2 - 2.0 * (t - 24) / 4
    return 0.2


def make_ch3_data():
    rng = np.random.default_rng(23)
    n, sz = 41, 2.5
    truth = np.empty(n)
    truth[0] = 0.0
    for k in range(1, n):
        truth[k] = truth[k - 1] + ch3_speed(k - 0.5)
    z = truth + rng.normal(0, sz, n)
    return dict(n=n, sz=sz, truth=truth, z=z, v_model=1.0)


def split_error(e, win=9):
    """把誤差拆成「慢慢偏一邊」（落後）和「快速上下跳」（抖動）兩部分，回傳兩者的 RMS。"""
    pad = win // 2
    ep = np.pad(e, pad, mode="edge")
    slow = np.convolve(ep, np.ones(win) / win, mode="valid")
    fast = e - slow
    return float(np.sqrt(np.mean(slow ** 2))), float(np.sqrt(np.mean(fast ** 2)))


class TuneCanvas(QWidget):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.setMinimumHeight(500)

    def paintEvent(self, _):
        ch, d = self.ch, self.ch.d
        kf = ch.kf
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()
        n = d["n"]
        ks = np.arange(n)
        regions = ((10, 24, "司機加速：比模型說的快"), (24, 40, "司機放慢：比模型說的慢"))

        def shade(pl):
            for a, b, _ in regions:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(254, 243, 199, 120)))
                p.drawRect(QRectF(pl.px(a), pl.r.top(), pl.px(b) - pl.px(a), pl.r.height()))

        # 位置
        lo, hi = -4, d["truth"].max() + 6
        pa = Plot(QRectF(56, 30, W - 84, 200), -0.5, n - 0.5, lo, hi)
        pa.frame(p, xticks=range(0, n, 5), yticks=nice_ticks(lo, hi, 5),
                 title="位置（m）—— 模型一直以為車速是 1 m/s")
        shade(pa)
        for a, b, s in regions:
            draw_text(p, (pa.px(a) + pa.px(b)) / 2, pa.r.top() + 6, s, QColor("#92400E"), 9, ha="c", va="t")
        sd = np.sqrt(kf["P_po"])
        pa.band(p, ks, kf["x_po"] - 2 * sd, kf["x_po"] + 2 * sd, alpha(C_FUSE, 40))
        pa.polyline(p, ks, d["truth"], C_TRUE, 1.8, dash=True)
        pa.dots(p, ks, d["z"], alpha(C_MEAS, 170), 3.2)
        pa.polyline(p, ks, kf["x_po"], C_FUSE, 2.8)

        # 誤差
        e_max = float(np.nanmax(np.abs(kf["x_po"] - d["truth"])))
        lim = 8 if e_max < 7 else (12 if e_max < 11 else 16)
        pe = Plot(QRectF(56, 268, W - 84, 130), -0.5, n - 0.5, -lim, lim)
        pe.frame(p, xticks=range(0, n, 5), yticks=[-lim, -lim / 2, 0, lim / 2, lim],
                 title="誤差 = 估計 − 真實（m）：整段偏一邊 = 落後；一直上下跳 = 抖動")
        shade(pe)
        p.setPen(QPen(C_TRUE, 1.2))
        p.drawLine(pe.pt(-0.5, 0), pe.pt(n - 0.5, 0))
        p.save()
        p.setClipRect(pe.r)
        pe.dots(p, ks, d["z"] - d["truth"], alpha(C_MEAS, 110), 2.8)
        pe.polyline(p, ks, kf["x_po"] - d["truth"], C_FUSE, 2.6)
        p.restore()

        # K
        pk = Plot(QRectF(56, 436, W - 84, 60), -0.5, n - 0.5, 0, 1)
        pk.frame(p, xticks=range(0, n, 5), yticks=[0, 0.5, 1], yfmt=lambda v: f"{v:g}",
                 title="K（每次往 GPS 修正的比例）", xlabel="秒")
        pk.polyline(p, ks, kf["K"], C_FUSE, 2.4)
        p.end()


class Ch3Tuning(Chapter):
    NUM = 3
    TITLE = "Q 與 R：要相信模型，還是相信感測器？"
    IDEA = ("Q = 你覺得「模型」每秒可能錯多少；R = 你覺得「GPS」有多吵。"
            "兩者的比例決定 K：太相信模型 → 平滑但跟不上變化（落後）；太相信 GPS → 跟得快但很抖。")
    TRIES = ("按「太相信模型」：綠線很平滑，但司機一加速它就落後（誤差圖整段偏一邊）",
             "按「太相信 GPS」：綠線幾乎跟著紅點跳，K 接近 1",
             "把兩個滑桿往右推「同樣多格」：K 和結果完全不變 —— 只有比例重要",
             "按「剛剛好」：這是電腦把所有 Q 都試過一遍後，誤差最小的那一個")

    def build(self):
        self.d = make_ch3_data()
        self.canvas = TuneCanvas(self)
        self.s_q = LabeledSlider("模型每秒可能錯 √Q", 0.03, 0.03 * 10 ** 2.5, 0.7, "± {:.2g} m",
                                 self.refresh, log=True, steps=100, color=HEX_PRED, title_w=180)
        self.s_r = LabeledSlider("GPS 有多吵 √R", 0.1, 0.1 * 10 ** 2.5, 2.5, "± {:.2g} m",
                                 self.refresh, log=True, steps=100, color=HEX_MEAS, title_w=140)
        row = QHBoxLayout()
        row.addWidget(self.s_q, 1)
        row.addSpacing(16)
        row.addWidget(self.s_r, 1)
        self.body.addLayout(row)

        # 「剛剛好」：固定 R = 實際的 GPS 雜訊，掃過所有 Q 找誤差最小
        best, best_err = 0.7, 1e9
        for sq in np.geomspace(0.03, 9.0, 80):
            kf = kf_1d(self.d["z"], self.d["v_model"], sq ** 2, self.d["sz"] ** 2)
            err = float(np.sqrt(np.mean((kf["x_po"] - self.d["truth"]) ** 2)))
            if err < best_err:
                best, best_err = sq, err
        self.best_q = best

        presets = QHBoxLayout()
        presets.addWidget(QLabel("快速設定："))
        for text, q, r in (("太相信模型", 0.05, 2.5), ("太相信 GPS", 0.7, 0.15),
                           ("剛剛好", self.best_q, 2.5)):
            presets.addWidget(button(text, lambda _, a=q, b=r: self.preset(a, b)))
        presets.addStretch(1)
        self.body.addLayout(presets)
        self.body.addWidget(self.canvas)
        self.refresh()

    def preset(self, q, r):
        self.s_q.set_value(q)
        self.s_r.set_value(r)
        self.refresh()

    def refresh(self):
        d = self.d
        sq, sr = self.s_q.value(), self.s_r.value()
        self.kf = kf_1d(d["z"], d["v_model"], sq ** 2, sr ** 2)
        e = (self.kf["x_po"] - d["truth"])[2:]
        lag, jit = split_error(e)
        total = float(np.sqrt(np.mean(e ** 2)))
        gps = float(np.sqrt(np.mean((d["z"] - d["truth"])[2:] ** 2)))
        K_ss = self.kf["K"][-1]
        if lag > 1.6 * jit and lag > 1.0:
            verdict = "太相信模型：很平滑，但跟不上司機加減速 → 誤差整段偏一邊"
        elif K_ss > 0.6:
            verdict = "太相信 GPS：幾乎照抄 GPS → 抖動跟 GPS 差不多大"
        else:
            verdict = "兩邊平衡：落後和抖動都不大"

        def bar(v, color):
            n_cells = max(1, int(round(min(v / 6.0, 1.0) * 30)))
            return f"<span style='background:{color};'>{'&nbsp;' * n_cells}</span>"

        self.now.setText(
            f"<b>穩態 K ≈ {K_ss:.2f}</b>：每次 GPS 進來，往 GPS 修正 {K_ss * 100:.0f}%。"
            f"　（K 由 Q ÷ R = {sq ** 2:.3g} ÷ {sr ** 2:.3g} 這個比例決定）<br>"
            f"落後（整段偏一邊）　<b>{lag:.2f} m</b>　{bar(lag, '#FDE68A')}<br>"
            f"抖動（一直上下跳）　<b>{jit:.2f} m</b>　{bar(jit, '#FECACA')}<br>"
            f"總誤差 RMSE = {green(f'{total:.2f} m')}　（直接用 GPS = {red(f'{gps:.2f} m')}）"
            f"　<span style='color:#475569'>→ {verdict}</span>")
        self.canvas.update()


# =========================================================
#  第 4 章　量不到的速度，怎麼估出來？
# =========================================================
def make_ch4_data():
    rng = np.random.default_rng(5)
    n, dt, v_true, sz, sa = 9, 1.0, 2.0, 2.0, 0.2
    truth_p = 0.5 + v_true * np.arange(n)
    z = truth_p + rng.normal(0, sz, n)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = sa ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    H = np.array([[1.0, 0.0]])
    R = sz ** 2
    x = np.array([z[0], 0.0])
    P = np.diag([R, 5.0 ** 2])
    rec = dict(x_po=[x.copy()], P_po=[P.copy()], x_pr=[None], P_pr=[None])
    for k in range(1, n):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        S = (H @ Pp @ H.T)[0, 0] + R
        K = (Pp @ H.T / S).ravel()
        x = xp + K * (z[k] - xp[0])
        P = (np.eye(2) - np.outer(K, H)) @ Pp
        rec["x_pr"].append(xp)
        rec["P_pr"].append(Pp)
        rec["x_po"].append(x.copy())
        rec["P_po"].append(P.copy())
    u = np.random.default_rng(99).standard_normal((320, 2))     # 固定的「點雲」種子
    return dict(n=n, v_true=v_true, sz=sz, truth_p=truth_p, z=z, F=F, rec=rec, u=u)


def sample_cloud(u, mean, cov):
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
    return np.asarray(mean) + u @ L.T


class HiddenCanvas(QWidget):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.setMinimumHeight(480)

    def paintEvent(self, _):
        ch, d = self.ch, self.ch.d
        rec = d["rec"]
        k, ph = ch.kp()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        W = self.width()

        draw_text(p, 56, 18, f"第 {k} 秒", C_INK, 13, bold=True, va="m")
        if k == 0:
            draw_pill(p, 140, 18, "起點：位置來自 GPS，速度完全不知道", QColor("#FFFFFF"), C_FUSE, ha="l",
                      pt=10.5)
        else:
            phase_pills(p, 140, 18, ["① 預測（斜切）", "② 量測位置", "③ 融合"], ph)

        split = int(W * 0.63)
        ps = Plot(QRectF(64, 62, split - 110, 380), -12, 28, -11, 13)
        ps.frame(p, xticks=range(-10, 29, 5), yticks=range(-10, 13, 5),
                 xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}",
                 title="狀態平面：橫軸 = 位置（m），縱軸 = 速度（m/s）", xlabel="位置 m")
        draw_text(p, ps.r.left() + 6, ps.r.top() + 6, "速度 m/s ↑", C_MUTED, 8.5, va="t")

        p.save()
        p.setClipRect(ps.r)
        # 量測帶：GPS 只告訴你位置
        if k >= 1 and ph >= 1:
            zk = d["z"][k]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(alpha(C_MEAS, 45 if ph == 1 else 28)))
            p.drawRect(QRectF(ps.px(zk - 2 * d["sz"]), ps.r.top(),
                              ps.px(zk + 2 * d["sz"]) - ps.px(zk - 2 * d["sz"]), ps.r.height()))
            pen = QPen(C_MEAS, 1.6)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(ps.px(zk), ps.r.top()), QPointF(ps.px(zk), ps.r.bottom()))

        # 點雲（每個點 = 一種「可能的 (位置, 速度)」）
        pts, colors = ch.cloud()
        p.setPen(Qt.PenStyle.NoPen)
        for (x, y), c in zip(pts, colors):
            p.setBrush(QBrush(c))
            p.drawEllipse(ps.pt(x, y), 2.4, 2.4)

        # 橢圓
        if k == 0:
            draw_ellipse(p, ps, rec["x_po"][0], rec["P_po"][0], C_FUSE, 2.6)
        elif ph == 0:
            draw_ellipse(p, ps, rec["x_po"][k - 1], rec["P_po"][k - 1], alpha(C_FUSE, 120), 1.4, dash=True)
            if ch.anim >= 1.0:
                draw_ellipse(p, ps, rec["x_pr"][k], rec["P_pr"][k], C_PRED, 2.6)
        else:
            draw_ellipse(p, ps, rec["x_pr"][k], rec["P_pr"][k], C_PRED, 2.2, dash=(ph == 2))
            if ph == 2:
                draw_ellipse(p, ps, rec["x_po"][k], rec["P_po"][k], C_FUSE, 3.0, fill=alpha(C_FUSE, 30))
        # 真值
        tx, ty = d["truth_p"][k], d["v_true"]
        p.setPen(QPen(QColor("#FFFFFF"), 1.5))
        p.setBrush(QBrush(C_TRUE))
        p.drawPolygon(QPolygonF([ps.pt(tx, ty) + QPointF(0, -7), ps.pt(tx, ty) + QPointF(7, 0),
                                 ps.pt(tx, ty) + QPointF(0, 7), ps.pt(tx, ty) + QPointF(-7, 0)]))
        p.restore()
        draw_text(p, ps.px(tx) + 9, ps.py(ty) - 9, "真實", C_TRUE, 8.5, bold=True)

        # 右側「速度範圍」括號：一眼看出速度變確定了多少
        def bracket(x, mean, var, color, text):
            s = 2 * math.sqrt(var)
            a, b = ps.py(mean + s), ps.py(mean - s)
            a, b = max(a, ps.r.top()), min(b, ps.r.bottom())
            p.setPen(QPen(color, 3))
            p.drawLine(QPointF(x, a), QPointF(x, b))
            p.drawLine(QPointF(x - 5, a), QPointF(x + 5, a))
            p.drawLine(QPointF(x - 5, b), QPointF(x + 5, b))
            draw_text(p, x + 9, (a + b) / 2, text, color, 9, bold=True, va="m")

        bx = ps.r.right() + 12
        if k == 0:
            P0 = rec["P_po"][0]
            bracket(bx, rec["x_po"][0][1], P0[1, 1], C_FUSE, f"σ {math.sqrt(P0[1, 1]):.1f}")
        else:
            Pp = rec["P_pr"][k]
            if ph < 2:
                bracket(bx, rec["x_pr"][k][1], Pp[1, 1], C_PRED, f"σ {math.sqrt(Pp[1, 1]):.1f}")
            else:
                Pf = rec["P_po"][k]
                bracket(bx, rec["x_pr"][k][1], Pp[1, 1], alpha(C_PRED, 120), "")
                bracket(bx + 4, rec["x_po"][k][1], Pf[1, 1], C_FUSE, f"σ {math.sqrt(Pf[1, 1]):.1f}")
        draw_text(p, bx - 4, ps.r.top() - 20, "速度範圍", C_MUTED, 8.5)
        draw_text(p, bx - 4, ps.r.top() - 6, "（±2σ）", C_MUTED, 8.5)

        # 右邊小圖：速度估計 vs 時間
        pv = Plot(QRectF(split + 60, 62, W - split - 84, 380), -0.5, d["n"] - 0.5, -11, 13)
        pv.frame(p, xticks=range(0, d["n"], 2), yticks=range(-10, 13, 5),
                 title="速度估計（m/s）—— 從沒量過！", xlabel="秒")
        pen = QPen(C_TRUE, 1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(pv.pt(-0.5, d["v_true"]), pv.pt(d["n"] - 0.5, d["v_true"]))
        draw_text(p, pv.r.right() - 4, pv.py(d["v_true"]) - 5, "真實 2 m/s", C_TRUE, 8.5, ha="r")
        upto = k + 1 if (k == 0 or ph == 2) else k
        ks = np.arange(upto)
        v = np.array([rec["x_po"][i][1] for i in range(upto)])
        sv = np.array([math.sqrt(rec["P_po"][i][1, 1]) for i in range(upto)])
        pv.band(p, ks, v - 2 * sv, v + 2 * sv, alpha(C_FUSE, 45))
        pv.polyline(p, ks, v, C_FUSE, 2.6)
        pv.dots(p, ks, v, C_FUSE, 3.5)
        p.end()


class Ch4Hidden(Chapter):
    NUM = 4
    TITLE = "量不到的速度，怎麼估出來？"
    IDEA = ("GPS 只量位置，卡爾曼卻能估出速度。秘密在「預測」：位置 += 速度 × 時間，"
            "會讓位置和速度綁在一起（橢圓變斜）。於是修正位置時，速度也被順便修正了。")
    TRIES = ("連按幾次，看橢圓：直立長條 →（預測）被斜切 →（融合）被紅帶切小 → 越來越小",
             "第 1 秒融合後，看右邊的「速度範圍」括號：一下子縮小一半以上",
             "看右圖：速度估計從 0 開始，幾秒內就靠近真實的 2 m/s，而且綠帶越來越窄",
             "想一想：如果預測時橢圓「不會」變斜，量位置還能修正速度嗎？（不能！）")

    def build(self):
        self.d = make_ch4_data()
        self.s = 0
        self.anim = 1.0
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(30)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.canvas = HiddenCanvas(self)
        self.stepper = Stepper(self, self.total, interval=1300)
        self.body.addLayout(self.stepper.bar)
        self.body.addWidget(self.canvas)
        self.on_step()

    def total(self):
        return 1 + 3 * (self.d["n"] - 1)

    def kp(self):
        if self.s == 0:
            return 0, 0
        return 1 + (self.s - 1) // 3, (self.s - 1) % 3

    def leave(self):
        self.stepper.stop()
        self.anim_timer.stop()

    def _anim_tick(self):
        self.anim = min(1.0, self.anim + 0.07)
        if self.anim >= 1.0:
            self.anim_timer.stop()
        self.canvas.update()

    def cloud(self):
        """目前要畫的點雲與顏色。"""
        d, rec, u = self.d, self.d["rec"], self.d["u"]
        k, ph = self.kp()
        if k == 0:
            pts = sample_cloud(u, rec["x_po"][0], rec["P_po"][0])
            return pts, [alpha(C_FUSE, 150)] * len(pts)
        before = sample_cloud(u, rec["x_po"][k - 1], rec["P_po"][k - 1])
        after = before @ d["F"].T                        # 每個點照自己的速度往前走 1 秒
        if ph == 0:
            t = self.anim
            t = t * t * (3 - 2 * t)
            pts = before + (after - before) * t
            col = QColor(int(22 + (37 - 22) * t), int(163 + (99 - 163) * t), int(74 + (235 - 74) * t), 150)
            return pts, [col] * len(pts)
        if ph == 1:
            return after, [alpha(C_PRED, 150)] * len(after)
        like = np.exp(-0.5 * ((after[:, 0] - d["z"][k]) / d["sz"]) ** 2)
        return after, [alpha(C_PRED, int(20 + 180 * w)) for w in like]

    def on_step(self, forward=True):
        d, rec = self.d, self.d["rec"]
        k, ph = self.kp()
        if forward and k >= 1 and ph == 0:
            self.anim = 0.0
            self.anim_timer.start()
        else:
            self.anim = 1.0
        zk, sz = d["z"][k], d["sz"]
        if k == 0:
            P0 = rec["P_po"][0]
            html = (f"<b>起點</b>：第一筆 GPS 說位置 {red(f'{zk:.1f} m')}（± {sz:.0f} m）；"
                    f"速度完全沒量過 → 只好先猜 0，不確定 ± {math.sqrt(P0[1, 1]):.0f} m/s。<br>"
                    f"所以橢圓又高又窄：{green('位置大概知道、速度幾乎不知道')}。"
                    f"每個小點 = 一種「可能的 (位置, 速度)」。")
        else:
            Po, Pp = rec["P_po"][k - 1], rec["P_pr"][k]
            sv_before = math.sqrt(Pp[1, 1])
            if ph == 0:
                corr = Pp[0, 1] / math.sqrt(Pp[0, 0] * Pp[1, 1])
                html = (f"<b>① 預測</b>：每個點照「自己的速度」往前走 1 秒（位置 += 速度 × 1）。"
                        f"速度大的點跑得遠、速度小的跑得近 →<br>整團點被{blue('斜切')}成斜的橢圓："
                        f"位置和速度從此綁在一起（相關係數 {corr:.2f}）。速度本身的不確定度沒變："
                        f"± {math.sqrt(Po[1, 1]):.2f} m/s。")
            elif ph == 1:
                html = (f"<b>② 量測</b>：GPS 只說位置 = {red(f'{zk:.1f} m')}（± {sz:.0f} m）"
                        f"→ 一條{red('直立的紅帶')}。<br>它對速度一個字都沒說 —— 紅帶上下一樣寬。")
            else:
                Pf = rec["P_po"][k]
                v_est = rec["x_po"][k][1]
                html = (f"<b>③ 融合</b>：紅帶外的點被淘汰（變淡）。因為橢圓是斜的，被淘汰的剛好是"
                        f"「速度太大」和「速度太小」的點！<br>"
                        f"速度的不確定度 ± {sv_before:.2f} → {green(f'± {math.sqrt(Pf[1, 1]):.2f} m/s')}，"
                        f"速度估計 = {green(f'{v_est:.2f} m/s')}（真實 2.00）。"
                        f"從頭到尾沒量過速度！")
        self.now.setText(html)
        self.canvas.update()


# =========================================================
#  第 5 章　工程版：2D 追蹤 + 閘門 + 訊號遺失（用 kalman_2d.py）
# =========================================================
SIGMA_A, SIGMA_Z = 0.8, 5.0


class TrackCanvas(QWidget):
    VIEW_HALF_W = 46.0           # 跟車視角：左右各看 46 m

    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.setMinimumHeight(440)

    def paintEvent(self, _):
        ch = self.ch
        sim, run, k = ch.sim, ch.run, ch.s
        st = run.steps[k]
        u = st.update
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)
        rect = QRectF(10, 30, self.width() - 20, self.height() - 40)
        center = run.est[k][:2]
        scale = rect.width() / (2 * self.VIEW_HALF_W)
        half_h = rect.height() / (2 * scale)
        pl = Plot(rect, center[0] - self.VIEW_HALF_W, center[0] + self.VIEW_HALF_W,
                  center[1] - half_h, center[1] + half_h)
        draw_text(p, rect.left(), 18, "跟車視角（格線 = 10 m，畫面跟著估計位置移動）", C_INK, 10, bold=True, va="m")
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawRect(rect)
        p.setClipRect(rect)
        p.setPen(QPen(C_GRID, 1))
        for gx in range(int(pl.x0 // 10) * 10, int(pl.x1) + 10, 10):
            p.drawLine(QPointF(pl.px(gx), rect.top()), QPointF(pl.px(gx), rect.bottom()))
        for gy in range(int(pl.y0 // 10) * 10, int(pl.y1) + 10, 10):
            p.drawLine(QPointF(rect.left(), pl.py(gy)), QPointF(rect.right(), pl.py(gy)))

        # 過去：真值灰虛線、GPS 紅點（被擋的畫 ✕）、估計綠線
        pl.polyline(p, sim.truth[:k + 1, 0], sim.truth[:k + 1, 1], C_TRUE, 1.6, dash=True)
        for i in range(k):
            z = sim.z[i]
            if z is None:
                continue
            if run.status[i] in ("rejected",):
                self._cross(p, pl.pt(z[0], z[1]), alpha(C_MEAS, 160), 5)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(alpha(C_MEAS, 120)))
                p.drawEllipse(pl.pt(z[0], z[1]), 3, 3)
        ok = ~np.isnan(run.est[:k + 1, 0])
        pl.polyline(p, run.est[:k + 1, 0][ok], run.est[:k + 1, 1][ok], C_FUSE, 2.6)

        # 這一秒：預測（藍虛線橢圓）→ 量測（紅）→ 融合（綠）
        if st.predict is not None:
            xp, Pp = st.predict.x_prior, st.predict.P_prior
            draw_ellipse(p, pl, xp[:2], Pp[:2, :2], C_PRED, 2.0, dash=True, fill=alpha(C_PRED, 18))
            pl.dots(p, [xp[0]], [xp[1]], C_PRED, 4.5)
        if u.z is not None:
            zp = pl.pt(u.z[0], u.z[1])
            if rect.contains(zp):
                if u.status in ("rejected",):
                    self._cross(p, zp, C_MEAS, 9)
                else:
                    pl.dots(p, [u.z[0]], [u.z[1]], C_MEAS, 6, outline=QColor("#FFFFFF"))
            else:
                self._edge_arrow(p, pl, rect, center, u.z, C_MEAS)
        xf, Pf = run.est[k], run.P[k]
        draw_ellipse(p, pl, xf[:2], Pf[:2, :2], C_FUSE, 2.8, fill=alpha(C_FUSE, 35))
        if st.predict is not None and u.status == "accepted":
            draw_arrow(p, pl.pt(*st.predict.x_prior[:2]), pl.pt(*xf[:2]), C_FUSE, 2.2, 7)
        vx, vy = xf[2], xf[3]
        draw_arrow(p, pl.pt(*xf[:2]), pl.pt(xf[0] + 2 * vx, xf[1] + 2 * vy), QColor("#0E7490"), 2.4, 8)
        pl.dots(p, [xf[0]], [xf[1]], C_FUSE, 5.5, outline=QColor("#FFFFFF"))
        pl.dots(p, [sim.truth[k, 0]], [sim.truth[k, 1]], C_TRUE, 4.5, outline=QColor("#FFFFFF"))
        p.restore()
        p.setPen(QPen(C_FRAME, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)

        if u.status == "missing":
            draw_pill(p, rect.center().x(), rect.top() + 22, "🚇 隧道中：收不到 GPS，只能預測",
                      QColor("#FFFFFF"), QColor("#475569"), pt=11)
        draw_pill(p, rect.left() + 8, rect.bottom() - 16,
                  f"青色箭頭 = 照目前速度，2 秒後會到哪（速度 {math.hypot(vx, vy):.1f} m/s）",
                  QColor("#0E7490"), alpha(QColor("#FFFFFF"), 230), border=QColor("#A5F3FC"), ha="l",
                  pt=9, bold=False)
        self._minimap(p, rect, pl)
        p.end()

    @staticmethod
    def _cross(p, c, color, r):
        p.setPen(QPen(color, 2.6))
        p.drawLine(c + QPointF(-r, -r), c + QPointF(r, r))
        p.drawLine(c + QPointF(-r, r), c + QPointF(r, -r))

    @staticmethod
    def _edge_arrow(p, pl, rect, center, z, color):
        """量測跑到畫面外（離群值）：在邊緣畫箭頭指向它，並標距離。"""
        d = np.asarray(z) - center
        dist = float(np.hypot(*d))
        c = rect.center()
        ux, uy = d[0] / dist, -d[1] / dist
        tx = (rect.width() / 2 - 30) / abs(ux) if abs(ux) > 1e-9 else 1e9
        ty = (rect.height() / 2 - 30) / abs(uy) if abs(uy) > 1e-9 else 1e9
        t = min(tx, ty)
        tip = QPointF(c.x() + ux * t, c.y() + uy * t)
        tail = QPointF(tip.x() - ux * 40, tip.y() - uy * 40)
        draw_arrow(p, tail, tip, color, 2.6, 10)
        draw_pill(p, tail.x() - ux * 30, tail.y() - uy * 30, f"GPS 在 {dist:.0f} m 外", QColor("#FFFFFF"),
                  color, pt=9)

    def _minimap(self, p, rect, pl):
        sim, run, k = self.ch.sim, self.ch.run, self.ch.s
        mw, mh = 150, 104
        box = QRectF(rect.right() - mw - 8, rect.top() + 8, mw, mh)
        p.setPen(QPen(C_FRAME, 1))
        p.setBrush(QBrush(alpha(QColor("#FFFFFF"), 235)))
        p.drawRoundedRect(box, 6, 6)
        xs, ys = sim.truth[:, 0], sim.truth[:, 1]
        x0, x1, y0, y1 = xs.min() - 15, xs.max() + 15, ys.min() - 15, ys.max() + 15
        s = min((mw - 16) / (x1 - x0), (mh - 26) / (y1 - y0))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inner = QRectF(box.center().x() - (x1 - x0) * s / 2, box.top() + 18 + ((mh - 26) - (y1 - y0) * s) / 2,
                       (x1 - x0) * s, (y1 - y0) * s)
        mp = Plot(inner, x0, x1, y0, y1)
        draw_text(p, box.left() + 6, box.top() + 4, "全程地圖", C_MUTED, 8, va="t")
        a, b = sim.is_dropout.argmax(), len(sim.is_dropout) - sim.is_dropout[::-1].argmax()
        if sim.is_dropout.any():
            mp.polyline(p, xs[a:b], ys[a:b], QColor("#CBD5E1"), 7)
        mp.polyline(p, xs, ys, C_FRAME, 1.4, dash=True)
        ok = ~np.isnan(run.est[:k + 1, 0])
        mp.polyline(p, run.est[:k + 1, 0][ok], run.est[:k + 1, 1][ok], C_FUSE, 2)
        p.setPen(QPen(C_AMBER, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(mp.pt(pl.x0, pl.y1), mp.pt(pl.x1, pl.y0)).intersected(inner.adjusted(-4, -4, 4, 4)))
        del cx, cy


class Ch5Engineering(Chapter):
    NUM = 5
    TITLE = "工程版：2D 追蹤 + 閘門 + 訊號遺失"
    IDEA = ("真正的工程版 = 前四章（x、y 兩個方向一起做，同時估位置和速度）"
            "＋ 兩個保護：太離譜的 GPS 直接丟掉（閘門）；收不到 GPS 時只預測（遺失）。")
    TRIES = ("一路按到第 12 秒：GPS 突然跳到 40 m 外，看右邊「閘門」怎麼把它擋掉",
             "取消勾選「閘門」再看第 12 秒：綠色估計被跳點硬拉過去",
             "走到第 38～43 秒（隧道）：沒有 GPS，橢圓每秒變大；出隧道第一筆 GPS 會被大大相信",
             "看左下「到目前為止的位置誤差」：卡爾曼比直接用 GPS 準多少")

    def build(self):
        self.s = 0
        bar = QHBoxLayout()
        self.chk_out = QCheckBox("GPS 偶爾亂跳（離群值）")
        self.chk_tun = QCheckBox("經過隧道（訊號遺失）")
        self.chk_gate = QCheckBox("開啟閘門")
        for c in (self.chk_out, self.chk_tun, self.chk_gate):
            c.setChecked(True)
            c.toggled.connect(self.rebuild)
            bar.addWidget(c)
        bar.addStretch(1)
        self.stepper = Stepper(self, self.total, interval=450, next_text="⏭ 下一秒")
        for i, (text, kk) in enumerate((("跳到第 12 秒（跳點）", 12), ("跳到隧道", 38))):
            self.stepper.bar.insertWidget(4 + i, button(text, lambda _, t=kk: self.jump(t)))
        self.body.addLayout(self.stepper.bar)
        self.body.addLayout(bar)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.canvas = TrackCanvas(self)
        row.addWidget(self.canvas, 1)
        cards = QVBoxLayout()
        cards.setSpacing(8)
        self.card_ok = QLabel()
        self.card_gate = QLabel()
        self.card_miss = QLabel()
        for c in (self.card_ok, self.card_gate, self.card_miss):
            c.setWordWrap(True)
            c.setTextFormat(Qt.TextFormat.RichText)
            c.setFixedWidth(270)
            c.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            cards.addWidget(c)
        cards.addStretch(1)
        row.addLayout(cards)
        self.body.addLayout(row)
        self.rebuild()

    def total(self):
        return self.sim.n

    def leave(self):
        self.stepper.stop()

    def jump(self, k):
        self.stepper.stop()
        self.s = min(k, self.total() - 1)
        self.on_step()

    def rebuild(self):
        self.sim = k2d.simulate_2d_track(n=60, sigma_z=SIGMA_Z, outliers=self.chk_out.isChecked(),
                                         dropout=self.chk_tun.isChecked())
        self.run = k2d.run_filter(self.sim, SIGMA_A, SIGMA_Z, use_gating=self.chk_gate.isChecked())
        self.s = min(getattr(self, "s", 0), self.sim.n - 1)
        self.on_step()

    @staticmethod
    def _card(label, title, body, active, color, bg):
        if active:
            style = (f"QLabel {{ font-size:13px; color:#0F172A; background:{bg}; border:2px solid {color};"
                     " border-radius:10px; padding:10px; }")
            label.setText(f"<span style='font-size:15px;font-weight:800;color:{color}'>{title}</span><br>{body}")
        else:
            style = ("QLabel { font-size:12px; color:#94A3B8; background:#FFFFFF; border:1px solid #E2E8F0;"
                     " border-radius:10px; padding:10px; }")
            label.setText(f"<span style='font-size:14px;font-weight:700'>{title}</span><br>{body}")
        label.setStyleSheet(style)

    def on_step(self, forward=True):
        sim, run, k = self.sim, self.run, self.s
        st = run.steps[k]
        u = st.update
        status = u.status
        dist = float(np.hypot(*u.y)) if u.y is not None else 0.0

        ok_body = "預測和 GPS 差不多 → 照第 1 章的方法融合。"
        gate_body = "GPS 離預測太遠、遠到不合理 → 判定為跳點，丟掉不用。"
        miss_body = "沒有 GPS → 跳過融合，只靠模型往前推。"
        ok_active = gate_active = miss_active = False
        if status == "init":
            ok_active = True
            ok_body = "第 0 秒：第一筆 GPS 直接當起點；速度還不知道（不確定度很大）。"
        elif status == "accepted":
            ok_active = True
            ok_body = (f"GPS 和預測差 <b>{dist:.1f} m</b>，在合理範圍內（NIS {u.nis:.1f} ≤ 9.21）"
                       f"→ 融合，位置權重 K ≈ <b>{u.K[0, 0]:.2f}</b>。")
            if sim.is_outlier[k]:
                ok_body += ("<br><b style='color:#B91C1C'>⚠ 這筆其實是跳點！閘門關著，"
                            f"它被照樣融合（NIS {u.nis:.0f}）→ 估計被拉歪了。</b>")
        elif status in ("rejected", "reinit"):
            gate_active = True
            gate_body = (f"這筆 GPS 離預測 <b>{dist:.0f} m</b>。以目前的不確定度，正常 GPS 幾乎不可能差這麼多"
                         f"（NIS {u.nis:.1f} &gt; 門檻 9.21）→ <b>丟掉</b>，這一秒只用預測。")
            if status == "reinit":
                gate_body += "<br>而且已經連續被擋 3 次 → 判斷是自己跟丟了，用這筆 GPS 重新開始。"
        elif status == "missing":
            miss_active = True
            Pp = st.predict.P_prior
            s_now = math.sqrt(max(Pp[0, 0], Pp[1, 1]))
            miss_body = (f"隧道裡沒有 GPS → 只預測，不確定度一直長大（現在約 ± <b>{s_now:.1f} m</b>）。"
                         "出隧道後第一筆 GPS 會被大大相信（K 很大）。")
        self._card(self.card_ok, "✅ 正常融合", ok_body, ok_active, "#15803D", "#F0FDF4")
        self._card(self.card_gate, "🚧 閘門：擋下離譜的 GPS", gate_body, gate_active, "#B91C1C", "#FEF2F2")
        self._card(self.card_miss, "🚇 訊號遺失：只預測", miss_body, miss_active, "#475569", "#F1F5F9")

        # 到目前為止的誤差
        upto = k + 1
        gps_err = [np.hypot(*(sim.z[i] - sim.truth[i, :2])) for i in range(upto) if sim.z[i] is not None]
        kf_err = np.hypot(*(run.est[:upto, :2] - sim.truth[:upto, :2]).T)
        gps_rmse = float(np.sqrt(np.mean(np.square(gps_err)))) if gps_err else float("nan")
        kf_rmse = float(np.sqrt(np.nanmean(kf_err ** 2)))
        names = {"init": "起點", "accepted": "正常融合", "rejected": "閘門擋下跳點",
                 "reinit": "重新初始化", "missing": "訊號遺失"}
        self.now.setText(
            f"<b>第 {k} 秒</b>・{names.get(status, status)}<br>"
            f"{blue('藍色虛線圈')} = 預測（還沒看 GPS）；{red('紅點')} = GPS；{green('綠色圈')} = 融合後的答案"
            f"（圈 = 95% 可能在這裡）。<br>"
            f"到目前為止的位置誤差：直接用 GPS {red(f'{gps_rmse:.1f} m')} → 卡爾曼 {green(f'{kf_rmse:.1f} m')}"
            f"　{gray('（灰點 = 真實位置）')}")
        self.canvas.update()


# =========================================================
#  第 6 章　把直覺翻成公式
# =========================================================
FORMULA_HTML = f"""
<p style='font-size:14px'>前五章看到的每一個動作，在程式裡都只是一行公式。左邊是你已經懂的直覺，
中間是一維寫法（第 1～3 章），右邊是 <code>kalman_2d.py</code> 用的矩陣寫法 —— <b>結構完全一樣</b>，
只是數字變成矩陣。</p>
<table border='1' cellspacing='0' cellpadding='8' style='border-color:#CBD5E1; font-size:14px'>
<tr style='background:#F1F5F9'><th>你看到的</th><th>一維寫法</th><th>矩陣寫法</th><th>程式</th></tr>
<tr><td>{blue('預測')}：照模型往前推</td><td>x⁻ = x + v·dt</td><td>x⁻ = F x</td><td>STEP 2、7</td></tr>
<tr><td>{blue('預測變胖')}：模型不完美</td><td>P⁻ = P + Q</td><td>P⁻ = F P Fᵀ + Q</td><td>STEP 4、7</td></tr>
<tr><td>{red('量測')}跟預測差多少</td><td>y = z − x⁻</td><td>y = z − H x⁻</td><td>STEP 3、8</td></tr>
<tr><td>兩個不確定加起來</td><td>S = P⁻ + R</td><td>S = H P⁻ Hᵀ + R</td><td>STEP 5、8</td></tr>
<tr><td>權重：誰比較準（第 1 章）</td><td>K = P⁻ ÷ (P⁻ + R)</td><td>K = P⁻ Hᵀ S⁻¹</td><td>STEP 10</td></tr>
<tr><td>{green('融合')}</td><td>x = x⁻ + K·y</td><td>x = x⁻ + K y</td><td>STEP 11</td></tr>
<tr><td>{green('融合變瘦')}</td><td>P = (1 − K)·P⁻</td><td>P = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ</td><td>STEP 11</td></tr>
<tr><td>閘門：太離譜就丟（第 5 章）</td><td>|y| &gt; 3√S 就丟</td><td>yᵀ S⁻¹ y &gt; 9.21 就丟</td><td>STEP 9</td></tr>
<tr><td>遺失：只預測（第 5 章）</td><td colspan='2'>沒有 z → 跳過 y、K、融合，只做預測</td><td>STEP 12</td></tr>
</table>
<p style='font-size:14px'><b>四個矩陣各是什麼？</b></p>
<table cellspacing='0' cellpadding='6' style='font-size:14px'>
<tr><td><b>F</b></td><td>運動模型：「位置 += 速度 × dt」。第 4 章把橢圓<b>斜切</b>的就是它。</td></tr>
<tr><td><b>H</b></td><td>量測模型：「GPS 只量位置、不量速度」→ 從狀態裡挑出位置那兩格。</td></tr>
<tr><td><b>Q</b></td><td>模型每秒可能錯多少（第 3 章左邊的滑桿）。工程上從「目標最猛會怎麼加速 σ<sub>a</sub>」推出來。</td></tr>
<tr><td><b>R</b></td><td>感測器有多吵（第 3 章右邊的滑桿）。直接看 GPS 規格書（例如 ± 5 m）。</td></tr>
</table>
<p style='font-size:14px'><b>工程版比課本多做的三件事</b>（都在 <code>kalman_2d.py</code>）：
① <b>閘門</b>：先用 NIS 檢查量測合不合理（STEP 9）；
② <b>Joseph 形式</b>更新 P：數值上永遠保持對稱、正定，跑幾萬步也不會壞掉（STEP 11）；
③ <b>健康監控</b>：長期平均 NIS 應該 ≈ 2，太大 = Q 或 R 設太小、太小 = 設太大（STEP 13）。</p>
<p style='font-size:14px;color:#475569'>下一步：打開 <code>kalman_2d.py</code>，從 STEP 1 往下讀，
每個 STEP 旁邊都對得到這張表的一列。</p>
"""


class Ch6Formulas(Chapter):
    NUM = 6
    TITLE = "把直覺翻成公式"
    IDEA = "公式沒有新東西：每一行都是前五章已經看過的一個動作。對照著看，矩陣就不可怕了。"
    TRIES = ()

    def build(self):
        lab = QLabel(FORMULA_HTML)
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setStyleSheet("QLabel { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px;"
                          " padding:14px; color:#0F172A; }")
        self.body.addWidget(lab)
        self.now.setText("<b>三句話總結</b><br>"
                         f"① {blue('預測')}讓不確定變大，{red('量測')}帶來新資訊，"
                         f"{green('融合')}照「誰比較準」加權平均，讓不確定變小。<br>"
                         "② Q、R 決定要相信模型還是感測器；它們的比例決定 K。<br>"
                         "③ 預測讓位置和速度產生關聯，所以只量位置也能估出速度；"
                         "工程版再加上閘門與遺失處理，就能上路。")


# =========================================================
#  主視窗：左邊章節目錄、右邊一次只顯示一章
# =========================================================
CHAPTER_SHORT = ["兩個資訊怎麼合併", "預測 → 量測 → 融合", "Q 與 R 的取捨",
                 "量不到的速度", "工程版 2D 追蹤", "翻成公式"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("卡爾曼濾波器：六堂互動課")
        self.chapters = [Ch1Fusion(), Ch2Cycle(), Ch3Tuning(), Ch4Hidden(), Ch5Engineering(), Ch6Formulas()]
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
        t = QLabel("卡爾曼濾波器")
        t.setStyleSheet("QLabel { font-size:20px; font-weight:800; color:#0F172A; }")
        v.addWidget(t)
        st = QLabel("六堂互動課・一章一個觀念")
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
            f"<span style='color:{HEX_PRED}'>■</span> 藍　預測（模型說的）<br>"
            f"<span style='color:{HEX_MEAS}'>■</span> 紅　量測（感測器說的）<br>"
            f"<span style='color:{HEX_FUSE}'>■</span> 綠　融合（卡爾曼的答案）<br>"
            f"<span style='color:{HEX_TRUE}'>■</span> 灰　真值（只有教學看得到）")
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
QCheckBox { font-size:13px; color:#0F172A; spacing:6px; }
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
