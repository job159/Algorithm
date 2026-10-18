# -*- coding: utf-8 -*-
"""
上採樣 / 重取樣 工程技巧動畫（解說版）
=========================================
這支程式用 PyQt6 把「取樣率轉換」最常用的幾個工程技巧一步一步畫出來，
並且每走一步都用白話告訴你「這一步算了什麼、為什麼要這樣算」，數字直接代進去給你看。

四個分頁：
  A. 1D 上採樣     ZOH / 線性 / 三次 / 插零+FIR：每個新點用了哪幾個原始點、權重多少
  B. 降採樣與混疊   先濾波再取 vs 直接取：高頻怎麼「折」成假的低頻（附頻譜）
  C. 2D 影像放大   最近鄰 / 雙線性 / 雙三次：像素中心對齊的座標映射與鄰居權重表
  D. 時間戳對齊     抖動的 IMU + 會掉包的 GPS → 共同時間軸；斷線處填 NaN、絕不硬插

核心觀念：
  1. 取樣定理：訊號頻寬 < fs/2 時，樣本之間的值早就被決定了 —— 上採樣是「算回來」，不是捏造。
  2. 上採樣 = 插零 + 低通 + ×L；線性 / 三次插值只是比較便宜、比較不準的低通。
  3. 降採樣一定要先低通，否則 f_alias = |f − k·fs_new| 的假頻會混進來，事後再也分不開。
  4. 2D 縮放用像素中心對齊 src = (dst + 0.5)/scale − 0.5；不等間距資料空洞超過 max_gap 就給 NaN。

操作：
  每個分頁都有 ⏮ 上一步 / ⏭ 單步 / ▶ 自動播放（⏸ 暫停）/ ↺ 從頭重來 / ⏩ 跳到下一個 STEP + 速度滑桿。
  「上一步」= 從頭確定性重播 n−1 步（資料都用固定的亂數 seed，每次都一模一樣）。
  改參數（L、方法、M、抗混疊、取樣率）會自動從頭開始。
  分頁 C 可以用滑鼠點右邊放大圖的任一像素，看那一格是怎麼算出來的。

計算全部交給 resample.py（只依賴 numpy）；這支程式只負責畫圖與解說。

需求： pip install PyQt6 numpy
執行： python upsampling_pyqt6_step.py
"""
from __future__ import annotations

import math
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen,
                         QPolygonF)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QScrollArea, QSlider, QTabWidget,
                             QVBoxLayout, QWidget)

import resample as rs

# ---------- 外觀（跟 Dijkstra 動畫同一套配色） ----------
FONT_UI = "Microsoft JhengHei"
FONT_NUM = "Arial"
CANVAS_W = 1040                     # 每個分頁的繪圖寬度

C_BG       = QColor("#F8FAFC")      # 視窗底色
C_PANEL    = QColor("#FFFFFF")      # 圖表面板
C_GRID     = QColor("#E2E8F0")      # 格線 / 面板邊框
C_TEXT     = QColor("#0F172A")
C_MUTED    = QColor("#64748B")
C_AXIS     = QColor("#94A3B8")
C_BLUE     = QColor("#2563EB")      # 原始樣本
C_BLUE_LT  = QColor("#BFDBFE")
C_AMBER    = QColor("#F59E0B")      # 這一步正在處理的東西
C_AMBER_LT = QColor("#FCD34D")
C_AMBER_DK = QColor("#B45309")
C_GREEN    = QColor("#22C55E")      # 算出來的新點 / 結果
C_GREEN_DK = QColor("#15803D")
C_RED      = QColor("#EF4444")      # 插零、假頻、錯誤示範
C_TRUTH    = QColor("#CBD5E1")      # 背後的「真值」連續曲線（淡色）
C_NAN      = QColor(100, 116, 139, 55)   # NaN / 不插值 區塊

AL_CENTER = Qt.AlignmentFlag.AlignCenter
AL_LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
AL_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


def qfont(size: float, bold: bool = False, family: str = FONT_UI) -> QFont:
    f = QFont(family)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def with_alpha(c: QColor, a: int) -> QColor:
    q = QColor(c)
    q.setAlpha(a)
    return q


def dashed(color: QColor, width: float = 1.5, style=Qt.PenStyle.DashLine) -> QPen:
    pen = QPen(color, width)
    pen.setStyle(style)
    return pen


def fmt_signed(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}"


def fmt_amp(v: float) -> str:
    """振幅：一般顯示 4 位小數，太小就用科學記號（避免印出 0.0000）。"""
    return f"{v:.4f}" if abs(v) >= 1e-3 else f"{v:.1e}"


def signed_sum(terms, nd: int = 4) -> str:
    """[(0.13, 'x[1]'), (−0.05, 'x[2]')] → '0.1300·x[1] − 0.0500·x[2]'（不會出現「+ −」）。"""
    out = ""
    for i, (w, name) in enumerate(terms):
        mag = f"{abs(w):.{nd}f}·{name}"
        if i == 0:
            out = ("−" if w < 0 else "") + mag
        else:
            out += (" − " if w < 0 else " + ") + mag
    return out


def draw_panel(p: QPainter, rect: QRectF, title: str = "", right: str = "") -> None:
    """白色圓角面板 + 左上標題 + 右上（目前 STEP）。"""
    p.setPen(QPen(C_GRID, 1))
    p.setBrush(QBrush(C_PANEL))
    p.drawRoundedRect(rect, 10, 10)
    if title:
        p.setPen(QPen(C_TEXT))
        p.setFont(qfont(10, True))
        p.drawText(QRectF(rect.left() + 12, rect.top() + 6, rect.width() - 24, 22), AL_LEFT, title)
    if right:
        p.setPen(QPen(C_AMBER_DK))
        p.setFont(qfont(9.5, True))
        p.drawText(QRectF(rect.left() + 12, rect.top() + 6, rect.width() - 24, 22), AL_RIGHT, right)


def label_box(p: QPainter, center: QPointF, text: str, color: QColor, size: float = 8.0,
              bold: bool = True, family: str = FONT_NUM, bg: Optional[QColor] = None) -> QRectF:
    """在圖上放一個白底小標籤（避免文字跟線條糊在一起）。"""
    f = qfont(size, bold, family)
    fm = QFontMetricsF(f)
    w = fm.horizontalAdvance(text) + 8
    h = fm.height() + 2
    r = QRectF(center.x() - w / 2, center.y() - h / 2, w, h)
    p.setPen(QPen(color, 1))
    p.setBrush(QBrush(bg if bg is not None else QColor(255, 255, 255, 235)))
    p.drawRoundedRect(r, 4, 4)
    p.setFont(f)
    p.setPen(QPen(color))
    p.drawText(r, AL_CENTER, text)
    return r


def draw_note(p: QPainter, top_left: QPointF, text: str, color: QColor) -> None:
    """圖表左上角的小註解：先鋪一層半透明白底，避免被曲線穿過去看不清楚。"""
    f = qfont(8.5, True)
    fm = QFontMetricsF(f)
    r = QRectF(top_left.x(), top_left.y(), fm.horizontalAdvance(text) + 10, fm.height() + 2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(255, 255, 255, 215)))
    p.drawRoundedRect(r, 4, 4)
    p.setFont(f)
    p.setPen(QPen(color))
    p.drawText(r, AL_CENTER, text)


def draw_dot(p: QPainter, pt: QPointF, r: float, fill: Optional[QColor],
             edge: Optional[QColor] = None, width: float = 1.3) -> None:
    p.setPen(QPen(edge, width) if edge is not None else Qt.PenStyle.NoPen)
    p.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)
    p.drawEllipse(pt, r, r)


def draw_arrow(p: QPainter, a: QPointF, b: QPointF, color: QColor, width: float = 2.0,
               bend: float = 0.0) -> None:
    """從 a 畫到 b 的箭頭；bend ≠ 0 時畫成弧線（用來表示「折回」）。"""
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath(a)
    if bend:
        mid = QPointF((a.x() + b.x()) / 2, min(a.y(), b.y()) - bend)
        path.quadTo(mid, b)
        d = QPointF(b.x() - mid.x(), b.y() - mid.y())
    else:
        path.lineTo(b)
        d = QPointF(b.x() - a.x(), b.y() - a.y())
    p.drawPath(path)
    ln = math.hypot(d.x(), d.y()) or 1.0
    ux, uy = d.x() / ln, d.y() / ln
    h = 9.0
    for s in (1, -1):
        p.drawLine(b, QPointF(b.x() - h * ux + s * 0.55 * h * uy, b.y() - h * uy - s * 0.55 * h * ux))


class Axes:
    """資料座標 ↔ 像素座標 的小幫手（自己用 QPainter 畫圖，不靠 matplotlib）。"""

    def __init__(self, rect: QRectF, xr: Tuple[float, float], yr: Tuple[float, float]):
        self.rect = rect
        self.x0, self.x1 = xr
        self.y0, self.y1 = yr

    def X(self, x: float) -> float:
        return self.rect.left() + (x - self.x0) / (self.x1 - self.x0) * self.rect.width()

    def Y(self, y: float) -> float:
        return self.rect.bottom() - (y - self.y0) / (self.y1 - self.y0) * self.rect.height()

    def P(self, x: float, y: float) -> QPointF:
        return QPointF(self.X(x), self.Y(y))

    def draw_grid(self, p: QPainter, xticks: Sequence[float], yticks: Sequence[float],
                  xfmt: Callable = str, yfmt: Callable = str, xlabel: str = "",
                  ylabel: str = "", tick_font: float = 7.5) -> None:
        r = self.rect
        p.setFont(qfont(tick_font, family=FONT_NUM))
        for xv in xticks:
            x = self.X(xv)
            p.setPen(QPen(C_GRID, 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 14), AL_CENTER, xfmt(xv))
        for yv in yticks:
            y = self.Y(yv)
            p.setPen(QPen(C_GRID, 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(r.left() - 46, y - 7, 42, 14), AL_RIGHT, yfmt(yv))
        if self.y0 < 0 < self.y1:
            p.setPen(QPen(C_AXIS, 1.2))
            p.drawLine(QPointF(r.left(), self.Y(0)), QPointF(r.right(), self.Y(0)))
        p.setPen(QPen(C_GRID, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        p.setFont(qfont(8))
        p.setPen(QPen(C_MUTED))
        if xlabel:
            p.drawText(QRectF(r.left(), r.bottom() + 17, r.width(), 16), AL_RIGHT, xlabel)
        if ylabel:
            p.drawText(QRectF(r.left() + 4, r.top() + 2, r.width(), 16), AL_LEFT, ylabel)

    def clip(self, p: QPainter) -> None:
        p.setClipRect(self.rect.adjusted(-2, -2, 2, 2))

    def polyline(self, p: QPainter, xs, ys, pen: QPen) -> None:
        """畫折線；遇到 NaN 就斷開（NaN = 沒資料，不該被連起來）。"""
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        seg: List[QPointF] = []
        for x, y in zip(xs, ys):
            if np.isfinite(x) and np.isfinite(y):
                seg.append(self.P(float(x), float(y)))
            else:
                if len(seg) > 1:
                    p.drawPolyline(QPolygonF(seg))
                seg = []
        if len(seg) > 1:
            p.drawPolyline(QPolygonF(seg))


def draw_stem(p: QPainter, ax: Axes, x: float, y: float, color: QColor, r: float = 4.0,
              w: float = 2.0, hollow: bool = False) -> None:
    """棒狀圖（stem）：取樣點 = 一根從 0 長到樣本值的棒子 + 頂端圓點。"""
    p.setPen(QPen(color, w))
    p.drawLine(ax.P(x, 0), ax.P(x, y))
    draw_dot(p, ax.P(x, y), r, C_PANEL if hollow else color, color, 1.5)


# =========================================================
#  共用：逐步動畫分頁的 base class
# =========================================================
class StepTab(QWidget):
    """
    每個分頁都長一樣：工具列 / 參數列 / 標題 / 畫布 / 🔑 提示 / 統計列 / 圖例 / 解說框。
    子類別只要提供 engine（reset / step / done / stage / explain）與畫布。
    「上一步」= engine.reset() 後確定性重播 n−1 步（跟 Dijkstra 動畫同一招）。
    """
    TITLE = ""
    HINT = ""

    def __init__(self):
        super().__init__()
        self.frame = 0                       # 已前進的步數（供上一步倒退）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.engine = self.make_engine()
        self.canvases: List[QWidget] = []
        self._build_ui()
        self.refresh()

    # ---------- 子類別實作 ----------
    def make_engine(self):
        raise NotImplementedError

    def build_params(self) -> Optional[QHBoxLayout]:
        return None

    def build_canvases(self) -> List[QWidget]:
        return []

    def legend_items(self) -> List[Tuple[QColor, str]]:
        return []

    def stats_text(self) -> str:
        return ""

    # ---------- 介面組裝 ----------
    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())
        params = self.build_params()
        if params is not None:
            root.addLayout(params)

        title = QLabel(self.TITLE)
        title.setStyleSheet("font-size:17px;font-weight:bold;color:#1D4ED8;")
        root.addWidget(title)

        self.canvases = self.build_canvases()
        for w in self.canvases:
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(w)
            row.addStretch(1)
            root.addLayout(row)

        self.explain = QLabel()
        self.explain.setWordWrap(True)
        self.explain.setTextFormat(Qt.TextFormat.PlainText)
        self.explain.setMinimumHeight(72)
        self.explain.setStyleSheet(
            "font-size:13px;color:#0F172A;background:#FFFFFF;"
            "border:1px solid #E2E8F0;border-left:4px solid #2563EB;"
            "border-radius:6px;padding:8px;")

        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:13px;color:#0F172A;padding:2px;")
        root.addWidget(self.stats)
        root.addWidget(self._build_legend(self.legend_items()))
        root.addWidget(self.explain)

        if self.HINT:
            hint = QLabel(self.HINT)
            hint.setWordWrap(True)
            hint.setStyleSheet(
                "font-size:12px;color:#1E3A8A;background:#EFF6FF;"
                "border:1px solid #BFDBFE;border-radius:6px;padding:7px;")
            root.addWidget(hint)
        root.addStretch(1)

        container = QWidget()
        container.setObjectName("root")
        container.setLayout(root)
        # 只設定 container 自己的底色（加 #root 選擇器），不要讓樣式往下蓋掉按鈕的 :checked 顏色
        container.setStyleSheet("QWidget#root { background:#F8FAFC; }")

        scroll = QScrollArea()                # 包進可捲動區：小螢幕也不會被裁切
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)
        btn_back = QPushButton("⏮ 上一步")
        btn_back.clicked.connect(self.step_back)
        bar.addWidget(btn_back)
        btn_step = QPushButton("⏭ 單步（下一步）")
        btn_step.clicked.connect(self.step_once)
        bar.addWidget(btn_step)
        self.btn_play = QPushButton("▶ 自動播放")
        self.btn_play.clicked.connect(self.toggle_play)
        bar.addWidget(self.btn_play)
        btn_reset = QPushButton("↺ 從頭重來")
        btn_reset.clicked.connect(self.reset_all)
        bar.addWidget(btn_reset)
        btn_skip = QPushButton("⏩ 跳到下一個 STEP")
        btn_skip.clicked.connect(self.skip_stage)
        bar.addWidget(btn_skip)
        bar.addStretch(1)
        bar.addWidget(QLabel("速度（慢→快）："))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(45)
        self.speed.setFixedWidth(150)
        self.speed.valueChanged.connect(self.update_speed)
        bar.addWidget(self.speed)
        return bar

    @staticmethod
    def _build_legend(items) -> QFrame:
        frame = QFrame()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 4, 0)
        for color, text in items:
            sw = QLabel()
            sw.setFixedSize(16, 16)
            sw.setStyleSheet(f"background:{color.name()};border:1px solid #94A3B8;border-radius:4px;")
            lay.addWidget(sw)
            lab = QLabel(text)
            lab.setStyleSheet("font-size:12px;color:#334155;")
            lay.addWidget(lab)
            lay.addSpacing(8)
        lay.addStretch(1)
        return frame

    @staticmethod
    def make_choice_group(parent: QWidget, choices, current, on_pick) -> Tuple[QButtonGroup, dict]:
        """一排可切換的按鈕（選中的變藍），跟 Dijkstra 的地形按鈕一樣。回傳 {key: 按鈕}。"""
        group = QButtonGroup(parent)
        btns = {}
        for key, text in choices:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(key == current)
            b.clicked.connect(lambda _=False, k=key: on_pick(k))
            group.addButton(b)
            btns[key] = b
        return group, btns

    @staticmethod
    def sync_choice(btns: dict, key) -> None:
        """用程式改參數時，讓按鈕的選取狀態跟著同步。"""
        if key in btns and not btns[key].isChecked():
            btns[key].setChecked(True)

    # ---------- 播放 / 速度 ----------
    def stop_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.btn_play.setText("▶ 自動播放")

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.stop_play()
        else:
            if self.engine.done:
                self.reset_all()
            self.update_speed()
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def update_speed(self) -> None:
        self.timer.setInterval(int(620 - 6 * self.speed.value()))

    # ---------- 前進 / 倒退 ----------
    def advance_one(self) -> bool:
        if self.engine.done:
            return False
        self.engine.step()
        self.frame += 1
        return True

    def tick(self) -> None:
        if not self.advance_one():
            self.timer.stop()
            self.btn_play.setText("▶ 重新播放")
        self.refresh()

    def step_once(self) -> None:
        self.stop_play()
        self.advance_one()
        self.refresh()

    def step_back(self) -> None:
        self.stop_play()
        if self.frame > 0:
            self.rebuild_to(self.frame - 1)

    def rebuild_to(self, n: int) -> None:
        """倒退做法：整個重設後，再快轉前進 n 步（資料固定 seed，可完全重現）。"""
        self.engine.reset()
        self.frame = 0
        for _ in range(n):
            if not self.advance_one():
                break
        self.refresh()

    def skip_stage(self) -> None:
        """一直前進，直到進入下一個 STEP（逐點的階段點太多時很好用）。"""
        self.stop_play()
        s0 = self.engine.stage
        for _ in range(100000):
            if not self.advance_one() or self.engine.stage != s0:
                break
        self.refresh()

    def reset_all(self) -> None:
        self.stop_play()
        self.frame = 0
        self.engine.reset()
        self.refresh()

    def params_changed(self) -> None:
        self.reset_all()

    # ---------- 更新顯示 ----------
    def refresh(self) -> None:
        for c in self.canvases:
            c.update()
        self.stats.setText(f"第 {self.frame} 步　|　" + self.stats_text())
        self.explain.setText("為何這一步：" + self.engine.explain)


# =========================================================
#  分頁 A：1D 上採樣
# =========================================================
METHODS_1D = [("zoh", "ZOH 零階保持"), ("linear", "線性"), ("cubic", "三次 Catmull-Rom"),
              ("fir", "插零+FIR")]
METHOD_NAME_1D = dict(METHODS_1D)


class UpsampleEngine:
    """
    畫面上 N 個原始樣本；兩側各多藏 PAD 個樣本（模擬串流資料「畫面外還有資料」），
    讓三次插值與 FIR 在畫面邊緣也有真鄰居可用，不會被邊界效應干擾比較。
    """
    N, PAD, K_HALF = 12, 8, 6        # FIR 核心半寬 = K_HALF 個原始樣本

    STAGE_TITLE = {"S1": "STEP 1 原始取樣", "S2": "STEP 2 建立新時間軸", "S3": "STEP 3 逐點產生新點",
                   "S3a": "STEP 3a 插零", "S3b": "STEP 3b 低通濾波", "S3c": "STEP 3c ×L 補增益",
                   "S4": "STEP 4 跟真值比較"}

    def __init__(self):
        self.L = 4
        self.method = "linear"
        self.reset()

    @staticmethod
    def truth(t):
        """帶限的「真正連續訊號」：三個正弦，最高 0.29 週期/樣本 < Nyquist 0.5。"""
        t = np.asarray(t, dtype=float)
        return (0.55 * np.sin(2 * np.pi * 0.07 * t + 0.4) + 0.35 * np.sin(2 * np.pi * 0.17 * t + 1.3)
                + 0.22 * np.sin(2 * np.pi * 0.29 * t + 2.2))

    def reset(self) -> None:
        N, L, PAD = self.N, self.L, self.PAD
        self.x_full = self.truth(np.arange(-PAD, N + PAD))
        self.x = self.x_full[PAD:PAD + N]
        self.J = (N - 1) * L + 1                         # 畫面範圍內的新軸點數
        self.t_vis = np.arange(self.J) / L
        sl = slice(PAD * L, PAD * L + self.J)
        self.h = rs.design_lowpass_fir(2 * self.K_HALF * L + 1, 1.0 / L)
        self.D = (len(self.h) - 1) // 2
        self.results = {
            "zoh": rs.upsample_zoh(self.x_full, L)[sl],
            "linear": rs.upsample_linear(self.x_full, L)[sl],
            "cubic": rs.upsample_cubic(self.x_full, L)[sl],
            "fir": rs.upsample_fir(self.x_full, L, h=self.h)[sl],
        }
        self.y_true = self.truth(self.t_vis)
        self.rmse = {k: float(np.sqrt(np.mean((v - self.y_true) ** 2))) for k, v in self.results.items()}
        self.y = self.results[self.method]
        self.new_js = [j for j in range(self.J) if j % L]
        self.stage = "S1"
        self.k = 0
        self.cur_j: Optional[int] = None
        self.taps: List[Tuple[int, float, float]] = []      # (原始樣本編號 n, 權重, x[n])
        self.done = False
        self.explain = (
            f"STEP 1｜原始取樣：畫面上有 {N} 個原始樣本 x[0]…x[{N - 1}]（藍色棒子），背後淡灰色曲線是"
            f"「真正的連續訊號」。它由 3 個正弦組成，最高頻率 0.29 週期/樣本，低於 Nyquist 0.5 → 依取樣定理，"
            f"樣本之間的值其實「早就被決定了」，上採樣就是想辦法把它們算回來。"
            f"（兩側淡色空心棒 = 畫面外還有資料。）按「單步」開始。")

    # ---------- 給畫布用的小工具 ----------
    def stage_title(self) -> str:
        return self.STAGE_TITLE[self.stage]

    def local_curve(self, ts):
        """目前方法在某段區間的插值曲線（畫出「這個點是在哪條線上」）。"""
        pos = np.asarray(ts) + self.PAD
        _, idx, w = rs.interp_taps(pos, len(self.x_full), self.method)
        return (self.x_full[idx] * w).sum(axis=1)

    # ---------- 前進一步 ----------
    def step(self) -> None:
        if self.done:
            return
        s, L = self.stage, self.L
        if s == "S1":
            self.stage = "S2"
            n_new = len(self.new_js)
            extra = ("插零+FIR 法不直接猜值：下一步先把新點全部填 0，再交給濾波器處理。"
                     if self.method == "fir" else "下一步開始一個一個把空心圈填上值。")
            self.explain = (
                f"STEP 2｜建立新時間軸：上採樣 L = {L} 倍 → 新的取樣間隔 Δt = 1/{L} = {1 / L:.3f}。"
                f"畫面範圍 t = 0~{self.N - 1} 內新軸共 (N−1)·L+1 = {self.J} 點，其中 {self.N} 點剛好跟原始點重合"
                f"（直接沿用），要新生 {n_new} 點（橫軸上的空心圈）。{extra}"
                f"（函式庫回傳 N·L 點，最後 L−1 點落在 x[{self.N - 1}] 之後，這裡不畫。）")
        elif s == "S2":
            if self.method == "fir":
                self.stage = "S3a"
                self.explain = (
                    f"STEP 3a｜插零：在每兩個原始點之間塞 {L - 1} 個 0（紅點）。時域看起來像一排尖刺；"
                    f"頻域上原本 0~0.5 的頻譜其實沒變，但在新的取樣率下，0.5~{L / 2:g}（舊取樣率單位）之間"
                    f"多出了 {L - 1} 份「鏡像」。下一步用低通濾波器把鏡像砍掉 —— 也就是把這些 0「填起來」。")
            else:
                self.stage = "S3"
                self.k = 1
                self._interp_point()
        elif s == "S3":
            if self.k < len(self.new_js):
                self.k += 1
                self._interp_point()
            else:
                self._finish()
        elif s == "S3a":
            self.stage = "S3b"
            self.k = 1
            self._fir_point()
        elif s == "S3b":
            if self.k < self.J:
                self.k += 1
                self._fir_point()
            else:
                self.stage = "S3c"
                self.cur_j = None
                self.taps = []
                j = int(np.argmax(np.abs(self.y)))
                v = self.y[j] / L
                self.explain = (
                    f"STEP 3c｜×L 補增益：插零後每 {L} 點只有 1 點有值，平均能量被稀釋成 1/{L}；濾波器的直流增益又是 1，"
                    f"所以濾完的綠點只有正確高度的 1/{L}（淡綠點）。全部乘上 L = {L}：例如 t = {self.t_vis[j]:.3f} 處 "
                    f"{fmt_signed(v, 4)} × {L} = {fmt_signed(self.y[j], 4)}（真值 {fmt_signed(self.y_true[j], 4)}）。"
                    f"現在整條曲線回到正確高度。")
        elif s == "S3c":
            self._finish()

    def _interp_point(self) -> None:
        j = self.new_js[self.k - 1]
        self.cur_j = j
        t = self.t_vis[j]
        raw, idx, w = rs.interp_taps(t + self.PAD, len(self.x_full), self.method)
        self.taps = [(int(r) - self.PAD, float(ww), float(self.x_full[i]))
                     for r, i, ww in zip(raw[0], idx[0], w[0])]
        y = sum(ww * xv for _, ww, xv in self.taps)
        assert abs(y - self.y[j]) < 1e-9          # 解說的數字 = 函式庫算出來的值
        yt = self.y_true[j]
        tail = f"（真值 {fmt_signed(yt, 4)}，誤差 {fmt_signed(y - yt, 4)}）"
        head = f"STEP 3｜逐點產生（第 {self.k}/{len(self.new_js)} 個新點）：t = {t:.3f}"
        i0 = int(math.floor(t))
        f = t - i0
        if self.method == "zoh":
            self.explain = (f"{head} → 零階保持直接抄左邊的 x[{i0}]：y = 1 × x[{i0}] = {fmt_signed(y, 4)}{tail}。"
                            f"不管 t 落在區間的哪裡答案都一樣 → 波形變成階梯，而且整體慢了半格。")
        elif self.method == "linear":
            (n0, w0, x0), (n1, w1, x1) = self.taps
            self.explain = (
                f"{head} 位於 x[{i0}] 與 x[{i0 + 1}] 之間 {f * 100:.1f}% 處 → 權重 (1−{f:.3f}, {f:.3f})。"
                f"y = {w0:.3f}·x[{i0}] + {w1:.3f}·x[{i0 + 1}] = {w0:.3f}×({fmt_signed(x0)}) + "
                f"{w1:.3f}×({fmt_signed(x1)}) = {fmt_signed(y, 4)}{tail}。離誰近、誰的權重就大；"
                f"但兩點之間只能走直線，波峰會被削平。")
        else:
            ns = [n for n, _, _ in self.taps]
            dists = [abs(t - n) for n in ns]
            terms = " + ".join(f"({ww:+.4f})×({fmt_signed(xv)})" for _, ww, xv in self.taps)
            self.explain = (
                f"{head}，i0 = {i0}、f = {f:.3f}。取 x[{ns[0]}]~x[{ns[-1]}] 四點，到 t 的距離 "
                f"{' / '.join(f'{d:.3f}' for d in dists)} → 權重 W(距離) = "
                f"{', '.join(f'{ww:+.4f}' for _, ww, _ in self.taps)}（總和 1）。y = {terms} = "
                f"{fmt_signed(y, 4)}{tail}。最外側兩點權重是負的 → 曲線能「預判轉彎」，比直線更貼近真值。")

    def _fir_point(self) -> None:
        L, D, h = self.L, self.D, self.h
        j = self.k - 1
        self.cur_j = j
        n = self.PAD * L + j                          # 在「插零後序列」上的位置
        k_lo = -((-(n + D - len(h) + 1)) // L)       # ceil
        k_hi = (n + D) // L
        self.taps = [(kk - self.PAD, float(h[n + D - kk * L]), float(self.x_full[kk]))
                     for kk in range(k_lo, k_hi + 1)]
        v = sum(ww * xv for _, ww, xv in self.taps)
        assert abs(v * L - self.y[j]) < 1e-9          # 跟 polyphase 函式庫結果一致
        top = sorted(self.taps, key=lambda tp: -abs(tp[1]))[:4]
        top.sort(key=lambda tp: tp[0])
        terms = signed_sum([(ww, f"x[{nn}]") for nn, ww, _ in top])
        rest = v - sum(ww * xv for _, ww, xv in top)
        t = self.t_vis[j]
        on_orig = "（這裡剛好是原始點：核心在其他原始點上幾乎都是 0，所以原值被保留）" if j % L == 0 else ""
        self.explain = (
            f"STEP 3b｜低通濾波（第 {self.k}/{self.J} 點）：把濾波器核心（加窗 sinc、{len(h)} 點、截止 = 舊 Nyquist）"
            f"的中心對準 t = {t:.3f}，跟插零後的序列逐點相乘再相加。核心蓋到的 {len(h)} 個位置只有 {len(self.taps)} 個"
            f"是原始樣本，其餘 {len(h) - len(self.taps)} 個都是 0（乘了白做 → polyphase 直接跳過，省 {L} 倍）。"
            f"主要貢獻：{terms}，其餘 {len(self.taps) - len(top)} 項合計 {fmt_signed(rest, 4)} → v = {fmt_signed(v, 4)}"
            f"{on_orig}。注意 v 只有真值 {fmt_signed(self.y_true[j])} 的大約 1/{L}，STEP 3c 會補回來。")

    def _finish(self) -> None:
        self.stage = "S4"
        self.cur_j = None
        self.taps = []
        self.done = True
        order = sorted(self.rmse.items(), key=lambda kv: kv[1])
        rank = " < ".join(f"{METHOD_NAME_1D[k]} {v:.4f}" for k, v in order)
        err = np.abs(self.y - self.y_true)
        jm = int(np.argmax(err))
        best = order[0][1]
        me = self.rmse[self.method]
        ratio = f"，是最準方法的 {me / best:.0f} 倍" if self.method != order[0][0] else "，四種裡最準"
        self.explain = (
            f"STEP 4｜跟真值比較（畫面範圍 {self.J} 點的 RMSE，下方長條圖）：{rank}。"
            f"目前方法「{METHOD_NAME_1D[self.method]}」RMSE = {me:.4f}{ratio}；最大誤差 {err[jm]:.4f} 出現在 "
            f"t = {self.t_vis[jm]:.3f}。紅色細線 = 每點跟真值的差。結論：ZOH 最差（階梯 + 延遲半格）、線性削峰、"
            f"三次好很多、插零+FIR（加窗 sinc ≈ 理想低通）最接近真值。換個 L 或方法再跑一次比較看看。")


class UpsampleCanvas(QWidget):
    def __init__(self, engine: UpsampleEngine):
        super().__init__()
        self.e = engine
        self.setFixedSize(CANVAS_W, 372)

    def paintEvent(self, _):
        e = self.e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        draw_panel(p, QRectF(1, 1, W - 2, H - 2),
                   f"時域：原始樣本 x[n]（藍）→ 上採樣 ×{e.L}　方法：{METHOD_NAME_1D[e.method]}",
                   e.stage_title())
        ax = Axes(QRectF(52, 36, W - 52 - 20, H - 36 - 42), (-1.6, e.N - 1 + 1.6), (-1.25, 1.25))
        ax.draw_grid(p, list(range(-1, e.N + 1)), [-1, -0.5, 0, 0.5, 1],
                     xfmt=lambda v: f"{v}", yfmt=lambda v: f"{v:g}",
                     xlabel="t（單位：原始取樣間隔；整數 = 原始樣本 n）")
        p.save()
        ax.clip(p)
        tt = np.linspace(ax.x0, ax.x1, 700)
        ax.polyline(p, tt, e.truth(tt), QPen(C_TRUTH, 3))            # 真值（淡色）
        for n in (-1, e.N):                                           # 畫面外的資料
            draw_stem(p, ax, n, e.x_full[n + e.PAD], C_AXIS, r=3.5, w=1.5, hollow=True)

        st, L = e.stage, e.L
        if st == "S2":
            for j in e.new_js:
                t = e.t_vis[j]
                p.setPen(dashed(with_alpha(C_AMBER, 110), 1, Qt.PenStyle.DotLine))
                p.drawLine(ax.P(t, -1.2), ax.P(t, 1.2))
                draw_dot(p, ax.P(t, 0), 4, C_PANEL, C_AMBER, 1.6)

        for i in range(e.N):                                          # 原始樣本
            draw_stem(p, ax, i, e.x[i], C_BLUE)

        if st == "S3":
            for j in e.new_js[e.k:]:
                draw_dot(p, ax.P(e.t_vis[j], 0), 3.2, C_PANEL, with_alpha(C_AMBER, 150), 1.3)
            for j in e.new_js[:e.k - 1]:
                draw_dot(p, ax.P(e.t_vis[j], e.y[j]), 4, C_GREEN, C_GREEN_DK, 1)
            self._draw_current_interp(p, ax)
        elif st == "S3a":
            for j in e.new_js:
                draw_dot(p, ax.P(e.t_vis[j], 0), 4, C_RED, QColor("#991B1B"), 1)
        elif st == "S3b":
            for j in e.new_js:
                draw_dot(p, ax.P(e.t_vis[j], 0), 3.2, with_alpha(C_RED, 120), None)
            self._draw_kernel(p, ax)
            for j in range(e.k - 1):
                draw_dot(p, ax.P(e.t_vis[j], e.y[j] / L), 3.6, C_GREEN, C_GREEN_DK, 1)
            j = e.cur_j
            draw_dot(p, ax.P(e.t_vis[j], e.y[j] / L), 6.5, C_AMBER, C_AMBER_DK, 2)
        elif st == "S3c":
            for j in range(e.J):
                a, b = ax.P(e.t_vis[j], e.y[j] / L), ax.P(e.t_vis[j], e.y[j])
                p.setPen(QPen(with_alpha(C_GREEN, 90), 1))
                p.drawLine(a, b)
                draw_dot(p, a, 2.6, with_alpha(C_GREEN, 110), None)
                draw_dot(p, b, 4, C_GREEN, C_GREEN_DK, 1)
        elif st == "S4":
            for j in range(e.J):
                if abs(e.y[j] - e.y_true[j]) > 0.004:
                    p.setPen(QPen(C_RED, 1.4))
                    p.drawLine(ax.P(e.t_vis[j], e.y[j]), ax.P(e.t_vis[j], e.y_true[j]))
            ax.polyline(p, e.t_vis, e.y, QPen(C_GREEN, 2))
            for j in range(e.J):
                if j % L:
                    draw_dot(p, ax.P(e.t_vis[j], e.y[j]), 3.2, C_GREEN, C_GREEN_DK, 1)
        p.restore()

        # 右上角小註解
        note = {"S2": "空心圈 = 待新生的點", "S3a": "紅點 = 插進去的 0",
                "S3b": "琥珀色 = 濾波器核心（×L 顯示）　綠點 = 濾完（約 1/L 高）",
                "S3c": "淡綠 → 綠：×L", "S4": f"RMSE = {e.rmse[e.method]:.4f}　紅線 = 誤差"}.get(st)
        if note:
            draw_note(p, QPointF(ax.rect.left() + 6, ax.rect.top() + 4), note, C_AMBER_DK)
        p.end()

    def _tap_label(self, p: QPainter, ax: Axes, n: int, w: float, xv: float) -> None:
        dy = 0.2 if xv >= 0 else -0.2
        y = max(-1.15, min(1.15, xv + dy))
        label_box(p, ax.P(n, y), f"×{w:.3f}", C_AMBER_DK, 7.5)

    def _draw_current_interp(self, p: QPainter, ax: Axes) -> None:
        e = self.e
        j = e.cur_j
        t, y = e.t_vis[j], e.y[j]
        p.setPen(dashed(C_AMBER, 1.2))
        p.drawLine(ax.P(t, -1.2), ax.P(t, 1.2))
        i0 = math.floor(t)
        ts = np.linspace(i0, i0 + 1 - 1e-9, 80)
        ax.polyline(p, ts, e.local_curve(ts), dashed(C_AMBER, 2.4))
        for n, w, xv in e.taps:
            draw_dot(p, ax.P(n, xv), 9, None, C_AMBER, 2.4)
        for n, w, xv in e.taps:
            self._tap_label(p, ax, n, w, xv)
        draw_dot(p, ax.P(t, y), 6.5, C_AMBER, C_AMBER_DK, 2)

    def _draw_kernel(self, p: QPainter, ax: Axes) -> None:
        e = self.e
        t = e.t_vis[e.cur_j]
        hk = e.h * e.L
        tk = t + (np.arange(len(e.h)) - e.D) / e.L
        path = QPainterPath(ax.P(tk[0], 0))
        for a, b in zip(tk, hk):
            path.lineTo(ax.P(a, b))
        path.lineTo(ax.P(tk[-1], 0))
        path.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(with_alpha(C_AMBER_LT, 80)))
        p.drawPath(path)
        ax.polyline(p, tk, hk, QPen(C_AMBER, 1.8))
        p.setPen(dashed(C_AMBER, 1.2))
        p.drawLine(ax.P(t, -1.2), ax.P(t, 1.2))
        for n, w, xv in e.taps:                       # 核心碰到原始樣本的位置
            draw_dot(p, ax.P(n, w * e.L), 3, C_AMBER_DK, None)
        top = sorted(e.taps, key=lambda tp: -abs(tp[1]))[:4]
        for n, w, xv in top:
            draw_dot(p, ax.P(n, xv), 9, None, C_AMBER, 2.4)
        for n, w, xv in top:
            self._tap_label(p, ax, n, w, xv)


class RmseBars(QWidget):
    """STEP 4 的四種方法誤差比較（對數刻度長條圖）。"""

    def __init__(self, engine: UpsampleEngine):
        super().__init__()
        self.e = engine
        self.setFixedSize(CANVAS_W, 132)

    def paintEvent(self, _):
        e = self.e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        draw_panel(p, QRectF(1, 1, W - 2, H - 2),
                   f"四種方法對真值的 RMSE（畫面範圍 {e.J} 點；對數刻度，越短越好）")
        if e.stage != "S4":
            p.setFont(qfont(10))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(0, 30, W, H - 34), AL_CENTER,
                       "（走到 STEP 4 才會揭曉。先猜猜看：ZOH、線性、三次、插零+FIR，誰最準？）")
            p.end()
            return
        x0, x1 = 170.0, W - 170.0
        lo, hi = -4.0, 0.0
        y = 34.0
        p.setFont(qfont(7.5, family=FONT_NUM))
        for d in range(int(lo), int(hi) + 1):
            x = x0 + (d - lo) / (hi - lo) * (x1 - x0)
            p.setPen(QPen(C_GRID, 1))
            p.drawLine(QPointF(x, y - 2), QPointF(x, y + 4 * 21))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(x - 24, y + 4 * 21, 48, 12), AL_CENTER, f"1e{d}")
        best = min(e.rmse.values())
        for key, name in METHODS_1D:
            v = e.rmse[key]
            frac = (math.log10(max(v, 1e-12)) - lo) / (hi - lo)
            bw = max(3.0, min(1.0, frac) * (x1 - x0))
            cur = key == e.method
            p.setFont(qfont(9.5, cur))
            p.setPen(QPen(C_TEXT if cur else C_MUTED))
            p.drawText(QRectF(14, y, x0 - 22, 18), AL_RIGHT, name)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(C_BLUE if cur else C_BLUE_LT))
            p.drawRoundedRect(QRectF(x0, y + 2, bw, 15), 4, 4)
            p.setFont(qfont(9, cur, FONT_NUM))
            p.setPen(QPen(C_TEXT))
            txt = f"{v:.4f}" + ("" if v == best else f"（{v / best:.0f}×）")
            p.drawText(QRectF(x0 + bw + 6, y, 160, 18), AL_LEFT, txt)
            y += 21
        p.end()


class UpsampleTab(StepTab):
    TITLE = "1D 上採樣　—　新點從哪裡來？用了哪幾個原始點、各佔多少權重"
    HINT = ("🔑 上採樣的本質：取樣定理保證「帶限訊號的樣本之間早就被決定了」。插值法 = 用一個小核心（ZOH 是方塊、"
            "線性是三角形、三次是 4 點曲線）加權鄰居；插零+FIR 則是把核心換成接近理想低通的「加窗 sinc」。"
            "核心越接近理想低通，結果越接近真值 —— 但也越長、越貴（polyphase 讓它省 L 倍）。")

    def make_engine(self):
        return UpsampleEngine()

    def build_params(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("倍率 L："))
        self.L_group, self.L_btns = self.make_choice_group(self, [(L, f"×{L}") for L in range(2, 9)],
                                                           self.engine.L, self.set_L)
        for b in self.L_btns.values():
            bar.addWidget(b)
        bar.addSpacing(14)
        bar.addWidget(QLabel("方法："))
        self.m_group, self.m_btns = self.make_choice_group(self, METHODS_1D, self.engine.method,
                                                           self.set_method)
        for b in self.m_btns.values():
            bar.addWidget(b)
        bar.addStretch(1)
        return bar

    def set_L(self, L: int) -> None:
        self.engine.L = L
        self.sync_choice(self.L_btns, L)
        self.params_changed()

    def set_method(self, m: str) -> None:
        self.engine.method = m
        self.sync_choice(self.m_btns, m)
        self.params_changed()

    def build_canvases(self):
        return [UpsampleCanvas(self.engine), RmseBars(self.engine)]

    def legend_items(self):
        return [(C_BLUE, "原始樣本"), (C_TRUTH, "真值（連續訊號）"), (C_AMBER, "這一步的新點 / 用到的鄰居"),
                (C_GREEN, "已產生的新點"), (C_RED, "插進去的 0 / 誤差"), (C_AMBER_LT, "FIR 核心")]

    def stats_text(self):
        e = self.engine
        if e.stage == "S3":
            prog = f"新點 {e.k}/{len(e.new_js)}"
        elif e.stage == "S3b":
            prog = f"濾波 {e.k}/{e.J}"
        else:
            prog = "—"
        return (f"{e.stage_title()}　|　L = {e.L}　|　方法：{METHOD_NAME_1D[e.method]}　|　"
                f"原始 {e.N} 點 → 新軸 {e.J} 點（新生 {len(e.new_js)}）　|　進度：{prog}")


# =========================================================
#  分頁 B：降採樣與混疊
# =========================================================
def flattop(n: int) -> np.ndarray:
    """平頂窗：頻譜峰值高度幾乎不受「頻率有沒有剛好落在 FFT 格點上」影響，拿來讀振幅最準。"""
    a = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
    k = 2 * np.pi * np.arange(n) / (n - 1)
    return a[0] - a[1] * np.cos(k) + a[2] * np.cos(2 * k) - a[3] * np.cos(3 * k) + a[4] * np.cos(4 * k)


def amp_spectrum(sig: np.ndarray, fs: float):
    w = flattop(len(sig))
    amp = 2 * np.abs(np.fft.rfft(sig * w)) / w.sum()
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    return f, amp


def spectrum_bars(sig: np.ndarray, fs: float) -> np.ndarray:
    """把精細頻譜收成「每 1 Hz 一根長條」：第 k 根 = [k−0.5, k+0.5) 內的最大振幅。"""
    f, amp = amp_spectrum(sig, fs)
    nb = int(math.floor(fs / 2 + 1e-9)) + 1
    bars = np.zeros(nb)
    idx = np.clip(np.floor(f + 0.5).astype(int), 0, nb - 1)
    np.maximum.at(bars, idx, amp)
    return bars


class AliasEngine:
    FS = 100.0
    F_LO, A_LO = 3.0, 1.0
    F_HI, A_HI, PH_HI = 31.0, 0.6, 0.5
    PRE, REC, VIS = 840, 3360, 100       # 前面暖機樣本（840 可被 1~8 整除）、頻譜用長度、畫面樣本數
    STAGE_TITLE = {"S1": "STEP 1 原始訊號", "S2": "STEP 2 抗混疊低通", "S3": "STEP 3 每 M 取一",
                   "S4": "STEP 4 結果與假頻"}

    def __init__(self):
        self.M = 4
        self.aa = True
        self.reset()

    @classmethod
    def truth(cls, t):
        t = np.asarray(t, dtype=float)
        return (cls.A_LO * np.sin(2 * np.pi * cls.F_LO * t)
                + cls.A_HI * np.sin(2 * np.pi * cls.F_HI * t + cls.PH_HI))

    def stage_title(self) -> str:
        if self.stage == "S2" and not self.aa:
            return "STEP 2（略過濾波）"
        return self.STAGE_TITLE[self.stage]

    def reset(self) -> None:
        M, FS, PRE, REC = self.M, self.FS, self.PRE, self.REC
        n = np.arange(-PRE, REC + PRE)
        self.t = n / FS
        x = self.truth(self.t)
        self.x = x
        self.h = rs.design_lowpass_fir(2 * 10 * M + 1, 1.0 / M)
        self.xf = rs.lowpass_aligned(x, self.h)
        self.dec_aa = rs.decimate(x, M, h=self.h)
        self.dec_naive = rs.decimate_naive(x, M)
        assert np.allclose(self.dec_aa, self.xf[::M])
        self.recon_aa = rs.upsample_fir(self.dec_aa, M)[:len(n)]
        self.recon_naive = rs.upsample_fir(self.dec_naive, M)[:len(n)]
        self.fs_new = FS / M
        self.ny = self.fs_new / 2
        self.k_alias = int(round(self.F_HI / self.fs_new))
        self.f_alias = abs(self.F_HI - self.k_alias * self.fs_new)
        # 頻譜（一律取 n = 0 開始的 REC 個樣本）
        rec = slice(PRE, PRE + REC)
        drec = slice(PRE // M, (PRE + REC) // M)
        self.bars_orig = spectrum_bars(x[rec], FS)
        self.bars_filt = spectrum_bars(self.xf[rec], FS)
        self.bars_dec = {True: spectrum_bars(self.dec_aa[drec], self.fs_new),
                         False: spectrum_bars(self.dec_naive[drec], self.fs_new)}
        self.amp_alias = {}
        self.amp_lo = {}
        for key, d in ((True, self.dec_aa), (False, self.dec_naive)):
            f, a = amp_spectrum(d[drec], self.fs_new)
            self.amp_alias[key] = float(a[np.abs(f - self.f_alias) < 0.15].max())
            self.amp_lo[key] = float(a[np.abs(f - self.F_LO) < 0.15].max())
        # 濾波器頻率響應 |H(f)|
        self.fH = np.linspace(0, FS / 2, 401)
        nn = np.arange(len(self.h)) - (len(self.h) - 1) / 2
        self.H = np.abs(np.exp(-2j * np.pi * np.outer(self.fH, nn) / FS) @ self.h)
        self.g_lo = float(np.interp(self.F_LO, self.fH, self.H))
        self.g_hi = float(np.interp(self.F_HI, self.fH, self.H))
        self.vis = slice(PRE, PRE + self.VIS)
        self.stage = "S1"
        self.done = False
        over = self.F_HI > self.ny
        self.explain = (
            f"STEP 1｜原始訊號：fs = {FS:.0f} Hz，訊號 = {self.A_LO:g}·sin(2π·{self.F_LO:g}t) + "
            f"{self.A_HI:g}·sin(2π·{self.F_HI:g}t)。下方頻譜在 {self.F_LO:g} Hz、{self.F_HI:g} Hz 各有一根，"
            f"都低於原本的 Nyquist {FS / 2:.0f} Hz，目前一切正常。接下來要降採樣 ÷{M} → 新 fs = "
            f"{self.fs_new:.2f} Hz、新 Nyquist = {self.ny:.2f} Hz：{self.F_HI:g} Hz "
            f"{'超過了新 Nyquist —— 麻煩要來了' if over else '沒超過新 Nyquist'}。"
            f"（{'已' if self.aa else '未'}勾選抗混疊濾波。）")

    def step(self) -> None:
        if self.done:
            return
        M = self.M
        if self.stage == "S1":
            self.stage = "S2"
            if self.aa:
                db = 20 * math.log10(max(self.g_hi, 1e-12))
                self.explain = (
                    f"STEP 2｜抗混疊低通：先用 FIR（{len(self.h)} 點、截止 = 新 Nyquist {self.ny:.2f} Hz）把高頻濾掉。"
                    f"濾波器在 {self.F_LO:g} Hz 增益 {self.g_lo:.3f}、在 {self.F_HI:g} Hz 只剩 {self.g_hi:.1e}"
                    f"（{db:.0f} dB）→ 綠色曲線只剩緩慢的 {self.F_LO:g} Hz；下方頻譜 {self.F_HI:g} Hz 那根消失了"
                    f"（琥珀色曲線 = 濾波器的 |H(f)|）。群延遲 {(len(self.h) - 1) // 2} 個樣本已補償，綠線沒有往右偏。")
            else:
                self.explain = (
                    f"STEP 2｜（未勾選抗混疊）直接跳過濾波 —— {self.F_HI:g} Hz 還留在訊號裡。"
                    f"下一步每 {M} 取一時，它沒有地方放，會被「折」進 0~{self.ny:.2f} Hz 之間。"
                    f"（這就是 decimate_naive 做的事。）")
        elif self.stage == "S2":
            self.stage = "S3"
            cnt = len(range(0, self.VIS, M))
            alias = (f"{self.F_HI:g} Hz 落在灰色區，新取樣率表示不了它 → 它會被折回 {self.f_alias:.2f} Hz。"
                     if not self.aa else
                     f"因為先濾過，灰色區裡已經幾乎沒有能量，沒什麼東西會折回來。")
            self.explain = (
                f"STEP 3｜每 {M} 取一：保留 n = 0, {M}, {2 * M}, …（琥珀色），其餘丟掉 → 畫面 1 秒內剩 {cnt} 點，"
                f"新 fs = {self.fs_new:.2f} Hz。下方頻譜改畫降採樣後的結果：新取樣率只表示得了 0~{self.ny:.2f} Hz"
                f"（紅虛線右邊灰色區都看不到）。{alias}")
        elif self.stage == "S3":
            self.stage = "S4"
            self.done = True
            k, fs_new, fa = self.k_alias, self.fs_new, self.f_alias
            formula = (f"f_alias = |f − k·fs_new|，k = round({self.F_HI:g}/{fs_new:.2f}) = {k} → "
                       f"|{self.F_HI:g} − {k}×{fs_new:.2f}| = {fa:.2f} Hz。")
            if self.aa:
                tail = (f"因為先濾波，{fa:.2f} Hz 處只剩 {fmt_amp(self.amp_alias[True])}（不濾波時是 "
                        f"{self.amp_alias[False]:.3f}）；紅線幾乎就是乾淨的 {self.F_LO:g} Hz（綠虛線）。代價："
                        f"{self.F_HI:g} Hz 的資訊永久消失 —— 這是物理限制，低取樣率本來就裝不下。")
            else:
                tail = (f"頻譜在 {fa:.2f} Hz 冒出振幅 {self.amp_alias[False]:.3f} 的假峰，紅線（低取樣率系統以為的訊號）"
                        f"明顯多了一個不存在的 {fa:.2f} Hz 波動。它跟真的 {self.F_LO:g} Hz 混在一起，事後再也分不開。"
                        f"勾選「抗混疊濾波」重跑：假峰會降到 {fmt_amp(self.amp_alias[True])}。")
            self.explain = f"STEP 4｜結果與假頻：紅線 = 把保留的點做帶限重建。{formula}{tail}"


class AliasTimeCanvas(QWidget):
    def __init__(self, engine: AliasEngine):
        super().__init__()
        self.e = engine
        self.setFixedSize(CANVAS_W, 268)

    def paintEvent(self, _):
        e = self.e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        draw_panel(p, QRectF(1, 1, W - 2, H - 2),
                   f"時域（前 1 秒；fs = {e.FS:.0f} Hz → ÷{e.M} = {e.fs_new:.2f} Hz）", e.stage_title())
        ax = Axes(QRectF(52, 34, W - 52 - 20, H - 34 - 40), (-0.012, 1.0), (-1.95, 1.95))
        ax.draw_grid(p, [i / 10 for i in range(11)], [-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
                     xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: f"{v:g}", xlabel="時間（秒）")
        p.save()
        ax.clip(p)
        tt = np.linspace(0, 1, 1200)
        ax.polyline(p, tt, e.truth(tt), QPen(C_TRUTH, 2))
        idx = np.arange(e.vis.start, e.vis.stop)
        tv = e.t[idx]
        st = e.stage
        base = e.xf if (e.aa and st != "S1") else e.x
        if st == "S1" or (st == "S2" and not e.aa):
            for t, v in zip(tv, e.x[idx]):
                draw_stem(p, ax, t, v, C_BLUE, r=2.6, w=1.1)
        elif st == "S2":
            for t, v in zip(tv, e.x[idx]):
                draw_stem(p, ax, t, v, with_alpha(C_AXIS, 110), r=2.2, w=1)
            ax.polyline(p, tv, e.xf[idx], QPen(C_GREEN, 2.2))
            for t, v in zip(tv, e.xf[idx]):
                draw_dot(p, ax.P(t, v), 2.4, C_GREEN, None)
        else:
            if e.aa:
                ax.polyline(p, tv, e.xf[idx], QPen(with_alpha(C_GREEN, 120), 1.6))
            for i, t, v in zip(idx, tv, base[idx]):
                if (i - e.PRE) % e.M:
                    draw_dot(p, ax.P(t, v), 2.2, with_alpha(C_AXIS, 140), None)
            if st == "S4":
                rec = e.recon_aa if e.aa else e.recon_naive
                ax.polyline(p, tv, e.A_LO * np.sin(2 * np.pi * e.F_LO * tv), dashed(C_GREEN_DK, 1.6))
                ax.polyline(p, tv, rec[idx], QPen(C_RED, 2.4))
            for i, t, v in zip(idx, tv, base[idx]):
                if (i - e.PRE) % e.M == 0:
                    draw_stem(p, ax, t, v, C_AMBER, r=4.6, w=2)
        p.restore()
        note = {"S2": ("綠線 = 濾波後" if e.aa else "沒有濾波：31 Hz 仍在"),
                "S3": f"琥珀色 = 保留的點（每 {e.M} 取 1）",
                "S4": "紅線 = 低取樣率系統以為的訊號　綠虛線 = 真正的 3 Hz 成分"}.get(st)
        if note:
            draw_note(p, QPointF(ax.rect.left() + 6, ax.rect.top() + 3), note,
                      C_AMBER_DK if (e.aa or st != "S2") else C_RED)
        p.end()


class AliasSpectrumCanvas(QWidget):
    def __init__(self, engine: AliasEngine):
        super().__init__()
        self.e = engine
        self.setFixedSize(CANVAS_W, 236)

    def _bars(self, p: QPainter, ax: Axes, bars: np.ndarray, color: QColor, half: float = 0.36) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        for k, a in enumerate(bars):
            if a < 0.004:
                continue
            p.drawRect(QRectF(QPointF(ax.X(k - half), ax.Y(min(a, 1.15))), QPointF(ax.X(k + half), ax.Y(0))))

    def _peak(self, p: QPainter, ax: Axes, f: float, a: float, color: QColor) -> None:
        label_box(p, QPointF(ax.X(f), ax.Y(min(a, 1.0)) - 12), f"{f:g} Hz：{a:.2f}", color, 7.5)

    def paintEvent(self, _):
        e = self.e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        st = e.stage
        which = {"S1": "原始訊號", "S2": "濾波後" if e.aa else "原始訊號（沒濾波）",
                 "S3": "降採樣後", "S4": "降採樣後"}[st]
        draw_panel(p, QRectF(1, 1, W - 2, H - 2),
                   f"振幅頻譜（numpy FFT + 平頂窗，每 1 Hz 一根）：{which}")
        ax = Axes(QRectF(52, 34, W - 52 - 20, H - 34 - 40), (-0.8, 50.8), (0, 1.22))
        ax.draw_grid(p, list(range(0, 51, 5)), [0, 0.5, 1.0], xfmt=lambda v: f"{v}",
                     yfmt=lambda v: f"{v:g}", xlabel="頻率（Hz）")
        p.save()
        ax.clip(p)
        if st == "S1" or (st == "S2" and not e.aa):
            self._bars(p, ax, e.bars_orig, C_BLUE)
        elif st == "S2":
            self._bars(p, ax, e.bars_orig, with_alpha(C_AXIS, 90))
            self._bars(p, ax, e.bars_filt, C_GREEN)
            ax.polyline(p, e.fH, e.H, QPen(C_AMBER, 2))
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(C_NAN))
            p.drawRect(QRectF(QPointF(ax.X(e.ny), ax.rect.top()), ax.rect.bottomRight()))
            self._bars(p, ax, e.bars_orig, with_alpha(C_AXIS, 110))
            self._bars(p, ax, e.bars_dec[e.aa], C_AMBER)
        if st != "S1":
            p.setPen(dashed(C_RED, 1.8))
            p.drawLine(ax.P(e.ny, 0), ax.P(e.ny, 1.22))
        p.restore()

        # 標籤
        if st == "S1" or (st == "S2" and not e.aa):
            self._peak(p, ax, e.F_LO, e.bars_orig[int(e.F_LO)], C_BLUE)
            self._peak(p, ax, e.F_HI, e.bars_orig[int(e.F_HI)], C_BLUE)
        elif st == "S2":
            self._peak(p, ax, e.F_LO, e.bars_filt[int(e.F_LO)], C_GREEN_DK)
            label_box(p, ax.P(e.F_HI, 0.25), f"{e.F_HI:g} Hz：{fmt_amp(e.bars_filt[int(e.F_HI)])}（濾掉）",
                      C_GREEN_DK, 7.5)
            label_box(p, ax.P(min(e.ny + 9, 44), 0.75), "琥珀線 = 濾波器 |H(f)|", C_AMBER_DK, 8, family=FONT_UI)
        else:
            amp = e.amp_alias[e.aa]
            lo = e.amp_lo[e.aa]
            fa = e.f_alias
            self._peak(p, ax, e.F_LO, lo, C_AMBER_DK)
            a = ax.P(e.F_HI, e.A_HI)
            b = ax.P(fa, max(amp, 0.05) + 0.06)
            draw_arrow(p, a, b, C_RED if not e.aa else C_AXIS, 2.2, bend=40)
            txt = (f"{e.F_HI:g} Hz 折回 → |{e.F_HI:g} − {e.k_alias}×{e.fs_new:.2f}| = {fa:.2f} Hz："
                   f"振幅 {amp:.3f}" if not e.aa else
                   f"折回位置 {fa:.2f} Hz：只剩 {fmt_amp(amp)}（已先濾掉）")
            cx = (ax.X(e.F_HI) + ax.X(fa)) / 2
            label_box(p, QPointF(min(max(cx, 230), W - 230), ax.Y(0.97)), txt,
                      C_RED if not e.aa else C_GREEN_DK, 8, family=FONT_UI)
            p.setFont(qfont(8.5, True))
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(ax.X(e.ny) + 6, ax.Y(0.2), max(10.0, ax.rect.right() - ax.X(e.ny) - 12), 16),
                       AL_LEFT, "新 Nyquist 以上：低取樣率表示不了")
        if st != "S1":
            label_box(p, QPointF(ax.X(e.ny), ax.rect.top() + 10), f"新 Nyquist {e.ny:.2f} Hz", C_RED, 7.5,
                      family=FONT_UI)
        p.end()


class AliasTab(StepTab):
    TITLE = "降採樣與混疊　—　沒先低通，高頻會「折」成假的低頻"
    HINT = ("🔑 降採樣的鐵律：新取樣率 fs_new 只裝得下 0~fs_new/2。比它高的成分不會消失，而是折回成 "
            "f_alias = |f − k·fs_new| 的假頻，跟真訊號混在一起、事後無法分離。所以一定要「先低通（截止 = 新 Nyquist）、"
            "再每 M 取一」。勾選 / 取消下方的抗混疊濾波各跑一次，比較最後的頻譜。")

    def make_engine(self):
        return AliasEngine()

    def build_params(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("降採樣倍率 M："))
        self.M_slider = QSlider(Qt.Orientation.Horizontal)
        self.M_slider.setRange(2, 8)
        self.M_slider.setValue(self.engine.M)
        self.M_slider.setFixedWidth(180)
        self.M_slider.setPageStep(1)
        self.M_slider.valueChanged.connect(self.set_M)
        bar.addWidget(self.M_slider)
        self.M_label = QLabel()
        self.M_label.setStyleSheet("font-size:13px;font-weight:bold;color:#1D4ED8;")
        bar.addWidget(self.M_label)
        bar.addSpacing(18)
        self.aa_box = QCheckBox("抗混疊濾波（先低通再取樣）")
        self.aa_box.setChecked(self.engine.aa)
        self.aa_box.setStyleSheet("QCheckBox { font-size:13px; color:#0F172A; }")
        self.aa_box.toggled.connect(self.set_aa)
        bar.addWidget(self.aa_box)
        bar.addStretch(1)
        self._update_M_label()
        return bar

    def _update_M_label(self) -> None:
        e = self.engine
        self.M_label.setText(f"M = {e.M}（{e.FS:.0f} Hz → {e.fs_new:.2f} Hz，新 Nyquist {e.ny:.2f} Hz）")

    def set_M(self, M: int) -> None:
        self.engine.M = int(M)
        self.params_changed()
        self._update_M_label()

    def set_aa(self, on: bool) -> None:
        self.engine.aa = bool(on)
        self.params_changed()

    def build_canvases(self):
        return [AliasTimeCanvas(self.engine), AliasSpectrumCanvas(self.engine)]

    def legend_items(self):
        return [(C_BLUE, "原始樣本 / 頻譜"), (C_GREEN, "濾波後"), (C_AMBER, "保留的點 / 降採樣後頻譜"),
                (C_RED, "新 Nyquist / 重建（含假頻）"), (C_NAN, "新取樣率看不到的頻段"), (C_TRUTH, "真值")]

    def stats_text(self):
        e = self.engine
        return (f"{e.stage_title()}　|　M = {e.M}　|　fs：{e.FS:.0f} → {e.fs_new:.2f} Hz　|　"
                f"抗混疊：{'開' if e.aa else '關'}　|　{e.F_HI:g} Hz 的折回位置 = {e.f_alias:.2f} Hz")


# =========================================================
#  分頁 C：2D 影像放大
# =========================================================
METHODS_2D = [("nearest", "最近鄰"), ("linear", "雙線性"), ("cubic", "雙三次")]
METHOD_NAME_2D = dict(METHODS_2D)


def make_test_image() -> np.ndarray:
    """6×6 小圖：左上到右下的漸層 + 右下一塊暗區（兩條邊緣）+ 一個亮點。"""
    img = np.zeros((6, 6))
    for r in range(6):
        for c in range(6):
            img[r, c] = 50 + 22 * c + 10 * r
    img[2:, 4:] = 10            # 暗區 → 垂直 + 水平邊緣（夠暗，雙三次的過衝會掉到 0 以下）
    img[1, 1] = 255             # 亮點
    return img


class ResizeEngine:
    S, SCALE = 6, 4
    STAGE_TITLE = {"S1": "STEP 1 座標映射", "S2": "STEP 2 逐像素計算", "S3": "STEP 3 完成"}

    def __init__(self):
        self.method = "linear"
        self.img = make_test_image()
        self.reset()

    def stage_title(self) -> str:
        return self.STAGE_TITLE[self.stage]

    @property
    def n_out(self) -> int:
        return self.S * self.SCALE

    def reset(self) -> None:
        self.out = rs.resize(self.img, self.SCALE, self.method)
        self.stage = "S1"
        self.k = 0                      # 已算出的像素數（raster 順序）
        self.cur: Optional[Tuple[int, int]] = None
        self.detail = None
        self.clicked = False
        self.done = False
        self.explain = (
            f"STEP 1｜座標映射：左邊是 6×6 原圖（格內數字 = 灰階值），右邊要放大 ×{self.SCALE} 成 24×24。"
            f"每個輸出像素都要問「我對應到原圖的哪個位置？」。用像素中心對齊：src = (dst + 0.5)/{self.SCALE} − 0.5。"
            f"為什麼要 +0.5 再 −0.5？因為像素索引 i 代表的是「方塊中心」，在連續座標上位於 i + 0.5；先換成連續座標、"
            f"縮放、再換回索引，整張圖才不會偏移半格。按「單步」從左上角 dst(0,0) 開始算（也可以直接點右圖任一格）。")

    def pixel_detail(self, r: int, c: int) -> dict:
        n_in, n_out = self.S, self.n_out
        sy = float(rs.map_coord(r, n_in, n_out))
        sx = float(rs.map_coord(c, n_in, n_out))
        ry, iy, wy = rs.interp_taps(sy, n_in, self.method)
        rx, ix, wx = rs.interp_taps(sx, n_in, self.method)
        ry, iy, wy, rx, ix, wx = ry[0], iy[0], wy[0], rx[0], ix[0], wx[0]
        val = float(sum(wy[a] * wx[b] * self.img[iy[a], ix[b]]
                        for a in range(len(iy)) for b in range(len(ix))))
        assert abs(val - self.out[r, c]) < 1e-9          # 跟函式庫 resize() 結果一致
        return dict(r=r, c=c, sy=sy, sx=sx, ry=ry, iy=iy, wy=wy, rx=rx, ix=ix, wx=wx, val=val)

    def _explain_pixel(self, d: dict, prefix: str) -> str:
        r, c, sy, sx = d["r"], d["c"], d["sy"], d["sx"]
        s = self.SCALE
        head = (f"{prefix}dst(r={r}, c={c}) → src_y = ({r}+0.5)/{s} − 0.5 = {sy:+.3f}、"
                f"src_x = ({c}+0.5)/{s} − 0.5 = {sx:+.3f}（左圖紅十字）。")
        clamp = []
        for name, raw, idx in (("y", d["ry"], d["iy"]), ("x", d["rx"], d["ix"])):
            bad = [f"{int(a)}→{int(b)}" for a, b in zip(raw, idx) if a != b]
            if bad:
                clamp.append(f"{name} 方向 {'、'.join(bad)}")
        clamp_txt = (f"邊界：{'；'.join(clamp)}（超出原圖就 clamp = 邊緣像素往外延伸）。" if clamp else "")
        val = d["val"]
        if self.method == "nearest":
            iy, ix = int(d["iy"][0]), int(d["ix"][0])
            body = (f"最近鄰：四捨五入 → 取 src(⌊{sy:+.3f}+0.5⌋, ⌊{sx:+.3f}+0.5⌋) = ({int(d['ry'][0])}, "
                    f"{int(d['rx'][0])})，直接抄原圖 ({iy},{ix}) 的值 {val:.0f}。不做任何平均 → 放大後是一塊塊 4×4 的方塊。")
        elif self.method == "linear":
            fy, fx = d["wy"][1], d["wx"][1]
            terms = []
            for a in range(2):
                for b in range(2):
                    w = d["wy"][a] * d["wx"][b]
                    v = self.img[d["iy"][a], d["ix"][b]]
                    terms.append(f"{w:.3f}×{v:.0f}")
            body = (f"雙線性：取 y∈{{{int(d['ry'][0])},{int(d['ry'][1])}}}、x∈{{{int(d['rx'][0])},{int(d['rx'][1])}}} "
                    f"的 2×2 鄰居；fy = {fy:.3f} → wy = ({d['wy'][0]:.3f}, {d['wy'][1]:.3f})，fx = {fx:.3f} → "
                    f"wx = ({d['wx'][0]:.3f}, {d['wx'][1]:.3f})。2D 權重 = wy × wx：{' + '.join(terms)} = {val:.2f}。")
        else:
            body = (f"雙三次：y、x 各取 4 個鄰居（4×4 = 16 格），wy = ({', '.join(f'{w:+.3f}' for w in d['wy'])})、"
                    f"wx = ({', '.join(f'{w:+.3f}' for w in d['wx'])})，2D 權重 = wy × wx（見右側權重表），"
                    f"16 項加權和 = {val:.2f}。")
            if val > 255 or val < 0:
                body += f"⚠ 超出 0~255 → 這就是雙三次的「過衝」（負權重造成），存回影像前要 clip 成 {min(max(val, 0), 255):.0f}。"
            elif any(w < 0 for w in list(d["wy"]) + list(d["wx"])):
                body += "外圈權重是負的 → 邊緣更銳利，但亮暗交界附近會出現光暈 / 過衝。"
        return head + clamp_txt + body

    def step(self) -> None:
        if self.done:
            return
        total = self.n_out ** 2
        if self.stage == "S1":
            self.stage = "S2"
            self.k = 1
        elif self.stage == "S2" and self.k < total:
            self.k += 1
        else:
            self.stage = "S3"
            self.done = True
            self.cur = None
            self.detail = None
            o, im = self.out, self.img
            over = int(((o > 255) | (o < 0)).sum())
            extra = (f"輸出範圍 {o.min():.1f}~{o.max():.1f}（原圖 {im.min():.0f}~{im.max():.0f}）。")
            if o.min() < im.min() - 1e-9 or o.max() > im.max() + 1e-9:
                extra += "超出原圖範圍 = 過衝（負權重造成的光暈）"
                extra += (f"，其中 {over} 個像素甚至超出 0~255，存回影像前一定要 clip。" if over else "。")
            else:
                extra += "沒有超出原圖範圍 → 不會過衝（權重全是正的，只是加權平均）。"
            look = {"nearest": "每個原圖像素變成 4×4 的方塊，邊緣鋸齒明顯。",
                    "linear": "漸層變平滑，但亮點與邊緣被「抹開」變糊。",
                    "cubic": "比雙線性銳利，亮點周圍可以看到一圈過衝的暗環 / 亮環。"}[self.method]
            self.explain = (f"STEP 3｜完成：{total} 個像素全部算完。{look}{extra}"
                            f"可以切換方法比較，或點右圖任一像素看它的計算過程。")
            return
        r, c = divmod(self.k - 1, self.n_out)
        self.cur = (r, c)
        self.clicked = False
        self.detail = self.pixel_detail(r, c)
        self.explain = self._explain_pixel(self.detail, f"STEP 2｜第 {self.k}/{total} 個像素：")

    def inspect(self, r: int, c: int) -> None:
        """滑鼠點右圖：只查看，不改變逐步進度。"""
        self.cur = (r, c)
        self.clicked = True
        self.detail = self.pixel_detail(r, c)
        self.explain = self._explain_pixel(self.detail, "（滑鼠點選）")


class ResizeCanvas(QWidget):
    CS_SRC, CS_DST = 38, 13
    LEFT_W, MID_W = 380, 352

    def __init__(self, engine: ResizeEngine, tab: "ResizeTab"):
        super().__init__()
        self.e = engine
        self.tab = tab
        self.setFixedSize(CANVAS_W, 462)
        self.src_origin = QPointF(1 + 76, 1 + 30 + 76)
        self.dst_origin = QPointF(self.LEFT_W + 8 + 20, 1 + 30 + 46)

    @staticmethod
    def gray(v: float) -> QColor:
        g = int(round(min(max(v, 0.0), 255.0)))
        return QColor(g, g, g)

    def paintEvent(self, _):
        e = self.e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        H = self.height()
        d = e.detail
        # ---------- 左：原圖 ----------
        left = QRectF(1, 1, self.LEFT_W, H - 2)
        draw_panel(p, left, "輸入 6×6（紅十字 = 映射回來的浮點座標）")
        cs = self.CS_SRC
        ox, oy = self.src_origin.x(), self.src_origin.y()
        p.setFont(qfont(7.5, family=FONT_NUM))
        for i in range(e.S):
            p.setPen(QPen(C_MUTED))
            p.drawText(QRectF(ox + i * cs, oy - 16, cs, 14), AL_CENTER, f"x={i}")
            p.drawText(QRectF(ox - 34, oy + i * cs, 30, cs), AL_RIGHT, f"y={i}")
        for r in range(e.S):
            for c in range(e.S):
                v = e.img[r, c]
                rect = QRectF(ox + c * cs, oy + r * cs, cs, cs)
                p.setPen(QPen(C_GRID, 1))
                p.setBrush(QBrush(self.gray(v)))
                p.drawRect(rect)
                p.setPen(QPen(QColor("#0F172A") if v > 140 else QColor("#F8FAFC")))
                p.setFont(qfont(8, True, FONT_NUM))
                p.drawText(rect, AL_CENTER, f"{v:.0f}")
        if d is not None:
            # 用到的鄰居（clamp 後的實際格子：實線框；超出邊界的「虛擬格」：虛線框）
            for a, (rawy, iy) in enumerate(zip(d["ry"], d["iy"])):
                for b, (rawx, ix) in enumerate(zip(d["rx"], d["ix"])):
                    w = d["wy"][a] * d["wx"][b]
                    real = QRectF(ox + ix * cs + 2, oy + iy * cs + 2, cs - 4, cs - 4)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(C_AMBER if w >= 0 else C_RED, 3))
                    p.drawRect(real)
                    if rawy != iy or rawx != ix:
                        virt = QRectF(ox + rawx * cs + 3, oy + rawy * cs + 3, cs - 6, cs - 6)
                        p.setPen(dashed(C_AMBER_DK, 1.4))
                        p.drawRect(virt)
            if e.method in ("nearest", "linear"):
                cpt = QPointF(ox + (d["sx"] + 0.5) * cs, oy + (d["sy"] + 0.5) * cs)
                for a, iy in enumerate(d["iy"]):
                    for b, ix in enumerate(d["ix"]):
                        p.setPen(dashed(C_AMBER_DK, 1.2, Qt.PenStyle.DotLine))
                        p.drawLine(cpt, QPointF(ox + (ix + 0.5) * cs, oy + (iy + 0.5) * cs))
            cx, cy = ox + (d["sx"] + 0.5) * cs, oy + (d["sy"] + 0.5) * cs
            p.setPen(QPen(C_RED, 2.4))
            p.drawLine(QPointF(cx - 11, cy), QPointF(cx + 11, cy))
            p.drawLine(QPointF(cx, cy - 11), QPointF(cx, cy + 11))
            draw_dot(p, QPointF(cx, cy), 3.5, C_PANEL, C_RED, 1.6)
        p.setFont(qfont(8))
        p.setPen(QPen(C_MUTED))
        p.drawText(QRectF(left.left() + 10, left.bottom() - 38, left.width() - 20, 34),
                   int(Qt.AlignmentFlag.AlignLeft.value | Qt.AlignmentFlag.AlignBottom.value
                       | Qt.TextFlag.TextWordWrap.value),
                   "實線框 = 用到的像素（紅框 = 負權重）；虛線框 = 超出邊界、被 clamp 到邊緣的「虛擬鄰居」")

        # ---------- 中：輸出 ----------
        mid = QRectF(self.LEFT_W + 8, 1, self.MID_W, H - 2)
        draw_panel(p, mid, f"輸出 24×24（{METHOD_NAME_2D[e.method]}；點任一格看計算）", "")
        cs2 = self.CS_DST
        ox2, oy2 = self.dst_origin.x(), self.dst_origin.y()
        n = e.n_out
        done = n * n if e.stage == "S3" else e.k
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)    # 像素格不要反鋸齒，免得格與格之間出現細縫
        for r in range(n):
            for c in range(n):
                rect = QRectF(ox2 + c * cs2, oy2 + r * cs2, cs2, cs2)
                if r * n + c < done or (e.clicked and e.cur == (r, c)):
                    p.fillRect(rect, self.gray(e.out[r, c]))
                else:
                    p.fillRect(rect, QColor("#F1F5F9") if (r + c) % 2 else QColor("#E2E8F0"))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(with_alpha(C_AXIS, 160), 1))           # 每 4 格一條線 = 原圖一個像素的範圍
        for k in range(0, n + 1, e.SCALE):
            p.drawLine(QPointF(ox2 + k * cs2, oy2), QPointF(ox2 + k * cs2, oy2 + n * cs2))
            p.drawLine(QPointF(ox2, oy2 + k * cs2), QPointF(ox2 + n * cs2, oy2 + k * cs2))
        if e.cur is not None:
            r, c = e.cur
            p.setPen(QPen(C_AMBER, 2.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(ox2 + c * cs2 - 1, oy2 + r * cs2 - 1, cs2 + 2, cs2 + 2))
        p.setFont(qfont(7.5, family=FONT_NUM))
        p.setPen(QPen(C_MUTED))
        for k in range(0, n, e.SCALE):
            p.drawText(QRectF(ox2 + k * cs2 - 4, oy2 - 15, 30, 13), AL_LEFT, f"{k}")
            p.drawText(QRectF(ox2 - 22, oy2 + k * cs2, 18, 13), AL_RIGHT, f"{k}")
        p.setFont(qfont(8))
        p.drawText(QRectF(mid.left() + 10, oy2 + n * cs2 + 6, mid.width() - 20, 34),
                   int(Qt.AlignmentFlag.AlignLeft.value | Qt.AlignmentFlag.AlignTop.value
                       | Qt.TextFlag.TextWordWrap.value),
                   "灰線 = 每 4×4 格對應原圖 1 個像素；棋盤格 = 還沒算。"
                   + (f"\n目前 dst({e.cur[0]}, {e.cur[1]}) → src({d['sy']:+.3f}, {d['sx']:+.3f}) = "
                      f"{e.out[e.cur]:.2f}" if (e.cur and d is not None) else ""))

        # ---------- 右：權重表 ----------
        right = QRectF(self.LEFT_W + 8 + self.MID_W + 8, 1, CANVAS_W - self.LEFT_W - self.MID_W - 18, H - 2)
        draw_panel(p, right, "權重表　w = wy × wx")
        if d is None:
            p.setFont(qfont(9.5))
            p.setPen(QPen(C_MUTED))
            p.drawText(right.adjusted(14, 40, -14, -14),
                       int(Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value),
                       "點右圖任一格（或用單步逐格走）\n這裡會列出用到的鄰居、\n各自的權重與灰階值")
        else:
            self._draw_table(p, right, d)
        p.end()

    def _draw_table(self, p: QPainter, right: QRectF, d: dict) -> None:
        ny, nx = len(d["iy"]), len(d["ix"])
        head_w, head_h = 58.0, 34.0
        avail_w = right.width() - 20 - head_w
        cw = min(56.0, avail_w / nx)
        ch = min(52.0, (right.height() - 150) / ny)
        x0 = right.left() + 10 + head_w
        y0 = right.top() + 34 + head_h
        p.setFont(qfont(7.5, True, FONT_NUM))
        for b in range(nx):                      # 欄標題：x 索引與 wx
            rect = QRectF(x0 + b * cw, y0 - head_h, cw, head_h)
            p.setPen(QPen(C_MUTED))
            raw, idx = int(d["rx"][b]), int(d["ix"][b])
            p.drawText(QRectF(rect.left(), rect.top(), cw, 15), AL_CENTER,
                       f"x={raw}" if raw == idx else f"x={raw}→{idx}")
            p.setPen(QPen(C_BLUE))
            p.drawText(QRectF(rect.left(), rect.top() + 15, cw, 15), AL_CENTER, f"{d['wx'][b]:+.3f}")
        for a in range(ny):                      # 列標題：y 索引與 wy
            rect = QRectF(x0 - head_w, y0 + a * ch, head_w, ch)
            p.setPen(QPen(C_MUTED))
            raw, idx = int(d["ry"][a]), int(d["iy"][a])
            p.drawText(QRectF(rect.left(), rect.top() + ch / 2 - 15, head_w - 4, 15), AL_RIGHT,
                       f"y={raw}" if raw == idx else f"y={raw}→{idx}")
            p.setPen(QPen(C_BLUE))
            p.drawText(QRectF(rect.left(), rect.top() + ch / 2, head_w - 4, 15), AL_RIGHT, f"{d['wy'][a]:+.3f}")
        for a in range(ny):
            for b in range(nx):
                w = d["wy"][a] * d["wx"][b]
                v = self.e.img[d["iy"][a], d["ix"][b]]
                rect = QRectF(x0 + b * cw + 1, y0 + a * ch + 1, cw - 2, ch - 2)
                fill = with_alpha(C_AMBER_LT, int(40 + 180 * min(1.0, abs(w))))
                p.setPen(QPen(C_RED if w < 0 else C_AMBER, 1.2))
                p.setBrush(QBrush(fill))
                p.drawRoundedRect(rect, 5, 5)
                p.setFont(qfont(8.5 if nx <= 2 else 7.5, True, FONT_NUM))
                p.setPen(QPen(C_RED if w < 0 else C_TEXT))
                p.drawText(QRectF(rect.left(), rect.top() + 3, rect.width(), rect.height() / 2 - 2),
                           AL_CENTER, f"{w:+.3f}" if nx > 2 else f"{w:.3f}")
                p.setFont(qfont(7.5, False, FONT_NUM))
                p.setPen(QPen(C_MUTED))
                p.drawText(QRectF(rect.left(), rect.top() + rect.height() / 2, rect.width(), rect.height() / 2 - 2),
                           AL_CENTER, f"v={v:.0f}")
        yb = y0 + ny * ch + 12
        val = d["val"]
        p.setFont(qfont(10, True, FONT_NUM))
        p.setPen(QPen(C_TEXT))
        p.drawText(QRectF(right.left() + 10, yb, right.width() - 20, 20), AL_LEFT, f"Σ w·v = {val:.2f}")
        p.setFont(qfont(8.5))
        p.setPen(QPen(C_MUTED))
        sw = float(np.sum(np.outer(d["wy"], d["wx"])))
        p.drawText(QRectF(right.left() + 10, yb + 22, right.width() - 20, 18), AL_LEFT,
                   f"Σ w = {sw:.3f}（權重總和永遠是 1）")
        if val > 255 or val < 0:
            p.setFont(qfont(8.5, True))
            p.setPen(QPen(C_RED))
            p.drawText(QRectF(right.left() + 10, yb + 42, right.width() - 20, 18), AL_LEFT,
                       f"超出 0~255 → clip 成 {min(max(val, 0), 255):.0f}（過衝）")

    def mousePressEvent(self, ev):
        pos = ev.position()
        n, cs2 = self.e.n_out, self.CS_DST
        c = int((pos.x() - self.dst_origin.x()) // cs2)
        r = int((pos.y() - self.dst_origin.y()) // cs2)
        if 0 <= r < n and 0 <= c < n:
            self.tab.inspect_pixel(r, c)


class ResizeTab(StepTab):
    TITLE = "2D 影像放大 ×4　—　每個輸出像素對應原圖哪裡？用了哪些鄰居？"
    HINT = ("🔑 兩個重點：① 座標要用「像素中心對齊」src = (dst + 0.5)/scale − 0.5，所以 dst = 0 會對到 −0.375"
            "（在第 0 個像素中心的左邊），超出邊界就 clamp；用 align_corners 或忘了 ±0.5，整張圖會偏移半格。"
            "② 2D 權重 = y 方向權重 × x 方向權重（可分離），所以雙線性 = 2×2、雙三次 = 4×4，總和永遠是 1。")

    def make_engine(self):
        return ResizeEngine()

    def build_params(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("方法："))
        self.m_group, self.m_btns = self.make_choice_group(self, METHODS_2D, self.engine.method,
                                                           self.set_method)
        for b in self.m_btns.values():
            bar.addWidget(b)
        bar.addSpacing(14)
        tip = QLabel("滑鼠點右圖任一格 = 查看那一格的計算（不影響逐步進度）")
        tip.setStyleSheet("font-size:11px;color:#64748B;")
        bar.addWidget(tip)
        bar.addStretch(1)
        return bar

    def set_method(self, m: str) -> None:
        self.engine.method = m
        self.sync_choice(self.m_btns, m)
        self.params_changed()

    def inspect_pixel(self, r: int, c: int) -> None:
        self.stop_play()
        self.engine.inspect(r, c)
        self.refresh()

    def build_canvases(self):
        return [ResizeCanvas(self.engine, self)]

    def legend_items(self):
        return [(C_RED, "映射回去的浮點座標 / 負權重"), (C_AMBER, "用到的鄰居"),
                (C_AMBER_LT, "權重表（越深權重越大）"), (QColor("#E2E8F0"), "還沒算的像素")]

    def stats_text(self):
        e = self.engine
        n = e.n_out
        done = n * n if e.stage == "S3" else e.k
        cur = f"dst({e.cur[0]}, {e.cur[1]})" if e.cur else "—"
        return (f"{e.stage_title()}　|　方法：{METHOD_NAME_2D[e.method]}　|　6×6 → 24×24　|　"
                f"已算 {done}/{n * n} 像素　|　目前：{cur}")


# =========================================================
#  分頁 D：時間戳對齊
# =========================================================
class AlignEngine:
    T_END = 3.0
    RATES = [10, 20, 50]
    STAGE_TITLE = {"S1": "STEP 1 原始不等間距資料", "S2": "STEP 2 建立共同時間軸",
                   "S3": "STEP 3 逐點插值", "S4": "STEP 4 結果與「為何不硬插」"}

    def __init__(self):
        self.rate = 20
        self._make_data()
        self.reset()

    @staticmethod
    def imu_truth(t):
        t = np.asarray(t, dtype=float)
        return 0.8 * np.sin(2 * np.pi * 1.1 * t) + 0.3 * np.sin(2 * np.pi * 3.3 * t + 0.5)

    @staticmethod
    def gps_truth(t):
        t = np.asarray(t, dtype=float)
        # 隧道裡其實先加速再減速（高斯鼓包）—— 只有事後才知道，掉包期間誰也看不到
        return (1.5 + 0.6 * t + 0.35 * np.sin(2 * np.pi * 0.7 * t)
                + 0.6 * np.exp(-((t - 1.55) / 0.16) ** 2))

    def _make_data(self) -> None:
        rng = np.random.default_rng(2024)                    # 固定 seed → 每次重播一模一樣
        t = np.arange(0, self.T_END, 0.01) + np.clip(rng.normal(0, 0.0015, 300), -0.004, 0.004)
        t = t[~((t > 2.335) & (t < 2.405))]                  # IMU 短暫掉包
        self.t_imu = t
        self.x_imu = self.imu_truth(t) + rng.normal(0, 0.03, len(t))
        tg = 0.03 + np.arange(0, self.T_END, 0.1) + rng.uniform(-0.006, 0.006, 30)
        tg = tg[~((tg > 1.2) & (tg < 1.9))]                  # GPS 長時間掉包（例如進隧道）
        self.t_gps = tg
        self.x_gps = self.gps_truth(tg) + rng.normal(0, 0.02, len(tg))
        self.gap_imu = rs.auto_max_gap(self.t_imu)
        self.gap_gps = rs.auto_max_gap(self.t_gps)

    def stage_title(self) -> str:
        return self.STAGE_TITLE[self.stage]

    def streams(self):
        return [("IMU", self.t_imu, self.x_imu, self.gap_imu), ("GPS", self.t_gps, self.x_gps, self.gap_gps)]

    def reset(self) -> None:
        self.tc, al = rs.align_streams({"IMU": (self.t_imu, self.x_imu), "GPS": (self.t_gps, self.x_gps)},
                                       rate=self.rate)
        self.al = al
        self.K = len(self.tc)
        self.stage = "S1"
        self.k = 0
        self.cur = None            # {name: detail}
        self.done = False
        di, dg = np.diff(self.t_imu), np.diff(self.t_gps)
        self.explain = (
            f"STEP 1｜原始資料：IMU 標稱 100 Hz，但時間戳有抖動（間隔 {di.min() * 1000:.1f}~"
            f"{np.median(di) * 1000:.1f}~{np.percentile(di, 99) * 1000:.1f} ms），2.34~2.40 s 還掉了幾包；"
            f"GPS 10 Hz，{self.t_gps[self.t_gps < 1.5].max():.2f}~{self.t_gps[self.t_gps > 1.5].min():.2f} s "
            f"整段沒資料（例如進隧道），最大間隔 {dg.max():.2f} s。兩個感測器的時間點完全對不上，"
            f"沒辦法直接逐列合併 → 要先重取樣到同一條等間距時間軸。紅色區塊 = 間隔超過 max_gap 的空洞。")

    def gaps(self, t: np.ndarray, max_gap: float):
        d = np.diff(t)
        return [(t[i], t[i + 1]) for i in np.nonzero(d > max_gap)[0]]

    def detail(self, name: str, t: np.ndarray, x: np.ndarray, max_gap: float, tq: float) -> dict:
        i0, frac, gap = rs.bracket(t, tq)
        i0, frac, gap = int(i0[0]), float(frac[0]), float(gap[0])
        ok = gap <= max_gap or tq == t[i0] or tq == t[i0 + 1]
        val = (1 - frac) * x[i0] + frac * x[i0 + 1] if ok else float("nan")
        lib = self.al[name][self.k - 1]
        assert (np.isnan(val) and np.isnan(lib)) or abs(val - lib) < 1e-9     # 跟 align_streams 一致
        return dict(i0=i0, frac=frac, gap=gap, ok=ok, val=val, t0=t[i0], t1=t[i0 + 1],
                    x0=x[i0], x1=x[i0 + 1], max_gap=max_gap)

    def step(self) -> None:
        if self.done:
            return
        if self.stage == "S1":
            self.stage = "S2"
            t0 = max(self.t_imu[0], self.t_gps[0])
            t1 = min(self.t_imu[-1], self.t_gps[-1])
            self.explain = (
                f"STEP 2｜建立共同時間軸：只在兩個感測器「都有資料」的範圍內對齊 → 起點取兩者起點的最大值 "
                f"{t0:.4f} s、終點取終點的最小值 {t1:.4f} s，再對齊到 1/{self.rate} s 的整數倍："
                f"{self.tc[0]:.2f} ~ {self.tc[-1]:.2f} s，共 {self.K} 點（橫線上的琥珀色虛線）。"
                f"對齊到整數倍的好處：不同批次、不同檔案的資料可以直接拼接。")
        elif self.stage == "S2" or (self.stage == "S3" and self.k < self.K):
            self.stage = "S3"
            self.k += 1
            tq = float(self.tc[self.k - 1])
            self.cur = {name: self.detail(name, t, x, g, tq) for name, t, x, g in self.streams()}
            parts = [f"STEP 3｜第 {self.k}/{self.K} 點 t = {tq:.3f} s。"]
            unit = {"IMU": "m/s²", "GPS": "m/s"}
            for name, _, _, _ in self.streams():
                d = self.cur[name]
                if d["ok"]:
                    parts.append(
                        f"{name}：落在 t[{d['i0']}] = {d['t0']:.4f} 與 t[{d['i0'] + 1}] = {d['t1']:.4f} 之間，"
                        f"frac = ({tq:.3f} − {d['t0']:.4f}) / {d['gap']:.4f} = {d['frac']:.3f} → "
                        f"{1 - d['frac']:.3f}×({d['x0']:+.3f}) + {d['frac']:.3f}×({d['x1']:+.3f}) = "
                        f"{d['val']:+.3f} {unit[name]}。")
                else:
                    parts.append(
                        f"{name}：左右鄰居 {d['t0']:.3f} 與 {d['t1']:.3f} 相隔 {d['gap']:.3f} s > max_gap "
                        f"{d['max_gap']:.3f} s（= 正常間隔中位數 × 2.5）→ 填 NaN，不插值（灰色區塊）。")
            self.explain = "　".join(parts)
        else:
            self.stage = "S4"
            self.done = True
            self.cur = None
            ni, ng = int(np.isnan(self.al["IMU"]).sum()), int(np.isnan(self.al["GPS"]).sum())
            g0, g1 = self.gaps(self.t_gps, self.gap_gps)[0]
            mid = (g0 + g1) / 2
            lin = float(np.interp(mid, [g0, g1], [self.gps_truth(g0), self.gps_truth(g1)]))
            real = float(self.gps_truth(mid))
            self.explain = (
                f"STEP 4｜完成：{self.K} 個共同時間點，IMU 有 {ni} 個 NaN、GPS 有 {ng} 個 NaN。為什麼斷線處不能硬插？"
                f"紅虛線就是「硬插」的結果 —— 它假裝 GPS 在那 {g1 - g0:.2f} 秒裡速度沿直線變化，"
                f"但事後才知道真實情況是淡色曲線（例如在 {mid:.2f} s：直線 {lin:.2f} vs 真實 {real:.2f} m/s）。"
                f"插出來的點看起來跟真量測一模一樣，下游（例如卡爾曼濾波、模型訓練）會把它當真的資料用。"
                f"正確做法：填 NaN + 遮罩，讓下游知道「這段沒量到」—— 卡爾曼濾波就只做預測、不做更新。")

    def counts(self, name: str) -> Tuple[int, int]:
        n = self.K if self.stage == "S4" else (self.k if self.stage == "S3" else 0)
        arr = self.al[name][:n]
        return int(np.isfinite(arr).sum()), int(np.isnan(arr).sum())


class AlignCanvas(QWidget):
    ROW_H = 238

    def __init__(self, engine: AlignEngine):
        super().__init__()
        self.e = engine
        self.setFixedSize(CANVAS_W, 2 * self.ROW_H + 8)

    def paintEvent(self, _):
        e = self.e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        specs = [("IMU", e.t_imu, e.x_imu, e.gap_imu, (-1.6, 1.6), [-1, 0, 1], C_BLUE,
                  f"IMU 加速度（m/s²）：標稱 100 Hz、時間戳抖動、2.34~2.40 s 掉包；max_gap = {e.gap_imu * 1000:.1f} ms"),
                 ("GPS", e.t_gps, e.x_gps, e.gap_gps, (1.0, 3.8), [1.5, 2.5, 3.5], C_GREEN_DK,
                  f"GPS 速度（m/s）：10 Hz、1.2~1.9 s 進隧道沒訊號；max_gap = {e.gap_gps:.2f} s")]
        for row, (name, t, x, mg, yr, yt, color, title) in enumerate(specs):
            top = 1 + row * (self.ROW_H + 8)
            panel = QRectF(1, top, CANVAS_W - 2, self.ROW_H)
            draw_panel(p, panel, title, e.stage_title() if row == 0 else "")
            ax = Axes(QRectF(52, top + 32, 712, self.ROW_H - 32 - 36), (-0.05, 3.05), yr)
            ax.draw_grid(p, [i * 0.5 for i in range(7)], yt, xfmt=lambda v: f"{v:.1f}",
                         yfmt=lambda v: f"{v:g}", xlabel="時間（秒）")
            self._draw_main(p, ax, name, t, x, mg, color)
            zoom = QRectF(812, top + 32, CANVAS_W - 812 - 20, self.ROW_H - 32 - 36)
            self._draw_zoom(p, zoom, name, t, x, color)
        p.end()

    def _draw_main(self, p: QPainter, ax: Axes, name: str, t, x, mg: float, color: QColor) -> None:
        e = self.e
        st = e.stage
        p.save()
        ax.clip(p)
        # 空洞（STEP 1 紅色、之後灰色 = 不插值）
        for g0, g1 in e.gaps(t, mg):
            r = QRectF(QPointF(ax.X(g0), ax.rect.top()), QPointF(ax.X(g1), ax.rect.bottom()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(239, 68, 68, 40) if st == "S1" else C_NAN))
            p.drawRect(r)
            p.setFont(qfont(8, True))
            p.setPen(QPen(C_RED if st == "S1" else QColor("#475569")))
            label = f"掉包 {g1 - g0:.2f} s" if st == "S1" else "NaN：不插值"
            ly = ax.rect.top() + 4 if st == "S1" else ax.rect.bottom() - 34
            p.drawText(QRectF(r.center().x() - 60, ly, 120, 14), AL_CENTER, label)
        if st != "S1":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(148, 163, 184, 35)))
            p.drawRect(QRectF(QPointF(ax.X(ax.x0), ax.rect.top()), QPointF(ax.X(e.tc[0]), ax.rect.bottom())))
            p.drawRect(QRectF(QPointF(ax.X(e.tc[-1]), ax.rect.top()), QPointF(ax.X(ax.x1), ax.rect.bottom())))
            for tq in e.tc:
                p.setPen(dashed(with_alpha(C_AMBER, 120), 1, Qt.PenStyle.DotLine))
                p.drawLine(ax.P(tq, ax.y0), ax.P(tq, ax.y1))
        # 原始資料
        faded = st in ("S3", "S4")
        ax.polyline(p, t, x, QPen(with_alpha(color, 60 if faded else 110), 1))
        for tt, xx in zip(t, x):
            draw_dot(p, ax.P(tt, xx), 2.3 if name == "IMU" else 3.6, with_alpha(color, 110 if faded else 230), None)
        # 已對齊的點
        n_done = e.K if st == "S4" else (e.k if st == "S3" else 0)
        vals = e.al[name]
        for i in range(n_done):
            tq, v = e.tc[i], vals[i]
            if np.isfinite(v):
                pt = ax.P(tq, v)
                p.setPen(QPen(QColor("#0F172A"), 1))
                p.setBrush(QBrush(C_AMBER_LT if st == "S3" and i == e.k - 1 else C_PANEL))
                p.drawRect(QRectF(pt.x() - 3.5, pt.y() - 3.5, 7, 7))
            else:
                y = ax.rect.bottom() - 10
                xx = ax.X(tq)
                p.setPen(QPen(QColor("#475569"), 1.8))
                p.drawLine(QPointF(xx - 4, y - 4), QPointF(xx + 4, y + 4))
                p.drawLine(QPointF(xx - 4, y + 4), QPointF(xx + 4, y - 4))
        if st == "S3" and e.cur is not None:
            d = e.cur[name]
            tq = e.tc[e.k - 1]
            p.setPen(QPen(C_AMBER, 1.8))
            p.drawLine(ax.P(tq, ax.y0), ax.P(tq, ax.y1))
            for tt, xx in ((d["t0"], d["x0"]), (d["t1"], d["x1"])):
                draw_dot(p, ax.P(tt, xx), 7, None, C_AMBER, 2.2)
            if d["ok"]:
                draw_dot(p, ax.P(tq, d["val"]), 5, C_AMBER, C_AMBER_DK, 1.6)
        if st == "S4" and name == "GPS":
            for g0, g1 in e.gaps(t, mg):
                i0 = int(np.searchsorted(t, g0))
                ts = np.linspace(g0, g1, 60)
                ax.polyline(p, ts, e.gps_truth(ts), QPen(with_alpha(C_GREEN, 150), 2))
                p.setPen(dashed(C_RED, 2.2))
                p.drawLine(ax.P(t[i0], x[i0]), ax.P(t[i0 + 1], x[i0 + 1]))
                mid = (g0 + g1) / 2
                y_lin = float(np.interp(mid, [t[i0], t[i0 + 1]], [x[i0], x[i0 + 1]]))
                y_true = float(e.gps_truth(ts).max())
                label_box(p, ax.P(mid + 0.08, y_lin - 0.42), "紅虛線 = 硬插（捏造）", C_RED, 7.5, family=FONT_UI)
                label_box(p, ax.P(mid, min(y_true + 0.25, ax.y1 - 0.12)), "淡綠 = 真實（事後才知道）",
                          C_GREEN_DK, 7.5, family=FONT_UI)
        p.restore()

    def _draw_zoom(self, p: QPainter, rect: QRectF, name: str, t, x, color: QColor) -> None:
        e = self.e
        p.setPen(QPen(C_GRID, 1))
        p.setBrush(QBrush(QColor("#FFFBEB")))
        p.drawRoundedRect(rect.adjusted(-6, -4, 6, 4), 8, 8)
        p.setFont(qfont(8.5, True))
        p.setPen(QPen(C_AMBER_DK))
        p.drawText(QRectF(rect.left(), rect.top() - 2, rect.width(), 16), AL_CENTER, f"🔍 {name} 放大鏡")
        if e.stage != "S3" or e.cur is None:
            if e.stage == "S4":
                v, n = e.counts(name)
                msg = f"完成\n有效 {v} 點、NaN {n} 點\n（NaN = 那段沒量到，\n留給下游判斷）"
            else:
                msg = "STEP 3 會在這裡\n放大看左右兩個鄰居\n與插值比例 frac"
            p.setFont(qfont(9))
            p.setPen(QPen(C_MUTED))
            p.drawText(rect.adjusted(6, 20, -6, -6),
                       int(Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value), msg)
            return
        d = e.cur[name]
        tq = float(e.tc[e.k - 1])
        span = d["t1"] - d["t0"]
        xa, xb = d["t0"] - 0.45 * span, d["t1"] + 0.45 * span
        m = (t >= xa) & (t <= xb)
        ys = list(x[m]) + [d["x0"], d["x1"]]
        lo, hi = min(ys), max(ys)
        pad = max(0.15 * (hi - lo), 0.05)
        ax = Axes(QRectF(rect.left() + 8, rect.top() + 22, rect.width() - 16, rect.height() - 44),
                  (xa, xb), (lo - pad, hi + pad))
        p.setPen(QPen(C_GRID, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(ax.rect)
        p.save()
        ax.clip(p)
        if not d["ok"]:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(C_NAN))
            p.drawRect(QRectF(QPointF(ax.X(d["t0"]), ax.rect.top()), QPointF(ax.X(d["t1"]), ax.rect.bottom())))
        ax.polyline(p, t[m], x[m], QPen(with_alpha(color, 90), 1))
        for tt, xx in zip(t[m], x[m]):
            draw_dot(p, ax.P(tt, xx), 3, with_alpha(color, 200), None)
        p.setPen(dashed(C_AMBER if d["ok"] else C_RED, 1.8))
        p.drawLine(ax.P(d["t0"], d["x0"]), ax.P(d["t1"], d["x1"]))
        p.setPen(QPen(C_AMBER, 1.5))
        p.drawLine(ax.P(tq, ax.y0), ax.P(tq, ax.y1))
        for tt, xx in ((d["t0"], d["x0"]), (d["t1"], d["x1"])):
            draw_dot(p, ax.P(tt, xx), 6.5, None, C_AMBER, 2.2)
        if d["ok"]:
            draw_dot(p, ax.P(tq, d["val"]), 5, C_AMBER, C_AMBER_DK, 1.6)
        p.restore()
        p.setFont(qfont(8, True, FONT_NUM))
        p.setPen(QPen(C_TEXT if d["ok"] else C_RED))
        txt = (f"frac = {d['frac']:.3f}　→ {d['val']:+.3f}" if d["ok"]
               else f"間隔 {d['gap']:.3f}s > {d['max_gap']:.3f}s → NaN")
        p.drawText(QRectF(rect.left(), ax.rect.bottom() + 3, rect.width(), 16), AL_CENTER, txt)


class AlignTab(StepTab):
    TITLE = "時間戳對齊　—　抖動 + 掉包的多感測器資料 → 共同等間距時間軸"
    HINT = ("🔑 不等間距重取樣的三個動作：① 只在所有感測器都有資料的「交集」範圍建立時間軸；"
            "② 每個時間點用二分搜尋找左右鄰居、按比例 frac 線性插值；③ 左右鄰居相隔超過 max_gap（正常間隔 × 2.5）"
            "就填 NaN —— 那段根本沒量到，插出來的值是捏造的。")

    def make_engine(self):
        return AlignEngine()

    def build_params(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("共同時間軸取樣率："))
        self.r_group, self.r_btns = self.make_choice_group(self, [(r, f"{r} Hz") for r in AlignEngine.RATES],
                                                           self.engine.rate, self.set_rate)
        for b in self.r_btns.values():
            bar.addWidget(b)
        bar.addStretch(1)
        return bar

    def set_rate(self, r: int) -> None:
        self.engine.rate = r
        self.sync_choice(self.r_btns, r)
        self.params_changed()

    def build_canvases(self):
        return [AlignCanvas(self.engine)]

    def legend_items(self):
        return [(C_BLUE, "IMU 原始"), (C_GREEN_DK, "GPS 原始"), (C_AMBER, "這一步的時間點 / 左右鄰居"),
                (C_PANEL, "對齊後的值（方塊）"), (C_NAN, "NaN：不插值"), (QColor(239, 68, 68, 90), "掉包空洞")]

    def stats_text(self):
        e = self.engine
        vi, ni = e.counts("IMU")
        vg, ng = e.counts("GPS")
        prog = f"{e.k}/{e.K}" if e.stage in ("S3", "S4") else f"0/{e.K}"
        return (f"{e.stage_title()}　|　共同軸 {e.rate} Hz：{e.tc[0]:.2f}~{e.tc[-1]:.2f} s、{e.K} 點（進度 {prog}）"
                f"　|　IMU 有效 {vi} / NaN {ni}　|　GPS 有效 {vg} / NaN {ng}")


# =========================================================
#  主視窗
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("上採樣 / 重取樣 工程技巧動畫（解說版）")
        self.tabs = QTabWidget()
        self.tab_up = UpsampleTab()
        self.tab_alias = AliasTab()
        self.tab_resize = ResizeTab()
        self.tab_align = AlignTab()
        self.tabs.addTab(self.tab_up, "A｜1D 上採樣")
        self.tabs.addTab(self.tab_alias, "B｜降採樣與混疊")
        self.tabs.addTab(self.tab_resize, "C｜2D 影像放大")
        self.tabs.addTab(self.tab_align, "D｜時間戳對齊")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self.resize(1120, 960)
        self.setStyleSheet("""
            QMainWindow { background:#F8FAFC; }
            QPushButton {
                background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
                padding:7px 12px; font-size:13px; color:#0F172A;
            }
            QPushButton:hover { background:#EFF6FF; border-color:#60A5FA; }
            QPushButton:checked { background:#2563EB; color:white; border-color:#2563EB; }
            QLabel { color:#0F172A; }
            QCheckBox { color:#0F172A; }
            QTabWidget::pane { border:none; }
            QTabBar::tab {
                background:#FFFFFF; border:1px solid #CBD5E1; border-bottom:none;
                border-top-left-radius:8px; border-top-right-radius:8px;
                padding:7px 16px; margin-right:4px; font-size:13px; color:#334155;
            }
            QTabBar::tab:selected { background:#2563EB; color:white; border-color:#2563EB; }
            QTabBar::tab:hover:!selected { background:#EFF6FF; }
        """)

    def all_tabs(self) -> List[StepTab]:
        return [self.tab_up, self.tab_alias, self.tab_resize, self.tab_align]

    def _on_tab_changed(self, _idx: int) -> None:
        for t in self.all_tabs():          # 切換分頁時，背景分頁的自動播放先停
            if t is not self.tabs.currentWidget():
                t.stop_play()


def main():
    # 讓 print() 的中文在任何 Windows 主控台編碼下都不會出錯
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
