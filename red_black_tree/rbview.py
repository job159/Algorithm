# -*- coding: utf-8 -*-
"""
紅黑樹視覺化畫布（PySide6）。

TreeCanvas 是一個 QGraphicsView：
- 用節點的「中序順序」決定 x、「深度」決定 y 來排版（保證左 < 父 < 右、不重疊）。
- 用節點永久 id 來辨識同一個節點，在兩個快照之間做平滑的位移動畫
  （旋轉看起來就像節點滑到新位置）。
- 可選擇顯示 NIL 葉節點（黑色小方塊），方便講解性質三、性質五。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

# 調色盤
COL_RED_FILL = QColor("#e53935")
COL_RED_BORDER = QColor("#8e1b18")
COL_BLACK_FILL = QColor("#37474f")
COL_BLACK_BORDER = QColor("#11181c")
COL_HIGHLIGHT = QColor("#ffb300")
COL_EDGE = QColor("#90a4ae")
COL_NEUTRAL_FILL = QColor("#1565c0")  # BST 節點（無紅黑色）用中性藍
COL_NEUTRAL_BORDER = QColor("#0d3c8b")
COL_AVL_FILL = QColor("#00897b")  # AVL 節點用中性綠（青）
COL_AVL_BORDER = QColor("#00564d")
COL_BADGE_BG = QColor("#ffffff")
COL_BADGE_BORDER = QColor("#90a4ae")
COL_BADGE_TEXT = QColor("#37474f")
COL_BADGE_WARN = QColor("#c62828")  # |平衡因子| ≥ 2 時的警示色
COL_NIL_FILL = QColor("#263238")
COL_TEXT = QColor("#ffffff")
COL_BG = QColor("#fafafa")


def compute_layout(snapshot: Dict, show_nil: bool):
    """回傳 (positions, nil_leaves)。
    positions: {id: (col, depth)}
    nil_leaves: [(parent_id, side, col, depth)]  （side: 'L'/'R'）
    """
    nodes = snapshot["nodes"]
    root = snapshot["root"]
    positions: Dict[int, Tuple[int, int]] = {}
    nil_leaves: List[Tuple[int, str, int, int]] = []
    col = [0]

    def walk(nid: Optional[int], depth: int, parent_id: Optional[int], side: Optional[str]):
        if nid is None:
            if show_nil and parent_id is not None:
                c = col[0]
                col[0] += 1
                nil_leaves.append((parent_id, side, c, depth))
            return
        n = nodes[nid]
        walk(n["left"], depth + 1, nid, "L")
        c = col[0]
        col[0] += 1
        positions[nid] = (c, depth)
        walk(n["right"], depth + 1, nid, "R")

    walk(root, 0, None, None)
    return positions, nil_leaves


class NodeItem(QGraphicsItem):
    R = 22

    def __init__(self, node_id: int, key, color: str):
        super().__init__()
        self.node_id = node_id
        self.key = key
        self.color = color
        self.tag = ""  # AVL 用：平衡因子徽章文字
        self.highlight = False
        self.scale_factor = 1.0
        self.setZValue(2)

    def boundingRect(self) -> QRectF:
        # 上方多留空間給平衡因子徽章
        r = self.R + 10
        return QRectF(-r, -r - 18, 2 * r, 2 * r + 18)

    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.R * self.scale_factor
        if r <= 0.5:
            return
        if self.highlight:
            pen = QPen(COL_HIGHLIGHT, 4)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), r + 5, r + 5)
        if self.color == "R":
            fill, border = COL_RED_FILL, COL_RED_BORDER
        elif self.color == "B":
            fill, border = COL_BLACK_FILL, COL_BLACK_BORDER
        elif self.color == "A":
            fill, border = COL_AVL_FILL, COL_AVL_BORDER
        else:
            fill, border = COL_NEUTRAL_FILL, COL_NEUTRAL_BORDER
        p.setBrush(QBrush(fill))
        p.setPen(QPen(border, 2))
        p.drawEllipse(QPointF(0, 0), r, r)
        if self.scale_factor > 0.6:
            p.setPen(QPen(COL_TEXT))
            f = QFont()
            f.setPointSize(12)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(-r, -r, 2 * r, 2 * r), Qt.AlignCenter, str(self.key))
            if self.tag:
                self._paint_badge(p, r)

    def _paint_badge(self, p: QPainter, r: float):
        bw, bh = 28.0, 16.0
        cy = -r - 11
        rect = QRectF(-bw / 2, cy - bh / 2, bw, bh)
        try:
            warn = abs(int(self.tag)) >= 2
        except ValueError:
            warn = False
        p.setBrush(QBrush(COL_BADGE_BG))
        p.setPen(QPen(COL_BADGE_WARN if warn else COL_BADGE_BORDER, 1.5 if warn else 1.0))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QPen(COL_BADGE_WARN if warn else COL_BADGE_TEXT))
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        p.setFont(f)
        p.drawText(rect, Qt.AlignCenter, self.tag)


class EdgeLayer(QGraphicsItem):
    """單一圖元，負責畫出所有的邊與（可選的）NIL 葉節點。"""

    def __init__(self, canvas: "TreeCanvas"):
        super().__init__()
        self.canvas = canvas
        self.setZValue(0)

    def boundingRect(self) -> QRectF:
        return self.canvas._content_rect()

    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.Antialiasing, True)
        pos = self.canvas.cur_pos
        # 邊
        p.setPen(QPen(COL_EDGE, 2.5))
        for parent_id, child_id in self.canvas.edges:
            if parent_id in pos and child_id in pos:
                p.drawLine(pos[parent_id], pos[child_id])
        # NIL 葉節點（只在靜止狀態畫）
        if self.canvas.show_nil and not self.canvas._animating:
            f = QFont()
            f.setPointSize(8)
            for (pid, side, x, y) in self.canvas.nil_draw:
                pp = pos.get(pid)
                cp = QPointF(x, y)
                if pp is not None:
                    p.setPen(QPen(COL_EDGE, 1.5, Qt.DashLine))
                    p.drawLine(pp, cp)
                p.setPen(QPen(COL_BLACK_BORDER, 1.5))
                p.setBrush(QBrush(COL_NIL_FILL))
                p.drawRect(QRectF(cp.x() - 11, cp.y() - 9, 22, 18))
                p.setPen(QPen(COL_TEXT))
                p.setFont(f)
                p.drawText(QRectF(cp.x() - 11, cp.y() - 9, 22, 18), Qt.AlignCenter, "NIL")


class TreeCanvas(QGraphicsView):
    XSP = 56  # 每一個中序欄位的水平間距
    YSP = 84  # 每一層的垂直間距

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setBackgroundBrush(QBrush(COL_BG))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.node_items: Dict[int, NodeItem] = {}
        self.cur_pos: Dict[int, QPointF] = {}
        self.edges: List[Tuple[int, int]] = []
        self.nil_draw: List[Tuple[int, str, float, float]] = []
        self.show_nil = False
        self._last_snap: Optional[Dict] = None

        self.edge_layer = EdgeLayer(self)
        self._scene.addItem(self.edge_layer)

        # 動畫狀態
        self._animating = False
        self._anim_t = 0.0
        self._duration_ms = 420
        self._frame_ms = 16
        self._start_pos: Dict[int, QPointF] = {}
        self._target_pos: Dict[int, QPointF] = {}
        self._start_scale: Dict[int, float] = {}
        self._target_scale: Dict[int, float] = {}
        self._to_remove: List[int] = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._on_anim_done = None

    # ------------------------------------------------------------------ #
    def _content_rect(self) -> QRectF:
        if not self.cur_pos:
            return QRectF(-200, -100, 400, 200)
        xs = [pt.x() for pt in self.cur_pos.values()]
        ys = [pt.y() for pt in self.cur_pos.values()]
        m = 80
        return QRectF(min(xs) - m, min(ys) - m, (max(xs) - min(xs)) + 2 * m,
                      (max(ys) - min(ys)) + 2 * m)

    def set_show_nil(self, value: bool):
        self.show_nil = value
        if self._last_snap is not None:
            self.render_snapshot(self._last_snap, animate=False)

    def set_duration(self, ms: int):
        self._duration_ms = max(60, ms)

    def clear_all(self):
        self._timer.stop()
        self._animating = False
        for it in self.node_items.values():
            self._scene.removeItem(it)
        self.node_items.clear()
        self.cur_pos.clear()
        self.edges = []
        self.nil_draw = []
        self._last_snap = None
        self.edge_layer.update()
        self._scene.setSceneRect(self._content_rect())

    # ------------------------------------------------------------------ #
    def _pixel_positions(self, snapshot: Dict):
        positions, nil_leaves = compute_layout(snapshot, self.show_nil)
        # 以整體置中
        if positions:
            cols = [c for (c, d) in positions.values()]
            min_c = min(cols)
        elif nil_leaves:
            min_c = min(c for (_, _, c, _) in nil_leaves)
        else:
            min_c = 0
        px: Dict[int, QPointF] = {}
        for nid, (c, d) in positions.items():
            px[nid] = QPointF((c - min_c) * self.XSP, d * self.YSP)
        nil_px = []
        for (pid, side, c, d) in nil_leaves:
            nil_px.append((pid, side, (c - min_c) * self.XSP, d * self.YSP))
        return px, nil_px

    def render_snapshot(self, snapshot: Dict, animate: bool = True):
        self._timer.stop()
        self._last_snap = snapshot
        target_px, nil_px = self._pixel_positions(snapshot)
        nodes = snapshot["nodes"]

        # 邊
        self.edges = []
        for nid, info in nodes.items():
            if info["left"] is not None:
                self.edges.append((nid, info["left"]))
            if info["right"] is not None:
                self.edges.append((nid, info["right"]))
        self.nil_draw = nil_px

        # 建立新節點、更新顏色 / 標籤 / 強調
        highlights = snapshot.get("highlights", set())
        for nid, info in nodes.items():
            it = self.node_items.get(nid)
            if it is None:
                it = NodeItem(nid, info["key"], info["color"])
                # 新節點從父節點位置「長出來」
                parent_id = info["parent"]
                if parent_id in self.cur_pos:
                    start = QPointF(self.cur_pos[parent_id])
                elif parent_id in target_px:
                    start = QPointF(target_px[parent_id])
                else:
                    start = QPointF(target_px.get(nid, QPointF(0, 0)))
                it.setPos(start)
                it.scale_factor = 0.01
                self.cur_pos[nid] = start
                self.node_items[nid] = it
                self._scene.addItem(it)
            it.key = info["key"]
            it.color = info["color"]
            it.tag = info.get("tag", "")
            it.highlight = nid in highlights
            it.update()

        # 找出要移除的節點（已不在快照裡）
        removed = [nid for nid in self.node_items if nid not in nodes]

        # 計算動畫起訖
        self._start_pos = {}
        self._target_pos = {}
        self._start_scale = {}
        self._target_scale = {}
        for nid, it in self.node_items.items():
            self._start_pos[nid] = QPointF(self.cur_pos.get(nid, it.pos()))
            self._start_scale[nid] = it.scale_factor
            if nid in target_px:
                self._target_pos[nid] = target_px[nid]
                self._target_scale[nid] = 1.0
            else:
                # 被移除：縮小並留在原處
                self._target_pos[nid] = self._start_pos[nid]
                self._target_scale[nid] = 0.01
        self._to_remove = removed

        if not animate or self._duration_ms <= 0:
            self._finalize()
            return

        self._animating = True
        self._anim_t = 0.0
        self._timer.start(self._frame_ms)

    def _ease(self, t: float) -> float:
        # ease-in-out
        return t * t * (3 - 2 * t)

    def _on_tick(self):
        self._anim_t += self._frame_ms / float(self._duration_ms)
        if self._anim_t >= 1.0:
            self._finalize()
            return
        e = self._ease(self._anim_t)
        for nid, it in self.node_items.items():
            sp = self._start_pos[nid]
            tp = self._target_pos[nid]
            x = sp.x() + (tp.x() - sp.x()) * e
            y = sp.y() + (tp.y() - sp.y()) * e
            pt = QPointF(x, y)
            it.setPos(pt)
            self.cur_pos[nid] = pt
            ss = self._start_scale[nid]
            ts = self._target_scale[nid]
            it.scale_factor = ss + (ts - ss) * e
            it.update()
        self.edge_layer.prepareGeometryChange()
        self.edge_layer.update()

    def _finalize(self):
        self._timer.stop()
        self._animating = False
        for nid in self._to_remove:
            it = self.node_items.pop(nid, None)
            if it is not None:
                self._scene.removeItem(it)
            self.cur_pos.pop(nid, None)
        for nid, it in self.node_items.items():
            tp = self._target_pos.get(nid, it.pos())
            it.setPos(tp)
            self.cur_pos[nid] = tp
            it.scale_factor = self._target_scale.get(nid, 1.0)
            it.update()
        self._scene.setSceneRect(self._content_rect())
        self.edge_layer.prepareGeometryChange()
        self.edge_layer.update()
        self._fit()
        if self._on_anim_done is not None:
            cb = self._on_anim_done
            self._on_anim_done = None
            cb()

    def _fit(self):
        rect = self._content_rect()
        self.fitInView(rect, Qt.KeepAspectRatio)
        # 避免過度放大：限制最大縮放
        if self.transform().m11() > 1.4:
            self.resetTransform()
            self.scale(1.4, 1.4)
            self.centerOn(rect.center())

    def fit_view(self):
        self._fit()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
