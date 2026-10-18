"""B+ 樹視覺化畫布:節點繪製、樹狀佈局、快照之間的補間動畫。

TreeCanvas.show_snapshot(snap) 會把畫面從「目前狀態」平滑動畫到
快照描述的狀態:既有節點移動、新節點淡入、消失節點淡出,
父子連線與葉節點鏈結箭頭每一格都跟著節點即時重算。
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView

CELL_W = 44      # 每個鍵一格的寬度
NODE_H = 38      # 節點高度
H_GAP = 30       # 葉節點之間的水平間距
LEVEL_H = 116    # 層與層的垂直距離
EDGE_OPACITY = 0.9

# 強調型別 → (填色, 邊框色, 邊框粗細)
NODE_STYLE = {
    "leaf":      ("#EDF5FF", "#86A8D8", 1.6),   # 一般葉節點
    "internal":  ("#FFFFFF", "#9AA4B2", 1.6),   # 一般內部節點
    "path":      ("#FFFBE8", "#D9B44A", 2.2),   # 已走過的路徑
    "visit":     ("#FFE9A8", "#DE9E00", 2.6),   # 目前走訪中
    "found":     ("#C9F2C7", "#3E9B4F", 2.8),   # 成功 / 找到
    "insert":    ("#C9F2C7", "#3E9B4F", 2.8),   # 剛插入
    "new":       ("#D8F6E3", "#27AE60", 2.8),   # 新建立的節點
    "overflow":  ("#FFD9D9", "#D64545", 3.0),   # 溢位
    "underflow": ("#FFD9D9", "#D64545", 3.0),   # 不足
    "split":     ("#ECDFFC", "#8E5FD3", 2.8),   # 分裂
    "merge":     ("#D8E7FF", "#3B6FD4", 2.8),   # 合併
    "lend":      ("#D3F2F1", "#2BA8A0", 2.8),   # 借出鍵的兄弟
    "borrow":    ("#D3F2F1", "#2BA8A0", 2.8),   # 借入鍵的節點
    "target":    ("#FFE7CC", "#E67E22", 2.8),   # 操作目標
    "error":     ("#FFD9D9", "#D64545", 2.8),   # 失敗
}

# 鍵格標記 → 底色
KEY_MARK = {
    "new": "#2ECC71",
    "found": "#2ECC71",
    "compare": "#F5A623",
    "removed": "#E74C3C",
    "moved": "#3498DB",
}

KEY_FONT = QFont("Consolas", 12)
KEY_FONT.setBold(True)


class NodeItem(QGraphicsItem):
    """一個 B+ 樹節點:圓角矩形 + 鍵格,依強調型別變色。"""

    def __init__(self):
        super().__init__()
        self.keys = []
        self.is_leaf = True
        self.highlight = None
        self.marks = {}
        self.setZValue(1)

    def width(self):
        return max(1, len(self.keys)) * CELL_W

    def set_content(self, keys, is_leaf, highlight, marks):
        self.prepareGeometryChange()
        self.keys = list(keys)
        self.is_leaf = is_leaf
        self.highlight = highlight
        self.marks = dict(marks or {})
        self.update()

    def boundingRect(self):
        w = self.width()
        return QRectF(-w / 2 - 3, -NODE_H / 2 - 3, w + 6, NODE_H + 6)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        rect = QRectF(-w / 2, -NODE_H / 2, w, NODE_H)

        if self.highlight in NODE_STYLE:
            fill, border, bw = NODE_STYLE[self.highlight]
        elif self.is_leaf:
            fill, border, bw = NODE_STYLE["leaf"]
        else:
            fill, border, bw = NODE_STYLE["internal"]

        pen = QPen(QColor(border), bw)
        if not self.keys:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(fill))
        painter.drawRoundedRect(rect, 8, 8)

        if not self.keys:
            painter.setPen(QColor("#8A94A6"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "空")
            return

        clip = QPainterPath()
        clip.addRoundedRect(rect, 8, 8)
        painter.save()
        painter.setClipPath(clip)
        for i in range(len(self.keys)):
            cell = QRectF(-w / 2 + i * CELL_W, -NODE_H / 2, CELL_W, NODE_H)
            mark = self.marks.get(i)
            if mark in KEY_MARK:
                c = QColor(KEY_MARK[mark])
                c.setAlpha(120)
                painter.fillRect(cell, c)
            if i > 0:
                sep = QColor(border)
                sep.setAlpha(110)
                painter.setPen(QPen(sep, 1))
                painter.drawLine(cell.topLeft(), cell.bottomLeft())
        painter.restore()

        painter.setPen(QColor("#1E2A3A"))
        painter.setFont(KEY_FONT)
        for i, k in enumerate(self.keys):
            cell = QRectF(-w / 2 + i * CELL_W, -NODE_H / 2, CELL_W, NODE_H)
            painter.drawText(cell, Qt.AlignmentFlag.AlignCenter, str(k))


def compute_layout(snap):
    """由快照計算每個節點的中心座標:葉節點由左到右排開,內部節點置中於子節點之上。"""
    nodes = snap["nodes"]
    pos = {}
    cursor = [0.0]

    def width_of(nid):
        return max(1, len(nodes[nid]["keys"])) * CELL_W

    def place(nid, depth):
        nd = nodes[nid]
        if not nd["children"]:
            w = width_of(nid)
            pos[nid] = QPointF(cursor[0] + w / 2, depth * LEVEL_H)
            cursor[0] += w + H_GAP
        else:
            for c in nd["children"]:
                place(c, depth + 1)
            x0 = pos[nd["children"][0]].x()
            x1 = pos[nd["children"][-1]].x()
            pos[nid] = QPointF((x0 + x1) / 2, depth * LEVEL_H)

    place(snap["root"], 0)

    minx = min(p.x() - width_of(nid) / 2 for nid, p in pos.items())
    maxx = max(p.x() + width_of(nid) / 2 for nid, p in pos.items())
    maxy = max(p.y() for p in pos.values())
    rect = QRectF(minx - 50, -70, (maxx - minx) + 100, maxy + 140)
    return pos, rect


class TreeCanvas(QGraphicsView):
    userZoomed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#FBFCFE"))
        self.auto_fit = True
        self.node_items = {}    # node_id -> NodeItem
        self.edge_items = {}    # (parent_id, child_id) -> QGraphicsPathItem
        self.link_items = {}    # (leaf_id, next_id)    -> QGraphicsPathItem(箭頭)
        self._edge_slot = {}    # (parent_id, child_id) -> 第幾個指標
        self._anim = None
        self._finalize = None   # 立即完成目前動畫的函式

    # ------------------------------------------------------------ 幾何
    def _edge_path(self, pid, cid):
        """父節點第 i 個指標位置(鍵與鍵之間)→ 子節點頂端,畫成 S 形曲線。"""
        p = self.node_items[pid]
        c = self.node_items[cid]
        slot = min(self._edge_slot.get((pid, cid), 0), len(p.keys))
        x0 = p.pos().x() - p.width() / 2 + slot * CELL_W
        y0 = p.pos().y() + NODE_H / 2
        x1 = c.pos().x()
        y1 = c.pos().y() - NODE_H / 2
        midy = (y0 + y1) / 2
        path = QPainterPath(QPointF(x0, y0))
        path.cubicTo(QPointF(x0, midy), QPointF(x1, midy), QPointF(x1, y1))
        return path

    def _link_path(self, a, b):
        """葉節點 a 右側 → 葉節點 b 左側 的水平箭頭(B+ 樹的鏈結串列)。"""
        ia = self.node_items[a]
        ib = self.node_items[b]
        x0 = ia.pos().x() + ia.width() / 2 + 3
        y0 = ia.pos().y()
        x1 = ib.pos().x() - ib.width() / 2 - 3
        y1 = ib.pos().y()
        path = QPainterPath(QPointF(x0, y0))
        path.lineTo(x1 - 6, y1)
        path.moveTo(x1, y1)
        path.lineTo(x1 - 8, y1 - 4.5)
        path.lineTo(x1 - 8, y1 + 4.5)
        path.closeSubpath()
        return path

    def _update_all_edges(self):
        for (pid, cid), it in self.edge_items.items():
            if pid in self.node_items and cid in self.node_items:
                it.setPath(self._edge_path(pid, cid))
        for (a, b), it in self.link_items.items():
            if a in self.node_items and b in self.node_items:
                it.setPath(self._link_path(a, b))

    # ------------------------------------------------------------ 主入口
    def show_snapshot(self, snap, duration=320, on_finished=None):
        """把畫面動畫到 snap 描述的狀態。duration<=0 代表立即套用。"""
        if self._finalize is not None:
            self._finalize()    # 先讓上一段動畫立刻收尾

        nodes = snap["nodes"]
        hl = snap["hl"]
        marks = snap["marks"]
        target_pos, target_rect = compute_layout(snap)
        scene = self.scene()

        cur_ids = set(self.node_items)
        tgt_ids = set(target_pos)
        new_ids = tgt_ids - cur_ids
        gone_ids = cur_ids - tgt_ids

        start_pos = {}
        for nid in tgt_ids:
            nd = nodes[nid]
            item = self.node_items.get(nid)
            if item is None:
                item = NodeItem()
                scene.addItem(item)
                self.node_items[nid] = item
                item.setPos(target_pos[nid])
                item.setOpacity(0.0)
            item.set_content(nd["keys"], nd["is_leaf"], hl.get(nid), marks.get(nid))
            start_pos[nid] = item.pos()

        # 目標的邊與葉鏈結
        tgt_edges = {}
        tgt_links = set()
        for nid, nd in nodes.items():
            for i, c in enumerate(nd["children"]):
                tgt_edges[(nid, c)] = i
            if nd["next"] is not None:
                tgt_links.add((nid, nd["next"]))
        self._edge_slot.update(tgt_edges)

        new_edges = []
        for key in tgt_edges:
            if key not in self.edge_items:
                it = QGraphicsPathItem()
                it.setPen(QPen(QColor("#A7B4C6"), 1.7))
                it.setZValue(-1)
                it.setOpacity(0.0)
                scene.addItem(it)
                self.edge_items[key] = it
                new_edges.append(key)
        gone_edges = [k for k in self.edge_items if k not in tgt_edges]

        new_links = []
        for key in tgt_links:
            if key not in self.link_items:
                it = QGraphicsPathItem()
                it.setPen(QPen(QColor("#6FA0CF"), 1.6))
                it.setBrush(QColor("#6FA0CF"))
                it.setZValue(-0.5)
                it.setOpacity(0.0)
                scene.addItem(it)
                self.link_items[key] = it
                new_links.append(key)
        gone_links = [k for k in self.link_items if k not in tgt_links]

        # 動畫期間,場景矩形先涵蓋新舊兩個範圍
        scene.setSceneRect(scene.itemsBoundingRect().united(target_rect))

        def apply(t):
            for nid in tgt_ids:
                sp, ep = start_pos[nid], target_pos[nid]
                self.node_items[nid].setPos(sp + (ep - sp) * t)
            for nid in new_ids:
                self.node_items[nid].setOpacity(t)
            for nid in gone_ids:
                it = self.node_items.get(nid)
                if it is not None:
                    it.setOpacity(1.0 - t)
            for k in new_edges:
                self.edge_items[k].setOpacity(t * EDGE_OPACITY)
            for k in gone_edges:
                if k in self.edge_items:
                    self.edge_items[k].setOpacity((1.0 - t) * EDGE_OPACITY)
            for k in new_links:
                self.link_items[k].setOpacity(t * EDGE_OPACITY)
            for k in gone_links:
                if k in self.link_items:
                    self.link_items[k].setOpacity((1.0 - t) * EDGE_OPACITY)
            self._update_all_edges()

        done = {"v": False}

        def finalize():
            if done["v"]:
                return
            done["v"] = True
            if self._anim is not None:
                anim, self._anim = self._anim, None
                anim.stop()
            apply(1.0)
            for nid in gone_ids:
                it = self.node_items.pop(nid, None)
                if it is not None:
                    scene.removeItem(it)
            for k in gone_edges:
                it = self.edge_items.pop(k, None)
                if it is not None:
                    scene.removeItem(it)
                self._edge_slot.pop(k, None)
            for k in gone_links:
                it = self.link_items.pop(k, None)
                if it is not None:
                    scene.removeItem(it)
            self._update_all_edges()
            scene.setSceneRect(target_rect)
            self._finalize = None
            if self.auto_fit:
                self.fit_all()
            if on_finished is not None:
                on_finished()

        self._finalize = finalize

        if duration <= 0:
            finalize()
            return

        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(int(duration))
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.valueChanged.connect(lambda v: apply(float(v)))
        anim.finished.connect(finalize)
        self._anim = anim
        anim.start()

    # ------------------------------------------------------------ 縮放
    def fit_all(self):
        rect = self.scene().sceneRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        m = self.transform().m11()
        if m > 1.1:                      # 不要放大超過 110%,小樹保持原尺寸
            self.scale(1.1 / m, 1.1 / m)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.userZoomed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_all()
