# -*- coding: utf-8 -*-
"""
DFS / BFS 路徑搜尋動畫（解說版）
=========================================
在「簡化版」基礎上加了這些東西：
  1. 固定地圖（不隨機）—— 每次打開都一樣，方便重複講解
  2. 「上一步」按鈕 —— 可以倒退，回去看剛剛那一步
  3. 每次單步都即時用白話解釋「為什麼走這一步」
     （從堆疊頂端／佇列前端拿了哪一格、為什麼是它、又把哪些鄰居放進清單）
  4. ★來源箭頭：每格都畫一支箭頭指向它的來源(parent)，連起來就是一棵樹
  5. ★回推動畫：找到終點後，從 G 沿著箭頭一路退回 S，金線一段一段長出來
  6. ★路徑容器：把回推撿回來的整條最短路，一格一格收集進一個盒子
  —— 這三樣就是要回答「為什麼這樣走，最後能找出這條路」

左 = DFS（堆疊 Stack / 後進先出）　右 = BFS（佇列 Queue / 先進先出）

需求： pip install PyQt6
執行： python dfs_bfs_pyqt6_step.py
"""

import sys
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QSlider,
    QHBoxLayout, QVBoxLayout, QButtonGroup, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF


# ---------- 版面與外觀 ----------
COLS, ROWS = 13, 9
CELL = 42

C_EMPTY    = QColor("#F1F5F9")
C_WALL     = QColor("#334155")
C_START    = QColor("#22C55E")
C_GOAL     = QColor("#EF4444")
C_VISITED  = QColor("#BAE6FD")
C_FRONTIER = QColor("#FCD34D")
C_CURRENT  = QColor("#FB923C")
C_PATH     = QColor("#FACC15")
C_PANEL_BG = QColor("#FFFFFF")
C_GRID_LINE= QColor("#E2E8F0")


def fmt(node):
    return f"({node[0]},{node[1]})"


# =========================================================
#  迷宮模型（固定地圖）
# =========================================================
class Maze:
    def __init__(self):
        self.grid = [[0] * COLS for _ in range(ROWS)]
        self.start = (1, ROWS // 2)
        self.goal  = (COLS - 2, ROWS // 2)
        self.generate_fixed()

    def is_wall(self, c, r):
        return self.grid[r][c] == 1

    def neighbors(self, node):
        c, r = node
        result = []
        for dc, dr in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nc, nr = c + dc, r + dr
            if 0 <= nc < COLS and 0 <= nr < ROWS and self.grid[nr][nc] == 0:
                result.append((nc, nr))
        return result

    def clear_walls(self):
        for r in range(ROWS):
            for c in range(COLS):
                self.grid[r][c] = 0

    def generate_fixed(self):
        """固定地圖：兩段牆製造岔路，方便觀察 DFS 為何提早轉彎。"""
        self.clear_walls()
        # 中間直牆，上方留缺口
        mid = COLS // 2
        for r in range(2, ROWS):
            self.grid[r][mid] = 1
        # 左側一小段橫牆，逼出一個岔路
        for c in range(3, 6):
            self.grid[3][c] = 1


# =========================================================
#  搜尋引擎（含「為何這一步」的解說）
# =========================================================
class Search:
    def __init__(self, maze, mode):
        self.maze = maze
        self.mode = mode
        self.reset()

    def reset(self):
        self.visited = set()
        self.parent = {}
        self.current = None
        self.frontier = deque([self.maze.start])
        self.in_frontier = {self.maze.start}
        self.done = False
        self.found = False
        self.path = []
        self.path_reveal = 0
        self.steps = 0
        self.backtrack = []
        self.backtrack_log = ""
        box = "堆疊" if self.mode == 'dfs' else "佇列"
        self.explain = f"起點 S{fmt(self.maze.start)} 已放進{box}，按「單步」開始。"

    def step(self):
        if self.done:
            return
        if not self.frontier:
            self.done = True
            self.found = False
            self.explain = "清單空了、沒有格子可拿 → 起點到終點沒有通路。"
            return

        if self.mode == 'dfs':
            node = self.frontier.pop()
            end_desc = "堆疊頂端（最後放進去的）"
            rule = "後進先出 LIFO"
        else:
            node = self.frontier.popleft()
            end_desc = "佇列前端（最早放進去的）"
            rule = "先進先出 FIFO"

        self.in_frontier.discard(node)
        self.current = node
        self.steps += 1

        if node in self.visited:
            self.explain = f"從{end_desc}取出 {fmt(node)}，但它已造訪過 → 直接跳過。"
            return
        self.visited.add(node)

        if node == self.maze.goal:
            self.done = True
            self.found = True
            self._build_path()
            self.explain = (f"從{end_desc}取出 {fmt(node)} —— 正是終點 G！"
                            f"接著沿 parent 來源簿回推路徑（座標見下方「回推紀錄」）。")
            return

        added = []
        for nb in self.maze.neighbors(node):
            if nb not in self.visited and nb not in self.in_frontier:
                self.parent[nb] = node
                self.frontier.append(nb)
                self.in_frontier.add(nb)
                added.append(nb)

        if added:
            self.explain = (f"依「{rule}」，從{end_desc}取出 {fmt(node)}；"
                            f"把它沒走過的鄰居 {'、'.join(fmt(n) for n in added)} 放進清單。")
        else:
            self.explain = (f"依「{rule}」，從{end_desc}取出 {fmt(node)}；"
                            f"四周沒有新格子（都走過或已在清單）→ 下一步會回到清單裡別的格子。")

    def _build_path(self):
        cur = self.maze.goal
        path = [cur]
        while cur in self.parent:
            cur = self.parent[cur]
            path.append(cur)
        path.reverse()
        self.path = path
        # 麵包屑回推紀錄：從終點沿 parent 一路撿回起點（終點 → 起點 順序）
        self.backtrack = list(reversed(path))
        self.backtrack_log = " ← ".join(fmt(n) for n in self.backtrack)
        name = "DFS" if self.mode == 'dfs' else "BFS"
        print(f"[{name}] 回推（終點→起點）：{self.backtrack_log}")
        print(f"[{name}] 路徑（起點→終點）：{' → '.join(fmt(n) for n in path)}")
        print(f"[{name}] 路徑長度：{len(path) - 1} 格、共 {len(path)} 個座標")


# =========================================================
#  迷宮繪圖元件
# =========================================================
class GridWidget(QWidget):
    def __init__(self, maze, search, main, accent):
        super().__init__()
        self.maze = maze
        self.search = search
        self.main = main
        self.accent = accent
        self.setFixedSize(COLS * CELL, ROWS * CELL)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_PANEL_BG)

        s = self.search
        # 回推動畫是「從 G 往回退」，所以已揭露的是路徑尾段（靠近 G 那幾格）
        reveal = s.path_reveal
        n = len(s.path)
        start_idx = n - reveal if (s.path and reveal) else n
        revealed = set(s.path[start_idx:]) if s.path else set()
        revealed_edges = set()
        for i in range(start_idx, n - 1):
            revealed_edges.add(frozenset((s.path[i], s.path[i + 1])))

        for r in range(ROWS):
            for c in range(COLS):
                node = (c, r)
                if self.maze.is_wall(c, r):
                    color = C_WALL
                elif node == self.maze.start:
                    color = C_START
                elif node == self.maze.goal:
                    color = C_GOAL
                elif node in revealed:
                    color = C_PATH
                elif node == s.current and not s.done:
                    color = C_CURRENT
                elif node in s.in_frontier:
                    color = C_FRONTIER
                elif node in s.visited:
                    color = C_VISITED
                else:
                    color = C_EMPTY

                rect = QRectF(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4)
                p.setPen(QPen(C_GRID_LINE, 1))
                p.setBrush(QBrush(color))
                p.drawRoundedRect(rect, 7, 7)

                if node == s.current and not s.done:
                    p.setPen(QPen(QColor("#9A3412"), 3))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRoundedRect(rect, 7, 7)

        # 【關鍵】來源箭頭：每個被探索到的格子，都記得自己是從哪一格來的(parent)。
        # 這些細灰箭頭連起來就是一棵「搜尋樹」——之所以最後能回推出路徑，靠的就是它。
        for node, par in s.parent.items():
            if frozenset((node, par)) in revealed_edges:
                continue                       # 這條邊改由下方粗色路徑呈現
            self._draw_came_from(p, node, par, QColor("#64748B"), 1.4)

        # 找到終點後，從 G 沿來源箭頭往回退；粗線一段一段往 S 生長
        if s.path and reveal >= 2:
            pts = [QPointF(c * CELL + CELL / 2, r * CELL + CELL / 2)
                   for (c, r) in s.path[start_idx:]]
            pen = QPen(self.accent, 5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))
            # 路徑上也用粗箭頭強調「方向是往回指向來源（最終指回 S）」
            for i in range(start_idx, n - 1):
                self._draw_came_from(p, s.path[i + 1], s.path[i], self.accent, 2.4)
            p.setPen(QPen(self.accent, 1))
            p.setBrush(QBrush(self.accent))
            for pt in (pts[0], pts[-1]):
                p.drawEllipse(pt, 5, 5)

        # 顯示格子座標（小字）方便對照解說
        p.setFont(QFont("Arial", 7))
        p.setPen(QPen(QColor("#94A3B8")))
        for r in range(ROWS):
            for c in range(COLS):
                if not self.maze.is_wall(c, r):
                    p.drawText(QRectF(c * CELL + 3, r * CELL + 1, CELL, 12),
                               Qt.AlignmentFlag.AlignLeft, f"{c},{r}")

        p.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        p.setPen(QPen(QColor("white")))
        for node, txt in ((self.maze.start, "S"), (self.maze.goal, "G")):
            c, r = node
            p.drawText(QRectF(c * CELL, r * CELL, CELL, CELL),
                       Qt.AlignmentFlag.AlignCenter, txt)
        p.end()

    def _draw_came_from(self, p, node, par, color, width):
        """在 node 上畫一支指向其 parent（來源）的小箭頭：『我是從那一格來的』。"""
        c, r = node
        pc, pr = par
        cx, cy = c * CELL + CELL / 2, r * CELL + CELL / 2
        dx, dy = pc - c, pr - r                    # 一定是上下左右其中一個方向
        x1, y1 = cx + dx * (CELL * 0.14), cy + dy * (CELL * 0.14)
        x2, y2 = cx + dx * (CELL * 0.42), cy + dy * (CELL * 0.42)
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        bx, by = -dx, -dy                          # 箭頭尖（指向 par）
        px, py = -dy, dx
        h = 4.0
        p.drawLine(QPointF(x2, y2), QPointF(x2 + (bx + px) * h, y2 + (by + py) * h))
        p.drawLine(QPointF(x2, y2), QPointF(x2 + (bx - px) * h, y2 + (by - py) * h))

    def _cell(self, ev):
        pos = ev.position()
        return int(pos.x()) // CELL, int(pos.y()) // CELL

    def mousePressEvent(self, ev):
        c, r = self._cell(ev)
        if 0 <= c < COLS and 0 <= r < ROWS:
            erase = ev.button() == Qt.MouseButton.RightButton
            self.main.start_stroke(c, r, erase)

    def mouseMoveEvent(self, ev):
        c, r = self._cell(ev)
        if 0 <= c < COLS and 0 <= r < ROWS:
            self.main.continue_stroke(c, r)


# =========================================================
#  Stack / Queue 內容繪圖
# =========================================================
class FrontierWidget(QWidget):
    BOX = 34
    MAX_SHOW = 11

    def __init__(self, search, accent):
        super().__init__()
        self.search = search
        self.accent = accent
        self.setFixedHeight(76)
        self.setMinimumWidth(COLS * CELL)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        items = list(self.search.frontier)
        is_stack = self.search.mode == 'dfs'
        active_index = len(items) - 1 if is_stack else 0

        p.setFont(QFont("Arial", 9))
        x, y = 4, 22
        shown = items[-self.MAX_SHOW:] if is_stack else items[:self.MAX_SHOW]
        offset = len(items) - len(shown) if is_stack else 0

        for i, node in enumerate(shown):
            is_active = (offset + i) == active_index
            rect = QRectF(x, y, self.BOX, self.BOX)
            p.setBrush(QBrush(C_CURRENT if is_active else C_FRONTIER))
            p.setPen(QPen(QColor("#92400E"), 2 if is_active else 1))
            p.drawRoundedRect(rect, 5, 5)
            p.setPen(QPen(QColor("#1F2937")))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{node[0]},{node[1]}")
            x += self.BOX + 5

        if len(items) > self.MAX_SHOW:
            p.setPen(QPen(QColor("#64748B")))
            p.drawText(QRectF(x, y, 44, self.BOX),
                       Qt.AlignmentFlag.AlignVCenter, f"… +{len(items) - self.MAX_SHOW}")

        p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        p.setPen(QPen(self.accent))
        label = "↑ 下一個取出（top）" if is_stack else "↑ 下一個取出（front）"
        ax = (x - self.BOX - 5) if is_stack else 4
        p.drawText(QRectF(max(4, ax), y + self.BOX + 3, 220, 18),
                   Qt.AlignmentFlag.AlignLeft, label)
        p.end()


# =========================================================
#  最短路徑「容器」繪圖
# =========================================================
class PathWidget(QWidget):
    """
    路徑容器：找到終點後，沿 parent 從終點回推得到的整條路，
    會一格一格『收集』進這個盒子（從 G 端往回填，對應回推方向）。
    每個方塊上排是座標、下排是它在路徑上的順序，方塊之間用箭頭連起來。
    """
    BOX_W = 44
    BOX_H = 38
    ARROW = 22
    PAD = 10
    TITLE_H = 24
    ROW_H = 56

    def __init__(self, search, accent):
        super().__init__()
        self.search = search
        self.accent = accent
        self.setMinimumWidth(COLS * CELL)
        self.setFixedHeight(self.TITLE_H + self.ROW_H + 2 * self.PAD)

    def _container_width(self):
        return COLS * CELL

    def _slots_per_row(self):
        usable = self._container_width() - 2 * self.PAD
        return max(1, int(usable // (self.BOX_W + self.ARROW)))

    def sync_height(self):
        s = self.search
        n = len(s.path) if (s.done and s.found and s.path) else 0
        per = self._slots_per_row()
        rows = max(1, (n + per - 1) // per) if n else 1
        h = self.TITLE_H + rows * self.ROW_H + 2 * self.PAD
        if self.height() != h:
            self.setFixedHeight(h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.search

        cw = self._container_width()
        x0 = max(0, (self.width() - cw) // 2)
        outer = QRectF(x0 + 2, 2, cw - 4, self.height() - 4)
        p.setBrush(QBrush(QColor("#FFFBEB")))
        p.setPen(QPen(self.accent, 2))
        p.drawRoundedRect(outer, 10, 10)

        found = s.done and s.found and bool(s.path)
        reveal = min(s.path_reveal, len(s.path)) if found else 0

        p.setFont(QFont("Microsoft JhengHei", 10, QFont.Weight.Bold))
        p.setPen(QPen(self.accent))
        if found:
            title = (f"🏁 路徑容器　|　長度 {len(s.path) - 1} 格　|　"
                     f"已收集 {reveal}/{len(s.path)}")
        else:
            title = "🏁 路徑容器（找到終點後，從 G 沿來源箭頭回推、一格一格收集進來）"
        p.drawText(QRectF(x0 + self.PAD, 5, cw - 2 * self.PAD, self.TITLE_H),
                   Qt.AlignmentFlag.AlignLeft, title)

        if not found:
            p.end()
            return

        per = self._slots_per_row()
        for i, node in enumerate(s.path):
            row, col = divmod(i, per)
            x = x0 + self.PAD + col * (self.BOX_W + self.ARROW)
            y = self.TITLE_H + self.PAD + row * self.ROW_H
            shown = i >= len(s.path) - reveal       # 從 G 端往回收集

            if i > 0:
                if col == 0:
                    p.setPen(QPen(self.accent))
                    p.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                    p.drawText(QRectF(x - 16, y, 14, self.BOX_H),
                               Qt.AlignmentFlag.AlignCenter, "↳")
                else:
                    ax = x - self.ARROW
                    midy = y + self.BOX_H / 2
                    p.setPen(QPen(self.accent if shown else QColor("#E2E8F0"), 2))
                    p.drawLine(QPointF(ax + 4, midy), QPointF(x - 4, midy))
                    p.drawLine(QPointF(x - 4, midy), QPointF(x - 9, midy - 4))
                    p.drawLine(QPointF(x - 4, midy), QPointF(x - 9, midy + 4))

            box = QRectF(x, y, self.BOX_W, self.BOX_H)
            if shown:
                if node == self.search.maze.start:
                    fill = C_START
                elif node == self.search.maze.goal:
                    fill = C_GOAL
                else:
                    fill = C_PATH
                p.setBrush(QBrush(fill))
                p.setPen(QPen(self.accent, 1.5))
                p.drawRoundedRect(box, 6, 6)
                p.setPen(QPen(QColor("#1F2937")))
                p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                p.drawText(QRectF(x, y + 2, self.BOX_W, 16),
                           Qt.AlignmentFlag.AlignCenter, f"{node[0]},{node[1]}")
                p.setFont(QFont("Arial", 7))
                p.setPen(QPen(QColor("#475569")))
                p.drawText(QRectF(x, y + 20, self.BOX_W, 13),
                           Qt.AlignmentFlag.AlignCenter, f"#{i}")
            else:
                pen = QPen(QColor("#CBD5E1"), 1.4)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(box, 6, 6)
        p.end()


# =========================================================
#  主視窗
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DFS / BFS 路徑搜尋動畫（解說版）")

        self.maze = Maze()
        self.dfs = Search(self.maze, 'dfs')
        self.bfs = Search(self.maze, 'bfs')

        self.edit_mode = 'wall'
        self.paint_value = 1
        self.frame = 0                      # 已前進的步數（供上一步倒退）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.reveal_timer = QTimer(self)    # 找到終點後，把路徑回推收集進容器的動畫
        self.reveal_timer.timeout.connect(self._tick_reveal)

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())
        root.addLayout(self._build_panels())
        root.addWidget(self._build_legend())

        why = QLabel(
            "🔑 為什麼這樣走能找到路：每格的細灰箭頭代表「我是從哪一格來的」(parent)，"
            "是把鄰居放進清單時順手記下的。這些箭頭組成一棵搜尋樹；找到 G 後，從 G 沿箭頭"
            "一路退回 S，就是這條路徑。下方容器的金/紫線＝沿箭頭回推出來的，不是另外算的。")
        why.setWordWrap(True)
        why.setStyleSheet(
            "font-size:12px;color:#1E3A8A;background:#EFF6FF;"
            "border:1px solid #BFDBFE;border-radius:6px;padding:7px;")
        root.addWidget(why)

        self.insight = QLabel()
        self.insight.setStyleSheet("font-size:13px;color:#0F172A;padding:2px;")
        root.addWidget(self.insight)

        self.backtrack_label = QLabel()
        self.backtrack_label.setWordWrap(True)
        self.backtrack_label.setStyleSheet(
            "font-size:12px;color:#0F172A;background:#FFFBEB;"
            "border:1px solid #FCD34D;border-radius:6px;padding:8px;")
        root.addWidget(self.backtrack_label)

        container = QWidget()
        container.setLayout(root)
        container.setStyleSheet("background:#F8FAFC;")

        # 包進可捲動區：多了容器後版面較高，小螢幕也不會被裁切
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        self.resize(1200, 940)
        self.setStyleSheet("""
            QPushButton {
                background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
                padding:7px 13px; font-size:13px; color:#0F172A;
            }
            QPushButton:hover { background:#EFF6FF; border-color:#60A5FA; }
            QPushButton:checked { background:#2563EB; color:white; border-color:#2563EB; }
            QLabel { color:#0F172A; }
        """)

    def _build_toolbar(self):
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
        btn_reset.clicked.connect(self.reset_search)
        bar.addWidget(btn_reset)

        btn_map = QPushButton("🧱 還原地圖")
        btn_map.clicked.connect(self.restore_map)
        bar.addWidget(btn_map)

        btn_clear = QPushButton("🧹 清空牆壁")
        btn_clear.clicked.connect(self.clear_walls)
        bar.addWidget(btn_clear)

        bar.addSpacing(12)
        bar.addWidget(QLabel("編輯："))
        self.mode_group = QButtonGroup(self)
        for key, text in (('wall', '牆壁'), ('start', '起點'), ('goal', '終點')):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(key == 'wall')
            b.clicked.connect(lambda _, k=key: self.set_edit_mode(k))
            self.mode_group.addButton(b)
            bar.addWidget(b)

        bar.addStretch(1)
        bar.addWidget(QLabel("速度（慢→快）："))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(20)
        self.speed.setFixedWidth(150)
        self.speed.valueChanged.connect(self.update_speed)
        bar.addWidget(self.speed)
        return bar

    def _panel(self, title, subtitle, accent, search):
        box = QVBoxLayout()
        box.setSpacing(4)
        head = QLabel(title)
        head.setStyleSheet(f"font-size:17px;font-weight:bold;color:{accent.name()};")
        sub = QLabel(subtitle)
        sub.setStyleSheet("font-size:12px;color:#475569;")
        box.addWidget(head)
        box.addWidget(sub)

        grid = GridWidget(self.maze, search, self, accent)
        box.addWidget(grid)

        stats = QLabel()
        stats.setStyleSheet("font-size:12px;color:#0F172A;padding:2px;")
        box.addWidget(stats)

        front = FrontierWidget(search, accent)
        box.addWidget(front)

        explain = QLabel()
        explain.setWordWrap(True)
        explain.setFixedWidth(COLS * CELL)
        explain.setMinimumHeight(54)
        explain.setStyleSheet(
            f"font-size:13px;color:#0F172A;background:#FFFFFF;"
            f"border:1px solid #E2E8F0;border-left:4px solid {accent.name()};"
            f"border-radius:6px;padding:8px;")
        box.addWidget(explain)

        path_box = PathWidget(search, accent)
        box.addWidget(path_box)

        wrapper = QWidget()
        wrapper.setLayout(box)
        return wrapper, grid, stats, front, explain, path_box

    def _build_panels(self):
        row = QHBoxLayout()
        row.setSpacing(20)
        left, self.dfs_grid, self.dfs_stats, self.dfs_front, self.dfs_explain, self.dfs_path_box = \
            self._panel("DFS 深度優先", "堆疊 Stack（後進先出 LIFO）",
                        QColor("#7C3AED"), self.dfs)
        right, self.bfs_grid, self.bfs_stats, self.bfs_front, self.bfs_explain, self.bfs_path_box = \
            self._panel("BFS 廣度優先", "佇列 Queue（先進先出 FIFO）",
                        QColor("#0891B2"), self.bfs)
        row.addStretch(1)
        row.addWidget(left)
        row.addWidget(right)
        row.addStretch(1)
        return row

    def _build_legend(self):
        frame = QFrame()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 4, 0)
        items = [
            (C_START, "起點 S"), (C_GOAL, "終點 G"), (C_WALL, "牆壁"),
            (C_FRONTIER, "待造訪"), (C_CURRENT, "處理中"),
            (C_VISITED, "已造訪"), (C_PATH, "最終路徑"),
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
            lay.addSpacing(10)
        lay.addStretch(1)
        return frame

    # ---------- 編輯 ----------
    def set_edit_mode(self, mode):
        self.edit_mode = mode

    def start_stroke(self, c, r, erase):
        node = (c, r)
        if self.edit_mode == 'wall':
            if node in (self.maze.start, self.maze.goal):
                return
            self.paint_value = 0 if erase else (0 if self.maze.is_wall(c, r) else 1)
            self.maze.grid[r][c] = self.paint_value
        elif self.edit_mode == 'start':
            if not self.maze.is_wall(c, r) and node != self.maze.goal:
                self.maze.start = node
        elif self.edit_mode == 'goal':
            if not self.maze.is_wall(c, r) and node != self.maze.start:
                self.maze.goal = node
        self.reset_search()

    def continue_stroke(self, c, r):
        if self.edit_mode != 'wall':
            return
        if (c, r) in (self.maze.start, self.maze.goal):
            return
        if self.maze.grid[r][c] != self.paint_value:
            self.maze.grid[r][c] = self.paint_value
            self.reset_search()

    # ---------- 播放 / 速度 ----------
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ 自動播放")
        else:
            if self.dfs.done and self.bfs.done:
                self.reset_search()          # 已跑完 → 重新播放從頭來
            self.update_speed()
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def update_speed(self):
        self.timer.setInterval(int(520 - 5 * self.speed.value()))

    # ---------- 「回推收集路徑」動畫 ----------
    def _needs_reveal(self, s):
        return s.done and s.found and s.path and s.path_reveal < len(s.path)

    def _maybe_animate_reveal(self):
        """任一邊剛找到終點時，啟動把路徑回推收集進容器的動畫。"""
        if any(self._needs_reveal(s) for s in (self.dfs, self.bfs)):
            if not self.reveal_timer.isActive():
                self.reveal_timer.start(110)

    def _tick_reveal(self):
        busy = False
        for s in (self.dfs, self.bfs):
            if self._needs_reveal(s):
                s.path_reveal += 1
                busy = True
                self._narrate_backtrack(s)
        self.refresh()
        if not busy:
            self.reveal_timer.stop()

    def _narrate_backtrack(self, s):
        """回推時，更新該搜尋的解說（逐格說明從哪退回哪）。"""
        n, k = len(s.path), s.path_reveal
        head = s.path[n - k]                       # 本次新退回到的格子（往 S 方向）
        name = "DFS" if s.mode == 'dfs' else "BFS"
        if k == 1:
            s.explain = (f"找到終點 G{fmt(head)}！開始回推：每格都記得自己從哪來"
                         f"（灰箭頭）。從 G 沿箭頭往回走，就能還原路徑。")
        elif k < n:
            child = s.path[n - k + 1]
            s.explain = (f"回推第 {k-1} 步：從 {fmt(child)} 沿來源箭頭退回 {fmt(head)}"
                         f"（即 parent[{child[0]},{child[1]}]={fmt(head)}）。{name} 的線又往 S 長一段。")
        else:
            s.explain = (f"退回起點 S{fmt(head)}，回推完成！這條線就是 {name} 找到的路徑，"
                         f"長度 {len(s.path)-1} 格。")

    # ---------- 前進一格 ----------
    def advance_one(self):
        """讓兩邊各前進一格（或揭露一段路徑）。回傳是否真的有動作。"""
        advancing = False
        for s in (self.dfs, self.bfs):
            if not s.done:
                s.step()
                advancing = True
        if advancing:
            self.frame += 1
        return advancing

    def tick(self):
        if not self.advance_one():
            self.timer.stop()
            self.btn_play.setText("▶ 重新播放")
        self.refresh()
        self._maybe_animate_reveal()

    def step_once(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ 自動播放")
        self.advance_one()
        self.refresh()
        self._maybe_animate_reveal()

    def step_back(self):
        """倒退一步：重設後重播到 frame-1。"""
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ 自動播放")
        if self.frame > 0:
            self.rebuild_to(self.frame - 1)

    def rebuild_to(self, n):
        self.reveal_timer.stop()
        self.dfs.reset()
        self.bfs.reset()
        self.frame = 0
        for _ in range(n):
            if not self.advance_one():
                break
        # 倒退/快轉時直接顯示完整路徑，不重播收集動畫
        for s in (self.dfs, self.bfs):
            if s.done and s.found and s.path:
                s.path_reveal = len(s.path)
        self.refresh()

    # ---------- 地圖操作 ----------
    def reset_search(self):
        self.timer.stop()
        self.reveal_timer.stop()
        self.btn_play.setText("▶ 自動播放")
        self.frame = 0
        self.dfs.reset()
        self.bfs.reset()
        self.refresh()

    def restore_map(self):
        self.maze.generate_fixed()
        self.maze.start = (1, ROWS // 2)
        self.maze.goal = (COLS - 2, ROWS // 2)
        self.reset_search()

    def clear_walls(self):
        self.maze.clear_walls()
        self.reset_search()

    # ---------- 更新顯示 ----------
    def refresh(self):
        self.dfs_grid.update()
        self.bfs_grid.update()
        self.dfs_front.update()
        self.bfs_front.update()
        for pb in (self.dfs_path_box, self.bfs_path_box):
            pb.sync_height()
            pb.update()
        self.dfs_stats.setText(self._stats_text(self.dfs))
        self.bfs_stats.setText(self._stats_text(self.bfs))
        self.dfs_explain.setText("為何走這一步：" + self.dfs.explain)
        self.bfs_explain.setText("為何走這一步：" + self.bfs.explain)
        self._update_insight()
        self.backtrack_label.setText(self._backtrack_text())

    def _stats_text(self, s):
        if not s.done:
            status = "搜尋中…"
        elif s.found:
            status = "✅ 找到終點"
        else:
            status = "❌ 無路徑"
        path_len = len(s.path) - 1 if s.path else 0
        return (f"狀態：{status}　|　步數：{s.steps}　|　"
                f"已造訪：{len(s.visited)}　|　路徑長度：{path_len}")

    def _backtrack_text(self):
        lines = []
        for s, name in ((self.dfs, "DFS"), (self.bfs, "BFS")):
            if s.done and s.found and s.backtrack:
                lines.append(f"{name} 沿麵包屑回推（終點 G → 起點 S）：{s.backtrack_log}")
        if not lines:
            return ("回推紀錄：找到終點後，這裡會列出沿 parent 來源簿"
                    "一路撿回起點的每一格座標。")
        return "\n".join(lines)

    def _update_insight(self):
        head = f"目前第 {self.frame} 步　|　用「上一步／單步」一格一格看，看左右兩邊的解說。"
        if self.dfs.done and self.bfs.done and self.bfs.found and self.dfs.found:
            dlen = len(self.dfs.path) - 1
            blen = len(self.bfs.path) - 1
            tail = f"　結果：BFS 路徑 {blen} 格（最短）、DFS 路徑 {dlen} 格。"
            self.insight.setText(head + tail)
        else:
            self.insight.setText(head)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
