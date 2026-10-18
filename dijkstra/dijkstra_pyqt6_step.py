# -*- coding: utf-8 -*-
"""
Dijkstra 最短路徑動畫（解說版）
=========================================
這支程式用 PyQt6 把「Dijkstra 最短路徑演算法」一步一步畫出來，
並且每走一步都用白話告訴你「為什麼是這一步」。

跟 DFS / BFS 最大的不同：
  ● DFS / BFS 的地圖每一格都一樣好走，求的是「最少格數」。
  ● Dijkstra 的地圖每一格有不同的「進入成本（地形權重）」，
    求的是「總成本最低」的路 —— 所以最短路不一定是格數最少的那條！

Dijkstra 的三個靈魂，本程式全部都看得到：
  1. dist[]      每一格「目前已知、從起點走過來的最短距離」（格子上的大數字）
  2. 優先佇列     一個 min-heap，永遠先拿出 dist 最小的格子（畫面下方那排方塊）
  3. relax 鬆弛   核心動作：發現「繞過某格更近」時，就更新它的 dist 與來源

操作：
  ⏮ 上一步 / ⏭ 單步 / ▶ 自動播放 / ↺ 重來 / 🧱 還原地圖 / 🧹 清空
  滑鼠可在地圖上畫牆、搬移起點終點、塗地形（讓某些格子變難走）

需求： pip install PyQt6
執行： python dijkstra_pyqt6_step.py
"""

import sys
import heapq

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QSlider,
    QHBoxLayout, QVBoxLayout, QButtonGroup, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF


# ---------- 版面與外觀 ----------
COLS, ROWS = 11, 8          # 地圖大小（格數）
CELL = 56                   # 每格像素（要夠大才放得下 dist 數字）
INF = float("inf")

C_EMPTY    = QColor("#F1F5F9")   # 還沒探索、且地形成本=1 的空地
C_WALL     = QColor("#334155")   # 牆（不能走）
C_START    = QColor("#22C55E")   # 起點 S
C_GOAL     = QColor("#EF4444")   # 終點 G
C_SETTLED  = QColor("#BAE6FD")   # 已確定（最短距離拍板定案）
C_FRONTIER = QColor("#FCD34D")   # 在優先佇列中（已碰到、但還沒確定）
C_CURRENT  = QColor("#FB923C")   # 這一步正在處理的格子
C_PATH     = QColor("#FACC15")   # 最終最短路徑
C_PANEL_BG = QColor("#FFFFFF")
C_GRID_LINE= QColor("#E2E8F0")

# 地形成本對應的按鈕：(權重, 顯示文字)
TERRAIN_CHOICES = [(1, "平地 1"), (3, "草地 3"), (6, "泥沼 6"), (9, "高山 9")]


def fmt(node):
    """把 (c, r) 印成 (c,r) 方便在解說裡引用某一格。"""
    return f"({node[0]},{node[1]})"


def terrain_tint(w):
    """地形權重 → 顏色。權重 1 = 很淺，越重越偏土黃/咖啡，讓人一眼看出哪裡難走。"""
    if w <= 1:
        return C_EMPTY
    t = min(1.0, (w - 1) / 8.0)          # 1..9 normalize 到 0..1
    # 從淺黃 (#FDE9A8) 漸層到深咖啡 (#B45309)
    r1, g1, b1 = 0xFD, 0xE9, 0xA8
    r2, g2, b2 = 0xB4, 0x53, 0x09
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return QColor(r, g, b)


# =========================================================
#  地圖模型（固定地圖 + 地形權重）
# =========================================================
class Maze:
    """
    grid[r][c]   : 0 = 可走、1 = 牆
    weight[r][c] : 「進入這一格」要付出的成本（>=1）。起點本身不算成本。
    """
    def __init__(self):
        self.grid = [[0] * COLS for _ in range(ROWS)]
        self.weight = [[1] * COLS for _ in range(ROWS)]
        self.start = (1, ROWS // 2)
        self.goal  = (COLS - 2, ROWS // 2)
        self.generate_fixed()

    def is_wall(self, c, r):
        return self.grid[r][c] == 1

    def weight_of(self, node):
        c, r = node
        return self.weight[r][c]

    def neighbors(self, node):
        """上下左右四方向、在界內、且不是牆的鄰居。"""
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

    def clear_terrain(self):
        for r in range(ROWS):
            for c in range(COLS):
                self.weight[r][c] = 1

    def generate_fixed(self):
        """
        固定地圖：中間放一大片『高成本沼澤』擋在起點與終點的直線上。
        直直走（格數最少）會踩過昂貴的沼澤；
        Dijkstra 會選擇『繞上方便宜平地』那條 —— 格數更多、但總成本更低。
        這正是 Dijkstra 與 BFS 的關鍵差異，一眼看穿。
        """
        self.clear_walls()
        self.clear_terrain()
        # 高成本沼澤：欄 4..7、列 2..7，權重 9（很貴）
        for r in range(2, ROWS):
            for c in range(4, 8):
                self.weight[r][c] = 9
        # 在右下放一小段牆，製造一點岔路（不會擋死上方便宜路線）
        for r in range(5, ROWS):
            self.grid[r][8] = 1


# =========================================================
#  Dijkstra 引擎（含「為何這一步」的解說）
# =========================================================
class Dijkstra:
    def __init__(self, maze):
        self.maze = maze
        self.reset()

    def reset(self):
        s = self.maze.start
        self.dist = {s: 0}          # 已知最短距離；沒出現的格子代表還是 ∞
        self.parent = {}            # 來源簿：parent[v] = u 代表「到 v 的最短路是經由 u」
        self.settled = set()        # 已『拍板定案』的格子（最短距離不會再變）
        self.counter = 0            # heap 的破平手序號，確保動畫可重現
        self.best_seq = {s: 0}      # 每個前緣格子最新一次入堆的序號（給佇列排序用）
        self.heap = [(0, 0, s)]     # 優先佇列：(dist, 序號, 格子)
        self.current = None
        self.done = False
        self.found = False
        self.path = []
        self.reveal = 0             # 已收集進「路徑容器」的格數（找到終點後做生長動畫）
        self.steps = 0
        self.stale_skipped = []     # 這一步略過了哪些『過期項目』
        self.explain = (f"起點 S{fmt(s)} 的 dist 設為 0、放進優先佇列；"
                        f"其它所有格子的 dist 都是 ∞（還沒走到）。按「單步」開始。")

    def dist_of(self, node):
        return self.dist.get(node, INF)

    def in_frontier(self, node):
        """在優先佇列中（已碰到但還沒拍板）= 有 dist 且尚未 settled。"""
        return node in self.dist and node not in self.settled

    def live_queue(self):
        """
        回傳『目前真正有效』的佇列內容，依（dist、入堆序號）由小到大排序。
        第一個就是下一步會被取出的格子。
        （heap 裡其實還有些過期的舊項目，這裡只顯示每格最新的有效狀態。）
        """
        live = [(self.dist[n], self.best_seq.get(n, 0), n)
                for n in self.dist if n not in self.settled]
        live.sort(key=lambda x: (x[0], x[1]))
        return live

    def step(self):
        """前進一步 = 從優先佇列拿出 dist 最小的一格，並鬆弛它的鄰居。"""
        if self.done:
            return
        self.stale_skipped = []

        # 1) 從 heap 取出 dist 最小者；途中略過『過期項目』（已 settled 又重複出現的）
        node = None
        while self.heap:
            d, _, u = heapq.heappop(self.heap)
            if u in self.settled:
                self.stale_skipped.append(u)        # 過期，跳過
                continue
            node, d = u, d
            break

        if node is None:
            # heap 空了還沒碰到終點 → 不可達
            self.done = True
            self.found = False
            self.explain = "優先佇列空了、沒有格子可拿 → 起點到終點之間沒有通路。"
            return

        # 2) 這一格 dist 最小，根據 Dijkstra 的貪婪性質，它的最短距離就此『拍板』
        self.settled.add(node)
        self.current = node
        self.steps += 1

        if node == self.maze.goal:
            self.done = True
            self.found = True
            self._build_path()
            self.explain = (f"取出 dist 最小的 {fmt(node)}（dist={d}）—— 正是終點 G！"
                            f"它的最短距離就此確定 = {d}。沿 parent 來源簿回推即得最短路（見下方）。")
            return

        # 3) 鬆弛（relax）每個鄰居：看看『繞過 node』會不會讓鄰居更近
        relaxed, skipped = [], []
        for v in self.maze.neighbors(node):
            if v in self.settled:
                continue
            w = self.maze.weight_of(v)
            nd = d + w                       # 經由 node 到 v 的新距離
            if nd < self.dist_of(v):
                old = self.dist_of(v)
                self.dist[v] = nd
                self.parent[v] = node
                self.counter += 1
                self.best_seq[v] = self.counter
                heapq.heappush(self.heap, (nd, self.counter, v))
                relaxed.append((v, old, nd, w))
            else:
                skipped.append((v, self.dist_of(v), nd, w))

        self.explain = self._explain(node, d, relaxed, skipped)

    def _explain(self, node, d, relaxed, skipped):
        """把這一步發生的事，逐項翻成白話。"""
        parts = []
        if self.stale_skipped:
            names = "、".join(fmt(n) for n in self.stale_skipped)
            parts.append(f"先略過 {len(self.stale_skipped)} 個過期項目（{names} 早已確定，"
                         f"是之前鬆弛時留下的舊副本）。")
        parts.append(f"取出 dist 最小的 {fmt(node)}（dist={d}）。貪婪關鍵：在所有「還沒確定」"
                     f"的格子裡它離 S 最近，不可能再被繞出更短的路，因此最短距離=「{d}」就此拍板。")
        for (v, old, nd, w) in relaxed:
            oldtxt = "∞" if old == INF else str(old)
            parts.append(f"鬆弛 {fmt(v)}：經由 {fmt(node)} 的距離 = {d}+{w}(進入成本) = {nd}，"
                         f"比原本 {oldtxt} 短 → 更新 dist[{v[0]},{v[1]}]={nd}，來源記成 {fmt(node)}。")
        for (v, cur, nd, w) in skipped:
            curtxt = "∞" if cur == INF else str(cur)
            parts.append(f"鬆弛 {fmt(v)}：經由 {fmt(node)} = {d}+{w} = {nd} ≥ 現有 {curtxt} → 不更新。")
        if not relaxed and not skipped:
            parts.append("四周沒有可鬆弛的鄰居（不是牆、就是早已確定）。")
        return "　".join(parts)

    def _build_path(self):
        """從終點沿 parent 一路回推到起點，組出最短路徑。"""
        cur = self.maze.goal
        path = [cur]
        while cur in self.parent:
            cur = self.parent[cur]
            path.append(cur)
        path.reverse()
        self.path = path
        total = self.dist_of(self.maze.goal)
        chain = " ← ".join(fmt(n) for n in reversed(path))
        print(f"[Dijkstra] 回推（終點→起點）：{chain}")
        print(f"[Dijkstra] 最短路（起點→終點）：{' → '.join(fmt(n) for n in path)}")
        print(f"[Dijkstra] 總成本 = {total}　（{len(path)-1} 步、{len(path)} 個座標）")


# =========================================================
#  地圖繪圖元件
# =========================================================
class GridWidget(QWidget):
    def __init__(self, maze, search, main):
        super().__init__()
        self.maze = maze
        self.search = search
        self.main = main
        self.setFixedSize(COLS * CELL, ROWS * CELL)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_PANEL_BG)

        s = self.search
        reveal = getattr(s, "reveal", 0)
        n = len(s.path)
        # 收集動畫是「從 G 往回退」，所以已揭露的是路徑的尾段（靠近 G 那幾格）
        start_idx = n - reveal if (s.path and reveal) else n
        revealed_path = set(s.path[start_idx:]) if s.path else set()
        revealed_edges = set()
        for i in range(start_idx, n - 1):
            revealed_edges.add(frozenset((s.path[i], s.path[i + 1])))

        for r in range(ROWS):
            for c in range(COLS):
                node = (c, r)
                w = self.maze.weight[r][c]

                # 決定底色（狀態優先；還沒探索的格子則用地形深淺呈現成本）
                if self.maze.is_wall(c, r):
                    color = C_WALL
                elif node == self.maze.start:
                    color = C_START
                elif node == self.maze.goal:
                    color = C_GOAL
                elif node in revealed_path:
                    color = C_PATH
                elif node == s.current and not s.done:
                    color = C_CURRENT
                elif node in s.settled:
                    color = C_SETTLED
                elif s.in_frontier(node):
                    color = C_FRONTIER
                else:
                    color = terrain_tint(w)

                rect = QRectF(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4)
                p.setPen(QPen(C_GRID_LINE, 1))
                p.setBrush(QBrush(color))
                p.drawRoundedRect(rect, 8, 8)

                if node == s.current and not s.done:
                    p.setPen(QPen(QColor("#9A3412"), 3))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRoundedRect(rect, 8, 8)

                if self.maze.is_wall(c, r):
                    continue

                # 左上角：座標（方便對照解說）
                p.setFont(QFont("Arial", 7))
                p.setPen(QPen(QColor("#64748B")))
                p.drawText(QRectF(c * CELL + 4, r * CELL + 2, CELL, 11),
                           Qt.AlignmentFlag.AlignLeft, f"{c},{r}")

                # 右上角：地形權重徽章（成本越高顏色越深）
                if w > 1:
                    badge = QRectF(c * CELL + CELL - 20, r * CELL + 4, 16, 14)
                    p.setBrush(QBrush(terrain_tint(w).darker(115)))
                    p.setPen(QPen(QColor("#78350F"), 1))
                    p.drawRoundedRect(badge, 4, 4)
                    p.setPen(QPen(QColor("#3F2400")))
                    p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                    p.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(w))

                # 正中央：dist（最短距離），這是 Dijkstra 的主角
                dist = s.dist_of(node)
                dtxt = "∞" if dist == INF else str(dist)
                p.setFont(QFont("Arial", 17, QFont.Weight.Bold))
                p.setPen(QPen(QColor("#0F172A")))
                p.drawText(QRectF(c * CELL, r * CELL + 8, CELL, CELL - 16),
                           Qt.AlignmentFlag.AlignCenter, dtxt)

                # 底部：S / G 小標籤
                if node == self.maze.start or node == self.maze.goal:
                    tag = "S" if node == self.maze.start else "G"
                    p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                    p.setPen(QPen(QColor("white")))
                    p.drawText(QRectF(c * CELL, r * CELL + CELL - 16, CELL, 14),
                               Qt.AlignmentFlag.AlignCenter, tag)

        # 【關鍵】來源箭頭：每個被探索到的格子，都記得自己是從哪一格來的（parent）。
        # 這些細藍箭頭連起來就是一棵「最短路徑樹」——之所以最後能回推出路徑，靠的就是它。
        for node, par in s.parent.items():
            if frozenset((node, par)) in revealed_edges:
                continue                       # 這條邊改由下方金色路徑呈現
            self._draw_came_from(p, node, par, QColor("#3B82F6"), 1.8)

        # 找到終點後，從 G 沿著來源箭頭往回退；金線一段一段往 S 生長
        if s.path and reveal >= 2:
            pts = [QPointF(c * CELL + CELL / 2, r * CELL + CELL / 2)
                   for (c, r) in s.path[start_idx:]]
            pen = QPen(QColor("#CA8A04"), 5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))
            # 金色路徑上也畫箭頭，強調「方向是往回指向來源（最終指回 S）」
            for i in range(start_idx, n - 1):
                self._draw_came_from(p, s.path[i + 1], s.path[i], QColor("#CA8A04"), 2.6)
            p.setBrush(QBrush(QColor("#CA8A04")))
            p.setPen(QPen(QColor("#CA8A04"), 1))
            for pt in (pts[0], pts[-1]):
                p.drawEllipse(pt, 5, 5)
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
        h = 5.0
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
#  優先佇列（min-heap）內容繪圖
# =========================================================
class QueueWidget(QWidget):
    BOX_W = 52
    BOX_H = 38
    MAX_SHOW = 11

    def __init__(self, search):
        super().__init__()
        self.search = search
        self.setFixedHeight(86)
        self.setMinimumWidth(COLS * CELL)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        live = self.search.live_queue()        # 已排序：第一個 = 下一個取出
        p.setFont(QFont("Arial", 8))
        x, y = 4, 8
        shown = live[:self.MAX_SHOW]

        for i, (d, _, node) in enumerate(shown):
            is_next = (i == 0)
            rect = QRectF(x, y, self.BOX_W, self.BOX_H)
            p.setBrush(QBrush(C_CURRENT if is_next else C_FRONTIER))
            p.setPen(QPen(QColor("#92400E"), 2 if is_next else 1))
            p.drawRoundedRect(rect, 6, 6)
            p.setPen(QPen(QColor("#1F2937")))
            p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            p.drawText(QRectF(x, y + 2, self.BOX_W, 16),
                       Qt.AlignmentFlag.AlignCenter, f"{node[0]},{node[1]}")
            p.setFont(QFont("Arial", 8))
            p.drawText(QRectF(x, y + 19, self.BOX_W, 14),
                       Qt.AlignmentFlag.AlignCenter, f"d={d}")
            x += self.BOX_W + 6

        if len(live) > self.MAX_SHOW:
            p.setPen(QPen(QColor("#64748B")))
            p.setFont(QFont("Arial", 9))
            p.drawText(QRectF(x, y, 60, self.BOX_H),
                       Qt.AlignmentFlag.AlignVCenter, f"… +{len(live) - self.MAX_SHOW}")

        # 提示：下一個取出的是 dist 最小者
        p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        p.setPen(QPen(QColor("#B45309")))
        if shown:
            p.drawText(QRectF(4, y + self.BOX_H + 4, 360, 18),
                       Qt.AlignmentFlag.AlignLeft, "↑ 下一個取出：dist 最小（min-heap 自動排序）")
        else:
            p.drawText(QRectF(4, y, 360, self.BOX_H),
                       Qt.AlignmentFlag.AlignVCenter, "（優先佇列已空）")
        p.end()


# =========================================================
#  最短路徑「容器」繪圖
# =========================================================
class PathWidget(QWidget):
    """
    最短路徑容器：找到終點後，沿 parent 從終點回推得到的整條路，
    會一格一格『收集』進這個盒子。每個方塊上排是座標、下排是到該格的累積 dist；
    方塊之間的箭頭標著 +進入成本，最後一格的 dist 就是總成本。
    """
    BOX_W = 50
    BOX_H = 42
    ARROW = 30           # 兩格之間箭頭區的寬度
    PAD = 12
    TITLE_H = 26
    ROW_H = 66           # 每一列（含箭頭與上下文字）的高度

    def __init__(self, search, maze):
        super().__init__()
        self.search = search
        self.maze = maze
        self.setMinimumWidth(COLS * CELL)
        self.setFixedHeight(self.TITLE_H + self.ROW_H + 2 * self.PAD)

    def _container_width(self):
        return COLS * CELL

    def _slots_per_row(self):
        usable = self._container_width() - 2 * self.PAD
        return max(1, int(usable // (self.BOX_W + self.ARROW)))

    def sync_height(self):
        """依路徑長度自動算需要幾列，調整容器高度（路徑越長，盒子越高）。"""
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

        # 容器外框（像一個托盤）
        cw = self._container_width()
        x0 = max(0, (self.width() - cw) // 2)
        outer = QRectF(x0 + 2, 2, cw - 4, self.height() - 4)
        p.setBrush(QBrush(QColor("#FFFBEB")))
        p.setPen(QPen(QColor("#F59E0B"), 2))
        p.drawRoundedRect(outer, 12, 12)

        found = s.done and s.found and bool(s.path)
        reveal = min(getattr(s, "reveal", 0), len(s.path)) if found else 0

        # 標題（含總成本與收集進度）
        p.setFont(QFont("Microsoft JhengHei", 11, QFont.Weight.Bold))
        p.setPen(QPen(QColor("#92400E")))
        if found:
            total = s.dist_of(self.maze.goal)
            title = (f"🏁 最短路徑容器　|　總成本 dist[G] = {total}　|　"
                     f"{len(s.path) - 1} 步　|　已收集 {reveal}/{len(s.path)} 格")
        else:
            title = "🏁 最短路徑容器　（找到終點後，沿 parent 從終點回推，一格一格收集進來）"
        p.drawText(QRectF(x0 + self.PAD, 6, cw - 2 * self.PAD, self.TITLE_H),
                   Qt.AlignmentFlag.AlignLeft, title)

        if not found:
            p.end()
            return

        per = self._slots_per_row()
        for i, node in enumerate(s.path):
            row, col = divmod(i, per)
            x = x0 + self.PAD + col * (self.BOX_W + self.ARROW)
            y = self.TITLE_H + self.PAD + row * self.ROW_H
            shown = i >= len(s.path) - reveal       # 從 G 端往回收集（對應回推方向）

            # 與前一格之間的連接（同列畫箭頭 + 邊成本；換列畫折返符號）
            if i > 0:
                w = self.maze.weight_of(node)
                if col == 0:
                    p.setPen(QPen(QColor("#B45309")))
                    p.setFont(QFont("Arial", 13, QFont.Weight.Bold))
                    p.drawText(QRectF(x - 18, y, 16, self.BOX_H),
                               Qt.AlignmentFlag.AlignCenter, "↳")
                else:
                    ax = x - self.ARROW
                    midy = y + self.BOX_H / 2
                    p.setPen(QPen(QColor("#B45309") if shown else QColor("#E5D5B0"), 2))
                    p.drawLine(QPointF(ax + 5, midy), QPointF(x - 4, midy))
                    p.drawLine(QPointF(x - 4, midy), QPointF(x - 10, midy - 4))
                    p.drawLine(QPointF(x - 4, midy), QPointF(x - 10, midy + 4))
                    if shown:
                        p.setPen(QPen(QColor("#92400E")))
                        p.setFont(QFont("Arial", 8))
                        p.drawText(QRectF(ax, y - 1, self.ARROW, 13),
                                   Qt.AlignmentFlag.AlignCenter, f"+{w}")

            box = QRectF(x, y, self.BOX_W, self.BOX_H)
            if shown:
                if node == self.maze.start:
                    fill = C_START
                elif node == self.maze.goal:
                    fill = C_GOAL
                else:
                    fill = C_PATH
                p.setBrush(QBrush(fill))
                p.setPen(QPen(QColor("#A16207"), 1.5))
                p.drawRoundedRect(box, 7, 7)
                p.setPen(QPen(QColor("#0F172A")))
                p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                p.drawText(QRectF(x, y + 3, self.BOX_W, 16),
                           Qt.AlignmentFlag.AlignCenter, f"{node[0]},{node[1]}")
                p.setFont(QFont("Arial", 8))
                p.drawText(QRectF(x, y + 22, self.BOX_W, 14),
                           Qt.AlignmentFlag.AlignCenter, f"d={s.dist_of(node)}")
            else:
                pen = QPen(QColor("#D6C28A"), 1.5)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(box, 7, 7)
        p.end()


# =========================================================
#  主視窗
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijkstra 最短路徑動畫（解說版）")

        self.maze = Maze()
        self.search = Dijkstra(self.maze)

        self.edit_mode = 'wall'
        self.terrain_value = 6          # 「地形」筆刷目前的權重
        self.paint_value = 1            # 牆筆刷：要塗成牆(1)還是清除(0)
        self.frame = 0                  # 已前進的步數（供上一步倒退）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.reveal_timer = QTimer(self)        # 找到終點後，把路徑一格一格收集進容器的動畫
        self.reveal_timer.timeout.connect(self._tick_reveal)

        self._build_ui()
        self.refresh()

    # ---------- 介面組裝 ----------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())
        root.addLayout(self._build_edit_bar())

        title = QLabel("Dijkstra 最短路徑　—　求「總成本最低」的路（不是格數最少）")
        title.setStyleSheet("font-size:17px;font-weight:bold;color:#1D4ED8;")
        root.addWidget(title)

        # 地圖置中
        grid_row = QHBoxLayout()
        grid_row.addStretch(1)
        self.grid = GridWidget(self.maze, self.search, self)
        grid_row.addWidget(self.grid)
        grid_row.addStretch(1)
        root.addLayout(grid_row)

        why = QLabel(
            "🔑 為什麼這樣走能找到路：每格的細藍箭頭代表「我是從哪一格來的」(parent)，"
            "是鬆弛時順手記下的。這些箭頭組成一棵樹；找到 G 後，從 G 沿箭頭一路退回 S，"
            "就是最短路徑。最後的金線＝沿著這些箭頭回推出來的，不是另外算的。")
        why.setWordWrap(True)
        why.setStyleSheet(
            "font-size:12px;color:#1E3A8A;background:#EFF6FF;"
            "border:1px solid #BFDBFE;border-radius:6px;padding:7px;")
        root.addWidget(why)

        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:13px;color:#0F172A;padding:2px;")
        root.addWidget(self.stats)

        self.queue = QueueWidget(self.search)
        root.addWidget(self.queue)

        root.addWidget(self._build_legend())

        self.explain = QLabel()
        self.explain.setWordWrap(True)
        self.explain.setMinimumHeight(72)
        self.explain.setStyleSheet(
            "font-size:13px;color:#0F172A;background:#FFFFFF;"
            "border:1px solid #E2E8F0;border-left:4px solid #2563EB;"
            "border-radius:6px;padding:8px;")
        root.addWidget(self.explain)

        # 最短路徑「容器」：找到終點後一格一格收集進來
        self.path_box = PathWidget(self.search, self.maze)
        root.addWidget(self.path_box)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size:12px;color:#475569;padding:2px;")
        root.addWidget(self.path_label)

        container = QWidget()
        container.setLayout(root)
        container.setStyleSheet("background:#F8FAFC;")

        # 包進可捲動區：元素多、視窗較高時，小螢幕也不會被裁切
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        self.resize(740, 900)
        self.setStyleSheet("""
            QPushButton {
                background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
                padding:7px 12px; font-size:13px; color:#0F172A;
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

        btn_clear = QPushButton("🧹 清空")
        btn_clear.clicked.connect(self.clear_all)
        bar.addWidget(btn_clear)

        bar.addStretch(1)
        bar.addWidget(QLabel("速度（慢→快）："))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(25)
        self.speed.setFixedWidth(150)
        self.speed.valueChanged.connect(self.update_speed)
        bar.addWidget(self.speed)
        return bar

    def _build_edit_bar(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("滑鼠編輯："))

        self.mode_group = QButtonGroup(self)
        for key, text in (('wall', '🧱 牆壁'), ('start', '🟢 起點'),
                          ('goal', '🔴 終點'), ('terrain', '⛰ 地形')):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(key == 'wall')
            b.clicked.connect(lambda _, k=key: self.set_edit_mode(k))
            self.mode_group.addButton(b)
            bar.addWidget(b)

        bar.addSpacing(14)
        bar.addWidget(QLabel("地形成本："))
        self.terrain_group = QButtonGroup(self)
        for w, text in TERRAIN_CHOICES:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(w == self.terrain_value)
            b.clicked.connect(lambda _, ww=w: self.set_terrain_value(ww))
            self.terrain_group.addButton(b)
            bar.addWidget(b)

        bar.addStretch(1)
        hint = QLabel("左鍵塗、右鍵清除；改地圖會自動從頭算")
        hint.setStyleSheet("font-size:11px;color:#64748B;")
        bar.addWidget(hint)
        return bar

    def _build_legend(self):
        frame = QFrame()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 4, 0)
        items = [
            (C_START, "起點 S"), (C_GOAL, "終點 G"), (C_WALL, "牆"),
            (C_FRONTIER, "在佇列"), (C_CURRENT, "處理中"),
            (C_SETTLED, "已確定"), (C_PATH, "最短路"),
            (terrain_tint(9), "高成本地形"),
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
            lay.addSpacing(8)
        lay.addStretch(1)
        return frame

    # ---------- 滑鼠編輯 ----------
    def set_edit_mode(self, mode):
        self.edit_mode = mode

    def set_terrain_value(self, w):
        self.terrain_value = w
        self.edit_mode = 'terrain'
        # 同步把「地形」編輯鈕也點亮
        for b in self.mode_group.buttons():
            b.setChecked(b.text().endswith('地形'))

    def start_stroke(self, c, r, erase):
        node = (c, r)
        if self.edit_mode == 'wall':
            if node in (self.maze.start, self.maze.goal):
                return
            self.paint_value = 0 if erase else (0 if self.maze.is_wall(c, r) else 1)
            self.maze.grid[r][c] = self.paint_value
        elif self.edit_mode == 'terrain':
            if not self.maze.is_wall(c, r):
                self.maze.weight[r][c] = 1 if erase else self.terrain_value
        elif self.edit_mode == 'start':
            if not self.maze.is_wall(c, r) and node != self.maze.goal:
                self.maze.start = node
        elif self.edit_mode == 'goal':
            if not self.maze.is_wall(c, r) and node != self.maze.start:
                self.maze.goal = node
        self.reset_search()

    def continue_stroke(self, c, r):
        node = (c, r)
        if self.edit_mode == 'wall':
            if node in (self.maze.start, self.maze.goal):
                return
            if self.maze.grid[r][c] != self.paint_value:
                self.maze.grid[r][c] = self.paint_value
                self.reset_search()
        elif self.edit_mode == 'terrain':
            if not self.maze.is_wall(c, r) and self.maze.weight[r][c] != self.terrain_value:
                self.maze.weight[r][c] = self.terrain_value
                self.reset_search()

    # ---------- 播放 / 速度 ----------
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ 自動播放")
        else:
            if self.search.done:
                self.reset_search()
            self.update_speed()
            self.timer.start()
            self.btn_play.setText("⏸ 暫停")

    def update_speed(self):
        self.timer.setInterval(int(620 - 6 * self.speed.value()))

    # ---------- 「收集路徑進容器」動畫 ----------
    def _maybe_animate_reveal(self):
        """剛找到終點時，啟動把路徑一格一格收集進容器的動畫。"""
        s = self.search
        if s.done and s.found and s.path and s.reveal < len(s.path):
            if not self.reveal_timer.isActive():
                self.reveal_timer.start(110)

    def _tick_reveal(self):
        s = self.search
        if s.done and s.found and s.path and s.reveal < len(s.path):
            s.reveal += 1
            n, k = len(s.path), s.reveal
            head = s.path[n - k]                 # 本次新退回到的格子（往 S 方向）
            if k == 1:
                s.explain = (f"找到終點 G{fmt(head)}！開始『回推』：每格都記得自己從哪來"
                             f"（藍箭頭）。我們從 G 出發，沿箭頭往回走，就能還原最短路。")
            elif k < n:
                child = s.path[n - k + 1]
                w = self.maze.weight_of(child)
                s.explain = (f"回推第 {k-1} 步：從 {fmt(child)} 沿來源箭頭退回 {fmt(head)}"
                             f"（即 parent[{child[0]},{child[1]}]={fmt(head)}，這段成本 {w}）。"
                             f"金線又往 S 長一段。")
            else:
                s.explain = (f"退回起點 S{fmt(head)}，回推完成！整條金線就是最短路徑，"
                             f"總成本 = dist[G] = {s.dist_of(self.maze.goal)}。")
            self.refresh()
        else:
            self.reveal_timer.stop()

    # ---------- 前進 / 倒退 ----------
    def advance_one(self):
        if self.search.done:
            return False
        self.search.step()
        self.frame += 1
        return True

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
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ 自動播放")
        if self.frame > 0:
            self.rebuild_to(self.frame - 1)

    def rebuild_to(self, n):
        """倒退做法：整個重設後，再快轉前進 n 步（因為 heap 用序號破平手，可完全重現）。"""
        self.reveal_timer.stop()
        self.search.reset()
        self.frame = 0
        for _ in range(n):
            if not self.advance_one():
                break
        # 倒退/快轉時直接顯示完整路徑，不重播收集動畫
        if self.search.done and self.search.found and self.search.path:
            self.search.reveal = len(self.search.path)
        self.refresh()

    # ---------- 地圖操作 ----------
    def reset_search(self):
        self.timer.stop()
        self.reveal_timer.stop()
        self.btn_play.setText("▶ 自動播放")
        self.frame = 0
        self.search.reset()
        self.refresh()

    def restore_map(self):
        self.maze.generate_fixed()
        self.maze.start = (1, ROWS // 2)
        self.maze.goal = (COLS - 2, ROWS // 2)
        self.reset_search()

    def clear_all(self):
        self.maze.clear_walls()
        self.maze.clear_terrain()
        self.reset_search()

    # ---------- 更新顯示 ----------
    def refresh(self):
        self.grid.update()
        self.queue.update()
        self.path_box.sync_height()
        self.path_box.update()
        self.stats.setText(self._stats_text())
        self.explain.setText("為何走這一步：" + self.search.explain)
        self.path_label.setText(self._path_text())

    def _stats_text(self):
        s = self.search
        if not s.done:
            status = "搜尋中…"
        elif s.found:
            status = "✅ 找到終點"
        else:
            status = "❌ 無路徑"
        goal_d = s.dist_of(self.maze.goal)
        goal_txt = "∞" if goal_d == INF else str(goal_d)
        return (f"第 {self.frame} 步　|　狀態：{status}　|　已確定格數：{len(s.settled)}　|　"
                f"佇列中：{len(s.live_queue())}　|　目前 dist[G] = {goal_txt}")

    def _path_text(self):
        s = self.search
        if s.done and s.found and s.path:
            back = " ← ".join(fmt(n) for n in reversed(s.path))
            return f"沿 parent 來源簿回推（終點 G → 起點 S）：{back}"
        return ("提示：按「單步」配合上方解說一格一格看；找到終點後，"
                "最短路徑會沿 parent 回推、一格一格收集進上面的容器。")


def main():
    # 讓 print() 的中文/箭號在任何 Windows 主控台編碼下都不會出錯
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = QApplication(sys.argv)
    font = QFont("Microsoft JhengHei", 10)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
