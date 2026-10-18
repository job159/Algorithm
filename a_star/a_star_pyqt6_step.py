# -*- coding: utf-8 -*-
"""
A*（A-star）最短路徑動畫（解說版）
=========================================
這支程式用 PyQt6 把「A* 演算法」一步一步畫出來，
並且每走一步都用白話告訴你「為什麼是這一步」。

A* = Dijkstra + 方向感：
  ● Dijkstra 只看「已經走了多少」(g)，像水波一樣往四面八方平均擴散 —— 很穩，但很盲目。
  ● A* 多加一個「離終點大概還有多遠」的估計 (h)，改用 f = g + h 排隊，
    會優先往終點方向探，通常少看非常多格；而且只要 h 不高估，找到的仍然是最短路。

A* 的四個靈魂，本程式全部都看得到：
  1. g(n)          從起點走到 n「實際」花了多少成本（格子下方 g）
  2. h(n)          從 n 到終點的「估計」成本：假裝沒牆、全是平地（格子下方 h、紫色虛線）
  3. f(n) = g + h  「經過 n 的整條路大概要多少」—— 排隊就看它（格子中央大數字）
  4. Open / Closed Open = 待處理的候選（min-heap 依 f 排序，畫面下方那排方塊）
                   Closed = 已處理、定案的格子（不再回頭）

操作：
  ⏮ 上一步 / ⏭ 單步 / ▶ 自動播放 / ↺ 重來 / 🧱 還原地圖 / 🧹 清空
  滑鼠可在地圖上畫牆、搬移起點終點、塗地形（讓某些格子變難走）
  上方可切換啟發函數 h：曼哈頓 / 歐幾里得 / h=0（就是 Dijkstra）/ 加權（貪快、不保證最短）

需求： pip install PyQt6
執行： python a_star_pyqt6_step.py
"""

import sys
import math
import heapq

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QSlider,
    QHBoxLayout, QVBoxLayout, QButtonGroup, QFrame, QScrollArea, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF


# ---------- 版面與外觀 ----------
COLS, ROWS = 13, 9          # 地圖大小（格數）
CELL = 56                   # 每格像素（要放得下 f 大字 + g/h 小字）
INF = float("inf")
DEFAULT_START = (1, 4)
DEFAULT_GOAL = (11, 4)

C_EMPTY    = QColor("#F1F5F9")   # 還沒探索、且地形成本=1 的空地
C_WALL     = QColor("#334155")   # 牆（不能走）
C_START    = QColor("#22C55E")   # 起點 S
C_GOAL     = QColor("#EF4444")   # 終點 G
C_CLOSED   = QColor("#BAE6FD")   # Closed：已處理、定案
C_OPEN     = QColor("#FDE68A")   # Open：已看到、等著被處理的候選（比路徑淡，結束後才分得清）
C_CURRENT  = QColor("#FB923C")   # 這一步正在處理的格子
C_PATH     = QColor("#FACC15")   # 最終最短路徑
C_H_LINE   = QColor("#8B5CF6")   # h 的「樂觀估計線」（無視牆壁）
C_PANEL_BG = QColor("#FFFFFF")
C_GRID_LINE= QColor("#E2E8F0")

# 地形成本對應的按鈕：(權重, 顯示文字)
TERRAIN_CHOICES = [(1, "平地 1"), (3, "草地 3"), (6, "泥沼 6"), (9, "高山 9")]

# 啟發函數：(代號, 按鈕文字, 簡稱)
HEURISTICS = [
    ("manhattan", "曼哈頓 |dx|+|dy|",    "A*（曼哈頓）"),
    ("euclid",    "歐幾里得 √(dx²+dy²)", "A*（歐幾里得）"),
    ("zero",      "h = 0（就是 Dijkstra）", "h=0（Dijkstra）"),
    ("weighted",  "加權 3×曼哈頓（貪快）", "加權 A*（3×曼哈頓）"),
]
H_WEIGHT = 3                # 加權 A* 的放大倍數


def fmt(node):
    """把 (c, r) 印成 (c,r) 方便在解說裡引用某一格。"""
    return f"({node[0]},{node[1]})"


def num(v):
    """數字好讀版：整數就印整數、小數印一位、無限大印 ∞。"""
    if v == INF:
        return "∞"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def terrain_tint(w):
    """地形權重 → 顏色。權重 1 = 很淺，越重越偏土黃/咖啡，讓人一眼看出哪裡難走。"""
    if w <= 1:
        return C_EMPTY
    t = min(1.0, (w - 1) / 8.0)          # 1..9 normalize 到 0..1
    r1, g1, b1 = 0xFD, 0xE9, 0xA8        # 淺黃
    r2, g2, b2 = 0xB4, 0x53, 0x09        # 深咖啡
    return QColor(int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t))


# =========================================================
#  啟發函數 h：「從這格到終點，最少大概還要多少成本？」
# =========================================================
def heuristic(kind, node, goal):
    """
    h 的設計原則：要「樂觀」—— 寧可低估、不可高估（admissible）。
    本地圖每格成本最少是 1、只能上下左右走，所以：
      曼哈頓   |dx|+|dy|       = 沒有牆、全是平地時的真實步數 → 剛好不高估，而且最準
      歐幾里得 √(dx²+dy²)      = 直線距離，比曼哈頓更小 → 也不高估，但比較不準（探索較多）
      h = 0                    = 完全沒方向感 → A* 退化成 Dijkstra
      加權     3×(|dx|+|dy|)   = 故意高估 → 衝得超快、看很少格，但不保證最短
    """
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    if kind == "manhattan":
        return dx + dy
    if kind == "euclid":
        return math.hypot(dx, dy)
    if kind == "weighted":
        return H_WEIGHT * (dx + dy)
    return 0


def heuristic_formula(kind, node, goal):
    """把 h 的計算過程寫成算式字串，放進解說裡（讓人看到數字是怎麼來的）。"""
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    h = heuristic(kind, node, goal)
    if kind == "manhattan":
        return f"{dx}+{dy} = {num(h)}"
    if kind == "euclid":
        return f"√({dx}²+{dy}²) = {num(h)}"
    if kind == "weighted":
        return f"{H_WEIGHT}×({dx}+{dy}) = {num(h)}"
    return "0"


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
        self.start = DEFAULT_START
        self.goal = DEFAULT_GOAL
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
        固定地圖（三個教學重點都設計在裡面）：
          ① 口袋牆：開口朝左（朝向起點），擋在 S 與 G 的直線上。
             A* 被 h 吸引，會先一頭衝進口袋（那裡 h 最小），撞牆後才退出來繞路 ——
             正好看到「h 只是估計、根本不知道有牆」的後果。
          ② 口袋上緣鋪一排泥沼（成本 6）：貼著口袋往上繞很貴，所以搜尋主要往下探。
          ③ 欄 8 一道「泥沼牆」（成本 6），只有最上、最下一列留平地缺口。
             最短路要老老實實繞到缺口（總成本 18）；
             加權 A*（h 放大 3 倍）太急著往 G 衝，會直接踩過泥沼 → 總成本 21，
             示範「h 高估 → 看的格子更少，但不保證最短」。
          同一張圖：Dijkstra 要展開約 100 格，A*（曼哈頓）不到一半。
        """
        self.clear_walls()
        self.clear_terrain()
        # ① 口袋牆：欄 6 的直牆 + 上下兩條橫臂（欄 3~6），開口朝左
        for r in range(2, 7):
            self.grid[r][6] = 1
        for c in range(3, 7):
            self.grid[2][c] = 1
            self.grid[6][c] = 1
        # ② 口袋上緣的泥沼
        for c in range(3, 8):
            self.weight[1][c] = 6
        # ③ 泥沼牆（上下各留一列平地缺口）
        for r in range(1, 8):
            self.weight[r][8] = 6


# =========================================================
#  A* 引擎（含「為何這一步」的解說）
# =========================================================
class AStar:
    def __init__(self, maze, h_kind="manhattan", verbose=True):
        self.maze = maze
        self.h_kind = h_kind
        self.verbose = verbose          # False = 安靜模式（拿來算對照數據，不印東西）
        self.reset()

    def h(self, node):
        return heuristic(self.h_kind, node, self.maze.goal)

    def reset(self):
        s = self.maze.start
        self.g = {s: 0}             # 已知「實際」最短成本；沒出現的格子代表還是 ∞
        self.parent = {}            # 來源簿：parent[v] = u 代表「目前到 v 最省的走法是經由 u」
        self.closed = set()         # Closed：已處理、定案（不再回頭）
        self.counter = 0            # heap 的破平手序號，確保動畫可重現
        self.best_seq = {s: 0}      # 每個 Open 格子最新一次入堆的序號（給佇列排序用）
        h0 = self.h(s)
        # 優先佇列（min-heap）：(f, h, 序號, 格子)
        #   先比 f；f 一樣比 h（h 小 = 離終點近，先處理它通常少繞路）；再一樣就比先來後到
        self.heap = [(h0, h0, 0, s)]
        self.current = None
        self.done = False
        self.found = False
        self.path = []
        self.reveal = 0             # 已收集進「路徑容器」的格數（找到終點後做生長動畫）
        self.steps = 0
        self.stale_skipped = []     # 這一步略過了哪些『過期項目』
        self.explain = (f"起點 S{fmt(s)}：g=0（還沒走）、h={heuristic_formula(self.h_kind, s, self.maze.goal)}"
                        f"（估計離 G 還要多少）、f=g+h={num(h0)}，放進 Open。"
                        f"其它格子 g 都是 ∞（還沒走到）。按「單步」開始。")

    def g_of(self, node):
        return self.g.get(node, INF)

    def f_of(self, node):
        g = self.g_of(node)
        return INF if g == INF else g + self.h(node)

    def in_open(self, node):
        """在 Open 中（已看到但還沒處理）= 有 g 且尚未 closed。"""
        return node in self.g and node not in self.closed

    def live_open(self):
        """
        回傳『目前真正有效』的 Open 內容，依（f、h、入堆序號）由小到大排序。
        第一個就是下一步會被取出的格子。
        （heap 裡其實還有些過期的舊項目，這裡只顯示每格最新的有效狀態。）
        """
        live = []
        for n in self.g:
            if n in self.closed:
                continue
            hv = self.h(n)
            live.append((self.g[n] + hv, hv, self.best_seq.get(n, 0), n))
        live.sort(key=lambda x: (x[0], x[1], x[2]))
        return live

    def step(self):
        """前進一步 = 從 Open 拿出 f 最小的一格，放進 Closed，並更新它的鄰居。"""
        if self.done:
            return
        self.stale_skipped = []

        # 1) 從 heap 取出 f 最小者；途中略過『過期項目』（已 closed 又重複出現的）
        node, f = None, INF
        while self.heap:
            ff, _, _, u = heapq.heappop(self.heap)
            if u in self.closed:
                self.stale_skipped.append(u)        # 過期，跳過
                continue
            node, f = u, ff
            break

        if node is None:
            self.done = True
            self.found = False
            self.current = None
            self.explain = "Open 空了、沒有候選可拿 → 起點到終點之間沒有通路。"
            return

        g = self.g[node]
        hval = self.h(node)
        # 平手分析：還有誰的 f 跟它一樣？（拿來解說「為什麼挑它」）
        ties = [(hh, n) for (fff, hh, _, n) in self.live_open()
                if n != node and abs(fff - f) < 1e-9]

        # 2) 放進 Closed：之後不再回頭處理它
        self.closed.add(node)
        self.current = node
        self.steps += 1

        if node == self.maze.goal:
            self.done = True
            self.found = True
            self._build_path()
            if self.h_kind == "weighted":
                why = ("但注意：加權 h 會高估，Open 裡其他格的 f 被灌水了，"
                       "所以這條路「不保證」最短（看上方對照列）。")
            else:
                why = ("此時 Open 裡其他格的 f 都 ≥ 它，而 h 從不高估 → 不可能還有更便宜的路，"
                       "這就是最短路。")
            self.explain = (f"取出 f 最小的 {fmt(node)}（f = g + h = {num(g)} + 0 = {num(f)}）"
                            f"—— 正是終點 G！到了終點 h=0，f 就是真實總成本 {num(g)}。"
                            f"{why}沿 parent 來源簿回推即得路徑（見下方）。")
            return

        # 3) 看每個鄰居：「經由 node 走過去」會不會比它現在已知的 g 更省？
        relaxed, skipped, closed_nb = [], [], []
        for v in self.maze.neighbors(node):
            if v in self.closed:
                closed_nb.append(v)
                continue
            w = self.maze.weight_of(v)
            ng = g + w                          # 經由 node 到 v 的實際成本
            if ng < self.g_of(v):
                old = self.g_of(v)
                hv = self.h(v)
                self.g[v] = ng
                self.parent[v] = node
                self.counter += 1
                self.best_seq[v] = self.counter
                heapq.heappush(self.heap, (ng + hv, hv, self.counter, v))
                relaxed.append((v, old, ng, w, hv))
            else:
                skipped.append((v, self.g_of(v), ng))

        self.explain = self._explain(node, g, hval, f, ties, relaxed, skipped, closed_nb)

    def _explain(self, node, g, hval, f, ties, relaxed, skipped, closed_nb):
        """把這一步發生的事，逐項翻成白話。"""
        goal = self.maze.goal
        parts = []
        if self.stale_skipped:
            names = "、".join(fmt(n) for n in self.stale_skipped)
            parts.append(f"先略過 {len(self.stale_skipped)} 個過期項目（{names} 早已在 Closed，"
                         f"是之前更新時留下的舊副本）。")

        tie_txt = ""
        if ties:
            bigger_h = [n for hh, n in ties if hh > hval + 1e-9]
            if bigger_h:
                tie_txt = (f"（另有 {len(ties)} 格 f 也是 {num(f)}；平手時先挑 h 較小 = 離終點較近的，"
                           f"這是常用的破平手技巧，可少繞很多格）")
            else:
                tie_txt = f"（另有 {len(ties)} 格 f、h 都一樣，就照先來後到）"
        parts.append(f"取出 Open 中 f 最小的 {fmt(node)}：f = g + h = {num(g)} + {num(hval)} = {num(f)}"
                     f"{tie_txt}。把它放進 Closed（定案，不再回頭）。")

        for (v, old, ng, w, hv) in relaxed:
            how = "第一次看到它 → 放進 Open" if old == INF else f"比原本 g={num(old)} 更省 → 更新"
            parts.append(f"→ 鄰居 {fmt(v)}：g = {num(g)}+{w}(進入成本) = {num(ng)}，"
                         f"h = {heuristic_formula(self.h_kind, v, goal)}，f = {num(ng + hv)}；"
                         f"{how}，來源記成 {fmt(node)}。")
        for (v, cur, ng) in skipped:
            parts.append(f"→ 鄰居 {fmt(v)}：經由這格 g = {num(ng)} ≥ 現有 {num(cur)} → 不更新。")
        if not relaxed and not skipped:
            parts.append("四周不是牆就是已在 Closed，沒有新候選（死路）。")

        # 前兩步額外點出「方向感」：朝 G 的鄰居 h 變小，f 不變 → 會繼續往 G 衝
        if self.steps <= 2 and relaxed and self.h_kind != "zero":
            best = min(relaxed, key=lambda t: (t[2] + t[4], t[4]))
            parts.append(f"👉 注意 {fmt(best[0])} 的 f={num(best[2] + best[4])} 最小：往 G 方向走，"
                         f"g 增加的被 h 減少的抵掉，f 幾乎不變；往反方向走 g、h 一起變大，f 就變大"
                         f" —— 這就是 A* 的「方向感」。")
        return "　".join(parts)

    def _build_path(self):
        """從終點沿 parent 一路回推到起點，組出路徑。"""
        cur = self.maze.goal
        path = [cur]
        while cur in self.parent:
            cur = self.parent[cur]
            path.append(cur)
        path.reverse()
        self.path = path
        if self.verbose:
            total = self.g_of(self.maze.goal)
            chain = " ← ".join(fmt(n) for n in reversed(path))
            print(f"[A*] 回推（終點→起點）：{chain}")
            print(f"[A*] 路徑（起點→終點）：{' → '.join(fmt(n) for n in path)}")
            print(f"[A*] 總成本 = {num(total)}　（{len(path)-1} 步、展開 {len(self.closed)} 格）")

    def run_to_end(self):
        """一口氣跑完（給『對照數據』用）。"""
        while not self.done:
            self.step()
        return self


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
                self._draw_cell(p, (c, r), revealed_path)

        # 【關鍵】來源箭頭：每個被看到的格子，都記得自己是從哪一格來的（parent）。
        for node, par in s.parent.items():
            if frozenset((node, par)) in revealed_edges:
                continue                       # 這條邊改由下方金色路徑呈現
            self._draw_came_from(p, node, par, QColor("#3B82F6"), 1.8)

        # h 的樂觀估計線：從「正在處理的格子」到 G，假裝沒有牆
        if s.current is not None and not s.done:
            self._draw_h_line(p, s.current)

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
            for i in range(start_idx, n - 1):
                self._draw_came_from(p, s.path[i + 1], s.path[i], QColor("#CA8A04"), 2.6)
            p.setBrush(QBrush(QColor("#CA8A04")))
            p.setPen(QPen(QColor("#CA8A04"), 1))
            for pt in (pts[0], pts[-1]):
                p.drawEllipse(pt, 5, 5)
        p.end()

    def _draw_cell(self, p, node, revealed_path):
        s = self.search
        c, r = node
        w = self.maze.weight[r][c]
        is_sg = node in (self.maze.start, self.maze.goal)

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
        elif node in s.closed:
            color = C_CLOSED
        elif s.in_open(node):
            color = C_OPEN
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
            return

        text_dark = QColor("white") if is_sg else QColor("#0F172A")
        text_soft = QColor("#F8FAFC") if is_sg else QColor("#334155")

        # 左上角：座標（方便對照解說）
        p.setFont(QFont("Arial", 7))
        p.setPen(QPen(QColor("#E2E8F0") if is_sg else QColor("#64748B")))
        p.drawText(QRectF(c * CELL + 5, r * CELL + 3, CELL, 11),
                   Qt.AlignmentFlag.AlignLeft, f"{c},{r}")

        # 右上角：S / G 標籤，或地形權重徽章（成本越高顏色越深）
        if is_sg:
            p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            p.setPen(QPen(QColor("white")))
            p.drawText(QRectF(c * CELL + CELL - 18, r * CELL + 3, 14, 13),
                       Qt.AlignmentFlag.AlignCenter, "S" if node == self.maze.start else "G")
        elif w > 1:
            badge = QRectF(c * CELL + CELL - 20, r * CELL + 4, 16, 14)
            p.setBrush(QBrush(terrain_tint(w).darker(115)))
            p.setPen(QPen(QColor("#78350F"), 1))
            p.drawRoundedRect(badge, 4, 4)
            p.setPen(QPen(QColor("#3F2400")))
            p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            p.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(w))

        hv = s.h(node)
        if node in s.g:
            # 正中央：f（排隊的依據，A* 的主角）；左下 g、右下 h 拆開看
            # （底部正中間刻意留空，給指向來源的箭頭走）
            p.setFont(QFont("Arial", 15, QFont.Weight.Bold))
            p.setPen(QPen(text_dark))
            p.drawText(QRectF(c * CELL, r * CELL + 11, CELL, 24),
                       Qt.AlignmentFlag.AlignCenter, num(s.g[node] + hv))
            p.setFont(QFont("Arial", 7))
            p.setPen(QPen(text_soft))
            low = QRectF(c * CELL + 5, r * CELL + CELL - 17, CELL - 10, 12)
            p.drawText(low, Qt.AlignmentFlag.AlignLeft, f"g{num(s.g[node])}")
            p.drawText(low, Qt.AlignmentFlag.AlignRight, f"h{num(hv)}")
        elif self.main.show_h:
            # 還沒看到的格子：f 未知（g=∞），但 h 是「每格都能先算好」的估計值
            if is_sg:
                soft = QColor("#F1F5F9")
            elif w > 1:
                soft = QColor("#5B3A10")
            else:
                soft = QColor("#94A3B8")
            p.setFont(QFont("Arial", 8))
            p.setPen(QPen(soft))
            p.drawText(QRectF(c * CELL, r * CELL + 14, CELL, CELL - 26),
                       Qt.AlignmentFlag.AlignCenter, f"h={num(hv)}")

    def _draw_h_line(self, p, node):
        """
        紫色虛線 = h 在「想像」的那條路：沒有牆、全是平地。
        曼哈頓 / 加權 → 先橫後直的 L 形；歐幾里得 → 直線；h=0 → 沒有方向感，不畫。
        """
        kind = self.search.h_kind
        if kind == "zero":
            return
        goal = self.maze.goal
        c, r = node
        gc, gr = goal

        def ctr(cc, rr):
            return QPointF(cc * CELL + CELL / 2, rr * CELL + CELL / 2)

        if kind == "euclid":
            pts = [ctr(c, r), ctr(gc, gr)]
        else:
            pts = [ctr(c, r), ctr(gc, r), ctr(gc, gr)]
        pen = QPen(C_H_LINE, 2.2)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(QPolygonF(pts))

        # 標籤放在最長那一段的中點
        best_mid, best_len = pts[0], -1.0
        for a, b in zip(pts, pts[1:]):
            seg = math.hypot(b.x() - a.x(), b.y() - a.y())
            if seg > best_len:
                best_len = seg
                best_mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        text = f"h = {heuristic_formula(kind, node, goal)}"
        p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        tw = p.fontMetrics().horizontalAdvance(text) + 12
        th = 18
        x = min(max(2.0, best_mid.x() - tw / 2), self.width() - tw - 2)
        y = min(max(2.0, best_mid.y() - th - 4), self.height() - th - 2)
        box = QRectF(x, y, tw, th)
        p.setBrush(QBrush(QColor("#F5F3FF")))
        p.setPen(QPen(C_H_LINE, 1.5))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QPen(QColor("#5B21B6")))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_came_from(self, p, node, par, color, width):
        """
        在 node 上畫一支指向其 parent（來源）的小箭頭：『我是從那一格來的』。
        箭頭只畫在格子外圈（離中心 0.27~0.47 格），不會蓋住中間的 f 與底部的 g/h。
        """
        c, r = node
        pc, pr = par
        cx, cy = c * CELL + CELL / 2, r * CELL + CELL / 2
        dx, dy = pc - c, pr - r                    # 一定是上下左右其中一個方向
        x1, y1 = cx + dx * (CELL * 0.27), cy + dy * (CELL * 0.27)
        x2, y2 = cx + dx * (CELL * 0.47), cy + dy * (CELL * 0.47)
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
#  Open List（min-heap，依 f 排序）內容繪圖
# =========================================================
class OpenListWidget(QWidget):
    BOX_W = 58
    BOX_H = 50
    GAP = 6

    def __init__(self, search):
        super().__init__()
        self.search = search
        self.setFixedHeight(84)
        self.setMinimumWidth(COLS * CELL)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        live = self.search.live_open()         # 已排序：第一個 = 下一個取出
        max_show = max(1, (self.width() - 70) // (self.BOX_W + self.GAP))
        x, y = 4, 4
        shown = live[:max_show]

        for i, (f, hv, _, node) in enumerate(shown):
            is_next = (i == 0)
            rect = QRectF(x, y, self.BOX_W, self.BOX_H)
            p.setBrush(QBrush(C_CURRENT if is_next else C_OPEN))
            p.setPen(QPen(QColor("#92400E"), 2 if is_next else 1))
            p.drawRoundedRect(rect, 6, 6)
            p.setPen(QPen(QColor("#1F2937")))
            p.setFont(QFont("Arial", 8))
            p.drawText(QRectF(x, y + 2, self.BOX_W, 14),
                       Qt.AlignmentFlag.AlignCenter, f"{node[0]},{node[1]}")
            p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            p.drawText(QRectF(x, y + 16, self.BOX_W, 16),
                       Qt.AlignmentFlag.AlignCenter, f"f={num(f)}")
            p.setFont(QFont("Arial", 7))
            p.setPen(QPen(QColor("#475569")))
            p.drawText(QRectF(x, y + 33, self.BOX_W, 13),
                       Qt.AlignmentFlag.AlignCenter, f"g{num(self.search.g[node])}+h{num(hv)}")
            x += self.BOX_W + self.GAP

        if len(live) > max_show:
            p.setPen(QPen(QColor("#64748B")))
            p.setFont(QFont("Arial", 9))
            p.drawText(QRectF(x, y, 60, self.BOX_H),
                       Qt.AlignmentFlag.AlignVCenter, f"… +{len(live) - max_show}")

        p.setFont(QFont("Microsoft JhengHei", 10, QFont.Weight.Bold))
        p.setPen(QPen(QColor("#B45309")))
        if shown:
            p.drawText(QRectF(4, y + self.BOX_H + 6, self.width() - 8, 20),
                       Qt.AlignmentFlag.AlignLeft,
                       "↑ Open List：下一個取出 f 最小的（f 一樣時挑 h 小 = 離終點近的）")
        else:
            p.drawText(QRectF(4, y, 360, self.BOX_H),
                       Qt.AlignmentFlag.AlignVCenter, "（Open List 已空）")
        p.end()


# =========================================================
#  最短路徑「容器」繪圖
# =========================================================
class PathWidget(QWidget):
    """
    路徑容器：找到終點後，沿 parent 從終點回推得到的整條路，
    會一格一格『收集』進這個盒子。每個方塊上排是座標、下排是到該格的累積 g；
    方塊之間的箭頭標著 +進入成本，最後一格的 g 就是總成本。
    """
    BOX_W = 50
    BOX_H = 42
    ARROW = 30
    PAD = 12
    TITLE_H = 26
    ROW_H = 66

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
        """依路徑長度自動算需要幾列，調整容器高度。"""
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
        p.setPen(QPen(QColor("#F59E0B"), 2))
        p.drawRoundedRect(outer, 12, 12)

        found = s.done and s.found and bool(s.path)
        reveal = min(getattr(s, "reveal", 0), len(s.path)) if found else 0

        p.setFont(QFont("Microsoft JhengHei", 11, QFont.Weight.Bold))
        p.setPen(QPen(QColor("#92400E")))
        if found:
            total = s.g_of(self.maze.goal)
            title = (f"🏁 路徑容器　|　總成本 g[G] = {num(total)}　|　"
                     f"{len(s.path) - 1} 步　|　已收集 {reveal}/{len(s.path)} 格")
        else:
            title = "🏁 路徑容器　（找到終點後，沿 parent 從終點回推，一格一格收集進來）"
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
                           Qt.AlignmentFlag.AlignCenter, f"g={num(s.g_of(node))}")
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
        self.setWindowTitle("A* 最短路徑動畫（解說版）")

        self.maze = Maze()
        self.h_kind = "manhattan"
        self.show_h = True
        self.search = AStar(self.maze, self.h_kind)

        self.edit_mode = 'wall'
        self.terrain_value = 6          # 「地形」筆刷目前的權重
        self.paint_value = 1            # 牆筆刷：要塗成牆(1)還是清除(0)
        self.frame = 0                  # 已前進的步數（供上一步倒退）
        self.ref_expanded = None        # 對照組：Dijkstra 要展開幾格
        self.ref_cost = INF             # 對照組：真正的最短成本
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.reveal_timer = QTimer(self)        # 找到終點後，把路徑一格一格收集進容器的動畫
        self.reveal_timer.timeout.connect(self._tick_reveal)

        self._build_ui()
        self._compute_reference()
        self.refresh()

    # ---------- 介面組裝 ----------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())
        root.addLayout(self._build_edit_bar())
        root.addLayout(self._build_heuristic_bar())

        title = QLabel("A* 最短路徑　—　f = g（已經走的實際成本）+ h（離終點的樂觀估計）")
        title.setStyleSheet("font-size:17px;font-weight:bold;color:#1D4ED8;")
        root.addWidget(title)

        grid_row = QHBoxLayout()
        grid_row.addStretch(1)
        self.grid = GridWidget(self.maze, self.search, self)
        grid_row.addWidget(self.grid)
        grid_row.addStretch(1)
        root.addLayout(grid_row)

        why = QLabel(
            "🔑 A* 跟 Dijkstra 只差一件事：排隊的依據從 g 換成 f = g + h。"
            "h 是「樂觀估計」—— 假裝沒有牆、全是平地，從這格到 G 最少還要多少（紫色虛線）。"
            "h 越準，A* 越會直直往 G 衝、少看很多格；只要 h 從不高估真實成本，找到的仍保證是最短路。"
            "細藍箭頭＝每格的來源 parent，找到 G 後沿箭頭回推就是答案。")
        why.setWordWrap(True)
        why.setStyleSheet(
            "font-size:12px;color:#1E3A8A;background:#EFF6FF;"
            "border:1px solid #BFDBFE;border-radius:6px;padding:7px;")
        root.addWidget(why)

        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:13px;color:#0F172A;padding:2px;")
        root.addWidget(self.stats)

        self.compare = QLabel()
        self.compare.setWordWrap(True)
        self.compare.setStyleSheet(
            "font-size:12px;color:#14532D;background:#F0FDF4;"
            "border:1px solid #BBF7D0;border-radius:6px;padding:6px;")
        root.addWidget(self.compare)

        self.open_list = OpenListWidget(self.search)
        root.addWidget(self.open_list)

        root.addWidget(self._build_legend())

        self.explain = QLabel()
        self.explain.setWordWrap(True)
        self.explain.setMinimumHeight(86)
        self.explain.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.explain.setStyleSheet(
            "font-size:13px;color:#0F172A;background:#FFFFFF;"
            "border:1px solid #E2E8F0;border-left:4px solid #2563EB;"
            "border-radius:6px;padding:8px;")
        root.addWidget(self.explain)

        self.path_box = PathWidget(self.search, self.maze)
        root.addWidget(self.path_box)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size:12px;color:#475569;padding:2px;")
        root.addWidget(self.path_label)

        container = QWidget()
        container.setLayout(root)
        # 背景色只套在這個容器本身：不加選擇器的話會一路傳給所有子按鈕，
        # 蓋掉 QPushButton:checked 的藍底（選中的按鈕會變成白底白字、看不到）
        container.setObjectName("root")
        container.setStyleSheet("QWidget#root { background:#F8FAFC; }")

        # 包進可捲動區：元素多、視窗較高時，小螢幕也不會被裁切
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        self.resize(860, 960)
        self.setStyleSheet("""
            QPushButton {
                background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px;
                padding:7px 12px; font-size:13px; color:#0F172A;
            }
            QPushButton:hover { background:#EFF6FF; border-color:#60A5FA; }
            QPushButton:checked { background:#2563EB; color:white; border-color:#2563EB; }
            QLabel { color:#0F172A; }
            QCheckBox { font-size:13px; color:#0F172A; }
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
        self.speed.setFixedWidth(140)
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
        return bar

    def _build_heuristic_bar(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("啟發函數 h："))

        self.h_group = QButtonGroup(self)
        for key, text, _ in HEURISTICS:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(key == self.h_kind)
            b.clicked.connect(lambda _, k=key: self.set_heuristic(k))
            self.h_group.addButton(b)
            bar.addWidget(b)

        bar.addSpacing(8)
        self.chk_h = QCheckBox("顯示每格 h")
        self.chk_h.setChecked(self.show_h)
        self.chk_h.toggled.connect(self.set_show_h)
        bar.addWidget(self.chk_h)
        bar.addStretch(1)
        return bar

    def _build_legend(self):
        frame = QFrame()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 4, 0)
        items = [
            (C_START, "起點 S"), (C_GOAL, "終點 G"), (C_WALL, "牆"),
            (C_OPEN, "Open 候選"), (C_CURRENT, "處理中"),
            (C_CLOSED, "Closed 定案"), (C_PATH, "路徑"),
            (terrain_tint(6), "高成本地形"), (C_H_LINE, "h 估計線"),
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
        lay.addStretch(1)
        return frame

    # ---------- 啟發函數 ----------
    def set_heuristic(self, kind):
        self.h_kind = kind
        self.search.h_kind = kind
        self.reset_search()

    def set_show_h(self, on):
        self.show_h = on
        self.grid.update()

    def _compute_reference(self):
        """
        對照組：在『同一張地圖』上安靜地跑一次 h=0（= Dijkstra），
        記下它要展開幾格、最短成本是多少，拿來跟目前的 A* 比。
        """
        ref = AStar(self.maze, "zero", verbose=False).run_to_end()
        if ref.found:
            self.ref_expanded = len(ref.closed)
            self.ref_cost = ref.g_of(self.maze.goal)
        else:
            self.ref_expanded = None
            self.ref_cost = INF

    # ---------- 滑鼠編輯 ----------
    def set_edit_mode(self, mode):
        self.edit_mode = mode

    def set_terrain_value(self, w):
        self.terrain_value = w
        self.edit_mode = 'terrain'
        for b in self.mode_group.buttons():
            b.setChecked(b.text().endswith('地形'))

    def start_stroke(self, c, r, erase):
        node = (c, r)
        is_sg = node in (self.maze.start, self.maze.goal)
        if self.edit_mode == 'wall':
            if is_sg:
                return
            self.paint_value = 0 if erase else (0 if self.maze.is_wall(c, r) else 1)
            self.maze.grid[r][c] = self.paint_value
        elif self.edit_mode == 'terrain':
            if is_sg or self.maze.is_wall(c, r):
                return
            self.maze.weight[r][c] = 1 if erase else self.terrain_value
        elif self.edit_mode == 'start':
            if self.maze.is_wall(c, r) or node == self.maze.goal:
                return
            self.maze.start = node
            self.maze.weight[r][c] = 1
        elif self.edit_mode == 'goal':
            if self.maze.is_wall(c, r) or node == self.maze.start:
                return
            self.maze.goal = node
            self.maze.weight[r][c] = 1
        self.reset_search()

    def continue_stroke(self, c, r):
        node = (c, r)
        if node in (self.maze.start, self.maze.goal):
            return
        if self.edit_mode == 'wall':
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
        s = self.search
        if s.done and s.found and s.path and s.reveal < len(s.path):
            if not self.reveal_timer.isActive():
                self.reveal_timer.start(110)

    def _tick_reveal(self):
        s = self.search
        if s.done and s.found and s.path and s.reveal < len(s.path):
            s.reveal += 1
            n, k = len(s.path), s.reveal
            head = s.path[n - k]
            if k == 1:
                s.explain = (f"找到終點 G{fmt(head)}！開始『回推』：每格都記得自己從哪來"
                             f"（藍箭頭）。我們從 G 出發，沿箭頭往回走，就能還原整條路。")
            elif k < n:
                child = s.path[n - k + 1]
                w = self.maze.weight_of(child)
                s.explain = (f"回推第 {k-1} 步：從 {fmt(child)} 沿來源箭頭退回 {fmt(head)}"
                             f"（即 parent[{child[0]},{child[1]}]={fmt(head)}，這段成本 {w}）。"
                             f"金線又往 S 長一段。")
            else:
                s.explain = (f"退回起點 S{fmt(head)}，回推完成！整條金線就是 A* 找到的路，"
                             f"總成本 = g[G] = {num(s.g_of(self.maze.goal))}。"
                             f"對照上方綠色列，看看它比 Dijkstra 少看了多少格。")
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
        """倒退做法：整個重設後，再快轉前進 n 步（heap 用序號破平手，可完全重現）。"""
        self.reveal_timer.stop()
        self.search.reset()
        self.frame = 0
        for _ in range(n):
            if not self.advance_one():
                break
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
        self._compute_reference()
        self.refresh()

    def restore_map(self):
        self.maze.start = DEFAULT_START
        self.maze.goal = DEFAULT_GOAL
        self.maze.generate_fixed()
        self.reset_search()

    def clear_all(self):
        self.maze.clear_walls()
        self.maze.clear_terrain()
        self.reset_search()

    # ---------- 更新顯示 ----------
    def refresh(self):
        self.grid.update()
        self.open_list.update()
        self.path_box.sync_height()
        self.path_box.update()
        self.stats.setText(self._stats_text())
        self.compare.setText(self._compare_text())
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
        return (f"第 {self.frame} 步　|　狀態：{status}　|　Closed（已展開）：{len(s.closed)} 格　|　"
                f"Open：{len(s.live_open())} 格　|　目前 g[G] = {num(s.g_of(self.maze.goal))}")

    def _compare_text(self):
        s = self.search
        name = dict((k, short) for k, _, short in HEURISTICS)[self.h_kind]
        if self.ref_expanded is None:
            return "📊 起點到終點沒有通路（連 Dijkstra 也找不到）。"
        if self.h_kind == "zero":
            text = (f"📊 h=0 時 A* 就是 Dijkstra：沒有方向感，像水波一樣往四面八方擴散，"
                    f"要展開 {self.ref_expanded} 格才走到 G。目前已展開 {len(s.closed)} 格。")
            if s.done and s.found:
                text += f"　|　總成本 {num(s.g_of(self.maze.goal))} ✓ 最短"
            return text
        text = (f"📊 同一張地圖：Dijkstra（h=0）要展開 {self.ref_expanded} 格才走到 G；"
                f"{name} 目前展開 {len(s.closed)} 格")
        if s.done and s.found:
            pct = round(100 * len(s.closed) / self.ref_expanded)
            cost = s.g_of(self.maze.goal)
            text += f"（只用了 {pct}%）。　總成本 {num(cost)}，真正最短 = {num(self.ref_cost)} → "
            if abs(cost - self.ref_cost) < 1e-9:
                text += "✓ 一樣是最短路"
            else:
                text += f"✗ 多花了 {num(cost - self.ref_cost)}（h 高估 → 不保證最短，換來看更少格）"
        else:
            text += "。"
        return text

    def _path_text(self):
        s = self.search
        if s.done and s.found and s.path:
            back = " ← ".join(fmt(n) for n in reversed(s.path))
            return f"沿 parent 來源簿回推（終點 G → 起點 S）：{back}"
        return ("提示：按「單步」配合下方解說一格一格看；試著切換上方的 h，"
                "比較 Closed（淺藍）範圍差多少。找到終點後，路徑會沿 parent 回推、收集進上面的容器。")


def main():
    # 讓 print() 的中文/箭號在任何 Windows 主控台編碼下都不會出錯
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft JhengHei", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
