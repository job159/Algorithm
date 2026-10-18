# -*- coding: utf-8 -*-
"""
資料結構互動教學（Qt6 / PySide6）
================================

一個用來「教學」二元搜尋樹與紅黑樹的可視化工具：
- 上方選單可切換「紅黑樹」與「二元搜尋樹（BST）」兩種模式。
- 插入、刪除、搜尋都會逐步播放，每一步都有中文解說。
- 旋轉、重新著色看得到平滑動畫。
- 左側面板有操作按鈕、逐步控制、解說、性質說明、統計（含樹高）。
- 內建「BST 退化示範」，對照紅黑樹的自動平衡，強化學習效果。

執行：
    python main.py
"""
from __future__ import annotations

import random
import sys
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QFont,
    QIntValidator,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from avl import AVL
from bst import BST
from rbtree import RBTree
from rbview import TreeCanvas

# 紅黑樹五大性質（步驟可用 prop=1..5 對應高亮）
RB_PROPERTIES = [
    "1. 每個節點不是紅色就是黑色。",
    "2. 根節點是黑色。",
    "3. 每個葉節點（NIL）都是黑色。",
    "4. 紅色節點的子節點都是黑色（不能有兩個連續紅色）。",
    "5. 從任一節點到其所有後代 NIL，黑色節點數目都相同（黑高一致）。",
]

# 二元搜尋樹說明（沒有 prop 高亮，純對照教學用）
BST_NOTES = [
    "• BST 性質：左子樹所有鍵 ＜ 節點 ＜ 右子樹所有鍵。",
    "• 沒有自動平衡：依序插入會「退化成鏈狀」，高度變 O(n)。",
    "• 對照紅黑樹：高度永遠保證 O(log n)，搜尋更快。",
    "• 試試選單的「BST 退化示範」感受兩者差異。",
]

# AVL 樹說明
AVL_NOTES = [
    "• AVL 也是 BST：左子樹 ＜ 節點 ＜ 右子樹。",
    "• 平衡因子 = 左子樹高 − 右子樹高，必須 ∈ {−1, 0, +1}。",
    "• 一旦變成 ±2 就旋轉：LL/RR 單旋、LR/RL 雙旋。",
    "• 比紅黑樹更嚴格平衡 → 查詢更快，但旋轉較多。",
    "• 節點上方的小徽章就是平衡因子，變紅代表 ±2。",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1240, 780)

        self.mode = "rb"  # "rb" 紅黑樹 / "bst" 二元搜尋樹
        self.tree = self._new_tree()
        self.steps: List[dict] = []
        self.idx = -1
        self.playing = False

        self.canvas = TreeCanvas()

        # 自動播放計時器
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        self._build_menu()
        self._build_ui()
        self._install_shortcuts()
        self._set_properties_for_mode()
        self._update_window_title()
        self._update_stats()
        self._refresh_step_ui()

    def _new_tree(self):
        if self.mode == "rb":
            return RBTree()
        if self.mode == "avl":
            return AVL()
        return BST()

    # ------------------------------------------------------------------ #
    #  選單列
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        mb = self.menuBar()

        ds = mb.addMenu("資料結構(&D)")
        self.act_rb = QAction("紅黑樹 Red-Black Tree", self, checkable=True)
        self.act_rb.setChecked(True)
        self.act_bst = QAction("二元搜尋樹 Binary Search Tree", self, checkable=True)
        self.act_avl = QAction("AVL 樹 AVL Tree", self, checkable=True)
        grp = QActionGroup(self)
        grp.setExclusive(True)
        grp.addAction(self.act_rb)
        grp.addAction(self.act_bst)
        grp.addAction(self.act_avl)
        self.act_rb.triggered.connect(lambda: self.set_mode("rb"))
        self.act_bst.triggered.connect(lambda: self.set_mode("bst"))
        self.act_avl.triggered.connect(lambda: self.set_mode("avl"))
        ds.addAction(self.act_rb)
        ds.addAction(self.act_bst)
        ds.addAction(self.act_avl)

        ex = mb.addMenu("範例(&E)")
        ex.addAction("建立範例樹（目前模式）", self.on_build_demo_current)
        ex.addSeparator()
        ex.addAction("BST 退化示範（依序插入 1..10）", self.on_bst_degenerate)
        ex.addAction("對照：AVL 依序插入 1..10", self.on_avl_sorted)
        ex.addAction("對照：紅黑樹依序插入 1..10", self.on_rb_sorted)
        ex.addSeparator()
        ex.addAction("清空／重置", self.on_reset)

        hp = mb.addMenu("說明(&H)")
        hp.addAction("操作說明", self.show_help)
        hp.addAction("關於", self.show_about)

    def _update_window_title(self):
        name = {
            "rb": "紅黑樹 Red-Black Tree",
            "bst": "二元搜尋樹 Binary Search Tree",
            "avl": "AVL 樹 AVL Tree",
        }[self.mode]
        self.setWindowTitle("資料結構互動教學 — %s（Qt6）" % name)

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        panel = self._build_panel()
        root.addWidget(panel, 0)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(340)
        v = QVBoxLayout(panel)
        v.setSpacing(10)

        self.title_label = QLabel("資料結構互動教學")
        tf = QFont()
        tf.setPointSize(15)
        tf.setBold(True)
        self.title_label.setFont(tf)
        v.addWidget(self.title_label)

        # --- 操作區 ---
        op_box = QGroupBox("操作")
        op = QVBoxLayout(op_box)
        row = QHBoxLayout()
        self.value_edit = QLineEdit()
        self.value_edit.setValidator(QIntValidator(-9999, 9999, self))
        self.value_edit.setPlaceholderText("輸入整數鍵值…")
        self.value_edit.returnPressed.connect(self.on_insert)
        row.addWidget(self.value_edit)
        op.addLayout(row)

        row2 = QHBoxLayout()
        btn_insert = QPushButton("插入")
        btn_insert.clicked.connect(self.on_insert)
        btn_delete = QPushButton("刪除")
        btn_delete.clicked.connect(self.on_delete)
        btn_search = QPushButton("搜尋")
        btn_search.clicked.connect(self.on_search)
        for b in (btn_insert, btn_delete, btn_search):
            row2.addWidget(b)
        op.addLayout(row2)

        row3 = QHBoxLayout()
        btn_rand = QPushButton("隨機插入")
        btn_rand.clicked.connect(self.on_random_insert)
        btn_build = QPushButton("範例樹")
        btn_build.clicked.connect(self.on_build_demo_current)
        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self.on_reset)
        for b in (btn_rand, btn_build, btn_reset):
            row3.addWidget(b)
        op.addLayout(row3)
        v.addWidget(op_box)

        # --- 逐步控制 ---
        step_box = QGroupBox("逐步播放")
        sb = QVBoxLayout(step_box)
        self.step_label = QLabel("步驟 0 / 0")
        sb.addWidget(self.step_label)

        srow = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一步")
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.on_toggle_play)
        self.btn_next = QPushButton("下一步 ▶")
        self.btn_next.clicked.connect(self.on_next)
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            srow.addWidget(b)
        sb.addLayout(srow)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("速度"))
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setMinimum(1)
        self.speed.setMaximum(10)
        self.speed.setValue(6)
        self.speed.valueChanged.connect(self._apply_speed)
        speed_row.addWidget(self.speed)
        sb.addLayout(speed_row)
        v.addWidget(step_box)

        # --- 解說 ---
        exp_box = QGroupBox("步驟解說")
        eb = QVBoxLayout(exp_box)
        self.explain = QTextEdit()
        self.explain.setReadOnly(True)
        self.explain.setMinimumHeight(90)
        ef = QFont()
        ef.setPointSize(11)
        self.explain.setFont(ef)
        eb.addWidget(self.explain)
        v.addWidget(exp_box)

        # --- 性質說明（隨模式切換內容）---
        self.prop_box = QGroupBox("性質")
        self.prop_layout = QVBoxLayout(self.prop_box)
        self.prop_labels: List[QLabel] = []
        v.addWidget(self.prop_box)

        # --- 統計 / 設定 ---
        stat_box = QGroupBox("狀態")
        stb = QVBoxLayout(stat_box)
        self.stat_label = QLabel("")
        self.stat_label.setWordWrap(True)
        stb.addWidget(self.stat_label)
        self.nil_check = QCheckBox("顯示空葉節點（NIL）")
        self.nil_check.toggled.connect(self.canvas.set_show_nil)
        stb.addWidget(self.nil_check)
        hint = QLabel("提示：滾輪縮放、拖曳平移；空白鍵播放/暫停、← → 切換步驟。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        stb.addWidget(hint)
        v.addWidget(stat_box)

        v.addStretch(1)
        self._apply_speed()
        return panel

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.on_next)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.on_prev)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.on_toggle_play)

    # ------------------------------------------------------------------ #
    #  操作
    # ------------------------------------------------------------------ #
    def _read_value(self) -> Optional[int]:
        text = self.value_edit.text().strip()
        if text == "" or text == "-":
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def on_insert(self):
        v = self._read_value()
        if v is None:
            return
        self._run(self.tree.insert(v))
        self.value_edit.clear()

    def on_delete(self):
        v = self._read_value()
        if v is None:
            return
        self._run(self.tree.delete(v))
        self.value_edit.clear()

    def on_search(self):
        v = self._read_value()
        if v is None:
            return
        self._run(self.tree.search(v))

    def on_random_insert(self):
        present = set(self.tree.keys_inorder())
        candidates = [k for k in range(1, 100) if k not in present]
        if not candidates:
            return
        v = random.choice(candidates)
        self._run(self.tree.insert(v))

    def _build_demo(self, demo: List[int]):
        self.on_reset()
        all_steps: List[dict] = []
        for k in demo:
            all_steps.extend(self.tree.insert(k))
        self._load(all_steps, autoplay=False, goto_end=True)

    def on_build_demo_current(self):
        if self.mode == "rb":
            self._build_demo([10, 20, 30, 15, 25, 5, 1, 8, 22, 28, 40, 35, 50])
        elif self.mode == "avl":
            self._build_demo([30, 10, 50, 20, 40, 60, 5, 15, 25, 35, 45])
        else:
            self._build_demo([50, 30, 70, 20, 40, 60, 80, 10, 25, 65, 90])

    def on_bst_degenerate(self):
        """BST 依序插入 1..10 → 退化成鏈狀。"""
        if self.mode != "bst":
            self.set_mode("bst", build_demo=False)
        self._build_demo(list(range(1, 11)))
        self.explain.setPlainText(
            "依序插入 1..10：BST 沒有平衡機制，每個新節點都掛在最右邊，"
            "整棵樹退化成「鏈狀」，高度 = 節點數 = O(n)。最壞情況搜尋要走遍每一個節點。"
        )

    def on_avl_sorted(self):
        """對照：AVL 依序插入 1..10 仍保持嚴格平衡。"""
        if self.mode != "avl":
            self.set_mode("avl", build_demo=False)
        self._build_demo(list(range(1, 11)))
        self.explain.setPlainText(
            "同樣依序插入 1..10，AVL 每次插入都用旋轉維持嚴格平衡，高度只有 4。"
            "可以按上一步／下一步，觀察每個節點的平衡因子何時變成 ±2 並觸發旋轉。"
        )

    def on_rb_sorted(self):
        """對照：紅黑樹依序插入 1..10 仍保持平衡。"""
        if self.mode != "rb":
            self.set_mode("rb", build_demo=False)
        self._build_demo(list(range(1, 11)))
        self.explain.setPlainText(
            "同樣依序插入 1..10，但紅黑樹會用旋轉與重新著色自動平衡，高度維持在 O(log n)（約 4），"
            "和 BST 退化成鏈狀（高度 10）形成強烈對比。"
        )

    def set_mode(self, mode: str, build_demo: bool = True):
        if mode == self.mode:
            return
        self.mode = mode
        self.act_rb.setChecked(mode == "rb")
        self.act_bst.setChecked(mode == "bst")
        self.act_avl.setChecked(mode == "avl")
        self._stop_play()
        self.tree = self._new_tree()
        self.steps = []
        self.idx = -1
        self.canvas.clear_all()
        self._set_properties_for_mode()
        self._update_window_title()
        self._update_stats()
        self._refresh_step_ui()
        if build_demo:
            self.on_build_demo_current()

    def on_reset(self):
        self._stop_play()
        self.tree = self._new_tree()
        self.steps = []
        self.idx = -1
        self.canvas.clear_all()
        self._update_stats()
        self._refresh_step_ui()
        self.explain.setPlainText("已重置為空樹。輸入鍵值並按「插入」開始，或按「範例樹」一鍵建立。")

    def show_help(self):
        QMessageBox.information(
            self,
            "操作說明",
            "● 上方「資料結構」選單可切換 紅黑樹 / 二元搜尋樹。\n"
            "● 輸入整數後按 插入 / 刪除 / 搜尋（或在輸入框按 Enter 插入）。\n"
            "● 逐步播放：◀ 上一步、▶ 播放/暫停、下一步 ▶、速度滑桿。\n"
            "● 鍵盤：空白鍵 播放/暫停、← → 切換步驟。\n"
            "● 滑鼠：滾輪縮放、拖曳平移。\n"
            "● 「範例」選單有 BST 退化示範，可對照紅黑樹的自動平衡。",
        )

    def show_about(self):
        QMessageBox.about(
            self,
            "關於",
            "資料結構互動教學（Qt6 / PySide6）\n\n"
            "支援 二元搜尋樹（BST）與 紅黑樹 的逐步可視化教學，\n"
            "每一步皆附中文解說，旋轉與重新著色有動畫。\n"
            "核心演算法採 CLRS 版本，並通過大量隨機正確性測試。",
        )

    # ------------------------------------------------------------------ #
    #  播放器
    # ------------------------------------------------------------------ #
    def _run(self, steps: List[dict]):
        self._load(steps, autoplay=True)

    def _load(self, steps: List[dict], autoplay: bool, goto_end: bool = False):
        self._stop_play()
        self.steps = steps
        self.idx = -1
        if not steps:
            self._update_stats()
            self._refresh_step_ui()
            return
        if goto_end:
            # 直接跳到最後狀態（不逐步），給「範例樹」用
            self.idx = len(steps) - 1
            self.canvas.render_snapshot(steps[self.idx], animate=False)
            self._update_stats()
            self._refresh_step_ui()
            return
        self._goto(0, animate=True)
        if autoplay:
            self._start_play()

    def _goto(self, i: int, animate: bool):
        if not self.steps:
            return
        i = max(0, min(i, len(self.steps) - 1))
        self.idx = i
        self.canvas.render_snapshot(self.steps[i], animate=animate)
        self._update_stats()
        self._refresh_step_ui()

    def on_next(self):
        self._stop_play()
        if self.steps and self.idx < len(self.steps) - 1:
            self._goto(self.idx + 1, animate=True)

    def on_prev(self):
        self._stop_play()
        if self.steps and self.idx > 0:
            self._goto(self.idx - 1, animate=True)

    def on_toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if not self.steps or self.idx >= len(self.steps) - 1:
            return
        self.playing = True
        self.btn_play.setText("⏸ 暫停")
        self.play_timer.start(self._interval_ms())

    def _stop_play(self):
        self.playing = False
        self.play_timer.stop()
        self.btn_play.setText("▶ 播放")

    def _on_play_tick(self):
        if not self.playing:
            return
        if self.idx < len(self.steps) - 1:
            self._goto(self.idx + 1, animate=True)
        if self.idx >= len(self.steps) - 1:
            self._stop_play()

    def _interval_ms(self) -> int:
        # 速度 1（慢）→ 1500ms，10（快）→ 300ms
        s = self.speed.value()
        return int(1500 - (s - 1) * (1200 / 9))

    def _apply_speed(self):
        interval = self._interval_ms()
        self.canvas.set_duration(min(int(interval * 0.8), 600))
        if self.playing:
            self.play_timer.start(interval)

    # ------------------------------------------------------------------ #
    #  顯示更新
    # ------------------------------------------------------------------ #
    def _refresh_step_ui(self):
        n = len(self.steps)
        cur = self.idx + 1 if self.idx >= 0 else 0
        self.step_label.setText("步驟 %d / %d" % (cur, n))
        self.btn_prev.setEnabled(self.idx > 0)
        self.btn_next.setEnabled(bool(self.steps) and self.idx < n - 1)
        self.btn_play.setEnabled(bool(self.steps) and self.idx < n - 1)

        active_prop = None
        if 0 <= self.idx < n:
            step = self.steps[self.idx]
            self.explain.setPlainText(step.get("desc", ""))
            active_prop = step.get("prop")
        self._highlight_property(active_prop)

    def _set_properties_for_mode(self):
        title = {
            "rb": "紅黑樹互動教學",
            "bst": "二元搜尋樹互動教學",
            "avl": "AVL 樹互動教學",
        }[self.mode]
        self.title_label.setText(title)
        while self.prop_layout.count():
            item = self.prop_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.prop_labels = []
        if self.mode == "rb":
            self.prop_box.setTitle("紅黑樹五大性質")
            texts = RB_PROPERTIES
        elif self.mode == "avl":
            self.prop_box.setTitle("AVL 樹")
            texts = AVL_NOTES
        else:
            self.prop_box.setTitle("二元搜尋樹 BST")
            texts = BST_NOTES
        for text in texts:
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#555;")
            self.prop_layout.addWidget(lab)
            self.prop_labels.append(lab)

    def _highlight_property(self, prop: Optional[int]):
        if self.mode != "rb":
            return
        for i, lab in enumerate(self.prop_labels, start=1):
            if prop is not None and i == prop:
                lab.setStyleSheet(
                    "color:#b71c1c; font-weight:bold; background:#fff3e0; padding:2px; border-radius:4px;")
            else:
                lab.setStyleSheet("color:#555;")

    def _update_stats(self):
        size = self.tree.size()
        height = self.tree.height() if size > 0 else 0
        ok, msg = self.tree.validate()
        keys = self.tree.keys_inorder()
        inorder = "、".join(str(k) for k in keys) if keys else "（空）"
        if len(inorder) > 56:
            inorder = inorder[:54] + "…"
        mark = "✓" if ok else "✗"
        color = "#2e7d32" if ok else "#c62828"
        if self.mode == "rb":
            bh = self.tree.black_height() if size > 0 else 0
            head = "節點數：%d　樹高：%d　黑高：%d" % (size, height, bh)
            check = "性質檢查"
        elif self.mode == "avl":
            head = "節點數：%d　樹高：%d" % (size, height)
            check = "AVL 檢查"
        else:
            head = "節點數：%d　樹高：%d" % (size, height)
            check = "BST 檢查"
        self.stat_label.setText(
            "%s\n中序：%s\n<span style='color:%s'>%s：%s %s</span>"
            % (head, inorder, color, check, mark, msg)
        )
        self.stat_label.setTextFormat(Qt.RichText)


def _apply_font(app: QApplication):
    """挑一個 Windows 上一定有、且支援繁體中文的字型，並設好 fallback。"""
    from PySide6.QtGui import QFontDatabase

    available = set(QFontDatabase.families())
    preferred = [
        "Microsoft JhengHei UI",
        "Microsoft JhengHei",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "PMingLiU",
        "MingLiU",
        "Noto Sans CJK TC",
        "Segoe UI",
    ]
    chosen = [f for f in preferred if f in available]
    font = QFont()
    if chosen:
        font.setFamilies(chosen)
        font.setFamily(chosen[0])
    font.setPointSize(10)
    app.setFont(font)


def setup_app(app: QApplication):
    """統一外觀：固定為乾淨的淺色主題（不受系統深色模式影響），並套用中文字型。"""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    _apply_font(app)

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f4f6f8"))
    pal.setColor(QPalette.WindowText, QColor("#263238"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#eceff1"))
    pal.setColor(QPalette.Text, QColor("#263238"))
    pal.setColor(QPalette.Button, QColor("#eceff1"))
    pal.setColor(QPalette.ButtonText, QColor("#263238"))
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipText, QColor("#263238"))
    pal.setColor(QPalette.Highlight, QColor("#1e88e5"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#9aa7ad"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9aa7ad"))
    app.setPalette(pal)

    app.setStyleSheet(
        """
        QWidget { color:#263238; }
        QGroupBox { font-weight:bold; border:1px solid #cfd8dc; border-radius:6px;
                    margin-top:10px; padding-top:10px; background:#ffffff; }
        QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px;
                           color:#37474f; }
        QPushButton { padding:6px 8px; border:1px solid #b0bec5; border-radius:5px;
                      background:#eceff1; color:#263238; }
        QPushButton:hover { background:#cfd8dc; }
        QPushButton:pressed { background:#b0bec5; }
        QPushButton:disabled { color:#aab4b9; background:#f5f7f8; border-color:#dfe4e6; }
        QLineEdit { padding:6px; border:1px solid #b0bec5; border-radius:5px;
                    background:#ffffff; }
        QTextEdit { border:1px solid #cfd8dc; border-radius:5px; background:#ffffff; }
        QSlider::groove:horizontal { height:6px; background:#cfd8dc; border-radius:3px; }
        QSlider::handle:horizontal { width:16px; margin:-6px 0; border-radius:8px;
                                     background:#1e88e5; }
        """
    )


def main():
    app = QApplication(sys.argv)
    setup_app(app)
    win = MainWindow()
    win.show()
    win.on_build_demo_current()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
