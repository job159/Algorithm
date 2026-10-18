"""B+ 樹視覺化教學 — PyQt6 互動動畫。

執行:python main.py

畫面組成:
  上方  操作列(插入 / 刪除 / 搜尋 / 範圍查詢 / 隨機插入 / 範例建樹 / 清空 / 階數)
  中央  樹的動畫畫布(滾輪縮放、拖曳平移) + 右側步驟列表 / 圖例 / 規則
  下方  播放控制(播放/暫停、單步前進後退、速度、自動縮放)
"""
from __future__ import annotations

import math
import random
import re
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from bplus_tree import BPlusTree
from tree_view import TreeCanvas

STYLESHEET = """
QMainWindow, QWidget { background: #F4F6FA; color: #1F2937; }
QLineEdit, QSpinBox, QComboBox, QListWidget {
    background: #FFFFFF; border: 1px solid #C9D2DE; border-radius: 6px; padding: 4px 6px;
}
QPushButton {
    background: #FFFFFF; border: 1px solid #B9C4D2; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover  { background: #E8F0FE; }
QPushButton:pressed{ background: #D5E3F8; }
QPushButton#primary {
    background: #2F6FDE; color: white; border: none; font-weight: bold;
}
QPushButton#primary:hover { background: #2A63C6; }
QLabel#desc {
    background: #FFFFFF; border: 1px solid #C9D2DE; border-radius: 8px;
    padding: 10px 14px; font-size: 12pt; font-weight: 600; color: #163A6B;
}
QLabel#card {
    background: #FFFFFF; border: 1px solid #C9D2DE; border-radius: 8px; padding: 8px 10px;
}
QGraphicsView { border: 1px solid #C9D2DE; border-radius: 8px; }
QListWidget::item { padding: 3px 4px; }
QListWidget::item:selected { background: #2F6FDE; color: white; }
"""

DEMO_SEQUENCE = [10, 20, 5, 15, 25, 30, 8, 12, 28, 40, 35, 22, 2, 18, 42, 33]


def vline():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f


def legend_html():
    items = [
        ("#FFE9A8", "走訪中"), ("#FFFBE8", "已走過路徑"), ("#C9F2C7", "成功/新增"),
        ("#FFD9D9", "溢位/不足/失敗"), ("#ECDFFC", "分裂"), ("#D8E7FF", "合併"),
        ("#D3F2F1", "借鍵"), ("#FFE7CC", "操作目標"),
    ]
    chips = "&nbsp; ".join(
        f"<span style='color:{c}'>██</span><span style='font-size:9pt'>{t}</span>"
        for c, t in items
    )
    return f"<b>顏色圖例</b><br>{chips}"


def rules_html(order):
    min_leaf = math.ceil((order - 1) / 2)
    min_child = math.ceil(order / 2)
    return (
        f"<b>B+ 樹規則(階數 m = {order})</b><br>"
        f"<span style='font-size:9pt'>"
        f"• 內部節點:子節點數 {min_child} ~ {order} 個,鍵數 = 子節點數 − 1<br>"
        f"• 葉節點:鍵數 {min_leaf} ~ {order - 1} 個,以鏈結串列由小到大相連<br>"
        f"• 根節點特例:不受最低限制(可少至 1 個鍵或單一葉節點)<br>"
        f"• 資料只存在葉節點,內部節點的鍵只是導引搜尋的「路標」<br>"
        f"• 插入溢位 → <span style='color:#8E5FD3'><b>分裂</b></span>;"
        f"刪除不足 → 先<span style='color:#2BA8A0'><b>借鍵</b></span>,"
        f"否則<span style='color:#3B6FD4'><b>合併</b></span>"
        f"</span>"
    )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("B+ 樹視覺化教學 — Qt6 互動動畫")
        self.resize(1380, 860)

        self.tree = BPlusTree(order=4)
        self.steps = []
        self.cur = -1
        self.playing = False
        self.dwell = QTimer(self)            # 每一步顯示完之後的停留時間
        self.dwell.setSingleShot(True)
        self.dwell.timeout.connect(self._dwell_done)

        self._build_ui()
        self.load_steps(
            [self.tree.current_state("歡迎使用 B+ 樹視覺化教學!輸入數字後按「插入」,"
                                     "或直接點「範例建樹」觀看完整動畫示範")],
            autoplay=False,
        )

    # ================================================================ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(8)

        # ---- 操作列 ----
        ops = QHBoxLayout()
        ops.setSpacing(6)
        ops.addWidget(QLabel("數值:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("輸入整數,可用逗號一次多個,如:5, 17, 42")
        self.input_edit.setFixedWidth(250)
        self.input_edit.returnPressed.connect(self._op_insert)
        ops.addWidget(self.input_edit)

        btn_insert = QPushButton("插入")
        btn_insert.setObjectName("primary")
        btn_insert.clicked.connect(self._op_insert)
        ops.addWidget(btn_insert)
        btn_delete = QPushButton("刪除")
        btn_delete.clicked.connect(self._op_delete)
        ops.addWidget(btn_delete)
        btn_search = QPushButton("搜尋")
        btn_search.clicked.connect(self._op_search)
        ops.addWidget(btn_search)

        ops.addWidget(vline())
        ops.addWidget(QLabel("範圍:"))
        self.range_lo = QSpinBox()
        self.range_lo.setRange(-999, 999)
        self.range_lo.setValue(10)
        ops.addWidget(self.range_lo)
        ops.addWidget(QLabel("~"))
        self.range_hi = QSpinBox()
        self.range_hi.setRange(-999, 999)
        self.range_hi.setValue(30)
        ops.addWidget(self.range_hi)
        btn_range = QPushButton("範圍查詢")
        btn_range.clicked.connect(self._op_range)
        ops.addWidget(btn_range)

        ops.addWidget(vline())
        btn_random = QPushButton("隨機插入")
        btn_random.clicked.connect(self._op_random)
        ops.addWidget(btn_random)
        btn_demo = QPushButton("範例建樹")
        btn_demo.clicked.connect(self._op_demo)
        ops.addWidget(btn_demo)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._op_clear)
        ops.addWidget(btn_clear)

        ops.addWidget(vline())
        ops.addWidget(QLabel("階數:"))
        self.order_combo = QComboBox()
        self.order_combo.addItems(["3", "4", "5", "6"])
        self.order_combo.setCurrentText("4")
        self.order_combo.currentTextChanged.connect(self._order_changed)
        ops.addWidget(self.order_combo)
        ops.addStretch(1)
        root.addLayout(ops)

        # ---- 步驟解說橫幅 ----
        self.desc_label = QLabel("")
        self.desc_label.setObjectName("desc")
        self.desc_label.setWordWrap(True)
        self.desc_label.setMinimumHeight(56)
        root.addWidget(self.desc_label)

        # ---- 中央:畫布 + 右側面板 ----
        split = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = TreeCanvas()
        self.canvas.userZoomed.connect(lambda: self.autofit_check.setChecked(False))
        split.addWidget(self.canvas)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        rv.addWidget(QLabel("📋 步驟列表(點擊可跳到該步)"))
        self.step_list = QListWidget()
        self.step_list.setWordWrap(True)
        self.step_list.itemClicked.connect(self._step_clicked)
        rv.addWidget(self.step_list, 1)
        self.legend_label = QLabel(legend_html())
        self.legend_label.setObjectName("card")
        self.legend_label.setWordWrap(True)
        rv.addWidget(self.legend_label)
        self.rules_label = QLabel(rules_html(self.tree.order))
        self.rules_label.setObjectName("card")
        self.rules_label.setWordWrap(True)
        rv.addWidget(self.rules_label)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 340])
        root.addWidget(split, 1)

        # ---- 播放控制列 ----
        bar = QHBoxLayout()
        bar.setSpacing(6)
        btn_first = QPushButton("⏮ 最前")
        btn_first.clicked.connect(lambda: self._manual_goto(0))
        bar.addWidget(btn_first)
        btn_prev = QPushButton("◀ 上一步")
        btn_prev.clicked.connect(lambda: self._manual_goto(self.cur - 1))
        bar.addWidget(btn_prev)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setObjectName("primary")
        self.btn_play.setFixedWidth(110)
        self.btn_play.clicked.connect(self._toggle_play)
        bar.addWidget(self.btn_play)
        btn_next = QPushButton("下一步 ▶")
        btn_next.clicked.connect(lambda: self._manual_goto(self.cur + 1))
        bar.addWidget(btn_next)
        btn_last = QPushButton("⏭ 最後")
        btn_last.clicked.connect(lambda: self._manual_goto(len(self.steps) - 1, animate=False))
        bar.addWidget(btn_last)

        bar.addWidget(vline())
        bar.addWidget(QLabel("速度:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 300)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_value_label.setText(f"{v}%"))
        bar.addWidget(self.speed_slider)
        self.speed_value_label = QLabel("100%")
        self.speed_value_label.setFixedWidth(44)
        bar.addWidget(self.speed_value_label)

        bar.addWidget(vline())
        self.progress_label = QLabel("步驟 0 / 0")
        bar.addWidget(self.progress_label)
        bar.addStretch(1)
        self.autofit_check = QCheckBox("自動縮放")
        self.autofit_check.setChecked(True)
        self.autofit_check.toggled.connect(self._autofit_toggled)
        bar.addWidget(self.autofit_check)
        btn_fit = QPushButton("置中")
        btn_fit.clicked.connect(self.canvas.fit_all)
        bar.addWidget(btn_fit)
        root.addLayout(bar)

        # ---- 狀態列 ----
        self.stats_label = QLabel("")
        self.statusBar().addPermanentWidget(self.stats_label)
        self.statusBar().showMessage("滑鼠滾輪:縮放|按住拖曳:平移畫面", 8000)

    # ================================================================ 播放器
    def _speed(self):
        return self.speed_slider.value() / 100.0

    def _move_ms(self):
        return max(40, int(380 / self._speed()))

    def _dwell_ms(self):
        return max(120, int(900 / self._speed()))

    def load_steps(self, steps, autoplay=True):
        if not steps:
            return
        self.dwell.stop()
        self.steps = steps
        self.step_list.clear()
        for j, s in enumerate(steps, 1):
            self.step_list.addItem(f"{j}. {s['desc']}")
        self.playing = autoplay
        self.btn_play.setText("⏸ 暫停" if self.playing else "▶ 播放")
        self.cur = -1
        self.goto(0)

    def goto(self, i, animate=True):
        if not self.steps:
            return
        self.dwell.stop()
        i = max(0, min(i, len(self.steps) - 1))
        self.cur = i
        snap = self.steps[i]
        self.desc_label.setText(snap["desc"])
        self.step_list.setCurrentRow(i)
        self.progress_label.setText(f"步驟 {i + 1} / {len(self.steps)}")
        self._update_stats(snap)
        self.canvas.show_snapshot(snap,
                                  duration=self._move_ms() if animate else 0,
                                  on_finished=self._step_settled)

    def _manual_goto(self, i, animate=True):
        self._set_playing(False)
        self.goto(i, animate=animate)

    def _step_clicked(self, item):
        self._set_playing(False)
        self.goto(self.step_list.row(item))

    def _step_settled(self):
        """一步的移動動畫完成後呼叫:若在自動播放,排定停留後前進。"""
        if not self.playing:
            return
        if self.cur >= len(self.steps) - 1:
            self._set_playing(False)
            return
        self.dwell.start(self._dwell_ms())

    def _dwell_done(self):
        if self.playing:
            self.goto(self.cur + 1)

    def _toggle_play(self):
        if not self.steps:
            return
        if self.playing:
            self._set_playing(False)
        else:
            if self.cur >= len(self.steps) - 1:
                self.cur = -1               # 播到底之後再按 → 從頭重播
            self._set_playing(True)
            self.goto(self.cur + 1)

    def _set_playing(self, on):
        self.playing = on
        self.dwell.stop()
        self.btn_play.setText("⏸ 暫停" if on else "▶ 播放")

    def _autofit_toggled(self, on):
        self.canvas.auto_fit = on
        if on:
            self.canvas.fit_all()

    def _update_stats(self, snap):
        nodes = snap["nodes"]
        n_keys = sum(len(nd["keys"]) for nd in nodes.values() if nd["is_leaf"])
        h, nid = 1, snap["root"]
        while nodes[nid]["children"]:
            nid = nodes[nid]["children"][0]
            h += 1
        self.stats_label.setText(
            f"階數:{self.tree.order}   樹高:{h}   節點數:{len(nodes)}   鍵數:{n_keys} ")

    # ================================================================ 操作
    def _parse_values(self):
        text = self.input_edit.text().strip()
        if not text:
            self.statusBar().showMessage("請先在輸入框輸入數字", 4000)
            return []
        vals = []
        for part in re.split(r"[,,、;;\s]+", text):
            if not part:
                continue
            try:
                v = int(part)
            except ValueError:
                self.statusBar().showMessage(f"「{part}」不是整數,請重新輸入", 5000)
                return []
            if not -999 <= v <= 999:
                self.statusBar().showMessage("數值請介於 -999 ~ 999", 5000)
                return []
            vals.append(v)
        return vals

    def _run_ops(self, values, op):
        steps = []
        for v in values:
            op(v)
            steps.extend(self.tree.steps)
        if steps:
            self.load_steps(steps)

    def _op_insert(self):
        vals = self._parse_values()
        if vals:
            self._run_ops(vals, self.tree.insert)
            self.input_edit.selectAll()

    def _op_delete(self):
        vals = self._parse_values()
        if vals:
            self._run_ops(vals, self.tree.delete)
            self.input_edit.selectAll()

    def _op_search(self):
        vals = self._parse_values()
        if vals:
            self._run_ops(vals, self.tree.search)

    def _op_range(self):
        lo, hi = self.range_lo.value(), self.range_hi.value()
        if lo > hi:
            lo, hi = hi, lo
            self.statusBar().showMessage(f"已自動把範圍調整為 {lo} ~ {hi}", 4000)
        self.tree.range_search(lo, hi)
        self.load_steps(self.tree.steps)

    def _op_random(self):
        existing = set(self.tree.all_keys())
        pool = [v for v in range(1, 100) if v not in existing]
        if not pool:
            self.statusBar().showMessage("1~99 已全部插入,沒有可用的隨機值", 5000)
            return
        v = random.choice(pool)
        self.input_edit.setText(str(v))
        self.tree.insert(v)
        self.load_steps(self.tree.steps)

    def _op_demo(self):
        self.tree = BPlusTree(int(self.order_combo.currentText()))
        steps = [self.tree.current_state(
            f"【範例建樹】從空樹開始,依序插入:{DEMO_SEQUENCE}(過程會看到分裂與長高)")]
        for v in DEMO_SEQUENCE:
            self.tree.insert(v)
            steps.extend(self.tree.steps)
        self.load_steps(steps)

    def _op_clear(self):
        if self.tree.all_keys():
            ret = QMessageBox.question(self, "清空", "確定要清空整棵樹嗎?")
            if ret != QMessageBox.StandardButton.Yes:
                return
        self.tree = BPlusTree(int(self.order_combo.currentText()))
        self.load_steps([self.tree.current_state("已清空,從空樹重新開始")], autoplay=False)

    def _order_changed(self, text):
        m = int(text)
        if m == self.tree.order:
            return
        keys = self.tree.all_keys()
        self.tree = BPlusTree(m)
        for k in keys:
            self.tree.insert(k)
        self.tree.steps = []
        self.rules_label.setText(rules_html(m))
        self.load_steps(
            [self.tree.current_state(f"階數已改為 {m},原有的 {len(keys)} 個鍵已重新建樹")],
            autoplay=False,
        )


def make_app_font():
    f = QFont()
    f.setFamilies(["Microsoft JhengHei UI", "Microsoft JhengHei",
                   "Noto Sans TC", "Segoe UI", "sans-serif"])
    f.setPointSize(10)
    return f


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(make_app_font())
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
