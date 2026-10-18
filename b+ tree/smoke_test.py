"""GUI 煙霧測試:以 offscreen 模式跑完整流程並輸出截圖(不開視窗)。"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtGui import QFontDatabase  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import main as appmod  # noqa: E402


def load_system_fonts():
    """offscreen 平台不會載入系統字型,手動註冊讓截圖有正確文字。"""
    for name in ("msjh.ttc", "consola.ttf", "segoeui.ttf", "seguisym.ttf"):
        path = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)

SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")


def grab(win, app, name):
    app.processEvents()
    path = os.path.join(SHOT_DIR, name)
    ok = win.grab().save(path)
    assert ok, f"save failed: {name}"
    print("saved", name)


def run():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(SHOT_DIR, exist_ok=True)
    app = QApplication([])
    load_system_fonts()
    app.setStyle("Fusion")
    app.setFont(appmod.make_app_font())
    app.setStyleSheet(appmod.STYLESHEET)
    win = appmod.MainWindow()
    win.resize(1500, 950)
    win.show()
    app.processEvents()

    # 1) 建一棵高度 3 的樹
    for v in [10, 20, 5, 15, 25, 30, 8, 12, 28, 40, 35, 42]:
        win.tree.insert(v)
    win.load_steps([win.tree.current_state("煙霧測試:已建立範例樹")], autoplay=False)
    win.goto(0, animate=False)
    grab(win, app, "01_tree.png")

    # 2) 插入觸發溢位 → 分裂
    win.tree.insert(13)
    win.tree.insert(14)   # 葉節點 [10,12,13] + 14 → 溢位
    steps = win.tree.steps
    assert any("溢位" in s["desc"] for s in steps), "預期出現溢位步驟"
    win.load_steps(steps, autoplay=False)
    idx = next(i for i, s in enumerate(steps) if "溢位" in s["desc"])
    win.goto(idx, animate=False)
    grab(win, app, "02_overflow.png")
    win.goto(len(steps) - 1, animate=False)
    grab(win, app, "03_after_split.png")

    # 3) 刪除觸發借鍵 / 合併
    merge_steps = None
    for v in [42, 40, 35, 30, 28, 25]:
        win.tree.delete(v)
        if any("合併" in s["desc"] for s in win.tree.steps):
            merge_steps = win.tree.steps
            break
    assert merge_steps, "預期出現合併步驟"
    win.load_steps(merge_steps, autoplay=False)
    idx = next(i for i, s in enumerate(merge_steps) if "合併" in s["desc"])
    win.goto(idx, animate=False)
    grab(win, app, "04_merge.png")
    win.goto(len(merge_steps) - 1, animate=False)
    grab(win, app, "05_after_delete.png")

    # 4) 範圍查詢
    win.tree.range_search(8, 20)
    win.load_steps(win.tree.steps, autoplay=False)
    win.goto(len(win.tree.steps) - 1, animate=False)
    grab(win, app, "06_range.png")

    # 5) 動畫路徑也跑一次(有 duration 的補間)確保不丟例外
    win.tree.insert(99)
    win.load_steps(win.tree.steps, autoplay=True)
    for _ in range(60):
        app.processEvents()
    win.goto(len(win.steps) - 1, animate=False)   # 中途打斷 → 立即收尾
    app.processEvents()

    # 6) 階數切換重建
    win.order_combo.setCurrentText("3")
    app.processEvents()
    assert win.tree.order == 3
    win.tree.validate()
    win.goto(0, animate=False)
    grab(win, app, "07_order3.png")

    print("SMOKE OK")


if __name__ == "__main__":
    run()
