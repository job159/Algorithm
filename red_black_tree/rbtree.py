# -*- coding: utf-8 -*-
"""
紅黑樹 (Red-Black Tree) 核心邏輯 —— 含「逐步教學」記錄功能。

這個模組不依賴任何 GUI，可單獨執行做正確性自我測試：
    python rbtree.py

設計重點：
- 採用 CLRS 教科書版本（含 NIL 哨兵節點），insert / delete / search 完全正確。
- 每個有意義的步驟都會呼叫 self._emit(...)，把「當下整棵樹的快照 + 中文說明 +
  要強調的節點 + 相關的紅黑性質」記錄成一個 step，供視覺化介面逐格播放。
- 每個節點有一個永久的內部 id，視覺化時用 id 來辨識同一個節點做位移動畫
  （旋轉時節點只是換位置，id 不變；插入時出現新 id；刪除時 id 消失）。
"""
from __future__ import annotations

import random
from typing import List, Dict, Optional, Iterable

RED = "R"
BLACK = "B"


class Node:
    __slots__ = ("id", "key", "color", "left", "right", "parent")

    def __init__(self, node_id: int, key, color: str):
        self.id = node_id
        self.key = key
        self.color = color
        self.left: "Node" = None  # type: ignore
        self.right: "Node" = None  # type: ignore
        self.parent: "Node" = None  # type: ignore


# 一個 step（快照）長這樣：
#   {
#     "desc": str,                 # 中文說明文字
#     "highlights": set[int],      # 要強調的節點 id
#     "prop": Optional[int],       # 相關的紅黑性質編號 1..5（給側欄高亮用）
#     "nodes": {id: {key,color,left,right,parent}},  # 整棵樹結構
#     "root": Optional[int],       # 根節點 id
#   }
Step = Dict


class RBTree:
    def __init__(self):
        self._next_id = 1
        # NIL 哨兵節點：id=0、永遠黑色、不會被畫出來
        self.NIL = Node(0, None, BLACK)
        self.NIL.left = self.NIL.right = self.NIL.parent = self.NIL
        self.root: Node = self.NIL
        self.steps: List[Step] = []
        self._recording = False

    # ------------------------------------------------------------------ #
    #  節點建立 / 步驟記錄
    # ------------------------------------------------------------------ #
    def _new_node(self, key) -> Node:
        n = Node(self._next_id, key, RED)
        self._next_id += 1
        n.left = n.right = n.parent = self.NIL
        return n

    def _begin(self):
        self.steps = []
        self._recording = True

    def _end(self):
        self._recording = False

    def _emit(self, desc: str, highlights: Iterable[Node] = (), prop: Optional[int] = None):
        if not self._recording:
            return
        nodes: Dict[int, Dict] = {}

        def walk(n: Node):
            if n is self.NIL:
                return None
            nodes[n.id] = {
                "key": n.key,
                "color": n.color,
                "left": n.left.id if n.left is not self.NIL else None,
                "right": n.right.id if n.right is not self.NIL else None,
                "parent": n.parent.id if n.parent is not self.NIL else None,
            }
            walk(n.left)
            walk(n.right)
            return n.id

        root_id = walk(self.root)
        hl = set()
        for h in highlights:
            if h is not None and h is not self.NIL:
                hl.add(h.id)
        self.steps.append(
            {
                "desc": desc,
                "highlights": hl,
                "prop": prop,
                "nodes": nodes,
                "root": root_id,
            }
        )

    # ------------------------------------------------------------------ #
    #  旋轉
    # ------------------------------------------------------------------ #
    def _left_rotate(self, x: Node):
        y = x.right
        x.right = y.left
        if y.left is not self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x: Node):
        y = x.left
        x.left = y.right
        if y.right is not self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self.NIL:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    # ------------------------------------------------------------------ #
    #  搜尋
    # ------------------------------------------------------------------ #
    def search_node(self, key) -> Node:
        x = self.root
        while x is not self.NIL and key != x.key:
            x = x.left if key < x.key else x.right
        return x

    def search(self, key) -> List[Step]:
        self._begin()
        x = self.root
        path: List[Node] = []
        if x is self.NIL:
            self._emit("空樹，找不到鍵值 %s。" % key)
            self._end()
            return self.steps
        while x is not self.NIL:
            path.append(x)
            if key == x.key:
                self._emit("在節點 %s 找到鍵值！搜尋成功。" % key, highlights=[x])
                self._end()
                return self.steps
            elif key < x.key:
                self._emit("%s ＜ %s，往左子樹繼續找。" % (key, x.key), highlights=path)
                x = x.left
            else:
                self._emit("%s ＞ %s，往右子樹繼續找。" % (key, x.key), highlights=path)
                x = x.right
        self._emit("走到 NIL 仍未找到，鍵值 %s 不在樹中。" % key, highlights=path)
        self._end()
        return self.steps

    # ------------------------------------------------------------------ #
    #  插入
    # ------------------------------------------------------------------ #
    def insert(self, key) -> List[Step]:
        self._begin()
        if self.search_node(key) is not self.NIL:
            self._emit("鍵值 %s 已存在，略過插入。" % key)
            self._end()
            return self.steps

        z = self._new_node(key)
        y = self.NIL
        x = self.root
        while x is not self.NIL:
            y = x
            x = x.left if z.key < x.key else x.right
        z.parent = y
        if y is self.NIL:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.NIL
        z.right = self.NIL
        z.color = RED
        self._emit(
            "依照二元搜尋樹規則找到位置，插入新節點 %s。新節點一律先塗成「紅色」。" % key,
            highlights=[z],
            prop=4,
        )
        self._insert_fixup(z)
        if self.root.color != BLACK:
            self.root.color = BLACK
        self._emit("插入完成，紅黑性質全部滿足。根節點保持黑色。",
                   highlights=[self.root], prop=2)
        self._end()
        return self.steps

    def _insert_fixup(self, z: Node):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                u = z.parent.parent.right  # 叔節點 (uncle)
                if u.color == RED:
                    # 情況一
                    self._emit(
                        "情況一：父節點與叔節點都是紅色（出現連續紅色，違反性質四）。"
                        "把父、叔塗黑，祖父塗紅。",
                        highlights=[z, z.parent, u, z.parent.parent],
                        prop=4,
                    )
                    z.parent.color = BLACK
                    u.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                    self._emit("重新著色完成，焦點上移到祖父節點，繼續往上檢查。",
                               highlights=[z], prop=4)
                else:
                    if z is z.parent.right:
                        # 情況二 → 轉成情況三
                        z = z.parent
                        self._emit("情況二：叔節點為黑、且目前節點是「右」子節點。先對父節點做左旋，轉成情況三。",
                                   highlights=[z], prop=4)
                        self._left_rotate(z)
                        self._emit("左旋完成。", highlights=[z, z.parent])
                    # 情況三
                    self._emit(
                        "情況三：叔節點為黑、且目前節點是「左」子節點。父節點塗黑、祖父塗紅，"
                        "再對祖父做右旋。",
                        highlights=[z, z.parent, z.parent.parent],
                        prop=4,
                    )
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
                    self._emit("右旋完成，這個分支的紅黑性質已恢復。", highlights=[z.parent])
            else:
                # 鏡像（父節點是祖父的右子節點）
                u = z.parent.parent.left
                if u.color == RED:
                    self._emit(
                        "情況一（鏡像）：父節點與叔節點都是紅色。把父、叔塗黑，祖父塗紅。",
                        highlights=[z, z.parent, u, z.parent.parent],
                        prop=4,
                    )
                    z.parent.color = BLACK
                    u.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                    self._emit("重新著色完成，焦點上移到祖父節點。", highlights=[z], prop=4)
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._emit("情況二（鏡像）：目前節點是左子節點，先對父節點做右旋。",
                                   highlights=[z], prop=4)
                        self._right_rotate(z)
                        self._emit("右旋完成。", highlights=[z, z.parent])
                    self._emit(
                        "情況三（鏡像）：父節點塗黑、祖父塗紅，再對祖父做左旋。",
                        highlights=[z, z.parent, z.parent.parent],
                        prop=4,
                    )
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
                    self._emit("左旋完成，紅黑性質已恢復。", highlights=[z.parent])

    # ------------------------------------------------------------------ #
    #  刪除
    # ------------------------------------------------------------------ #
    def _transplant(self, u: Node, v: Node):
        if u.parent is self.NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, x: Node) -> Node:
        while x.left is not self.NIL:
            x = x.left
        return x

    def delete(self, key) -> List[Step]:
        self._begin()
        z = self.search_node(key)
        if z is self.NIL:
            self._emit("找不到鍵值 %s，無法刪除。" % key)
            self._end()
            return self.steps

        self._emit("找到要刪除的節點 %s。" % key, highlights=[z])
        y = z
        y_original_color = y.color
        if z.left is self.NIL:
            x = z.right
            self._emit("節點 %s 沒有左子節點，直接用它的右子樹頂上來。" % key, highlights=[z])
            self._transplant(z, z.right)
        elif z.right is self.NIL:
            x = z.left
            self._emit("節點 %s 沒有右子節點，直接用它的左子樹頂上來。" % key, highlights=[z])
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            self._emit(
                "節點 %s 有兩個子節點。找出「中序後繼」節點 %s（右子樹最小者）來頂替它。"
                % (key, y.key),
                highlights=[z, y],
            )
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            self._emit("後繼節點 %s 移到原位置，並接管被刪節點的「顏色」與左右子樹。" % y.key,
                       highlights=[y])

        if y_original_color == BLACK:
            self._emit(
                "被移走的節點原本是「黑色」，少了一個黑色會破壞黑高（性質五），"
                "出現一個「雙重黑色」，進入刪除修復。",
                highlights=[x] if x is not self.NIL else [x.parent],
                prop=5,
            )
            self._delete_fixup(x)
        if self.root is not self.NIL and self.root.color != BLACK:
            self.root.color = BLACK
        self._emit("刪除完成，紅黑性質全部恢復。",
                   highlights=[self.root] if self.root is not self.NIL else [])
        self._end()
        return self.steps

    def _delete_fixup(self, x: Node):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right  # 兄弟節點 (sibling)
                if w.color == RED:
                    self._emit(
                        "刪除情況一：兄弟節點 %s 是紅色。把兄弟塗黑、父節點塗紅，對父節點左旋，"
                        "換一個黑色兄弟出來。" % w.key,
                        highlights=[x.parent, w],
                        prop=5,
                    )
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                    self._emit("左旋後得到新的（黑色）兄弟，轉成情況二／三／四。", highlights=[w])
                if w.left.color == BLACK and w.right.color == BLACK:
                    self._emit(
                        "刪除情況二：兄弟為黑、且兄弟的兩個子節點都是黑。把兄弟塗紅，"
                        "讓「雙重黑色」往上移到父節點，繼續處理。",
                        highlights=[w, x.parent],
                        prop=5,
                    )
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        self._emit(
                            "刪除情況三：兄弟為黑、兄弟右子為黑但左子為紅。把兄弟左子塗黑、兄弟塗紅，"
                            "對兄弟右旋，轉成情況四。",
                            highlights=[w, w.left],
                            prop=5,
                        )
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    self._emit(
                        "刪除情況四：兄弟為黑、兄弟右子為紅。兄弟接管父節點顏色，父節點與兄弟右子塗黑，"
                        "對父節點左旋，修復完成。",
                        highlights=[w, x.parent, w.right],
                        prop=5,
                    )
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                # 鏡像
                w = x.parent.left
                if w.color == RED:
                    self._emit(
                        "刪除情況一（鏡像）：兄弟節點 %s 是紅色，對父節點右旋。" % w.key,
                        highlights=[x.parent, w],
                        prop=5,
                    )
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                    self._emit("右旋後得到新的兄弟。", highlights=[w])
                if w.right.color == BLACK and w.left.color == BLACK:
                    self._emit(
                        "刪除情況二（鏡像）：兄弟與其子節點皆黑，兄弟塗紅，雙重黑色往上移。",
                        highlights=[w, x.parent],
                        prop=5,
                    )
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        self._emit(
                            "刪除情況三（鏡像）：對兄弟做左旋，轉成情況四。",
                            highlights=[w, w.right],
                            prop=5,
                        )
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    self._emit(
                        "刪除情況四（鏡像）：兄弟接管父顏色，父與兄弟左子塗黑，對父節點右旋，修復完成。",
                        highlights=[w, x.parent, w.left],
                        prop=5,
                    )
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        if x.color != BLACK:
            x.color = BLACK
            self._emit("把目前節點塗黑，吸收掉多出來的黑色，刪除修復結束。",
                       highlights=[x] if x is not self.NIL else [])

    # ------------------------------------------------------------------ #
    #  工具：中序鍵值、黑高、驗證
    # ------------------------------------------------------------------ #
    def keys_inorder(self) -> List:
        out: List = []

        def walk(n: Node):
            if n is self.NIL:
                return
            walk(n.left)
            out.append(n.key)
            walk(n.right)

        walk(self.root)
        return out

    def size(self) -> int:
        return len(self.keys_inorder())

    def height(self) -> int:
        """樹高（最長路徑的邊數；空樹為 0，單一節點為 1）。"""

        def h(n: Node) -> int:
            if n is self.NIL:
                return 0
            return 1 + max(h(n.left), h(n.right))

        return h(self.root)

    def black_height(self) -> int:
        """根節點到任一 NIL 路徑上的黑色節點數（不含根本身、含 NIL）。"""
        n = self.root
        bh = 0
        while n is not self.NIL:
            if n.color == BLACK:
                bh += 1
            n = n.left
        return bh + 1  # 加上 NIL

    def validate(self):
        """檢查五大紅黑性質，回傳 (是否正確, 說明字串)。"""
        if self.root is self.NIL:
            return True, "空樹（合法）"
        if self.root.color != BLACK:
            return False, "違反性質二：根節點不是黑色"

        bad = [None]

        def check(n: Node) -> int:
            if n is self.NIL:
                return 1  # NIL 視為一個黑色
            if n.color == RED:
                if n.left.color == RED or n.right.color == RED:
                    if bad[0] is None:
                        bad[0] = "違反性質四：紅色節點有紅色子節點"
            lh = check(n.left)
            rh = check(n.right)
            if lh != rh and bad[0] is None:
                bad[0] = "違反性質五：黑高不一致"
            return lh + (1 if n.color == BLACK else 0)

        check(self.root)
        if bad[0]:
            return False, bad[0]
        return True, "符合全部五大紅黑性質"


# ---------------------------------------------------------------------- #
#  自我測試：大量隨機 insert / delete 後驗證紅黑性質
# ---------------------------------------------------------------------- #
def _self_test():
    rng = random.Random(20260610)
    for trial in range(40):
        t = RBTree()
        present = set()
        keys = list(range(1, 120))
        rng.shuffle(keys)
        for k in keys:
            t.insert(k)
            present.add(k)
            ok, msg = t.validate()
            assert ok, "insert %s 後 %s" % (k, msg)
        assert t.keys_inorder() == sorted(present)
        rng.shuffle(keys)
        for k in keys:
            t.delete(k)
            present.discard(k)
            ok, msg = t.validate()
            assert ok, "delete %s 後 %s" % (k, msg)
            assert t.keys_inorder() == sorted(present)
        assert t.root is t.NIL
    print("自我測試通過：40 輪隨機插入/刪除，全程滿足紅黑性質。")


if __name__ == "__main__":
    _self_test()
