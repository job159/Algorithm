# -*- coding: utf-8 -*-
"""
二元搜尋樹 (Binary Search Tree) 核心邏輯 —— 含「逐步教學」記錄功能。

刻意與 rbtree.RBTree 提供相同的介面與「step 快照格式」，
所以可以直接套用同一套視覺化畫布：
    insert / delete / search 都回傳一串 step；
    keys_inorder / size / height / validate 提供統計與驗證。

教學重點：BST 沒有自動平衡，依序插入會「退化成鏈狀」，高度變成 O(n)，
正好和紅黑樹（高度保證 O(log n)）形成對照。

可單獨執行做自我測試：
    python bst.py
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional

NEUTRAL = "N"  # BST 節點沒有紅黑顏色，畫布會把 'N' 畫成中性藍色


class Node:
    __slots__ = ("id", "key", "color", "left", "right", "parent")

    def __init__(self, node_id: int, key):
        self.id = node_id
        self.key = key
        self.color = NEUTRAL
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.parent: Optional["Node"] = None


Step = Dict


class BST:
    def __init__(self):
        self._next_id = 1
        self.root: Optional[Node] = None
        self.steps: List[Step] = []
        self._recording = False

    # ------------------------------------------------------------------ #
    #  節點建立 / 步驟記錄（與 RBTree 相同的快照格式）
    # ------------------------------------------------------------------ #
    def _new_node(self, key) -> Node:
        n = Node(self._next_id, key)
        self._next_id += 1
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

        def walk(n: Optional[Node]):
            if n is None:
                return None
            nodes[n.id] = {
                "key": n.key,
                "color": n.color,
                "left": n.left.id if n.left is not None else None,
                "right": n.right.id if n.right is not None else None,
                "parent": n.parent.id if n.parent is not None else None,
            }
            walk(n.left)
            walk(n.right)
            return n.id

        root_id = walk(self.root)
        hl = set()
        for h in highlights:
            if h is not None:
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
    #  搜尋
    # ------------------------------------------------------------------ #
    def search_node(self, key) -> Optional[Node]:
        x = self.root
        while x is not None and key != x.key:
            x = x.left if key < x.key else x.right
        return x

    def search(self, key) -> List[Step]:
        self._begin()
        x = self.root
        path: List[Node] = []
        if x is None:
            self._emit("空樹，找不到鍵值 %s。" % key)
            self._end()
            return self.steps
        while x is not None:
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
        self._emit("走到底仍未找到，鍵值 %s 不在樹中。" % key, highlights=path)
        self._end()
        return self.steps

    # ------------------------------------------------------------------ #
    #  插入
    # ------------------------------------------------------------------ #
    def insert(self, key) -> List[Step]:
        self._begin()
        if self.search_node(key) is not None:
            self._emit("鍵值 %s 已存在，略過插入。" % key)
            self._end()
            return self.steps

        z = self._new_node(key)
        if self.root is None:
            self.root = z
            self._emit("空樹：節點 %s 直接成為根節點。" % key, highlights=[z])
            self._end()
            return self.steps

        x = self.root
        y = None
        path: List[Node] = []
        while x is not None:
            y = x
            path.append(x)
            if z.key < x.key:
                self._emit("%s ＜ %s，往左子樹找插入位置。" % (key, x.key), highlights=path)
                x = x.left
            else:
                self._emit("%s ＞ %s，往右子樹找插入位置。" % (key, x.key), highlights=path)
                x = x.right
        z.parent = y
        if z.key < y.key:
            y.left = z
            side = "左"
        else:
            y.right = z
            side = "右"
        self._emit("找到空位，把節點 %s 接到 %s 的「%s」子節點。" % (key, y.key, side),
                   highlights=[z, y])
        self._end()
        return self.steps

    # ------------------------------------------------------------------ #
    #  刪除（標準 BST 刪除，含三種情況）
    # ------------------------------------------------------------------ #
    def _transplant(self, u: Node, v: Optional[Node]):
        if u.parent is None:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None:
            v.parent = u.parent

    def _minimum(self, x: Node) -> Node:
        while x.left is not None:
            x = x.left
        return x

    def delete(self, key) -> List[Step]:
        self._begin()
        z = self.search_node(key)
        if z is None:
            self._emit("找不到鍵值 %s，無法刪除。" % key)
            self._end()
            return self.steps

        self._emit("找到要刪除的節點 %s。" % key, highlights=[z])
        if z.left is None:
            self._emit("情況：節點 %s 沒有左子節點，直接用右子樹（可能是空）頂上來。" % key,
                       highlights=[z])
            self._transplant(z, z.right)
        elif z.right is None:
            self._emit("情況：節點 %s 沒有右子節點，直接用左子樹頂上來。" % key, highlights=[z])
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            self._emit(
                "情況：節點 %s 有兩個子節點。找出「中序後繼」%s（右子樹最小值）來頂替它的位置。"
                % (key, y.key),
                highlights=[z, y],
            )
            if y.parent is not z:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            self._emit("後繼節點 %s 移到原位置，刪除完成。" % y.key, highlights=[y])

        self._emit("刪除完成。",
                   highlights=[self.root] if self.root is not None else [])
        self._end()
        return self.steps

    # ------------------------------------------------------------------ #
    #  工具：中序鍵值、樹高、驗證
    # ------------------------------------------------------------------ #
    def keys_inorder(self) -> List:
        out: List = []

        def walk(n: Optional[Node]):
            if n is None:
                return
            walk(n.left)
            out.append(n.key)
            walk(n.right)

        walk(self.root)
        return out

    def size(self) -> int:
        return len(self.keys_inorder())

    def height(self) -> int:
        def h(n: Optional[Node]) -> int:
            if n is None:
                return 0
            return 1 + max(h(n.left), h(n.right))

        return h(self.root)

    def validate(self):
        """檢查 BST 性質（中序遞增），回傳 (是否正確, 說明字串)。"""
        keys = self.keys_inorder()
        ok = all(keys[i] < keys[i + 1] for i in range(len(keys) - 1))
        if not ok:
            return False, "違反 BST 性質：中序走訪沒有遞增"
        n = len(keys)
        if n == 0:
            return True, "空樹（合法）"
        h = self.height()
        if n >= 4 and h == n:
            return True, "符合 BST 性質（注意：已退化成鏈狀，高度 = %d）" % h
        return True, "符合 BST 性質（左 ＜ 節點 ＜ 右）"


# ---------------------------------------------------------------------- #
#  自我測試
# ---------------------------------------------------------------------- #
def _self_test():
    rng = random.Random(20260610)
    for _ in range(40):
        t = BST()
        present = set()
        keys = list(range(1, 80))
        rng.shuffle(keys)
        for k in keys:
            t.insert(k)
            present.add(k)
            ok, msg = t.validate()
            assert ok, "insert %s -> %s" % (k, msg)
            assert t.keys_inorder() == sorted(present)
        rng.shuffle(keys)
        for k in keys:
            t.delete(k)
            present.discard(k)
            ok, msg = t.validate()
            assert ok, "delete %s -> %s" % (k, msg)
            assert t.keys_inorder() == sorted(present)
        assert t.root is None
    # 退化測試：依序插入應變成高度 == n 的鏈狀
    t = BST()
    for k in range(1, 11):
        t.insert(k)
    assert t.height() == 10, "依序插入應退化成鏈狀，height=%d" % t.height()
    print("BST 自我測試通過：隨機插入/刪除全程維持 BST 性質，且依序插入會退化成鏈狀。")


if __name__ == "__main__":
    _self_test()
