# -*- coding: utf-8 -*-
"""
AVL 樹核心邏輯 —— 含「逐步教學」記錄功能。

AVL 是「嚴格平衡」的二元搜尋樹：每個節點的「平衡因子」
（左子樹高 − 右子樹高）必須是 −1、0、+1。一旦插入/刪除讓某節點的
平衡因子變成 ±2，就用旋轉修正：
    LL（左左）→ 單次右旋
    RR（右右）→ 單次左旋
    LR（左右）→ 先左旋再右旋
    RL（右左）→ 先右旋再左旋

刻意與 rbtree.RBTree / bst.BST 提供相同的介面與「step 快照格式」，
可直接套用同一套視覺化畫布；另外每個節點的快照多帶一個 "tag"
欄位（平衡因子字串），畫布會把它畫成節點上的小徽章。

可單獨執行做自我測試：
    python avl.py
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional

AVL_COLOR = "A"  # 畫布會把 'A' 畫成中性綠色


class Node:
    __slots__ = ("id", "key", "color", "height", "left", "right", "parent")

    def __init__(self, node_id: int, key):
        self.id = node_id
        self.key = key
        self.color = AVL_COLOR
        self.height = 1
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.parent: Optional["Node"] = None


Step = Dict


class AVL:
    def __init__(self):
        self._next_id = 1
        self.root: Optional[Node] = None
        self.steps: List[Step] = []
        self._recording = False

    # ------------------------------------------------------------------ #
    #  基本工具
    # ------------------------------------------------------------------ #
    def _new_node(self, key) -> Node:
        n = Node(self._next_id, key)
        self._next_id += 1
        return n

    @staticmethod
    def _h(n: Optional[Node]) -> int:
        return n.height if n is not None else 0

    def _bf(self, n: Optional[Node]) -> int:
        return self._h(n.left) - self._h(n.right) if n is not None else 0

    def _update_height(self, n: Node):
        n.height = 1 + max(self._h(n.left), self._h(n.right))

    @staticmethod
    def _fmt_bf(b: int) -> str:
        return "0" if b == 0 else ("%+d" % b)

    # ------------------------------------------------------------------ #
    #  步驟記錄（與其他樹相同的快照格式，外加每個節點的 tag = 平衡因子）
    # ------------------------------------------------------------------ #
    def _begin(self):
        self.steps = []
        self._recording = True

    def _end(self):
        self._recording = False

    def _emit(self, desc: str, highlights: Iterable[Node] = (), prop: Optional[int] = None):
        if not self._recording:
            return
        nodes: Dict[int, Dict] = {}

        def walk(n: Optional[Node], parent_id: Optional[int]):
            if n is None:
                return None
            bf = self._h(n.left) - self._h(n.right)
            nodes[n.id] = {
                "key": n.key,
                "color": n.color,
                "left": n.left.id if n.left is not None else None,
                "right": n.right.id if n.right is not None else None,
                "parent": parent_id,
                "tag": self._fmt_bf(bf),
            }
            walk(n.left, n.id)
            walk(n.right, n.id)
            return n.id

        root_id = walk(self.root, None)
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
    #  旋轉（會重新連結父節點並更新高度）
    # ------------------------------------------------------------------ #
    def _replace_child(self, parent: Optional[Node], old: Node, new: Optional[Node]):
        if parent is None:
            self.root = new
        elif parent.left is old:
            parent.left = new
        else:
            parent.right = new
        if new is not None:
            new.parent = parent

    def _left_rotate(self, x: Node) -> Node:
        y = x.right
        p = x.parent
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.left = x
        x.parent = y
        self._replace_child(p, x, y)
        self._update_height(x)
        self._update_height(y)
        return y

    def _right_rotate(self, x: Node) -> Node:
        y = x.left
        p = x.parent
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.right = x
        x.parent = y
        self._replace_child(p, x, y)
        self._update_height(x)
        self._update_height(y)
        return y

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
            if key < x.key:
                self._emit("%s ＜ %s，往左子樹找插入位置。" % (key, x.key), highlights=path)
                x = x.left
            else:
                self._emit("%s ＞ %s，往右子樹找插入位置。" % (key, x.key), highlights=path)
                x = x.right
        z.parent = y
        if key < y.key:
            y.left = z
            side = "左"
        else:
            y.right = z
            side = "右"
        self._emit("把節點 %s 接到 %s 的「%s」子節點，從這裡往上回溯檢查平衡。"
                   % (key, y.key, side), highlights=[z, y])
        self._rebalance_after_insert(z)
        self._end()
        return self.steps

    def _rebalance_after_insert(self, z: Node):
        n = z.parent
        while n is not None:
            old_h = n.height
            self._update_height(n)
            b = self._bf(n)
            self._emit("回溯到節點 %s：更新高度後平衡因子 = %s。"
                       % (n.key, self._fmt_bf(b)), highlights=[n])
            if b > 1 or b < -1:
                self._rotate_case(n)
                self._emit("旋轉完成，子樹高度恢復原狀，上層不受影響，插入修復結束。",
                           highlights=[])
                break
            if n.height == old_h:
                self._emit("節點 %s 高度沒有改變，上層平衡不受影響，插入完成。" % n.key,
                           highlights=[n])
                break
            n = n.parent

    # ------------------------------------------------------------------ #
    #  刪除
    # ------------------------------------------------------------------ #
    def _transplant(self, u: Node, v: Optional[Node]):
        p = u.parent
        if p is None:
            self.root = v
        elif u is p.left:
            p.left = v
        else:
            p.right = v
        if v is not None:
            v.parent = p

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
        if z.left is not None and z.right is not None:
            s = self._minimum(z.right)
            self._emit("節點 %s 有兩個子節點，找中序後繼 %s（右子樹最小值）來頂替。"
                       % (z.key, s.key), highlights=[z, s])
            if s.parent is z:
                start = s
            else:
                start = s.parent
                self._transplant(s, s.right)
                s.right = z.right
                if s.right is not None:
                    s.right.parent = s
            self._transplant(z, s)
            s.left = z.left
            if s.left is not None:
                s.left.parent = s
            self._update_height(s)
            self._emit("後繼 %s 移到原位置，從受影響的最深節點往上回溯。" % s.key, highlights=[s])
        else:
            child = z.left if z.left is not None else z.right
            start = z.parent
            self._transplant(z, child)
            self._emit("節點 %s 至多一個子節點，直接用%s接上，往上回溯檢查平衡。"
                       % (key, "子樹" if child is not None else "空（移除）"),
                       highlights=[start] if start is not None else [])

        self._rebalance_after_delete(start)
        self._end()
        return self.steps

    def _rebalance_after_delete(self, n: Optional[Node]):
        while n is not None:
            parent = n.parent  # 旋轉會改變 n.parent，先記下來
            self._update_height(n)
            b = self._bf(n)
            self._emit("回溯到節點 %s：更新高度後平衡因子 = %s。"
                       % (n.key, self._fmt_bf(b)), highlights=[n])
            if b > 1 or b < -1:
                self._rotate_case(n)
                self._emit("旋轉完成，繼續往上檢查（刪除可能需要多次旋轉）。", highlights=[])
            n = parent
        self._emit("回溯到根節點，刪除完成，整棵樹維持 AVL 平衡。",
                   highlights=[self.root] if self.root is not None else [])

    # ------------------------------------------------------------------ #
    #  四種旋轉情況
    # ------------------------------------------------------------------ #
    def _rotate_case(self, n: Node):
        b = self._bf(n)
        if b > 1:
            if self._bf(n.left) < 0:
                self._emit("左右（LR）情況：先對 %s 的左子節點 %s 做左旋。"
                           % (n.key, n.left.key), highlights=[n, n.left])
                self._left_rotate(n.left)
                self._emit("再對 %s 做右旋。" % n.key, highlights=[n])
            else:
                self._emit("左左（LL）情況：對 %s 做一次右旋即可。" % n.key, highlights=[n])
            self._right_rotate(n)
        else:  # b < -1
            if self._bf(n.right) > 0:
                self._emit("右左（RL）情況：先對 %s 的右子節點 %s 做右旋。"
                           % (n.key, n.right.key), highlights=[n, n.right])
                self._right_rotate(n.right)
                self._emit("再對 %s 做左旋。" % n.key, highlights=[n])
            else:
                self._emit("右右（RR）情況：對 %s 做一次左旋即可。" % n.key, highlights=[n])
            self._left_rotate(n)

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
        return self._h(self.root)

    def validate(self):
        """檢查 BST 性質 + AVL 平衡 + 高度欄位，回傳 (是否正確, 說明字串)。"""
        keys = self.keys_inorder()
        if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
            return False, "違反 BST 性質：中序走訪沒有遞增"

        bad = [None]

        def chk(n: Optional[Node]) -> int:
            if n is None:
                return 0
            lh = chk(n.left)
            rh = chk(n.right)
            if abs(lh - rh) > 1 and bad[0] is None:
                bad[0] = "違反 AVL 平衡：節點 %s 的 |平衡因子| > 1" % n.key
            real_h = 1 + max(lh, rh)
            if n.height != real_h and bad[0] is None:
                bad[0] = "節點 %s 的高度欄位不正確" % n.key
            return real_h

        chk(self.root)
        if bad[0]:
            return False, bad[0]
        if not keys:
            return True, "空樹（合法）"
        return True, "符合 AVL：每個節點 |平衡因子| ≤ 1（樹高 %d）" % self.height()


# ---------------------------------------------------------------------- #
#  自我測試
# ---------------------------------------------------------------------- #
def _self_test():
    rng = random.Random(20260610)
    for _ in range(40):
        t = AVL()
        present = set()
        keys = list(range(1, 100))
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
    # 依序插入應仍保持平衡（高度 ≈ log n，遠小於節點數）
    t = AVL()
    n = 1023
    for k in range(1, n + 1):
        t.insert(k)
    bound = 1.44 * math.log2(n + 2)
    assert t.height() <= bound, "height %d 超過 AVL 上界 %.1f" % (t.height(), bound)
    print("AVL 自我測試通過：隨機插入/刪除維持 AVL 平衡；"
          "依序插入 1..%d 高度仍只有 %d（約 log n，不會退化）。" % (n, t.height()))


if __name__ == "__main__":
    _self_test()
