"""B+ 樹資料結構(教學版),內建「步驟記錄」功能。

每個操作(插入 / 刪除 / 搜尋 / 範圍查詢)執行時,會在每個關鍵時刻
記錄一份整棵樹的快照(snapshot),連同中文解說文字、要強調的節點
與鍵,提供給 GUI 逐步播放動畫使用。

階數(order)m 的規則(本實作採用「m = 內部節點最大子節點數」):
  - 內部節點:最多 m-1 個鍵、m 個子節點;非根節點至少 ceil(m/2) 個子節點
  - 葉節點:鍵數介於 ceil((m-1)/2) ~ m-1,並以鏈結串列由小到大相連
  - 根節點特例:可以少到 1 個鍵,或本身就是唯一的葉節點
  - 所有資料都存放在葉節點;內部節點的鍵只是導引搜尋的「路標」
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right


class Node:
    """B+ 樹節點。葉節點存資料鍵;內部節點存索引鍵與子節點指標。"""

    _counter = 0
    __slots__ = ("id", "is_leaf", "keys", "children", "next", "parent")

    def __init__(self, is_leaf):
        Node._counter += 1
        self.id = Node._counter          # 穩定識別碼,讓動畫能跨快照追蹤同一節點
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []               # 僅內部節點使用
        self.next = None                 # 僅葉節點使用:指向右邊的葉節點
        self.parent = None


class BPlusTree:
    def __init__(self, order=4):
        if order < 3:
            raise ValueError("階數至少為 3")
        self.order = order
        self.root = Node(is_leaf=True)
        self.steps = []                  # 最近一次操作產生的教學步驟

    # ------------------------------------------------------------------ 規則
    @property
    def max_keys(self):
        return self.order - 1

    @property
    def min_leaf_keys(self):
        return math.ceil((self.order - 1) / 2)

    @property
    def min_internal_keys(self):         # 非根的內部節點
        return math.ceil(self.order / 2) - 1

    # ------------------------------------------------------------------ 快照
    def current_state(self, desc="", hl=None, marks=None):
        """把整棵樹序列化成一個步驟(快照 + 解說 + 強調標記)。

        hl:    {node_id: 強調型別}    例如 "visit" / "split" / "merge"
        marks: {node_id: {鍵索引: 型別}} 例如 {"5": {0: "new"}}
        """
        nodes = {}

        def walk(n):
            nodes[n.id] = {
                "keys": list(n.keys),
                "is_leaf": n.is_leaf,
                "children": [c.id for c in n.children],
                "next": n.next.id if (n.is_leaf and n.next is not None) else None,
            }
            for c in n.children:
                walk(c)

        walk(self.root)
        return {
            "root": self.root.id,
            "nodes": nodes,
            "desc": desc,
            "hl": dict(hl or {}),
            "marks": {k: dict(v) for k, v in (marks or {}).items()},
        }

    def _record(self, desc, hl=None, marks=None):
        self.steps.append(self.current_state(desc, hl, marks))

    # ------------------------------------------------------------------ 共用:往下找葉節點
    def _descend(self, key, purpose="搜尋"):
        """從根節點走到 key 應該在的葉節點,沿途記錄比較過程。

        回傳 (葉節點, 走訪路徑的強調字典)。
        """
        node = self.root
        acc = {node.id: "visit"}
        if node.is_leaf:
            self._record(f"根節點本身就是葉節點,直接在裡面{purpose} {key}", hl=acc)
            return node, acc
        self._record(f"從根節點開始往下,{purpose}鍵 {key} 應該所在的葉節點", hl=acc)
        while not node.is_leaf:
            i = bisect_right(node.keys, key)
            marks = {node.id: {j: "compare" for j in range(len(node.keys))}}
            if i == 0:
                how = f"{key} < {node.keys[0]},走最左邊的指標"
            elif i == len(node.keys):
                how = f"{key} ≥ {node.keys[-1]},走最右邊的指標"
            else:
                how = f"{node.keys[i - 1]} ≤ {key} < {node.keys[i]},走第 {i + 1} 個指標"
            acc[node.id] = "path"
            node = node.children[i]
            acc[node.id] = "visit"
            kind = "葉節點" if node.is_leaf else "內部節點"
            self._record(f"比較索引鍵:{how} → 抵達{kind}", hl=acc, marks=marks)
        return node, acc

    # ------------------------------------------------------------------ 搜尋
    def search(self, key):
        self.steps = []
        self._record(f"【搜尋 {key}】從根節點出發,沿著索引鍵一路往下")
        leaf, acc = self._descend(key)
        if key in leaf.keys:
            i = leaf.keys.index(key)
            self._record(f"在葉節點的第 {i + 1} 格找到 {key},搜尋成功 ✓",
                         hl={**acc, leaf.id: "found"}, marks={leaf.id: {i: "found"}})
            return True
        self._record(f"葉節點中沒有 {key} → 搜尋失敗 ✗(B+ 樹中若存在,必定在這個葉節點)",
                     hl={**acc, leaf.id: "error"})
        return False

    # ------------------------------------------------------------------ 插入
    def insert(self, key):
        self.steps = []
        self._record(f"【插入 {key}】B+ 樹的資料一律存放在葉節點,先找出 {key} 應放入的葉節點")
        leaf, acc = self._descend(key, "尋找")
        if key in leaf.keys:
            i = leaf.keys.index(key)
            self._record(f"鍵 {key} 已存在,B+ 樹不允許重複鍵 → 插入取消 ✗",
                         hl={**acc, leaf.id: "error"}, marks={leaf.id: {i: "found"}})
            return False
        pos = bisect_left(leaf.keys, key)
        leaf.keys.insert(pos, key)
        self._record(f"把 {key} 依大小順序放進葉節點(第 {pos + 1} 格)",
                     hl={**acc, leaf.id: "insert"}, marks={leaf.id: {pos: "new"}})
        if len(leaf.keys) <= self.max_keys:
            self._record(f"葉節點現有 {len(leaf.keys)} 個鍵 ≤ 上限 {self.max_keys},符合規則,插入完成 ✓",
                         hl={leaf.id: "found"}, marks={leaf.id: {leaf.keys.index(key): "new"}})
        else:
            self._record(f"溢位!葉節點有 {len(leaf.keys)} 個鍵,超過上限 {self.max_keys},必須分裂",
                         hl={leaf.id: "overflow"})
            self._split_up(leaf)
            self._record(f"樹已恢復平衡,插入 {key} 完成 ✓")
        return True

    def _split_up(self, node):
        """node 溢位 → 不斷往上分裂,直到所有節點都符合規則。"""
        while len(node.keys) > self.max_keys:
            if node.is_leaf:
                mid = math.ceil(len(node.keys) / 2)
                right = Node(is_leaf=True)
                right.keys = node.keys[mid:]
                node.keys = node.keys[:mid]
                right.next = node.next
                node.next = right
                up = right.keys[0]
                how = (f"葉節點分裂:左半 {node.keys} 留在原節點,右半 {right.keys} 搬進新節點並接回鏈結;"
                       f"右半的最小鍵 {up} 會「複製」一份上推作為索引(資料本身仍留在葉節點)")
            else:
                mid = len(node.keys) // 2
                up = node.keys[mid]
                right = Node(is_leaf=False)
                right.keys = node.keys[mid + 1:]
                right.children = node.children[mid + 1:]
                for c in right.children:
                    c.parent = right
                node.keys = node.keys[:mid]
                node.children = node.children[:mid + 1]
                how = (f"內部節點分裂:中間鍵 {up} 整個「上推」(搬走、不複製),"
                       f"左半 {node.keys} 與右半 {right.keys} 各自成為一個節點")

            parent = node.parent
            if parent is None:
                parent = Node(is_leaf=False)
                parent.keys = [up]
                parent.children = [node, right]
                node.parent = right.parent = parent
                self.root = parent
                self._record(how + f";被分裂的是根節點,因此建立新根 [{up}],樹高 +1",
                             hl={parent.id: "new", node.id: "split", right.id: "split"},
                             marks={parent.id: {0: "new"}})
            else:
                idx = parent.children.index(node)
                parent.keys.insert(idx, up)
                parent.children.insert(idx + 1, right)
                right.parent = parent
                self._record(how + f";把 {up} 放進父節點",
                             hl={parent.id: "insert", node.id: "split", right.id: "split"},
                             marks={parent.id: {idx: "new"}})
                if len(parent.keys) > self.max_keys:
                    self._record(f"父節點也溢位了({len(parent.keys)} > {self.max_keys}),繼續往上分裂",
                                 hl={parent.id: "overflow"})
            node = parent

    # ------------------------------------------------------------------ 刪除
    def delete(self, key):
        self.steps = []
        self._record(f"【刪除 {key}】先找到 {key} 所在的葉節點")
        leaf, acc = self._descend(key, "尋找")
        if key not in leaf.keys:
            self._record(f"葉節點中沒有 {key},無法刪除 ✗", hl={**acc, leaf.id: "error"})
            return False
        pos = leaf.keys.index(key)
        was_first = (pos == 0)
        self._record(f"在葉節點找到 {key}(第 {pos + 1} 格),將它移除",
                     hl={**acc, leaf.id: "target"}, marks={leaf.id: {pos: "removed"}})
        leaf.keys.pop(pos)
        self._record(f"{key} 已從葉節點移除", hl={leaf.id: "visit"})

        # 若 key 同時是某個祖先的索引鍵(路標),更新成新的最小鍵,讓圖面與教科書一致
        if was_first and leaf.keys:
            anc = leaf.parent
            while anc is not None:
                if key in anc.keys:
                    j = anc.keys.index(key)
                    anc.keys[j] = leaf.keys[0]
                    self._record(f"{key} 同時是內部節點的索引鍵(路標),把它更新為新的最小鍵 {leaf.keys[0]}",
                                 hl={anc.id: "visit", leaf.id: "path"},
                                 marks={anc.id: {j: "new"}})
                    break
                anc = anc.parent

        self._rebalance(leaf)
        if self.root.is_leaf and not self.root.keys:
            self._record(f"刪除 {key} 完成 ✓ 樹現在是空的")
        else:
            self._record(f"樹已恢復平衡,刪除 {key} 完成 ✓")
        return True

    def _rebalance(self, node):
        """node 可能不足(underflow)→ 借鍵或合併,必要時往上層繼續處理。"""
        merged = False
        while True:
            if node is self.root:
                if not node.is_leaf and len(node.children) == 1:
                    child = node.children[0]
                    child.parent = None
                    self.root = child
                    self._record("根節點已經沒有索引鍵,讓唯一的子節點成為新的根,樹高 −1",
                                 hl={child.id: "new"})
                elif merged:
                    self._record("根節點沒有最低鍵數限制,不需再調整", hl={node.id: "found"})
                return

            min_keys = self.min_leaf_keys if node.is_leaf else self.min_internal_keys
            if len(node.keys) >= min_keys:
                if merged:
                    self._record(f"父節點仍有 {len(node.keys)} 個鍵 ≥ 下限 {min_keys},符合規則,調整結束",
                                 hl={node.id: "found"})
                return

            parent = node.parent
            idx = parent.children.index(node)
            left = parent.children[idx - 1] if idx > 0 else None
            right = parent.children[idx + 1] if idx + 1 < len(parent.children) else None
            kind = "葉節點" if node.is_leaf else "內部節點"
            self._record(f"{kind}只剩 {len(node.keys)} 個鍵 < 下限 {min_keys},"
                         f"發生「不足」(underflow),需要調整",
                         hl={node.id: "underflow"})

            def rich(sib):
                if sib is None:
                    return False
                m = self.min_leaf_keys if sib.is_leaf else self.min_internal_keys
                return len(sib.keys) > m

            if rich(left):
                self._borrow_from_left(parent, node, left, idx)
                return
            if rich(right):
                self._borrow_from_right(parent, node, right, idx)
                return

            # 兄弟都借不出來 → 合併
            if left is not None:
                self._merge(parent, idx - 1)
            else:
                self._merge(parent, idx)
            merged = True
            node = parent

    def _borrow_from_left(self, parent, node, left, idx):
        if node.is_leaf:
            k = left.keys.pop()
            node.keys.insert(0, k)
            parent.keys[idx - 1] = k
            self._record(f"左兄弟的鍵數有餘裕,借出它最大的鍵 {k} 放到本節點最前面;"
                         f"父節點的索引鍵同步改成 {k}",
                         hl={left.id: "lend", node.id: "borrow", parent.id: "visit"},
                         marks={node.id: {0: "new"}, parent.id: {idx - 1: "new"}})
        else:
            sep = parent.keys[idx - 1]
            node.keys.insert(0, sep)
            parent.keys[idx - 1] = left.keys.pop()
            child = left.children.pop()
            child.parent = node
            node.children.insert(0, child)
            self._record(f"向左兄弟「旋轉」借鍵:父節點索引鍵 {sep} 下移到本節點,"
                         f"左兄弟最大的鍵 {parent.keys[idx - 1]} 上移進父節點,"
                         f"左兄弟最右邊的子樹也一併過繼",
                         hl={left.id: "lend", node.id: "borrow", parent.id: "visit"},
                         marks={node.id: {0: "new"}, parent.id: {idx - 1: "new"}})

    def _borrow_from_right(self, parent, node, right, idx):
        if node.is_leaf:
            k = right.keys.pop(0)
            node.keys.append(k)
            parent.keys[idx] = right.keys[0]
            self._record(f"右兄弟的鍵數有餘裕,借出它最小的鍵 {k} 接到本節點最後面;"
                         f"父節點的索引鍵改成右兄弟新的最小鍵 {right.keys[0]}",
                         hl={right.id: "lend", node.id: "borrow", parent.id: "visit"},
                         marks={node.id: {len(node.keys) - 1: "new"}, parent.id: {idx: "new"}})
        else:
            sep = parent.keys[idx]
            node.keys.append(sep)
            parent.keys[idx] = right.keys.pop(0)
            child = right.children.pop(0)
            child.parent = node
            node.children.append(child)
            self._record(f"向右兄弟「旋轉」借鍵:父節點索引鍵 {sep} 下移到本節點,"
                         f"右兄弟最小的鍵 {parent.keys[idx]} 上移進父節點,"
                         f"右兄弟最左邊的子樹也一併過繼",
                         hl={right.id: "lend", node.id: "borrow", parent.id: "visit"},
                         marks={node.id: {len(node.keys) - 1: "new"}, parent.id: {idx: "new"}})

    def _merge(self, parent, i):
        """把 parent.children[i+1] 併入 parent.children[i]。"""
        left = parent.children[i]
        right = parent.children[i + 1]
        sep = parent.keys[i]
        self._record("兩個兄弟節點都沒有多餘的鍵可以借 → 進行「合併」",
                     hl={left.id: "merge", right.id: "merge", parent.id: "visit"},
                     marks={parent.id: {i: "removed"}})
        if left.is_leaf:
            left.keys.extend(right.keys)
            left.next = right.next
            desc = (f"葉節點合併:右節點的鍵全部搬進左節點,葉節點鏈結重新接好;"
                    f"父節點中的索引鍵 {sep} 不再需要,移除")
        else:
            left.keys.append(sep)
            left.keys.extend(right.keys)
            for c in right.children:
                c.parent = left
            left.children.extend(right.children)
            desc = (f"內部節點合併:父節點索引鍵 {sep} 「下移」進左節點,"
                    f"再把右節點的鍵與子樹全部併入;父節點移除 {sep}")
        parent.keys.pop(i)
        parent.children.pop(i + 1)
        self._record(desc, hl={left.id: "merge", parent.id: "visit"})

    # ------------------------------------------------------------------ 範圍查詢
    def range_search(self, lo, hi):
        self.steps = []
        self._record(f"【範圍查詢 {lo} ~ {hi}】這是 B+ 樹的強項:先找到第一個可能 ≥ {lo} 的葉節點,"
                     f"之後只要沿著葉節點之間的鏈結往右掃描即可")
        leaf, _ = self._descend(lo, "尋找")
        found = []
        hl = {}
        marks = {}
        node = leaf
        while node is not None:
            hits = [i for i, k in enumerate(node.keys) if lo <= k <= hi]
            if hits:
                found.extend(node.keys[i] for i in hits)
                hl[node.id] = "found"
                marks[node.id] = {i: "found" for i in hits}
                self._record(f"掃描葉節點:{[node.keys[i] for i in hits]} 在範圍內,收集起來",
                             hl=hl, marks=marks)
            else:
                hl[node.id] = "visit"
                self._record("掃描葉節點:這裡沒有符合範圍的鍵", hl=hl, marks=marks)
            if node.keys and node.keys[-1] > hi:
                self._record(f"此葉節點已出現 > {hi} 的鍵,右邊不可能再有答案,掃描結束",
                             hl=hl, marks=marks)
                break
            if node.next is None:
                self._record("已到達最右端的葉節點,掃描結束", hl=hl, marks=marks)
                break
            node = node.next
            hl[node.id] = "visit"
            self._record("沿著葉節點之間的鏈結指標 → 移到下一個葉節點(完全不必回到樹根!)",
                         hl=hl, marks=marks)
        if found:
            self._record(f"範圍查詢完成 ✓ 共找到 {len(found)} 個鍵:{found}", hl=hl, marks=marks)
        else:
            self._record("範圍查詢完成:範圍內沒有任何鍵", hl=hl)
        return found

    # ------------------------------------------------------------------ 工具
    def all_keys(self):
        """由左到右走訪葉節點鏈,回傳所有鍵(必為遞增)。"""
        n = self.root
        while not n.is_leaf:
            n = n.children[0]
        out = []
        while n is not None:
            out.extend(n.keys)
            n = n.next
        return out

    def height(self):
        h, n = 1, self.root
        while not n.is_leaf:
            n = n.children[0]
            h += 1
        return h

    def node_count(self):
        cnt, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            cnt += 1
            stack.extend(n.children)
        return cnt

    # ------------------------------------------------------------------ 不變量驗證(測試用)
    def validate(self):
        leaves_inorder = []
        depths = set()

        assert self.root.parent is None

        def walk(n, depth):
            if n.is_leaf:
                assert not n.children
                leaves_inorder.append(n)
                depths.add(depth)
            else:
                assert len(n.children) == len(n.keys) + 1, "子節點數必須 = 鍵數 + 1"
                for c in n.children:
                    assert c.parent is n, "parent 指標錯誤"
                    walk(c, depth + 1)
            assert n.keys == sorted(n.keys), "節點內的鍵必須遞增"

        walk(self.root, 0)
        assert len(depths) == 1, "所有葉節點必須在同一層"

        # 葉節點鏈結必須等於由左到右的葉節點順序
        chain, n = [], self.root
        while not n.is_leaf:
            n = n.children[0]
        while n is not None:
            chain.append(n)
            n = n.next
        assert chain == leaves_inorder, "葉節點鏈結順序錯誤"

        keys = [k for lf in leaves_inorder for k in lf.keys]
        assert keys == sorted(keys), "全部鍵必須整體遞增"
        assert len(keys) == len(set(keys)), "鍵不可重複"

        # 容量限制
        def all_nodes():
            stack = [self.root]
            while stack:
                x = stack.pop()
                yield x
                stack.extend(x.children)

        for x in all_nodes():
            assert len(x.keys) <= self.max_keys, "鍵數超過上限"
            if x is self.root:
                if not x.is_leaf:
                    assert 2 <= len(x.children) <= self.order
                continue
            if x.is_leaf:
                assert len(x.keys) >= self.min_leaf_keys, f"葉節點鍵數不足: {x.keys}"
            else:
                assert len(x.keys) >= self.min_internal_keys, f"內部節點鍵數不足: {x.keys}"
                assert math.ceil(self.order / 2) <= len(x.children) <= self.order

        # 索引鍵的導引正確性:child[i] 的所有鍵 ∈ [keys[i-1], keys[i])
        def check_range(x, lo, hi):
            if x.is_leaf:
                for k in x.keys:
                    if lo is not None:
                        assert k >= lo, f"鍵 {k} 小於下界 {lo}"
                    if hi is not None:
                        assert k < hi, f"鍵 {k} 不小於上界 {hi}"
                return
            bounds = [lo] + list(x.keys) + [hi]
            for i, c in enumerate(x.children):
                check_range(c, bounds[i], bounds[i + 1])

        check_range(self.root, None, None)
