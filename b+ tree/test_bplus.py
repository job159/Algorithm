"""B+ 樹邏輯的隨機模糊測試:以 Python set 為對照組,逐步驗證所有不變量。"""
import random
import sys

from bplus_tree import BPlusTree


def check_snapshots(tree):
    """快照本身的健全性:所有引用到的節點 id 都必須存在於該快照中。"""
    for s in tree.steps:
        nodes = s["nodes"]
        assert s["root"] in nodes
        for nd in nodes.values():
            for c in nd["children"]:
                assert c in nodes, "快照中引用了不存在的子節點"
            if nd["next"] is not None:
                assert nd["next"] in nodes, "快照中引用了不存在的 next 葉節點"
        for nid in s["hl"]:
            assert nid in nodes, "hl 引用了不存在的節點"
        for nid, mm in s["marks"].items():
            assert nid in nodes, "marks 引用了不存在的節點"
            for idx in mm:
                assert 0 <= idx < len(nodes[nid]["keys"]), \
                    f"marks 索引 {idx} 超出鍵數 {len(nodes[nid]['keys'])}"


def fuzz(order, rounds, seed):
    rnd = random.Random(seed)
    tree = BPlusTree(order)
    ref = set()
    for _ in range(rounds):
        op = rnd.random()
        v = rnd.randint(0, 120)
        if op < 0.50:
            assert tree.insert(v) == (v not in ref)
            ref.add(v)
        elif op < 0.90:
            assert tree.delete(v) == (v in ref)
            ref.discard(v)
        elif op < 0.96:
            assert tree.search(v) == (v in ref)
        else:
            lo = rnd.randint(0, 120)
            hi = lo + rnd.randint(0, 40)
            got = tree.range_search(lo, hi)
            want = sorted(x for x in ref if lo <= x <= hi)
            assert got == want, f"range({lo},{hi}) got {got} want {want}"
        tree.validate()
        assert tree.all_keys() == sorted(ref)
        check_snapshots(tree)
    return tree


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for order in (3, 4, 5, 6, 7):
        t = fuzz(order, rounds=2500, seed=42 + order)
        print(f"order={order}: OK  (final keys={len(t.all_keys())}, "
              f"height={t.height()}, nodes={t.node_count()})")
    # 邊界:清到全空再重建
    t = BPlusTree(3)
    for v in range(30):
        t.insert(v)
    for v in range(30):
        t.delete(v)
        t.validate()
    assert t.all_keys() == []
    assert t.height() == 1
    for v in range(30, 0, -1):
        t.insert(v)
        t.validate()
    print("edge cases: OK")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
