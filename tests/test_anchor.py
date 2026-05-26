"""测试 anchor 查找正确性。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lopt_core import find_anchor


def test_simple_consecutive():
    """连续匹配：两个 chunk 的 overlap 完全一致。"""
    off_p = [(0, 2), (2, 5), (5, 8)]   # 前 chunk offsets
    off_c = [(5, 8), (8, 10), (10, 13)]  # 后 chunk offsets
    a_start, a_len, q_start = find_anchor(off_p, off_c, 5, 8)
    assert a_start == 2 and a_len == 1 and q_start == 0, f"Got ({a_start},{a_len},{q_start})"


def test_offset_shift():
    """BPE 偏移场景：ca 在 chunk 边界处多了 1 个 token。"""
    # chunk_i overlap: [169-175)(175-178)(178-182)(182-185)...
    # chunk_{i+1} overlap: [170-171)(171-175)(175-178)(178-182)...
    # 第一个 token 不同（169-175 vs 170-171），但第二个开始匹配
    off_p = [(169, 175), (175, 178), (178, 182), (182, 185), (185, 188)]
    off_c = [(170, 171), (171, 175), (175, 178), (178, 182), (182, 185)]
    a_start, a_len, q_start = find_anchor(off_p, off_c, 170, 185)
    # pa[1]=(175,178) == ca[2]=(175,178), anchor 从 pa_idx=1 开始, ca_idx=2
    assert a_start is not None, "Should find anchor"
    assert a_len >= 2, f"Anchor too short: {a_len}"


def test_no_overlap():
    """无重叠区 → 返回 None。"""
    off_p = [(0, 2), (2, 5)]
    off_c = [(10, 12), (12, 15)]
    a_start, a_len, q_start = find_anchor(off_p, off_c, 10, 12)
    assert a_start is None


def test_exact_match_three():
    """三段完全一致的重叠。"""
    off_p = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    off_c = [(6, 9), (9, 12), (12, 15), (15, 18), (18, 21)]
    a_start, a_len, q_start = find_anchor(off_p, off_c, 6, 15)
    assert a_start == 2 and a_len == 3 and q_start == 0


def test_partial_overlap():
    """部分重叠 + 中间有匹配。"""
    off_p = [(0, 4), (4, 8), (8, 12), (12, 16)]
    off_c = [(8, 12), (12, 16), (16, 20), (20, 24)]
    a_start, a_len, q_start = find_anchor(off_p, off_c, 8, 16)
    assert a_start == 2 and a_len == 2 and q_start == 0


if __name__ == "__main__":
    test_simple_consecutive()
    test_offset_shift()
    test_no_overlap()
    test_exact_match_three()
    test_partial_overlap()
    print("All anchor tests passed ✅")
