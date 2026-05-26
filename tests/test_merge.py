"""测试合并精度。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lopt_core import merge_chunks


def test_two_chunks():
    """两个 chunk 合并。"""
    res = [
        ([10, 20, 30, 40, 50], [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]),  # chunk 0
        ([40, 50, 60, 70], [(6, 8), (8, 10), (10, 12), (12, 14)]),          # chunk 1
    ]
    anchors = [(3, 2, 0)]  # anchor 在 pa[3]=40, ca[0]=40, 长度 2
    merged = merge_chunks(res, anchors)
    assert merged == [10, 20, 30, 40, 50, 60, 70], f"Got {merged}"


def test_three_chunks():
    """三个 chunk 合并。"""
    res = [
        ([1, 2, 3, 4, 5], [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]),
        ([4, 5, 6, 7, 8], [(6, 8), (8, 10), (10, 12), (12, 14), (14, 16)]),
        ([7, 8, 9, 10], [(12, 14), (14, 16), (16, 18), (18, 20)]),
    ]
    anchors = [(3, 2, 0), (3, 2, 0)]
    merged = merge_chunks(res, anchors)
    assert merged == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f"Got {merged}"


def test_ca_shift_merge():
    """ca 偏移场景：anchor 从 ca[1] 开始。"""
    res = [
        ([10, 20, 30, 40, 50, 60], [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]),
        ([99, 30, 40, 50, 60, 70], [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)]),
    ]
    anchors = [(1, 4, 1)]  # pa[1]=20 -> 但实际 anchor 是 30,40,50,60
    # ca[1]=30 -> ca[4]=60，ca 的 ca[0]=99 是 chunk_{i+1} 独有的
    merged = merge_chunks(res, anchors)
    assert merged == [10, 20, 30, 40, 50, 60, 70], f"Got {merged}"


def test_single_chunk():
    """单个 chunk → 直接返回。"""
    res = [([1, 2, 3], [(0, 1), (1, 2), (2, 3)])]
    anchors = []
    merged = merge_chunks(res, anchors)
    assert merged == [1, 2, 3], f"Got {merged}"


if __name__ == "__main__":
    test_two_chunks()
    test_three_chunks()
    test_ca_shift_merge()
    test_single_chunk()
    print("All merge tests passed ✅")
