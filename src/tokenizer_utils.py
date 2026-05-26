"""Tokenizer 加载和编码工具函数。"""

from typing import List, Tuple
from transformers import AutoTokenizer

_rust = None
tok = None


def load_tokenizer(path: str, trust_remote_code: bool = True):
    """加载 tokenizer 并缓存底层 Rust 绑定。"""
    global tok, _rust
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    _rust = tok._tokenizer
    return tok


def _enc(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Rust 底层 encode，返回 (ids, offsets)。"""
    e = _rust.encode(text, add_special_tokens=False)
    return e.ids, e.offsets


def _enc_ids(text: str) -> List[int]:
    """仅返回 token IDs，不返回 offsets。"""
    return _rust.encode(text, add_special_tokens=False).ids
