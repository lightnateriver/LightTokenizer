#!/usr/bin/env python3
"""Build real-web corpora for tokenizer benchmarks.

The corpora are derived from visible text fetched from real public webpages.
Each language corpus is normalized and language-filtered so the benchmark
receives:

- pure English text
- pure Chinese text

The target corpus size is character-based because the benchmark requirements
specify text lengths such as 1k, 4k, ..., 1024k.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Iterable

ENGLISH_URLS = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://docs.python.org/3/library/asyncio.html",
    "https://docs.python.org/3/library/concurrent.futures.html",
    "https://docs.python.org/3/library/multiprocessing.html",
    "https://docs.python.org/3/library/pathlib.html",
    "https://docs.python.org/3/library/queue.html",
    "https://docs.python.org/3/reference/lexical_analysis.html",
    "https://docs.python.org/3/reference/compound_stmts.html",
    "https://numpy.org/doc/stable/user/absolute_beginners.html",
    "https://numpy.org/doc/stable/reference/arrays.ndarray.html",
    "https://pandas.pydata.org/docs/user_guide/index.html",
    "https://pytorch.org/docs/stable/index.html",
    "https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
    "https://pytorch.org/docs/stable/data.html",
    "https://fastapi.tiangolo.com/",
    "https://www.sqlite.org/lang.html",
    "https://www.sqlite.org/queryplanner.html",
    "https://www.postgresql.org/docs/current/index.html",
    "https://www.rfc-editor.org/rfc/rfc9110.html",
    "https://www.rfc-editor.org/rfc/rfc6455.html",
    "https://www.rfc-editor.org/rfc/rfc9000.html",
    "https://www.gnu.org/software/bash/manual/bash.html",
    "https://www.w3.org/TR/html52/",
    "https://developer.mozilla.org/en-US/docs/Web/HTML",
    "https://developer.mozilla.org/en-US/docs/Web/CSS",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
]

CHINESE_URLS = [
    "https://www.gov.cn/zhengce/index.htm",
    "https://www.gov.cn/yaowen/index.htm",
    "https://www.news.cn/politics/",
    "https://www.news.cn/fortune/",
    "https://www.news.cn/tech/",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_01_0001.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_01_0002.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_01_0003.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_01_0017.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_05_0001.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_05_0002.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_05_0003.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_25_0001.html",
    "https://support.huaweicloud.com/usermanual-modelarts/modelarts_23_0001.html",
    "https://cloud.tencent.com/developer/doc/1024",
    "https://cloud.tencent.com/developer/doc/1034",
    "https://help.aliyun.com/zh/ecs/",
    "https://help.aliyun.com/zh/oss/",
    "https://help.aliyun.com/zh/model-studio/",
    "https://help.aliyun.com/zh/pai/",
]

TAG_BLOCK_RE = re.compile(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>")
COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
TAG_RE = re.compile(r"(?is)<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
ASCII_PRINTABLE_RE = re.compile(r"[ -~]")

ZH_KEEP_RE = re.compile(r"[0-9\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
EN_KEEP_RE = re.compile(r"[A-Za-z0-9]")


@dataclass
class SourceRecord:
    url: str
    extracted_chars: int
    kept_chars: int


def fetch_url(url: str, timeout_s: float = 20.0) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="ignore")


def extract_visible_text(html: str) -> str:
    text = TAG_BLOCK_RE.sub(" ", html)
    text = COMMENT_RE.sub(" ", text)
    text = TAG_RE.sub("\n", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    lines = []
    for raw_line in text.splitlines():
        line = MULTISPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_english_line(line: str) -> str:
    if CJK_RE.search(line):
        return ""
    chars = [ch if ord(ch) < 128 else " " for ch in line]
    cleaned = "".join(chars)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) < 40:
        return ""
    ascii_count = sum(1 for ch in cleaned if ASCII_PRINTABLE_RE.match(ch))
    keep_count = sum(1 for ch in cleaned if EN_KEEP_RE.match(ch))
    if ascii_count == 0 or keep_count == 0:
        return ""
    if keep_count / max(len(cleaned), 1) < 0.35:
        return ""
    return cleaned


def normalize_chinese_line(line: str) -> str:
    filtered = []
    for ch in line:
        if ZH_KEEP_RE.match(ch):
            filtered.append(ch)
        elif ch.isspace():
            filtered.append(" ")
    cleaned = "".join(filtered)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    cjk_count = sum(1 for ch in cleaned if CJK_RE.match(ch))
    if len(cleaned) < 20 or cjk_count == 0:
        return ""
    if cjk_count / max(len(cleaned), 1) < 0.60:
        return ""
    return cleaned


def normalize_text(text: str, language: str) -> str:
    normalizer = normalize_english_line if language == "en" else normalize_chinese_line
    lines = []
    seen = set()
    for line in text.splitlines():
        cleaned = normalizer(line)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            lines.append(cleaned)
    return "\n".join(lines)


def cycle_join(blocks: Iterable[str], target_chars: int) -> str:
    chunks = [block.strip() for block in blocks if block.strip()]
    if not chunks:
        raise RuntimeError("No usable text blocks were extracted from the provided URLs.")
    corpus = []
    current = 0
    index = 0
    while current < target_chars:
        block = chunks[index % len(chunks)]
        if corpus:
            corpus.append("\n\n")
            current += 2
        corpus.append(block)
        current += len(block)
        index += 1
    return "".join(corpus)[:target_chars]


def corpus_paths(output_dir: Path, language: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{language}_web_corpus.txt",
        output_dir / f"{language}_sources.json",
    )


def build_language_corpus(
    language: str,
    output_dir: Path,
    target_chars: int,
    timeout_s: float = 20.0,
    force: bool = False,
) -> tuple[Path, Path]:
    text_path, meta_path = corpus_paths(output_dir, language)
    if (
        not force
        and text_path.exists()
        and meta_path.exists()
        and len(text_path.read_text(encoding="utf-8")) >= target_chars
    ):
        return text_path, meta_path

    output_dir.mkdir(parents=True, exist_ok=True)
    urls = ENGLISH_URLS if language == "en" else CHINESE_URLS

    kept_blocks = []
    records: list[SourceRecord] = []
    errors: list[dict[str, str]] = []
    accumulated_chars = 0

    for url in urls:
        try:
            html = fetch_url(url, timeout_s=timeout_s)
            visible = extract_visible_text(html)
            kept = normalize_text(visible, language)
            kept_chars = len(kept)
            if kept_chars > 0:
                kept_blocks.append(kept)
                accumulated_chars += kept_chars + 2
            records.append(
                SourceRecord(
                    url=url,
                    extracted_chars=len(visible),
                    kept_chars=kept_chars,
                )
            )
            if accumulated_chars >= target_chars:
                break
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            errors.append({"url": url, "error": str(exc)})

    corpus = cycle_join(kept_blocks, target_chars)
    text_path.write_text(corpus, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "language": language,
                "target_chars": target_chars,
                "final_chars": len(corpus),
                "sources": [asdict(record) for record in records],
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return text_path, meta_path


def ensure_corpora(
    output_dir: Path,
    target_chars: int,
    languages: Iterable[str] = ("en", "zh"),
    timeout_s: float = 20.0,
    force: bool = False,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    result: dict[str, Path] = {}
    for language in languages:
        text_path, _ = build_language_corpus(
            language=language,
            output_dir=output_dir,
            target_chars=target_chars,
            timeout_s=timeout_s,
            force=force,
        )
        result[language] = text_path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/assets"),
        help="Directory used to store built corpus files.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=1024 * 1024,
        help="Target corpus size per language, measured in characters.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh"],
        choices=["en", "zh"],
        help="Languages to build.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the corpora even if matching files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_corpora(
        output_dir=args.output_dir,
        target_chars=args.target_chars,
        languages=args.languages,
        timeout_s=args.timeout_s,
        force=args.force,
    )
    summary = "\n".join(
        f"- {language}: {path}" for language, path in sorted(paths.items())
    )
    print(
        textwrap.dedent(
            f"""\
            Built corpus files:
            {summary}
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
