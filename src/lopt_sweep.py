#!/usr/bin/env python3
"""
LoPT exhaustive parameter sweep — v2.
Usage: python3 lopt_sweep.py --lang zh --md /path/to/results.md

Generates self-contained markdown results file.

Search space:
  N (tokens): 1K, 16K, 64K, 128K, 256K, 512K, 860K, 1024K
  Lc (chars): auto-computed from N (2^5 to N/2, 8 values per size)
  LO (chars): zh=[16,32,64,128,256], en=[32,64,128,256,512]
  threads:    1, 2, 4, 8, 16, 32
"""

import time, bisect, sys, argparse, json, os, math, textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════
_rust = None
tok = None

def init_tokenizer(path: str):
    global tok, _rust
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    _rust = tok._tokenizer

def _enc(t: str):
    e = _rust.encode(t, add_special_tokens=False)
    return e.ids, e.offsets

def _enc_ids(t: str):
    return _rust.encode(t, add_special_tokens=False).ids

# ═══════════════════════════════════════════════════════════════
# Text corpora
# ═══════════════════════════════════════════════════════════════
ZH_BLOCKS = [
    "新华社北京五月二十六日电 人工智能技术正在深刻改变经济社会发展的各个方面。从智能制造到智慧医疗，从自动驾驶到智能客服，人工智能的应用场景不断拓展。",
    "专家表示，大模型技术的突破为人工智能发展注入了新动力。深度学习、强化学习等技术的持续进步，使得机器在视觉、语音、自然语言处理等领域取得了显著成就。",
    "量子计算作为一种新兴计算范式，在特定问题上具有经典计算机无法比拟的优势。中国科研团队在量子计算领域取得重要突破，成功构建了新一代量子计算原型机。",
    "气候变化是全人类面临的共同挑战，需要各国携手合作应对。中国积极推动绿色低碳发展，提出了碳达峰碳中和目标，并制定了详细的实施路线图和政策措施。",
    "科技创新是引领发展的第一动力。中国持续加大研发投入，在基础研究、前沿技术、战略高技术等领域取得了一系列重大原创成果，为经济社会发展提供了有力支撑。",
    "教育数字化转型正加速推进。智慧课堂、在线学习平台、虚拟实验室等新型教育模式蓬勃发展，有效促进了优质教育资源的共享和教育教学质量的提升。",
    "数字经济的发展为传统产业转型升级提供了新路径。工业互联网、大数据、云计算等技术与制造业深度融合，推动了智能制造和柔性生产的发展。",
    "粤港澳大湾区建设取得显著成效。基础设施互联互通水平不断提升，科技创新要素加速集聚，现代产业体系加快构建，区域协同发展格局初步形成。",
    "乡村振兴战略深入实施，农业农村现代化步伐加快。数字乡村建设、农业科技创新、农村人居环境整治等工作全面推进，农民收入持续增长。",
    "健康中国建设稳步推进，医疗卫生体制改革不断深化。远程医疗、智能诊断、精准医疗等新技术应用越来越广泛，人民群众健康水平持续提高。",
    "中国航天事业取得新的重大成就。载人航天工程、嫦娥探月工程、天问火星探测等任务圆满完成，空间站建设稳步推进，航天科技水平跃上新台阶。",
    "新能源产业发展势头强劲。光伏、风电、氢能等清洁能源技术不断突破，储能技术加速进步，新能源汽车产销量连续多年位居全球第一。",
    "数据安全和个人信息保护受到社会广泛关注。相关法律法规不断完善，数据要素市场培育有序推进，为数字经济的健康发展提供了制度保障。",
    "生物医药领域创新成果丰硕。创新药研发取得重要突破，基因编辑、细胞治疗等前沿技术快速发展，为疾病治疗提供了新的手段和希望。",
    "文化遗产保护与传承取得新进展。数字化技术在文物修复、非遗保护、古籍整理等领域得到广泛应用，让传统文化焕发新的生机与活力。",
    "半导体产业加快自主创新步伐。芯片设计、制造工艺、封装测试等关键环节取得重要进展，产业链供应链韧性和安全水平持续提升。",
    "智慧城市建设从概念走向实践。城市大脑、数字孪生、智能交通等系统在越来越多城市落地运行，城市治理能力和公共服务水平显著提升。",
    "金融科技深度赋能实体经济。移动支付、数字人民币、供应链金融等创新应用快速发展，金融服务覆盖面和便利性持续提高。",
    "新材料产业蓬勃发展。碳纤维、石墨烯、超导材料等新型材料在航空航天、电子信息、新能源等领域应用前景广阔，产业规模持续扩大。",
    "绿色低碳发展理念深入人心。生态文明建设取得新成效，污染防治攻坚战持续推进，生态系统保护修复力度不断加大，美丽中国建设迈出坚实步伐。",
]

EN_BLOCKS = [
    "Artificial intelligence is profoundly transforming every aspect of economic and social development. From smart manufacturing and intelligent healthcare to autonomous driving, the application scenarios of AI continue to expand.",
    "Breakthroughs in large language models have injected new momentum into AI development. Advances in deep learning and reinforcement learning have enabled machines to achieve remarkable results in vision and natural language processing.",
    "Quantum computing, as an emerging computational paradigm, offers advantages over classical computers for specific problems. Chinese research teams have made significant breakthroughs in this field.",
    "Climate change is a shared challenge facing all of humanity, requiring global cooperation. China is actively pursuing green and low-carbon development with ambitious carbon peak and neutrality targets.",
    "Technological innovation serves as the primary engine of development. China continues to increase R&D investment, achieving major original results in basic research and strategic high-tech fields.",
    "Digital transformation of education is accelerating rapidly. Smart classrooms, online learning platforms, and virtual laboratories are thriving, effectively promoting educational resource sharing.",
    "The digital economy is providing new pathways for traditional industry transformation. Industrial internet, big data, and cloud computing are deeply integrating with manufacturing industries.",
    "The development of the Greater Bay Area has achieved significant results. Infrastructure connectivity continues to improve, with innovation factors accelerating agglomeration across the region.",
    "Rural revitalization strategy is being implemented deeply across China, accelerating agricultural modernization and improving rural living conditions comprehensively.",
    "Healthy China construction is progressing steadily with deepening healthcare reform. Telemedicine and intelligent diagnosis are becoming increasingly widespread across the country.",
    "China's space program has achieved major new accomplishments. Crewed spaceflight and lunar exploration missions were completed successfully with space station construction advancing.",
    "The new energy industry is growing strongly worldwide. Solar, wind, and hydrogen energy technologies continue to break through with energy storage technology accelerating rapidly.",
    "Data security and personal information protection have received widespread attention globally. Relevant laws and regulations are being continuously improved and updated.",
    "Biomedical innovation has yielded fruitful results globally. Significant breakthroughs in drug development and gene editing are providing new means of disease treatment.",
    "Cultural heritage protection has made new progress worldwide. Digital technologies are widely applied in cultural relic restoration and intangible heritage preservation.",
    "The semiconductor industry is accelerating independent innovation globally. Key progress in chip design and manufacturing processes continues to improve industrial resilience.",
    "Smart city construction is moving from concept to practice worldwide. Urban digital twins and intelligent transportation systems are being deployed in more cities.",
    "Financial technology is deeply empowering the real economy globally. Mobile payments and digital currency applications are developing rapidly across many countries.",
    "New materials industries are booming worldwide. Carbon fiber and graphene have broad application prospects in aerospace and electronics industries.",
    "Sustainable development concepts are gaining traction globally. Environmental protection efforts continue to strengthen with ecosystem restoration making steady progress.",
]


# ═══════════════════════════════════════════════════════════════
# Text generation
# ═══════════════════════════════════════════════════════════════
_text_cache = {}

def generate_text(lang: str, target_tokens: int) -> str:
    """Generate text ~target_tokens tokens."""
    key = (lang, target_tokens)
    if key in _text_cache:
        return _text_cache[key]
    blocks = ZH_BLOCKS if lang == 'zh' else EN_BLOCKS
    single = "".join(blocks)
    single_tok = len(_enc_ids(single))
    repeats = max(1, target_tokens // max(single_tok, 1))
    text = single * repeats
    _text_cache[key] = text
    return text


# ═══════════════════════════════════════════════════════════════
# Lc values auto-computation
# ═══════════════════════════════════════════════════════════════
def compute_lc_values(total_chars: int) -> List[int]:
    """Compute Lc values for given text length.
    Returns ~6 values from total_chars/128 to total_chars/2,
    clamped to [64, 524288], rounded to nice numbers (2^n or nearby)."""
    if total_chars < 128:
        return [total_chars // 4, total_chars // 2]
    
    vals = set()
    # Generate from small to large, roughly geometric
    # Starting from total_chars/128, doubling until total_chars/2
    start = max(64, total_chars // 128)
    val = start
    while val <= total_chars // 2:
        # Round to nice number (nearest 2^n)
        if val > 1:
            power = 2 ** round(math.log2(val))
            vals.add(power)
        vals.add(val)
        val *= 2
    
    # Add specific important values
    for v in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
        if v > start // 2 and v <= total_chars:
            vals.add(v)
    
    # Filter: only keep values that give >= 2 chunks or are close
    result = sorted(v for v in vals if v < total_chars)
    
    # Ensure we don't have too many
    max_vals = 8
    if len(result) > max_vals:
        # Keep every nth value to get ~max_vals
        step = len(result) / max_vals
        result = [result[int(i * step)] for i in range(max_vals)]
        result = sorted(set(result))
    
    return result


# ═══════════════════════════════════════════════════════════════
# LoPT core
# ═══════════════════════════════════════════════════════════════
def find_anchor(off_p, off_c, ostart, oend):
    n_p, n_c = len(off_p), len(off_c)
    l = bisect.bisect_left(off_p, (ostart, 0))
    ps = None
    if l > 0 and off_p[l-1][1] > ostart:
        ps = l - 1
    elif l < n_p and off_p[l][0] < oend:
        ps = l
    if ps is None:
        return None, None, None
    r = bisect.bisect_left(off_c, (oend, 0))
    ce = r - 1
    if ce >= 0 and off_c[ce][1] <= ostart:
        while ce >= 0 and off_c[ce][1] <= ostart:
            ce -= 1
    if ce < 0:
        return None, None, None
    pa = off_p[ps:]
    ca = off_c[:ce+1]
    bl, bp, bq = 0, None, None
    for pv in range(len(pa)-1, -1, -1):
        for qv in range(len(ca)-1, -1, -1):
            mk = min(len(pa)-pv, len(ca)-qv)
            if mk <= bl:
                continue
            k = 0
            while k < mk and pa[pv+k] == ca[qv+k]:
                k += 1
            if k > bl:
                bl, bp, bq = k, pv, qv
    if bl == 0:
        return None, None, None
    return ps + bp, bl, bq


def merge_chunks(res, anchors):
    segs = []
    tl = 0
    s = res[0][0][:anchors[0][0] + anchors[0][1]]
    segs.append(s)
    tl += len(s)
    for i in range(1, len(res) - 1):
        st = anchors[i-1][2] + anchors[i-1][1]
        en = anchors[i][0] + anchors[i][1]
        if en > st:
            s = res[i][0][st:en]
            segs.append(s)
            tl += len(s)
    s = res[-1][0][anchors[-1][2] + anchors[-1][1]:]
    segs.append(s)
    tl += len(s)
    m = [0] * tl
    p = 0
    for seg in segs:
        m[p:p+len(seg)] = seg
        p += len(seg)
    return m


# ═══════════════════════════════════════════════════════════════
# Single test
# ═══════════════════════════════════════════════════════════════
def run_lopt(prompt_str, ser_ids, Lc, LO, n_threads):
    n = len(prompt_str)
    n_chunks = (n + Lc - 1) // Lc
    if n_chunks <= 1:
        return None
    
    ci = [(prompt_str[i*Lc:min(n, (i+1)*Lc+LO)], i*Lc) for i in range(n_chunks)]
    
    t0 = time.perf_counter()
    # Sequential pool to avoid ThreadPoolExecutor overhead for very small tests
    if n_threads <= 1:
        res = [((lambda ids, off: (ids, [(x[0]+a[1], x[1]+a[1]) for x in off]))(*_enc(a[0]))) for a in ci]
    else:
        with ThreadPoolExecutor(n_threads) as ex:
            res = list(ex.map(
                lambda a: (lambda ids, off: (ids, [(x[0]+a[1], x[1]+a[1]) for x in off]))(*_enc(a[0])),
                ci
            ))
    pool_ms = (time.perf_counter() - t0) * 1000
    
    anchors = []
    anc_fails = 0
    anc_tokens = 0
    for i in range(1, len(res)):
        a_start, a_len, q_start = find_anchor(res[i-1][1], res[i][1], i*Lc, i*Lc+LO)
        if a_len is None or a_len == 0:
            anc_fails += 1
        else:
            anchors.append((a_start, a_len, q_start))
            anc_tokens += a_len
    
    has_fallback = anc_fails > 0
    if has_fallback:
        ids = _enc_ids(prompt_str)
    else:
        ids = merge_chunks(res, anchors)
    
    lopt_ms = (time.perf_counter() - t0) * 1000
    merge_ms = lopt_ms - pool_ms
    correct = ids == ser_ids
    avg_anchor = anc_tokens / max(len(anchors), 1)
    
    return {
        "Lc": Lc, "LO": LO, "n_threads": n_threads,
        "n_chunks": n_chunks, "pool_ms": round(pool_ms, 1),
        "merge_ms": round(merge_ms, 1), "total_ms": round(lopt_ms, 1),
        "avg_anchor": round(avg_anchor, 1), "anc_fails": anc_fails,
        "has_fallback": has_fallback, "correct": correct,
    }


# ═══════════════════════════════════════════════════════════════
# Sweep runner
# ═══════════════════════════════════════════════════════════════
LO_VALUES = {"zh": [16, 32, 64, 128, 256], "en": [32, 64, 128, 256, 512]}
THREAD_VALUES = [1, 2, 4, 8, 16, 32]
SIZE_VALUES = [1000, 16000, 64000, 128000, 256000, 512000, 860000, 1024000]

MACHINE_INFO = {
    "host": "dev-modelarts.cn-southwest-2.huaweicloud.com:30028",
    "cpu": "ARM aarch64, 1 core",
    "ram": "2TB",
    "tokenizer": "Qwen3.5-35B-A3B (248K vocab)",
    "python": "3.11",
}


def run_one_lang(lang: str, out_lines: List[str], md_path: str):
    print(f"\n{'='*70}")
    print(f"  {lang.upper()} — Sweep starting: N={SIZE_VALUES}")
    print(f"{'='*70}")

    lo_vals = LO_VALUES[lang]
    all_sizes_data = {}
    
    for target in SIZE_VALUES:
        text = generate_text(lang, target)
        messages = [{"role": "user", "content": text}]
        ser_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        t0 = time.perf_counter()
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ser_ms = (time.perf_counter() - t0) * 1000
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_chars = len(prompt_str)
        n_tok = len(ser_ids)
        n_label = f"{target//1000}K" if target >= 1000 else f"{target}"
        
        lc_vals = compute_lc_values(n_chars)
        
        print(f"\n── [{lang.upper()} {n_label}] {n_tok} tok, {n_chars} chars, "
              f"Lc={lc_vals}, LO={lo_vals}, threads={THREAD_VALUES}")
        print(f"    Serial: {ser_ms:.1f}ms ({n_tok/ser_ms*1000:.0f} tok/s)")
        
        results = []
        total_tests = len(lc_vals) * len(lo_vals) * len(THREAD_VALUES)
        test_i = 0
        
        for Lc in lc_vals:
            for LO in lo_vals:
                for nt in THREAD_VALUES:
                    test_i += 1
                    r = run_lopt(prompt_str, ser_ids, Lc, LO, nt)
                    if r is None:
                        continue
                    r["ser_ms"] = round(ser_ms, 1)
                    r["ratio"] = round(r["total_ms"] / ser_ms, 3)
                    r["size_label"] = n_label
                    results.append(r)
                    
                    fb = " FB" if r["has_fallback"] else ""
                    cor = "✓" if r["correct"] else "✗"
                    print(f"  [{test_i:>3}/{total_tests}] Lc={Lc:>6} LO={LO:>3} t={nt:>2} | "
                          f"chk={r['n_chunks']:>4} | "
                          f"pool={r['pool_ms']:>7.1f} merge={r['merge_ms']:>7.1f} "
                          f"total={r['total_ms']:>8.1f} | "
                          f"{r['ratio']:>5.2f}x{fb} {cor}", flush=True)
        
        # Find best per size
        valid = [r for r in results if r["correct"] and not r["has_fallback"]]
        if not valid:
            valid = [r for r in results if r["correct"]]
        
        if valid:
            best = min(valid, key=lambda r: r["ratio"])
            best["speedup"] = round(1 / best["ratio"], 2)
        else:
            best = None
        
        all_sizes_data[n_label] = {
            "n_tok": n_tok, "n_chars": n_chars, "ser_ms": round(ser_ms, 1),
            "results": results, "best": best, "lc_vals": lc_vals,
        }
        
        if best:
            print(f"\n  >> BEST [{n_label}]: Lc={best['Lc']} LO={best['LO']} "
                  f"t={best['n_threads']} → {best['ratio']}x "
                  f"({best['speedup']}× speedup)")
        
        # Write intermediate results to md
        _write_md(md_path, lang, all_sizes_data, ser_ms, n_tok)

    return all_sizes_data


def _write_md(md_path: str, lang: str, all_sizes_data: dict,
              final_ser_ms: float = None, final_n_tok: int = None):
    lines = []
    lines.append(f"# LoPT Parameter Sweep Results\n")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Machine**: {MACHINE_INFO['host']}  ")
    lines.append(f"**CPU**: {MACHINE_INFO['cpu']} ({MACHINE_INFO['ram']} RAM)  ")
    lines.append(f"**Tokenizer**: {MACHINE_INFO['tokenizer']}  ")
    lines.append(f"**Python**: {MACHINE_INFO['python']}  ")
    lines.append(f"\n---\n")

    for lang_key in ([lang] if lang else ["zh", "en"]):
        sizes = all_sizes_data if lang else all_sizes_data.get(lang_key, {})
        
        lines.append(f"\n## Language: {lang_key.upper()}\n")
        
        # Summary table
        lines.append("### Best Configuration Per Size\n")
        lines.append("| Size | Tokens | Chars | Serial(ms) | Best Lc | Best LO | Best Thr | "
                     "Pool(ms) | Merge(ms) | LoPT(ms) | Ratio | Speedup |")
        lines.append("|:--|:--|:--|:--|:--|:--|:--|"
                     ":--|:--|:--|:--|:--|")
        
        for n_label in sorted(sizes.keys(), key=lambda x: {
            "1K": 1, "16K": 16, "64K": 64, "128K": 128, "256K": 256,
            "512K": 512, "860K": 860, "1024K": 1024}.get(x, 999)):
            if n_label not in sizes:
                continue
            d = sizes[n_label]
            b = d.get("best")
            if b:
                lines.append(f"| {n_label} | {d['n_tok']:,} | {d['n_chars']:,} | "
                             f"{d['ser_ms']:.1f} | {b['Lc']} | {b['LO']} | {b['n_threads']} | "
                             f"{b['pool_ms']:.1f} | {b['merge_ms']:.1f} | "
                             f"{b['total_ms']:.1f} | {b['ratio']:.3f}x | "
                             f"{b['speedup']:.2f}× |")
            else:
                lines.append(f"| {n_label} | {d['n_tok']:,} | {d['n_chars']:,} | "
                             f"{d['ser_ms']:.1f} | - | - | - | - | - | - | - | - |")
        
        # Full sweep for each size
        for n_label in sorted(sizes.keys(), key=lambda x: {
            "1K": 1, "16K": 16, "64K": 64, "128K": 128, "256K": 256,
            "512K": 512, "860K": 860, "1024K": 1024}.get(x, 999)):
            if n_label not in sizes:
                continue
            d = sizes[n_label]
            results = d["results"]
            lc_vals = d.get("lc_vals", [])
            
            lines.append(f"\n<details>\n<summary>")
            lines.append(f"<b>Full sweep: {n_label}</b> — {d['n_tok']:,} tok, "
                         f"{d['n_chars']:,} chars, serial {d['ser_ms']:.1f}ms, "
                         f"Lc ∈ {lc_vals}")
            lines.append(f"</summary>\n")
            
            # Group by Lc for readability
            for Lc in sorted(set(r["Lc"] for r in results)):
                lines.append(f"\n**Lc = {Lc}**  ")
                lines.append("| LO | Thr | Chk | Pool(ms) | Merge(ms) | Total(ms) | "
                             "Ratio | Anchor | Fail | Corr |")
                lines.append("|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|")
                
                for r in sorted(results, key=lambda x: (x["LO"], x["n_threads"])):
                    if r["Lc"] != Lc:
                        continue
                    fb = "FB" if r["has_fallback"] else ""
                    cor = "✓" if r["correct"] else "✗"
                    lines.append(f"| {r['LO']} | {r['n_threads']} | {r['n_chunks']} | "
                                 f"{r['pool_ms']:.1f} | {r['merge_ms']:.1f} | "
                                 f"{r['total_ms']:.1f} | {r['ratio']:.3f}x | "
                                 f"{r['avg_anchor']:.1f} | {fb} | {cor} |")
            
            lines.append("\n</details>\n")
    
    content = "\n".join(lines)
    with open(md_path, "w") as f:
        f.write(content)
    print(f"  [MD] Wrote {md_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoPT exhaustive sweep")
    parser.add_argument("--lang", default="zh,en")
    parser.add_argument("--md", default="/root/lopt_results.md",
                       help="Output markdown path")
    parser.add_argument("--tokenizer", default="/mnt/sfs_turbo/models/qwen3.5-35b-a3b")
    args = parser.parse_args()

    init_tokenizer(args.tokenizer)
    langs = [l.strip() for l in args.lang.split(",")]

    all_sizes_data = {}
    for lang in langs:
        lines = []
        data = run_one_lang(lang, lines, args.md)
        all_sizes_data[lang] = data
        _write_md(args.md, lang, all_sizes_data)
    
    print(f"\n{'='*70}")
    print(f"  ALL DONE. Results → {args.md}")
    print(f"{'='*70}")
