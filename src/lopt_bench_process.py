#!/usr/bin/env python3
"""
ProcessPool vs ThreadPool vs Serial — 全量对比。
用法:
  python3 lopt_bench_process.py --lang zh --sizes 128K,512K,1024K --workers 2,4,8,16,32
  python3 lopt_bench_process.py --lang en --tokenizer /path/to/dsv4_tokenizer

输出: 中位值 3 次测量，含 warmup，ProcessPool(spawn) 模式。
"""
import time, statistics, argparse, math, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# 20 短段中文新闻
ZH = [
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
EN = [
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
BLOCKS = {"zh": ZH, "en": EN}
TOK_PATH = "/mnt/sfs_turbo/models/qwen3.5-35b-a3b"


def pick_lc(chars):
    """从搜索脚本的 Lc 候选中取中位值。"""
    vals = set()
    start = max(64, chars // 128)
    val = start
    while val <= chars // 2:
        vals.add(2 ** round(math.log2(val)))
        vals.add(val)
        val *= 2
    for v in [8192, 16384, 32768, 65536, 131072, 262144, 524288]:
        if v > start//2 and v <= chars:
            vals.add(v)
    r = sorted(v for v in vals if v < chars)
    if len(r) > 8:
        step = len(r)/8
        r = [r[int(i*step)] for i in range(8)]
        r = sorted(set(r))
    return r[len(r)//2] if r else 65536


# ═══ Worker（每个 spawn 进程有独立 tokenizer） ═══
_R = {}
def init_worker(path):
    import transformers
    global _R
    if "_rust" not in _R:
        _R["_rust"] = transformers.AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)._tokenizer

def worker(args):
    text, offset = args
    global _R
    e = _R["_rust"].encode(text, add_special_tokens=False)
    return e.ids, [(x[0]+offset, x[1]+offset) for x in e.offsets]


# ═══ 入口 ═══
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--tokenizer", default=TOK_PATH)
    parser.add_argument("--sizes", default="128K,512K,1024K")
    parser.add_argument("--workers", default="2,4,8,16,32")
    parser.add_argument("--lo", type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = "{{ messages[0]['content'] }}"
    _rust = tok._tokenizer
    worker_list = [int(w) for w in args.workers.split(",")]

    sz = []
    for s in args.sizes.split(","):
        s = s.strip().upper()
        if s.endswith("K"):
            sz.append(int(s[:-1])*1000)
        elif s.endswith("M"):
            sz.append(int(s[:-1])*1000000)
        else:
            sz.append(int(s))

    print(f"# Lang={args.lang} Tok={args.tokenizer} Workers={worker_list} LO={args.lo}")
    cols = f"{'Size':>8} {'Tok':>10} {'Chars':>10} {'Lc':>8} {'Chk':>5}"
    cols += f" {'Serial':>9} {'TPool':>9} {'TPx':>6}"
    cols += f" {'PPool':>9} {'PPx':>6} {'Warm':>6} {'Wkr':>5}"
    print(f"# {cols}")
    sep = f"{'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*5}"
    sep += f" {'─'*9} {'─'*9} {'─'*6} {'─'*9} {'─'*6} {'─'*6} {'─'*5}"
    print(f"# {sep}")

    for target in sz:
        # generate text
        single = "".join(BLOCKS[args.lang])
        rust = _rust
        single_tok = len(rust.encode(single, add_special_tokens=False).ids)
        text = single * max(1, target // max(single_tok, 1))
        prompt_str = tok.apply_chat_template(
            [{"role":"user","content":text}],
            tokenize=False, add_generation_prompt=True)
        ser_ids = rust.encode(prompt_str, add_special_tokens=False).ids
        n, n_tok = len(prompt_str), len(ser_ids)
        lc = pick_lc(n)
        nc = (n + lc - 1) // lc
        if nc <= 1:
            continue
        label = f"{target//1000}K"
        chunks = [(prompt_str[i*lc:min(n,(i+1)*lc+args.lo)], i*lc) for i in range(nc)]

        # ── Serial（3次取中位） ──
        st = []
        for _ in range(3):
            t0 = time.perf_counter()
            rust.encode(prompt_str, add_special_tokens=False)
            st.append((time.perf_counter()-t0)*1000)
        ser = statistics.median(st)

        # ── ThreadPool（3次取中位） ──
        def _enc(t):
            e = rust.encode(t, add_special_tokens=False)
            return e.ids, e.offsets
        tp_nt = max(1, min(worker_list))
        tt = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(tp_nt) as ex:
                list(ex.map(lambda a: (
                    lambda ids, off: (ids, [(x[0]+a[1], x[1]+a[1]) for x in off])
                )(*_enc(a[0])), chunks))
            tt.append((time.perf_counter()-t0)*1000)
        tp = statistics.median(tt)

        # ── ProcessPool（每个 worker 数创建一次 pool） ──
        for nt in worker_list:
            pool = mp.get_context("spawn").Pool(
                nt, initializer=init_worker, initargs=(args.tokenizer,))
            try:
                # warmup
                pool.map(worker, chunks[:min(nt, nc)])
                wt = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    pool.map(worker, chunks[:min(nt, nc)])
                    wt.append((time.perf_counter()-t0)*1000)
                warm = statistics.median(wt)
                # measure
                pt = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    pool.map(worker, chunks)
                    pt.append((time.perf_counter()-t0)*1000)
                pp = statistics.median(pt)
                print(f"  {label:>8} {n_tok:>10} {n:>10} {lc:>8} {nc:>5} "
                      f"{ser:>9.1f} {tp:>9.1f} {ser/tp:>5.2f}× "
                      f"{pp:>9.1f} {ser/pp:>5.2f}× "
                      f"{warm:>6.1f} {nt:>5}", flush=True)
            finally:
                pool.close()
                pool.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
