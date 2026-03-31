"""
extract_paint4poem_imagery.py
─────────────────────────────────────────────────────────────────────────────
从 Paint4Poem 数据集提取意象词，生成权威意象词表。

任务：
  1. 读取 Paint4Poem 所有诗文（Zikai-poem 301首 + Web-famous 6152首 + Web-regular 83052首）
  2. 与本项目52种花名做重合度分析
  3. 基于高频字符 + 扩展意象词典，提取意象词表
  4. 输出 paint4poem_imagery.csv（核心意象词表）

输出（output/）：
  paint4poem_imagery.csv          完整意象词表（带频次、分类、与项目对比）
  paint4poem_flower_overlap.csv   花名重合度分析
  paint4poem_summary.txt          文字版摘要报告

运行：
    cd /Users/rzhang/Documents/poem-flower-recommend
    source flower_env/bin/activate
    python 02.sample_label_phase2/step2b_bert/extract_paint4poem_imagery.py
─────────────────────────────────────────────────────────────────────────────
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ─── 路径 ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent.parent
PAINT4POEM_DIR = PROJECT_ROOT / "models" / "paint4poem"
OUTPUT_DIR    = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_CSV = PROJECT_ROOT / "01.sample_label" / "output" / "sample_200.csv"

# ─── 本项目花名列表 ───────────────────────────────────────────────────────────
def load_project_flowers() -> dict:
    """返回 {花名: 核心字} 字典，用于模糊匹配诗文"""
    df = pd.read_csv(SAMPLE_CSV)
    flowers = sorted(df["花名"].dropna().unique())
    # 每种花的"核心字"（去掉「花」「梅」等通用后缀后的核心词）
    core_map = {}
    for f in flowers:
        core_map[f] = f
    # 增加单字/双字别名
    extra = {
        "梅": "梅花", "兰": "兰花", "菊": "菊花", "荷": "荷花",
        "桃": "桃花", "桂": "桂花", "杏": "杏花", "柳": "杨花",
        "玫瑰": "蔷薇", "牡丹": "牡丹", "梅花": "梅花",
        "芍药": "芍药", "莲": "荷花", "芙蓉": "荷花",
        "水仙": "水仙花", "月季": "月季", "山茶": "山茶花",
        "杜鹃": "杜鹃", "蜡梅": "蜡梅", "腊梅": "蜡梅",
        "虞美人": "虞美人", "紫薇": "紫薇", "木兰": "木兰",
        "玉兰": "玉兰", "榴花": "榴花", "石榴": "榴花",
        "桂花": "桂花", "金银花": "金银花", "栀子": "栀子花",
    }
    return flowers, {**{f: f for f in flowers}, **extra}


# ─── 扩展意象词典（来源：学界共识 + Paint4Poem论文Table2 + 本项目已有词典）──
# 论文 Table 2 最高频意象（Zikai-Poem数据集）：日/风/花/春/山/水/月/天/美人
IMAGERY_DB = {
    # ── 天象时令（论文最高频类别）────────────────────────────────────────────
    "天象时令": [
        "日", "月", "风", "雨", "雪", "霜", "露", "云", "雷", "电",
        "虹", "霞", "烟", "雾", "冰", "霰", "晓", "暮", "夕", "晨",
        "春", "夏", "秋", "冬", "夜", "昼", "午", "黄昏", "清晨",
        "寒", "暖", "凉", "热", "冷", "熏", "薰",
    ],
    # ── 山水地理 ──────────────────────────────────────────────────────────────
    "山水地理": [
        "山", "水", "江", "河", "湖", "海", "溪", "涧", "泉", "潭",
        "峰", "岭", "崖", "岩", "石", "沙", "洲", "渚", "岸", "滩",
        "波", "涛", "浪", "流", "潮", "渊", "谷", "林", "野", "原",
        "天", "地", "空", "苍", "碧",
        # 文化地理意象
        "关山", "阳关", "玉门", "长亭", "短亭", "南浦", "渭城",
        "江南", "塞北", "边塞", "天涯", "海角", "天涯海角",
    ],
    # ── 植物花卉（项目核心，重点扩充）────────────────────────────────────────
    "植物花卉": [
        # 花卉（与项目52种高度重叠）
        "梅", "梅花", "兰", "兰花", "竹", "菊", "菊花", "莲", "荷", "荷花",
        "桃", "桃花", "杏", "杏花", "柳", "杨柳", "桂", "桂花",
        "牡丹", "芍药", "蔷薇", "玫瑰", "月季", "海棠",
        "水仙", "蜡梅", "腊梅", "梨花", "李花",
        "芙蓉", "木兰", "玉兰", "山茶", "紫薇",
        "杜鹃", "虞美人", "石榴", "榴花",
        "金银花", "栀子", "栀子花", "迎春",
        "芦", "芦花", "蓼", "蓼花",
        # 非花植物意象
        "松", "柏", "梧", "梧桐", "椿", "桑", "柘", "槐", "桐",
        "草", "芳草", "绿草", "衰草", "枯草", "萋萋",
        "叶", "枝", "根", "干", "藤", "蔓",
        "香", "芳", "芬", "馨", "幽香", "暗香", "清香", "余香",
        "枯", "凋", "零落", "飘零", "残",
        # 诗词常见植物意象词组
        "疏影", "暗香", "暗香疏影", "折梅", "折柳", "攀折",
        "岁寒三友", "四君子",
    ],
    # ── 动物禽鸟 ──────────────────────────────────────────────────────────────
    "动物禽鸟": [
        "雁", "鸿", "燕", "鹤", "鹰", "鹰隼", "鸟", "禽",
        "蝉", "蝶", "蛱蝶", "萤", "蜂", "莺", "鹃", "杜鹃",
        "鱼", "鲤", "鲸", "蛟", "龙", "凤", "麟",
        "鸡", "犬", "马", "牛", "羊", "猿", "猴",
        # 文化意象
        "孤鸿", "征雁", "宿燕", "归燕", "哀猿", "啼鸟",
    ],
    # ── 器物人文 ──────────────────────────────────────────────────────────────
    "器物人文": [
        "剑", "刀", "弓", "矢", "戈", "矛", "旌", "旗",
        "琴", "瑟", "筝", "笛", "箫", "笙", "鼓", "筑",
        "酒", "杯", "觥", "壶", "尊", "斛",
        "灯", "烛", "炉", "镜", "帘", "帷", "幕", "幔",
        "舟", "帆", "桨", "楫", "篷",
        "楼", "台", "阁", "榭", "亭", "轩", "堂", "室",
        "书", "笔", "墨", "砚", "笺",
        "玉", "珠", "珍", "宝", "金", "银", "锦", "绫",
        # 文化符号
        "东篱", "南山", "西楼", "北斗", "蓬莱", "瑶池",
        "折桂", "蟾宫", "雁塔", "鸿雁传书",
    ],
    # ── 行为动态意象 ──────────────────────────────────────────────────────────
    "行为动态": [
        "飞", "落", "零", "凋", "散", "消", "逝", "沉",
        "归", "去", "来", "行", "别", "离", "断",
        "望", "思", "忆", "念", "怀", "愁", "恨",
        "寄", "托", "书", "题", "吟", "咏", "赋",
        "醉", "饮", "把酒", "临风", "凭栏",
        "折柳", "折梅", "寄梅", "采莲", "弄笛",
        "登高", "望月", "赏花", "对酒", "把盏",
    ],
    # ── 人物意象 ──────────────────────────────────────────────────────────────
    "人物意象": [
        "美人", "佳人", "才子", "游子", "征人", "旅人",
        "渔翁", "隐士", "高士", "逸士", "布衣",
        "思妇", "闺妇", "红颜", "红袖", "白发",
        "英雄", "豪杰", "壮士", "战士",
        # 历史典故人物（高频）
        "陶潜", "陶令", "陶公", "东坡", "太白", "少陵",
        "昭君", "西施", "貂蝉", "虞姬",
    ],
    # ── 情感抽象意象 ──────────────────────────────────────────────────────────
    "情感抽象": [
        "愁", "恨", "悲", "哀", "苦", "怨", "泪", "涕",
        "喜", "乐", "欢", "笑", "幸", "福",
        "孤", "独", "寂", "寞", "闲", "静", "幽",
        "豪", "壮", "雄", "烈", "慷", "慨",
        "高洁", "清高", "傲骨", "铁骨", "凌寒",
        "忠", "义", "节", "操", "气节", "风骨",
    ],
}

# 扁平化所有意象词，建立 词→分类 的映射
WORD_TO_CATEGORY = {}
for cat, words in IMAGERY_DB.items():
    for w in words:
        if w not in WORD_TO_CATEGORY:
            WORD_TO_CATEGORY[w] = cat

# 所有意象词的集合（含多字词）
ALL_IMAGERY_WORDS = set(WORD_TO_CATEGORY.keys())


# ─── 文本清洗 ─────────────────────────────────────────────────────────────────
PUNCT = re.compile(r"[，。！？、；：「」『』（）《》〔〕【】…—·\s\n\r\t]")

def clean_poem(text: str) -> str:
    return PUNCT.sub("", str(text))


def extract_chars_and_words(text: str, multi_word_set: set) -> list:
    """提取单字 + 多字意象词（优先匹配多字词）"""
    cleaned = clean_poem(text)
    found = []
    i = 0
    while i < len(cleaned):
        # 尝试最长匹配（从4字到2字）
        matched = False
        for length in [4, 3, 2]:
            chunk = cleaned[i:i+length]
            if chunk in multi_word_set:
                found.append(chunk)
                i += length
                matched = True
                break
        if not matched:
            if cleaned[i]:
                found.append(cleaned[i])
            i += 1
    return found


# ─── 读取所有诗文 ─────────────────────────────────────────────────────────────
def load_all_poems() -> dict:
    """
    返回 { subset_name: [poem_text, ...] }
    """
    poems = {}

    # 1. Zikai-poem (301首，最权威，有完整正文)
    zikai_csv = PAINT4POEM_DIR / "Paint4Poem-Zikai-poem-subset" / "POEM-IMAGE.csv"
    if zikai_csv.exists():
        df = pd.read_csv(zikai_csv)
        texts = df["PoemText"].dropna().tolist()
        poems["zikai_poem"] = texts
        print(f"  Zikai-poem: {len(texts)} 首")

    # 2. Web-famous (6152首)
    famous_csv = PAINT4POEM_DIR / "Paint4Poem-Web-famous-subset" / "POEM_IMAGE.csv"
    if famous_csv.exists():
        df = pd.read_csv(famous_csv, sep="\t")
        texts = df["poem"].dropna().tolist()
        poems["web_famous"] = texts
        print(f"  Web-famous: {len(texts)} 首/句")

    # 3. Web-regular（多个子集，名称含空格）
    regular_texts = []
    for subdir in PAINT4POEM_DIR.iterdir():
        if "Web-regular" in subdir.name and subdir.is_dir():
            csv_f = subdir / "POEM_IMAGE.csv"
            if csv_f.exists():
                try:
                    df = pd.read_csv(csv_f, sep="\t")
                    texts = df["poem"].dropna().tolist()
                    regular_texts.extend(texts)
                except Exception as e:
                    print(f"  [WARN] 读取 {subdir.name} 失败: {e}")
    if regular_texts:
        poems["web_regular"] = regular_texts
        print(f"  Web-regular（合并）: {len(regular_texts)} 首/句")

    return poems


# ─── 主分析 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Paint4Poem 意象提取分析")
    print("=" * 65)

    # 1. 加载花名
    project_flowers, flower_alias = load_project_flowers()
    print(f"\n本项目花名：{len(project_flowers)} 种")

    # 2. 加载诗文
    print("\n[1/4] 读取 Paint4Poem 诗文 ...")
    all_poem_sets = load_all_poems()
    all_texts = [t for texts in all_poem_sets.values() for t in texts]
    print(f"  总计：{len(all_texts)} 首/句诗文")

    # 多字意象词集合（用于优先匹配）
    multi_words = {w for w in ALL_IMAGERY_WORDS if len(w) >= 2}

    # 3. 统计字符频次（全量）
    print("\n[2/4] 统计字符频次 ...")
    char_counter: Counter = Counter()
    word_counter: Counter = Counter()  # 意象多字词频次

    for text in all_texts:
        tokens = extract_chars_and_words(text, multi_words)
        for t in tokens:
            if len(t) == 1:
                char_counter[t] += 1
            else:
                word_counter[t] += 1

    # Zikai-poem 单独统计（权威子集）
    zikai_char_counter: Counter = Counter()
    zikai_word_counter: Counter = Counter()
    for text in all_poem_sets.get("zikai_poem", []):
        tokens = extract_chars_and_words(text, multi_words)
        for t in tokens:
            if len(t) == 1:
                zikai_char_counter[t] += 1
            else:
                zikai_word_counter[t] += 1

    print(f"  不同字符数: {len(char_counter)}, 多字词数: {len(word_counter)}")

    # 4. 花名重合度分析
    print("\n[3/4] 花名重合度分析 ...")
    flower_rows = []
    for flower in project_flowers:
        # 从花名提取核心字（用于在诗文中搜索）
        core = flower.rstrip("花")  # 去掉末尾的"花"字
        core2 = flower  # 完整花名

        # 在所有诗文中直接搜索
        full_text = "".join(all_texts)
        cnt_full   = full_text.count(flower)
        cnt_core   = full_text.count(core) if len(core) >= 2 else char_counter.get(core, 0)
        cnt_zikai  = "".join(all_poem_sets.get("zikai_poem", [])).count(core2)

        flower_rows.append({
            "花名":           flower,
            "核心词":         core,
            "全数据集频次":   cnt_full,
            "核心词频次":     cnt_core,
            "Zikai子集频次":  cnt_zikai,
            "在P4P中出现":    cnt_full > 0 or cnt_core > 0,
        })

    df_flower = pd.DataFrame(flower_rows).sort_values("全数据集频次", ascending=False)
    overlap_count = df_flower["在P4P中出现"].sum()
    print(f"  本项目 {len(project_flowers)} 种花中，在 Paint4Poem 有出现：{overlap_count} 种")

    df_flower.to_csv(str(OUTPUT_DIR / "paint4poem_flower_overlap.csv"),
                     index=False, encoding="utf-8-sig")

    # 5. 提取意象词表
    print("\n[4/4] 生成意象词表 ...")

    # 当前项目已有意象词
    CURRENT_VOCAB = {
        "月","日","风","雪","霜","云","水","山","雨","露",
        "松","竹","梅","兰","菊","莲","柳","桃","杏","桂",
        "鸿","雁","燕","鹤","蝉","萤","蝶","鱼","花","草",
        "叶","枝","根","香","寒","春","秋","冬","夏","天",
        "江","湖","河","海","溪","涧","石","峰","岭",
        "剑","琴","酒","烛","灯","镜","楼","舟","帆",
        "东篱","南山","折柳","芳草","茅屋","寒窗",
        "飞","落","凋","零","断","寄","归","别","残","空",
        "散","消","逝","沉","望","思","忆",
    }

    imagery_rows = []
    for word, category in WORD_TO_CATEGORY.items():
        if len(word) == 1:
            freq_total  = char_counter.get(word, 0)
            freq_zikai  = zikai_char_counter.get(word, 0)
        else:
            freq_total  = word_counter.get(word, 0)
            freq_zikai  = zikai_word_counter.get(word, 0)
        # 在所有诗文中的粗略频次（对于多字词用直接搜索）
        if len(word) >= 2 and freq_total == 0:
            full_text_cache = "".join(all_texts)
            freq_total = full_text_cache.count(word)
            freq_zikai = "".join(all_poem_sets.get("zikai_poem", [])).count(word)

        imagery_rows.append({
            "意象词":                word,
            "分类":                  category,
            "Paint4Poem全量频次":    freq_total,
            "Zikai权威子集频次":     freq_zikai,
            "字数":                  len(word),
            "在当前项目词典中":       word in CURRENT_VOCAB,
            "在项目花名中":           word in project_flowers or word in flower_alias,
            "新增词(当前词典不含)":   word not in CURRENT_VOCAB,
        })

    df_imagery = pd.DataFrame(imagery_rows)
    df_imagery = df_imagery.sort_values(
        ["Paint4Poem全量频次", "分类"], ascending=[False, True]
    )

    df_imagery.to_csv(str(OUTPUT_DIR / "paint4poem_imagery.csv"),
                      index=False, encoding="utf-8-sig")

    # 6. 生成报告
    total_words  = len(df_imagery)
    in_current   = df_imagery["在当前项目词典中"].sum()
    new_words    = df_imagery["新增词(当前词典不含)"].sum()
    zero_freq    = (df_imagery["Paint4Poem全量频次"] == 0).sum()
    high_freq    = (df_imagery["Paint4Poem全量频次"] >= 100).sum()

    cat_summary = df_imagery.groupby("分类").agg(
        词数=("意象词", "count"),
        总频次=("Paint4Poem全量频次", "sum"),
        高频词=("意象词", lambda x: "、".join(
            df_imagery[df_imagery["意象词"].isin(x)]
            .nlargest(5, "Paint4Poem全量频次")["意象词"].tolist()
        ))
    ).reset_index()

    # 花名重合详情
    in_p4p = df_flower[df_flower["在P4P中出现"]]["花名"].tolist()
    not_in_p4p = df_flower[~df_flower["在P4P中出现"]]["花名"].tolist()

    top20_imagery = df_imagery.nlargest(20, "Paint4Poem全量频次")[
        ["意象词", "分类", "Paint4Poem全量频次", "Zikai权威子集频次"]
    ]

    lines = [
        "=" * 68,
        "Paint4Poem 数据集意象分析报告",
        "=" * 68,
        "",
        f"── 数据集规模 ──────────────────────────────────────",
        f"  Zikai-poem（权威手工配对）: {len(all_poem_sets.get('zikai_poem',[]))} 首",
        f"  Web-famous（名句）        : {len(all_poem_sets.get('web_famous',[]))} 首/句",
        f"  Web-regular（大规模）     : {len(all_poem_sets.get('web_regular',[]))} 首/句",
        f"  合计                      : {len(all_texts)} 首/句",
        "",
        f"── 意象词表统计 ────────────────────────────────────",
        f"  本次生成意象词总数  : {total_words}",
        f"  已在当前项目词典中  : {in_current}",
        f"  新增（当前词典不含）: {new_words}  ← 可直接扩充 IMAGERY_VOCAB",
        f"  在P4P中频次=0的词   : {zero_freq}（词典中有但语料未见，属于生僻意象）",
        f"  频次≥100的高频意象  : {high_freq} 个",
        "",
        f"── Top-20 高频意象（Paint4Poem 全量）──────────────",
    ]
    for _, r in top20_imagery.iterrows():
        lines.append(
            f"  {r['意象词']:4s}  [{r['分类']}]  "
            f"全量={r['Paint4Poem全量频次']:5d}  Zikai={r['Zikai权威子集频次']:4d}"
        )

    lines += [
        "",
        f"── 各分类统计 ──────────────────────────────────────",
    ]
    for _, r in cat_summary.iterrows():
        lines.append(
            f"  {r['分类']:8s}  词数={r['词数']:3d}  "
            f"总频次={r['总频次']:7d}  "
            f"高频词：{r['高频词']}"
        )

    lines += [
        "",
        f"── 花名重合度（本项目 {len(project_flowers)} 种 vs Paint4Poem）────────",
        f"  有重合的花（{overlap_count} 种）: {' / '.join(in_p4p[:20])}{'...' if len(in_p4p)>20 else ''}",
        f"  无重合的花（{len(not_in_p4p)} 种）: {' / '.join(not_in_p4p)}",
        "",
        f"── Top-15 花名在 Paint4Poem 出现频次 ──────────────",
    ]
    for _, r in df_flower.head(15).iterrows():
        lines.append(
            f"  {r['花名']:8s}  全量={r['全数据集频次']:5d}  "
            f"核心词={r['核心词']}({r['核心词频次']:5d})  "
            f"Zikai={r['Zikai子集频次']:3d}"
        )

    lines += [
        "",
        f"── 建议扩充 IMAGERY_VOCAB 的新词（频次≥50，当前词典不含）─",
    ]
    new_high = df_imagery[
        (df_imagery["新增词(当前词典不含)"] == True) &
        (df_imagery["Paint4Poem全量频次"] >= 50)
    ].sort_values("Paint4Poem全量频次", ascending=False)
    for _, r in new_high.head(30).iterrows():
        lines.append(
            f"  {r['意象词']:6s}  [{r['分类']}]  频次={r['Paint4Poem全量频次']}"
        )

    report_path = OUTPUT_DIR / "paint4poem_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # 终端打印
    print("\n".join(lines))

    print("\n" + "=" * 65)
    print("✅ 完成！输出文件：")
    for f in ["paint4poem_imagery.csv", "paint4poem_flower_overlap.csv", "paint4poem_summary.txt"]:
        p = OUTPUT_DIR / f
        if p.exists():
            print(f"  - {f}  ({p.stat().st_size/1024:.0f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
