"""
step2c_imagery/imagery_extract.py
===================================
基于词典匹配（路径B），对 1075 首诗的正文提取"物象"，
并分析 79 种花名 × 物象 的共现关系。

思路：
  - 论文方法（RoBERTa-BiLSTM-CRF）是训练好的NER模型直接推理
  - 路径B 用"词典匹配 + 最长优先"模拟同样的效果
  - 词典来源：
      1. 古汉语常用名词词库（天象/山水/植物/动物/器物/人物/情感）
      2. 项目79种花名（直接加入词典）
      3. 常见文化地理意象（关山、长亭等）

输出（output/）：
  - imagery_per_poem.csv        每首诗提取的物象列表
  - imagery_frequency.csv       全局物象词频排名
  - flower_imagery_cooccur.csv  花名 × 物象 共现计数矩阵
  - flower_top_imagery.csv      每种花 Top-20 共现物象
  - summary_report.txt          文字摘要
"""

from pathlib import Path
import re
from collections import Counter, defaultdict
import pandas as pd

# ─── 路径 ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = ROOT / "00.poems_dataset" / "poems_dataset_merged_done.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 物象词典（论文 Table1 四类 + 扩展）────────────────────────────────────────
# 编写原则：
#   - 优先多字词（避免被单字截断）
#   - 覆盖论文 Table4 高频未登录词：春风、故人、秋风、千里、相思等
#   - 涵盖花卉、动物、器物、天象、地理等各类意象
IMAGERY_VOCAB = {
    # ── 天象时令 ──────────────────────────────────────────────────────────────
    "天象时令": [
        # 复合词优先
        "春风", "秋风", "寒风", "东风", "西风", "晓风", "夜风", "暮风",
        "春雨", "秋雨", "寒雨", "夜雨", "细雨", "苦雨",
        "白雪", "残雪", "飞雪", "积雪", "寒雪",
        "白日", "落日", "斜日", "夕阳", "朝日", "旭日",
        "明月", "残月", "新月", "孤月", "寒月", "秋月", "圆月", "弯月",
        "白露", "寒露", "朝露", "晨露",
        "浮云", "白云", "黑云", "闲云", "暮云", "乱云",
        "秋霜", "寒霜", "白霜",
        "烟雨", "烟云", "烟霞", "暮霭",
        "黄昏", "清晨", "黎明", "五更",
        "清明", "重阳", "中秋", "除夕",
        # 单字
        "日", "月", "风", "雨", "雪", "霜", "露", "云", "雷", "电",
        "虹", "霞", "烟", "雾", "冰", "霰",
        "晓", "暮", "夕", "晨", "昼", "夜",
        "春", "夏", "秋", "冬",
        "寒", "暖", "凉", "热", "冷",
    ],

    # ── 山水地理 ──────────────────────────────────────────────────────────────
    "山水地理": [
        # 文化地理意象（复合词）
        "关山", "阳关", "玉门关", "玉门", "嘉峪关",
        "长亭", "短亭", "南浦", "渭城", "渭水",
        "江南", "塞北", "边塞", "天涯", "海角",
        "故乡", "故国", "故园", "他乡", "异乡",
        "洞庭", "潇湘", "巴山", "巫山", "峨眉",
        "黄河", "长江", "汉江", "淮河",
        "东海", "南海", "沧海",
        "春山", "秋山", "寒山", "青山", "远山", "空山",
        "寒江", "春江", "秋江", "晓江",
        "碧波", "碧水", "流水", "春水",
        "沙场", "战场",
        # 单字
        "山", "水", "江", "河", "湖", "海", "溪", "涧", "泉", "潭",
        "峰", "岭", "崖", "岩", "石", "沙", "洲", "渚", "岸", "滩",
        "波", "涛", "浪", "流", "潮", "渊", "谷",
        "林", "野", "原", "天", "地", "空",
        "关", "塞", "亭", "桥", "渡", "津",
    ],

    # ── 植物花卉（含项目79种花名）────────────────────────────────────────────
    "植物花卉": [
        # 项目79种花名（完整词优先）
        "梅花", "红梅", "蜡梅", "腊梅",
        "杏花", "桃花", "李花", "梨花", "棠梨花", "樱桃花", "棣棠",
        "荷花", "莲花", "芙蓉", "水芙蓉",
        "菊花", "甘菊", "野菊",
        "桂花",
        "兰花", "秋兰",
        "海棠", "秋海棠",
        "牡丹",
        "芍药",
        "蔷薇", "玫瑰", "月季",
        "木兰", "辛夷", "玉兰", "木芙蓉", "木槿", "木棉花", "木香",
        "水仙", "水仙花",
        "紫薇", "紫荆",
        "杜鹃",
        "迎春花", "迎春",
        "虞美人",
        "芦花",
        "萱草",
        "玉簪花", "玉蕊花",
        "琼花",
        "茉莉",
        "素馨",
        "栀子花",
        "金银花",
        "蓼花",
        "鸡冠花",
        "凤仙花",
        "牵牛花",
        "向日葵",
        "含笑",
        "合欢",
        "瑞香",
        "绣球花", "绣球",
        "酴醿",
        "榴花",
        "石竹",
        "秋葵", "蜀葵",
        "凌霄花",
        "剪春罗",
        "楝花",
        "桐花",
        "山茶花",
        "山枇杷",
        "曼陀罗",
        "滴滴金",
        "罂粟",
        "菜花",
        "蘋花",
        "雁来红",
        "金钱花",
        "金沙",
        "郁李花",
        "棠梨花",
        "杨花",
        # 其他常见植物意象（复合词）
        "芳草", "绿草", "衰草", "枯草", "萋萋芳草",
        "杨柳", "垂柳", "芳柳", "烟柳",
        "松柏", "青松", "老松",
        "梧桐", "芭蕉",
        "绿竹", "修竹", "翠竹", "竹影",
        "落叶", "黄叶", "红叶", "枯叶", "残叶",
        "疏影", "暗香", "余香", "幽香", "清香",
        "落花", "残花", "飞花", "繁花",
        "花落", "花开", "花谢",
        # 单字
        "梅", "兰", "竹", "菊", "荷", "莲", "桃", "杏", "桂",
        "柳", "松", "柏", "梧", "椿", "桑", "槐", "桐",
        "草", "叶", "枝", "根", "藤", "蔓",
        "香", "芳", "芬", "馨",
        "花",
    ],

    # ── 动物禽鸟 ──────────────────────────────────────────────────────────────
    "动物禽鸟": [
        # 复合词
        "孤鸿", "征雁", "归雁", "孤雁", "塞雁", "鸿雁",
        "归燕", "宿燕", "双燕", "乳燕",
        "子规", "杜鹃", "啼鸟", "哀猿", "啼猿",
        "锦鸡", "黄莺", "黄鹂",
        "蝴蝶", "蛱蝶", "粉蝶",
        "萤火", "萤虫",
        # 单字
        "雁", "鸿", "燕", "鹤", "鹰", "鸟", "禽",
        "蝉", "蝶", "萤", "蜂", "莺", "鹃",
        "鱼", "龙", "凤", "麟",
        "马", "牛", "羊", "猿",
    ],

    # ── 器物人文 ──────────────────────────────────────────────────────────────
    "器物人文": [
        # 乐器
        "古琴", "琵琶", "羌笛", "玉笛", "横笛", "短笛",
        "洞箫", "羌笙",
        # 武器
        "宝剑", "长剑", "铁剑", "干戈",
        # 饮酒
        "把酒", "举杯", "金樽", "玉樽",
        # 建筑
        "玉楼", "高楼", "西楼", "危楼", "小楼",
        "画阁", "朱阁", "香阁",
        "茅屋", "草堂",
        # 交通
        "孤舟", "扁舟", "轻舟", "渔舟",
        "马蹄", "征马",
        # 书写
        "锦书", "尺书", "鸿书",
        # 文化符号
        "东篱", "南山", "蓬莱", "瑶池",
        "折柳", "折梅", "寄梅", "采莲",
        "登高", "望月", "凭栏", "倚楼",
        # 单字
        "剑", "刀", "弓", "戈", "旌", "旗",
        "琴", "瑟", "筝", "笛", "箫", "笙", "鼓",
        "酒", "杯", "壶",
        "灯", "烛", "炉", "镜", "帘", "帷",
        "舟", "帆", "桨",
        "楼", "台", "阁", "榭", "亭", "轩",
        "书", "笔", "墨", "砚",
        "玉", "珠", "金", "银", "锦",
    ],

    # ── 人物意象 ──────────────────────────────────────────────────────────────
    "人物意象": [
        # 复合词（高频）
        "美人", "佳人", "丽人",
        "游子", "羁旅", "旅人", "行人", "征人",
        "思妇", "闺中人", "红颜",
        "渔翁", "渔父", "老渔翁",
        "隐士", "高士",
        "故人", "旧人",
        "千里人",
        # 历史典故人物
        "昭君", "西施", "虞姬",
        "陶潜", "陶令",
        # 单字
        "人", "客", "翁",
    ],

    # ── 情感抽象 ──────────────────────────────────────────────────────────────
    "情感抽象": [
        # 复合词
        "相思", "离愁", "乡愁", "旅愁", "万古愁",
        "哀怨", "悲愁", "苦愁",
        "孤独", "寂寞", "孤寂",
        "豪情", "壮志", "雄心",
        "高洁", "清高", "傲骨", "凌寒",
        "气节", "风骨",
        # 单字
        "愁", "恨", "悲", "哀", "苦", "怨", "泪",
        "喜", "乐", "欢",
        "孤", "寂", "幽",
        "豪", "壮", "烈",
        "忠", "义",
    ],

    # ── 时空行为（动态意象）───────────────────────────────────────────────────
    "时空行为": [
        "千里", "万里", "百里",
        "今日", "今朝", "今夕", "昨日", "他日",
        "少年", "白发", "平生",
        "人间", "天上", "世间",
        "春色", "秋色", "夜色",
        "月色", "月光", "月影",
        "水色", "山色", "江色",
    ],
}

# ─── 扁平化：词 → 类别，构建搜索集合 ─────────────────────────────────────────
WORD_TO_CAT: dict[str, str] = {}
for cat, words in IMAGERY_VOCAB.items():
    for w in words:
        if w not in WORD_TO_CAT:
            WORD_TO_CAT[w] = cat

ALL_WORDS: set[str] = set(WORD_TO_CAT.keys())
# 多字词集合（用于最长优先匹配）
MULTI_WORDS: set[str] = {w for w in ALL_WORDS if len(w) > 1}
MAX_WORD_LEN = max(len(w) for w in ALL_WORDS)

# ─── 文本清洗 ─────────────────────────────────────────────────────────────────
PUNCT_RE = re.compile(
    r"[，。！？、；：「」『』（）《》〔〕【】…—·\s\n\r\t\u3000，。、；：！？]"
)

def clean(text: str) -> str:
    return PUNCT_RE.sub("", str(text))


def extract_imagery(text: str) -> list[str]:
    """
    最长优先匹配：优先识别多字词，识别不到再取单字。
    返回该文本中出现的所有物象词（允许重复）。
    """
    s = clean(text)
    found = []
    i = 0
    while i < len(s):
        matched = False
        for length in range(min(MAX_WORD_LEN, len(s) - i), 1, -1):
            chunk = s[i: i + length]
            if chunk in MULTI_WORDS:
                found.append(chunk)
                i += length
                matched = True
                break
        if not matched:
            ch = s[i]
            if ch in ALL_WORDS:
                found.append(ch)
            i += 1
    return found


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  step2c：词典匹配物象提取（路径B）")
    print("=" * 60)

    df = pd.read_csv(DATA_CSV)
    print(f"\n数据集：{len(df)} 首诗，{df['花名'].nunique()} 种花\n")

    # ── 1. 对每首诗提取物象 ────────────────────────────────────────────────────
    records = []
    global_counter: Counter = Counter()
    flower_imagery: dict[str, Counter] = defaultdict(Counter)   # 花名 → 物象计数

    for _, row in df.iterrows():
        poem_id   = row["ID"]
        flower    = str(row["花名"]).strip()
        title     = str(row["诗名"]).strip()
        dynasty   = str(row["朝代"]).strip()
        author    = str(row["作者"]).strip()
        poem_text = str(row["正文"]) if pd.notna(row["正文"]) else ""

        imagery_list = extract_imagery(poem_text)
        # 去重（同一首诗中同一物象只计1次共现）
        imagery_set  = list(dict.fromkeys(imagery_list))

        global_counter.update(imagery_set)
        flower_imagery[flower].update(imagery_set)

        records.append({
            "ID":      poem_id,
            "花名":    flower,
            "诗名":    title,
            "朝代":    dynasty,
            "作者":    author,
            "物象数量": len(imagery_set),
            "物象列表": "｜".join(imagery_set),
            "物象分类": "｜".join(WORD_TO_CAT.get(w, "其他") for w in imagery_set),
        })

    df_poems = pd.DataFrame(records)
    df_poems.to_csv(OUT_DIR / "imagery_per_poem.csv", index=False, encoding="utf-8-sig")
    print(f"[1] imagery_per_poem.csv  ({len(df_poems)} 行，"
          f"平均每首 {df_poems['物象数量'].mean():.1f} 个物象)")

    # ── 2. 全局物象词频 ────────────────────────────────────────────────────────
    df_freq = pd.DataFrame(
        [(w, n, WORD_TO_CAT.get(w, "其他")) for w, n in global_counter.most_common()],
        columns=["物象", "出现诗篇数", "类别"],
    )
    df_freq.to_csv(OUT_DIR / "imagery_frequency.csv", index=False, encoding="utf-8-sig")
    print(f"[2] imagery_frequency.csv ({len(df_freq)} 种物象)")

    # ── 3. 花名 × 物象 共现矩阵 ──────────────────────────────────────────────
    # 取出现次数 ≥ 2 的物象，避免矩阵过大
    top_imagery = [w for w, _ in global_counter.most_common() if global_counter[w] >= 2]
    print(f"    共现矩阵物象列数（频次≥2）：{len(top_imagery)}")

    flowers = sorted(flower_imagery.keys())
    matrix_rows = []
    for f in flowers:
        row_data = {"花名": f}
        for w in top_imagery:
            row_data[w] = flower_imagery[f].get(w, 0)
        matrix_rows.append(row_data)

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(OUT_DIR / "flower_imagery_cooccur.csv", index=False, encoding="utf-8-sig")
    print(f"[3] flower_imagery_cooccur.csv ({len(flowers)} 种花 × {len(top_imagery)} 种物象)")

    # ── 4. 每种花 Top-20 共现物象 ──────────────────────────────────────────────
    top_rows = []
    for f in flowers:
        cnt = flower_imagery[f]
        for rank, (w, n) in enumerate(cnt.most_common(20), 1):
            top_rows.append({
                "花名": f,
                "排名": rank,
                "物象": w,
                "共现诗篇数": n,
                "类别": WORD_TO_CAT.get(w, "其他"),
            })

    df_top = pd.DataFrame(top_rows)
    df_top.to_csv(OUT_DIR / "flower_top_imagery.csv", index=False, encoding="utf-8-sig")
    print(f"[4] flower_top_imagery.csv ({len(df_top)} 行)")

    # ── 5. 文字摘要 ────────────────────────────────────────────────────────────
    _write_summary(df_poems, df_freq, df_top, flowers, flower_imagery)

    print("\n✓ 全部完成，输出目录：", OUT_DIR)


def _write_summary(df_poems, df_freq, df_top, flowers, flower_imagery):
    lines = []
    lines.append("=" * 60)
    lines.append("  step2c 物象提取摘要报告（词典匹配 路径B）")
    lines.append("=" * 60)

    lines.append(f"\n【数据规模】")
    lines.append(f"  诗篇总数       : {len(df_poems)}")
    lines.append(f"  花名种类       : {df_poems['花名'].nunique()}")
    lines.append(f"  物象词典总词数  : {len(WORD_TO_CAT)}")
    lines.append(f"  提取物象种类数  : {len(df_freq)}")
    lines.append(f"  平均每首物象数  : {df_poems['物象数量'].mean():.2f}")
    lines.append(f"  最多物象（首诗）: {df_poems['物象数量'].max()}")
    lines.append(f"  最少物象（首诗）: {df_poems['物象数量'].min()}")

    lines.append(f"\n【全局 Top-30 高频物象】")
    for i, row in df_freq.head(30).iterrows():
        lines.append(f"  {i+1:>3}. {row['物象']:<6}  {row['出现诗篇数']:>4} 首  [{row['类别']}]")

    lines.append(f"\n【各类别物象数量】")
    cat_counts = df_freq["类别"].value_counts()
    for cat, cnt in cat_counts.items():
        lines.append(f"  {cat:<10}: {cnt} 种")

    lines.append(f"\n【每种花 Top-5 共现物象】")
    for flower in flowers:
        cnt = flower_imagery[flower]
        top5 = "、".join(f"{w}({n})" for w, n in cnt.most_common(5))
        poem_count = sum(1 for _, row in df_top[df_top["花名"] == flower].iterrows()
                         if row["排名"] == 1) + \
                     len([r for r in df_top[df_top["花名"] == flower]["共现诗篇数"].tolist()])
        lines.append(f"  {flower:<8}: {top5}")

    report = "\n".join(lines)
    out_path = OUT_DIR / "summary_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"[5] summary_report.txt")

    # 打印摘要片段
    print()
    for line in lines[:40]:
        print(line)


if __name__ == "__main__":
    main()
