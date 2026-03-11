"""
step5_rules/rule_labeler.py
==============================
规则初标注：场合规则库 + 关系句式规则

基于赏析文本中的触发词和句式模式，为每首诗生成高置信度标签。
预期覆盖率：30-40%（剩余由LLM + 人工处理）

标注维度：
  occasion    赠送场合（12类）
  relation    赠送关系（6类）
  symbolism   花卉象征（从PMI结果提取）

输出：
  rule_labels.csv           → 每首诗的规则标注结果
  rule_coverage_report.txt  → 规则覆盖率和效果报告

用法：
    python3 rule_labeler.py
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 场合规则库（20条核心规则）
# ══════════════════════════════════════════════════════════════════════════════
#
# 格式：{
#   'label':      场合标签（英文key，与标注体系一致）
#   'label_cn':   场合标签（中文，用于展示）
#   'triggers':   触发词列表（任意一个命中即触发）
#   'patterns':   正则句式（更精确的匹配，可选）
#   'confidence': 规则置信度（0-1）
#   'weight':     规则权重（多规则命中时累加）
# }
#
OCCASION_RULES = [
    # ── 送别 / 离别 ───────────────────────────────────────────────────────────
    {
        'label': 'farewell', 'label_cn': '送别',
        'triggers': ['送别', '饯行', '折柳', '话别', '惜别', '别离', '临别',
                     '赠别', '离别', '分别', '践行', '送行', '依依惜别'],
        'patterns': [r'送\w{1,4}赴', r'赠别\w', r'临行\w{0,4}赠', r'折柳\w{0,4}送'],
        'confidence': 0.90, 'weight': 2.0,
    },
    # ── 生日 / 寿辰 ───────────────────────────────────────────────────────────
    {
        'label': 'birthday', 'label_cn': '生日祝寿',
        'triggers': ['生辰', '寿辰', '祝寿', '贺寿', '寿宴', '生日', '华诞',
                     '寿星', '松鹤延年', '福寿'],
        'patterns': [r'贺\w{0,2}寿', r'\w{0,2}岁寿', r'寿\w{0,2}辰'],
        'confidence': 0.88, 'weight': 2.0,
    },
    # ── 升职 / 入仕 ───────────────────────────────────────────────────────────
    {
        'label': 'promotion', 'label_cn': '升职祝贺',
        'triggers': ['升迁', '荣升', '擢升', '晋升', '入仕', '及第', '高中',
                     '金榜题名', '科举', '进士', '登科', '状元', '仕途'],
        'patterns': [r'贺\w{0,4}升', r'恭喜\w{0,4}擢', r'登科\w{0,4}贺'],
        'confidence': 0.85, 'weight': 1.8,
    },
    # ── 悼念 / 哀思 ───────────────────────────────────────────────────────────
    {
        'label': 'memorial', 'label_cn': '悼念',
        'triggers': ['悼念', '哀悼', '祭奠', '祭扫', '清明', '悼亡', '哀思',
                     '追思', '亡故', '逝世', '长眠', '已故', '悼词'],
        'patterns': [r'祭\w{0,4}墓', r'悼\w{0,4}亡', r'清明\w{0,4}扫'],
        'confidence': 0.92, 'weight': 2.0,
    },
    # ── 思乡 / 怀人 ───────────────────────────────────────────────────────────
    {
        'label': 'homesick', 'label_cn': '思乡怀人',
        'triggers': ['思乡', '望乡', '故乡', '游子', '客居', '羁旅', '漂泊',
                     '异乡', '故园', '乡愁', '归乡', '还乡', '思归'],
        'patterns': [r'客居\w{0,4}思', r'独在异乡', r'举头望\w{0,2}月'],
        'confidence': 0.85, 'weight': 1.8,
    },
    # ── 庆贺 / 喜庆 ───────────────────────────────────────────────────────────
    {
        'label': 'celebration', 'label_cn': '喜庆祝贺',
        'triggers': ['庆贺', '庆祝', '喜庆', '祝贺', '道贺', '恭贺', '祥瑞',
                     '吉祥', '大喜', '普天同庆', '欢庆'],
        'patterns': [r'恭\w{0,2}贺', r'祝\w{1,4}大吉'],
        'confidence': 0.82, 'weight': 1.5,
    },
    # ── 婚嫁 ──────────────────────────────────────────────────────────────────
    {
        'label': 'wedding', 'label_cn': '婚嫁',
        'triggers': ['婚嫁', '新婚', '成婚', '完婚', '嫁娶', '洞房', '花烛',
                     '鸳鸯', '百年好合', '喜结良缘', '嫁', '娶'],
        'patterns': [r'新婚\w{0,4}贺', r'成婚\w{0,4}赠'],
        'confidence': 0.88, 'weight': 2.0,
    },
    # ── 赴任 / 出行 ───────────────────────────────────────────────────────────
    {
        'label': 'departure', 'label_cn': '赴任出行',
        'triggers': ['赴任', '赴京', '赴考', '赴职', '出使', '出征', '远行',
                     '启程', '上任', '赴边', '从军', '出塞'],
        'patterns': [r'赴\w{1,4}任', r'出\w{0,2}征\w{0,4}赠', r'从军\w{0,4}别'],
        'confidence': 0.87, 'weight': 1.8,
    },
    # ── 归隐 / 隐居 ───────────────────────────────────────────────────────────
    {
        'label': 'retirement', 'label_cn': '归隐',
        'triggers': ['归隐', '隐居', '归田', '致仕', '辞官', '归园', '退隐',
                     '林泉', '山居', '隐逸', '渔隐'],
        'patterns': [r'归\w{0,2}隐\w{0,4}赠', r'辞官\w{0,4}别'],
        'confidence': 0.85, 'weight': 1.8,
    },
    # ── 病愈 / 探望 ───────────────────────────────────────────────────────────
    {
        'label': 'getwell', 'label_cn': '祝愿康复',
        'triggers': ['病愈', '康复', '痊愈', '探病', '慰问', '病中', '养病',
                     '病体', '卧病', '沉疴'],
        'patterns': [r'探\w{0,2}病\w{0,4}赠', r'病\w{0,4}康\w{0,4}贺'],
        'confidence': 0.83, 'weight': 1.8,
    },
    # ── 新居 / 乔迁 ───────────────────────────────────────────────────────────
    {
        'label': 'housewarming', 'label_cn': '乔迁新居',
        'triggers': ['乔迁', '新居', '迁居', '新宅', '新家', '落成'],
        'patterns': [r'乔迁\w{0,4}贺', r'新居\w{0,4}赠'],
        'confidence': 0.88, 'weight': 2.0,
    },
    # ── 自赏 / 托物言志 ───────────────────────────────────────────────────────
    {
        'label': 'self_expression', 'label_cn': '托物言志',
        'triggers': ['托物言志', '自比', '自喻', '借花抒怀', '以花明志',
                     '自况', '自励', '明志', '言志', '抒怀', '寄托'],
        'patterns': [r'以\w{1,4}自比', r'借\w{1,4}言志', r'托\w{1,4}抒怀'],
        'confidence': 0.78, 'weight': 1.5,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 关系规则库
# ══════════════════════════════════════════════════════════════════════════════

RELATION_RULES = [
    {
        'label': 'friend', 'label_cn': '友人',
        'triggers': ['友人', '挚友', '好友', '朋友', '旧友', '故友', '知己',
                     '同窗', '同学', '同僚', '同年', '莫逆', '至交'],
        'patterns': [r'赠\w{1,4}友', r'与\w{1,4}友', r'故\w{0,2}友'],
        'confidence': 0.85, 'weight': 1.5,
    },
    {
        'label': 'lover', 'label_cn': '爱人',
        'triggers': ['爱人', '情人', '相思', '伊人', '佳人', '红颜', '倾心',
                     '心上人', '意中人', '鸳鸯', '比翼', '连理'],
        'patterns': [r'寄\w{1,4}伊', r'思\w{1,4}人', r'赠\w{0,2}佳人'],
        'confidence': 0.82, 'weight': 1.5,
    },
    {
        'label': 'teacher', 'label_cn': '师长',
        'triggers': ['恩师', '师长', '先生', '夫子', '师父', '老师', '座师',
                     '业师', '授业', '门生', '弟子', '学生'],
        'patterns': [r'赠\w{0,2}师', r'谢\w{0,2}师', r'奉\w{1,4}先生'],
        'confidence': 0.87, 'weight': 1.8,
    },
    {
        'label': 'family', 'label_cn': '家人',
        'triggers': ['父母', '兄弟', '姐妹', '子女', '妻子', '夫君', '家人',
                     '骨肉', '亲人', '手足', '父亲', '母亲', '妻', '夫'],
        'patterns': [r'寄\w{0,2}妻', r'赠\w{0,2}弟', r'致\w{0,2}母'],
        'confidence': 0.85, 'weight': 1.8,
    },
    {
        'label': 'colleague', 'label_cn': '同僚',
        'triggers': ['同僚', '同事', '同官', '幕僚', '属下', '僚属',
                     '同朝', '共事'],
        'patterns': [r'赠\w{1,4}官', r'与\w{1,4}同\w{0,2}赋'],
        'confidence': 0.80, 'weight': 1.5,
    },
    {
        'label': 'self', 'label_cn': '自赠/自述',
        'triggers': ['自赠', '自题', '自序', '自述', '自吟', '自咏', '自励',
                     '自勉', '自况', '自比'],
        'patterns': [r'题\w{0,4}自\w{0,2}居', r'自\w{0,4}咏'],
        'confidence': 0.88, 'weight': 2.0,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 花卉象征规则（基于文化常识，PMI会进一步验证）
# ══════════════════════════════════════════════════════════════════════════════

SYMBOLISM_RULES = {
    '梅花':  ['坚韧', '高洁', '傲寒', '不屈'],
    '蜡梅':  ['坚韧', '高洁', '傲寒'],
    '荷花':  ['高洁', '纯洁', '出淤泥而不染', '清廉'],
    '莲花':  ['高洁', '纯洁', '佛教象征', '清廉'],
    '菊花':  ['隐逸', '清雅', '傲霜', '高洁'],
    '牡丹':  ['富贵', '繁荣', '国色', '雍容'],
    '桃花':  ['爱情', '美人', '春天', '桃源'],
    '兰花':  ['高洁', '清雅', '君子', '淡泊'],
    '水仙':  ['高洁', '清雅', '凌波', '思念'],
    '桂花':  ['富贵', '吉祥', '月亮', '科举'],
    '芍药':  ['离别', '爱情', '送别', '芳华'],
    '蔷薇':  ['爱情', '美丽', '春天'],
    '木芙蓉':['高洁', '美丽', '秋天'],
    '杏花':  ['春天', '喜悦', '美好'],
    '迎春花':['春天', '希望', '新生'],
}


# ══════════════════════════════════════════════════════════════════════════════
# 规则匹配引擎
# ══════════════════════════════════════════════════════════════════════════════

def match_rules(text: str, rules: list[dict]) -> list[dict]:
    """
    对文本应用规则列表，返回所有命中的规则及分数
    """
    if not text or not str(text).strip():
        return []

    matches = []
    for rule in rules:
        score = 0.0
        hit_triggers = []
        hit_patterns = []

        # 触发词匹配
        for trigger in rule.get('triggers', []):
            if trigger in text:
                score += rule['weight']
                hit_triggers.append(trigger)

        # 正则匹配（权重×1.2，更精确）
        for pattern in rule.get('patterns', []):
            if re.search(pattern, text):
                score += rule['weight'] * 1.2
                hit_patterns.append(pattern)

        if score > 0:
            matches.append({
                'label':        rule['label'],
                'label_cn':     rule['label_cn'],
                'score':        round(score, 3),
                'confidence':   rule['confidence'],
                'hit_triggers': hit_triggers,
                'hit_patterns': hit_patterns,
            })

    # 按分数降序
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches


def label_row(row: pd.Series) -> dict:
    """对单行数据应用所有规则，返回标注结果"""
    # 优先用赏析，再补充诗名信息
    analysis = str(row.get('赏析', '') or '')
    poem     = str(row.get('正文', '') or '')
    title    = str(row.get('诗名', '') or '')
    flower   = str(row.get('花名', '') or '')

    # 赠送场合
    occ_matches = match_rules(analysis + title, OCCASION_RULES)
    # 赠送关系
    rel_matches = match_rules(analysis + title, RELATION_RULES)
    # 花卉象征（直接查字典）
    symbolism = SYMBOLISM_RULES.get(flower, [])

    # 主标签（最高分）
    occ_label    = occ_matches[0]['label']    if occ_matches else ''
    occ_label_cn = occ_matches[0]['label_cn'] if occ_matches else ''
    occ_conf     = occ_matches[0]['confidence'] * min(occ_matches[0]['score'] / 2.0, 1.0) \
                   if occ_matches else 0.0
    rel_label    = rel_matches[0]['label']    if rel_matches else ''
    rel_label_cn = rel_matches[0]['label_cn'] if rel_matches else ''
    rel_conf     = rel_matches[0]['confidence'] * min(rel_matches[0]['score'] / 1.5, 1.0) \
                   if rel_matches else 0.0

    # 综合置信度（场合+关系都有时更可信）
    overall_conf = (occ_conf + rel_conf) / 2 if (occ_conf > 0 and rel_conf > 0) \
                   else max(occ_conf, rel_conf)

    return {
        'ID':              row.get('ID', ''),
        'sample_id':       row.get('sample_id', ''),
        '花名':            flower,
        '月份':            row.get('月份', ''),
        '朝代':            row.get('朝代', ''),
        '作者':            row.get('作者', ''),
        '诗名':            row.get('诗名', ''),
        # 场合标注
        'occasion':        occ_label,
        'occasion_cn':     occ_label_cn,
        'occasion_conf':   round(occ_conf, 3),
        'occasion_all':    json.dumps([m['label'] for m in occ_matches], ensure_ascii=False),
        'occasion_hits':   json.dumps([m['hit_triggers'][:3] for m in occ_matches[:2]],
                                       ensure_ascii=False),
        # 关系标注
        'relation':        rel_label,
        'relation_cn':     rel_label_cn,
        'relation_conf':   round(rel_conf, 3),
        'relation_all':    json.dumps([m['label'] for m in rel_matches], ensure_ascii=False),
        # 象征标注
        'symbolism':       json.dumps(symbolism, ensure_ascii=False),
        'symbolism_preview': '、'.join(symbolism[:3]),
        # 综合
        'overall_conf':    round(overall_conf, 3),
        'is_high_conf':    overall_conf >= 0.65,      # 高置信度标签
        'needs_review':    overall_conf < 0.4,        # 需要重点人工审核
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_csv', default='../step0_sample/output/sample_200.csv')
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), args.sample_csv)
    if not os.path.exists(csv_path):
        csv_path = args.sample_csv
    if not os.path.exists(csv_path):
        print(f"❌ 找不到: {csv_path}")
        return

    print("=" * 55)
    print("Step 5: 规则初标注")
    print("=" * 55)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"✅ 读取 {len(df)} 条")

    results = [label_row(row) for _, row in df.iterrows()]
    result_df = pd.DataFrame(results)

    # 输出标注结果
    out_path = os.path.join(OUTPUT_DIR, "rule_labels.csv")
    result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 rule_labels.csv ({len(result_df)} 条)")

    # ── 统计报告 ──────────────────────────────────────────────────────────────
    total = len(result_df)
    occ_covered   = (result_df['occasion'] != '').sum()
    rel_covered   = (result_df['relation'] != '').sum()
    high_conf     = result_df['is_high_conf'].sum()
    needs_review  = result_df['needs_review'].sum()

    report_lines = [
        "=" * 50,
        "Step 5 规则初标注报告",
        "=" * 50,
        f"总条数:          {total}",
        f"场合标注覆盖:    {occ_covered} / {total} = {occ_covered/total:.1%}",
        f"关系标注覆盖:    {rel_covered} / {total} = {rel_covered/total:.1%}",
        f"高置信度(≥0.65): {high_conf} / {total} = {high_conf/total:.1%}",
        f"需重点审核(<0.4):{needs_review} / {total} = {needs_review/total:.1%}",
        "",
        "── 场合分布 ─────────────────────────────",
    ]
    for label, cnt in result_df[result_df['occasion'] != '']['occasion_cn'].value_counts().items():
        report_lines.append(f"  {label:12s}: {cnt:3d} 条")

    report_lines += ["", "── 关系分布 ─────────────────────────────"]
    for label, cnt in result_df[result_df['relation'] != '']['relation_cn'].value_counts().items():
        report_lines.append(f"  {label:12s}: {cnt:3d} 条")

    report_lines += ["", "── 象征覆盖 ─────────────────────────────"]
    sym_covered = (result_df['symbolism_preview'] != '').sum()
    report_lines.append(f"  花卉象征已覆盖: {sym_covered} / {total} 条")

    report_text = '\n'.join(report_lines)
    print('\n' + report_text)

    report_path = os.path.join(OUTPUT_DIR, "rule_coverage_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text + '\n')
    print(f"\n💾 rule_coverage_report.txt")
    print("\n✅ Step 5 完成")


if __name__ == '__main__':
    main()
