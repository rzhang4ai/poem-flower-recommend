"""
step1_preprocess/preprocess.py
================================
文本预处理：jieba分词 + 词性标注 + 停用词过滤
输出三个版本：正文、赏析、正文+赏析

保留词性：
  n  (名词)    → 花名、人物、意象
  a  (形容词)  → 情感、象征描述
  v  (动词)    → 场合动作（赠/别/思）
  d  (副词)    → 情感强度修饰（选择性保留）

用法：
    python3 preprocess.py
    python3 preprocess.py --input ../step0_sample/output/sample_200.csv
"""

import argparse
import os
import re
import json
import pandas as pd
import jieba
import jieba.posseg as pseg

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 保留的词性 ────────────────────────────────────────────────────────────────
# jieba词性标注说明：
#   n=名词, nr=人名, ns=地名, nz=其他专名
#   a=形容词, ad=副形词, an=名形词
#   v=动词, vd=副动词, vn=名动词
#   d=副词
KEEP_POS = {
    'n', 'nr', 'ns', 'nz',          # 名词类
    'a', 'ad', 'an',                  # 形容词类
    'v', 'vd', 'vn',                  # 动词类
    'd',                              # 副词（保留用于强度修饰）
}

# ── 自定义停用词（在通用停用词基础上增加） ────────────────────────────────────
CUSTOM_STOPWORDS = {
    # 通用虚词
    '的','了','在','是','我','他','她','它','们','这','那','有','也','都',
    '与','和','及','或','但','而','却','就','不','非','无','未','已',
    '很','更','最','较','甚','极','颇','略','稍',
    # 文学评论套话（对标注无意义）
    '诗人','作者','全诗','一首','这首','此诗','本诗','诗词','诗句','词句',
    '通过','表达','表现','描写','描绘','刻画','体现','反映','展现','呈现',
    '可以','可谓','可见','所以','因此','然而','不仅','不但','同时','其中',
    '之中','之上','之下','之间','一种','一个','一片','一番',
    # 时间虚词
    '时','时候','之时','之际','之间','当时','此时','彼时',
    # 程度虚词
    '十分','非常','格外','尤为','尤其','特别',
}

# ── 花卉相关专有词汇（加入jieba词典，防止被切分） ────────────────────────────
FLOWER_TERMS = [
    '梅花','红梅','蜡梅','迎春花','樱桃花','玉兰','月季','杏花','李花','桃花',
    '芍药','牡丹','荷花','莲花','菊花','桂花','水仙','兰花','蔷薇','玫瑰',
    '木芙蓉','茉莉','凤仙花','木槿','紫薇','牵牛花','虞美人','榴花','萱草',
    '棣棠','凌霄花','杜鹃','山茶花','栀子花','秋葵','芦花','向日葵',
    # 情感象征词
    '高洁','坚韧','清雅','思念','离别','豪迈','忧愁','喜悦','哀愁',
    '凌寒','傲霜','淡泊','隐逸','富贵','吉祥','相思','怀人',
    # 赠送场合词
    '送别','饯行','祝寿','生辰','贺喜','悼念','思归','相赠','寄情',
]

def load_stopwords() -> set:
    """加载停用词：自定义 + 可选外部文件"""
    stopwords = set(CUSTOM_STOPWORDS)
    # 尝试加载外部停用词文件（如有）
    ext_path = os.path.join(OUTPUT_DIR, "stopwords_custom.txt")
    if os.path.exists(ext_path):
        with open(ext_path, encoding='utf-8') as f:
            stopwords.update(line.strip() for line in f if line.strip())
    return stopwords

def init_jieba():
    """初始化jieba：关闭默认输出，添加花卉专词"""
    jieba.setLogLevel('ERROR')
    for term in FLOWER_TERMS:
        jieba.add_word(term, freq=1000)

def clean_text(text: str) -> str:
    """清洗文本：去除标点、数字、英文、多余空白"""
    if not isinstance(text, str):
        return ''
    # 保留中文字符和句号（用于TextRank分句）
    text = re.sub(r'[^\u4e00-\u9fff，。！？、；：""''（）【】]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_with_pos(text: str, stopwords: set) -> list[dict]:
    """
    jieba词性标注，过滤停用词和低信息量词性
    返回：[{'word': '高洁', 'pos': 'a'}, ...]
    """
    if not text.strip():
        return []
    tokens = []
    for word, flag in pseg.cut(text):
        word = word.strip()
        if len(word) < 2:          # 过滤单字（古文单字噪音多）
            continue
        if word in stopwords:
            continue
        # 取词性前缀匹配
        pos_prefix = flag[0] if flag else ''
        if pos_prefix in KEEP_POS or flag in KEEP_POS:
            tokens.append({'word': word, 'pos': flag})
    return tokens

def process_row(row: pd.Series, stopwords: set) -> dict:
    """对单行数据生成三个版本的分词结果"""
    poem_raw     = clean_text(str(row.get('正文', '') or ''))
    analysis_raw = clean_text(str(row.get('赏析', '') or ''))
    combined_raw = poem_raw + '。' + analysis_raw if poem_raw and analysis_raw else poem_raw or analysis_raw

    poem_tokens     = tokenize_with_pos(poem_raw, stopwords)
    analysis_tokens = tokenize_with_pos(analysis_raw, stopwords)
    combined_tokens = tokenize_with_pos(combined_raw, stopwords)

    return {
        'ID':              row.get('ID', ''),
        'sample_id':       row.get('sample_id', ''),
        '花名':            row.get('花名', ''),
        '月份':            row.get('月份', ''),
        '朝代':            row.get('朝代', ''),
        '作者':            row.get('作者', ''),
        '诗名':            row.get('诗名', ''),
        # 原始清洗文本（供后续模块直接使用）
        'text_poem':       poem_raw,
        'text_analysis':   analysis_raw,
        'text_combined':   combined_raw,
        # 分词结果（JSON字符串）
        'tokens_poem':     json.dumps([t['word'] for t in poem_tokens],     ensure_ascii=False),
        'tokens_analysis': json.dumps([t['word'] for t in analysis_tokens], ensure_ascii=False),
        'tokens_combined': json.dumps([t['word'] for t in combined_tokens], ensure_ascii=False),
        # 词性标注（JSON字符串）
        'pos_poem':        json.dumps(poem_tokens,     ensure_ascii=False),
        'pos_analysis':    json.dumps(analysis_tokens, ensure_ascii=False),
        'pos_combined':    json.dumps(combined_tokens, ensure_ascii=False),
        # 统计
        'token_count_poem':     len(poem_tokens),
        'token_count_analysis': len(analysis_tokens),
        'token_count_combined': len(combined_tokens),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../step0_sample/output/sample_200.csv')
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), args.input)
    if not os.path.exists(csv_path):
        csv_path = args.input
    if not os.path.exists(csv_path):
        print(f"❌ 找不到输入文件: {csv_path}")
        return

    print(f"✅ 读取: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"   {len(df)} 条数据")

    print("🔧 初始化 jieba...")
    init_jieba()
    stopwords = load_stopwords()
    print(f"   停用词: {len(stopwords)} 个")

    print("⚙️  分词处理中...")
    results = []
    for _, row in df.iterrows():
        results.append(process_row(row, stopwords))

    result_df = pd.DataFrame(results)

    # ── 输出1：完整预处理结果 ──────────────────────────────────────────────────
    full_path = os.path.join(OUTPUT_DIR, "preprocessed.csv")
    result_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整预处理结果: {full_path}")

    # ── 输出2：三个版本的纯token列表（供TF-IDF等模块直接读取） ────────────────
    for version in ['poem', 'analysis', 'combined']:
        cols = ['ID', 'sample_id', '花名', '月份', '朝代', '作者', '诗名',
                f'text_{version}', f'tokens_{version}', f'pos_{version}',
                f'token_count_{version}']
        ver_df = result_df[cols].copy()
        ver_df.columns = ['ID', 'sample_id', '花名', '月份', '朝代', '作者', '诗名',
                          'text', 'tokens', 'pos_tags', 'token_count']
        out_path = os.path.join(OUTPUT_DIR, f"tokens_{version}.csv")
        ver_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"💾 tokens_{version}.csv  (平均{ver_df['token_count'].mean():.1f}词/条)")

    # ── 输出3：停用词文件（供人工检查和补充） ────────────────────────────────
    sw_path = os.path.join(OUTPUT_DIR, "stopwords_custom.txt")
    with open(sw_path, 'w', encoding='utf-8') as f:
        f.write("# 自定义停用词\n# 每行一个词，#开头为注释\n\n")
        for w in sorted(CUSTOM_STOPWORDS):
            f.write(w + '\n')
    print(f"💾 停用词文件: {sw_path}")

    # ── 统计报告 ──────────────────────────────────────────────────────────────
    print("\n── 分词统计 ─────────────────────────────────")
    for version in ['poem', 'analysis', 'combined']:
        col = f'token_count_{version}'
        print(f"  {version:10s}: 平均 {result_df[col].mean():.1f} / "
              f"最少 {result_df[col].min()} / 最多 {result_df[col].max()} tokens")
    zero_analysis = (result_df['token_count_analysis'] == 0).sum()
    if zero_analysis > 0:
        print(f"\n⚠️  赏析为空的条目: {zero_analysis} 条（这些条目仅能依赖正文）")

    print("\n✅ Step 1 完成")

if __name__ == '__main__':
    main()
