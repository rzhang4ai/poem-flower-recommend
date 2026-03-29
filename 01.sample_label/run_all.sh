#!/bin/bash
# run_all.sh
# ===========
# 按顺序运行 Step 0-6，在项目根目录执行：
#   bash 01.sample_label/run_all.sh
#
# 如果某步失败，脚本会停止并提示。
# 也可以单独运行某一步：
#   python3 01.sample_label/step3_unsupervised/cluster_lda.py --n_topics 7

set -e  # 任意步骤失败即停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "  诗花雅送 · 传统AI标注流水线"
echo "  01.sample_label  全流程运行"
echo "========================================================"
echo ""

run_step() {
    local step_name=$1
    local script=$2
    echo "────────────────────────────────────────────────────"
    echo "▶ $step_name"
    echo "────────────────────────────────────────────────────"
    python3 "$script"
    echo ""
}

run_step "Step 0: 分层抽样 200条"         "step0_sample/sample_200.py"
run_step "Step 1: 文本预处理（分词+词性）" "step1_preprocess/preprocess.py"
run_step "Step 2: 特征提取（TF-IDF+TextRank+PMI）" "step2_features/extract_features.py"
run_step "Step 3: 无监督聚类（LDA+k-Means+层次）"  "step3_unsupervised/cluster_lda.py"
run_step "Step 4: 情感基线（SnowNLP+NTUSD）"       "step4_sentiment/sentiment.py"
run_step "Step 5: 规则初标注"              "step5_rules/rule_labeler.py"
run_step "Step 6: 汇总报告 & 标注底稿"    "step6_report/generate_report.py"

echo "========================================================"
echo "✅ 全流程完成！"
echo ""
echo "关键输出文件："
echo "  step6_report/output/annotation_draft.csv   ← 标注底稿"
echo "  step6_report/output/low_confidence.csv      ← 需重点审核"
echo "  step6_report/output/high_confidence.csv     ← 高置信度可直接采用"
echo "  step6_report/output/figures/                ← 可视化图表"
echo "========================================================"
