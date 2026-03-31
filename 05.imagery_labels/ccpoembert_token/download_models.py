"""
下载 BERT-CCPoem 和 SikuRoBERTa 模型到本地 models/ 目录
─────────────────────────────────────────────────────────────────────────────
模型选型说明：
  正文（古诗词原文）→ synpjh/BERT_CCPoem_v1-finetuned-poem
      THUNLP-AIPoet BERT-CCPoem 的 HuggingFace 版本
      备选：ethanyt/guwenbert-base（GuwenBERT，768维，古文大规模预训练）
  赏析（现代汉语鉴赏）→ SIKU-BERT/sikuroberta
      南农大四库全书 RoBERTa，支持古文/半文言分析

运行方式：
    cd /Users/rzhang/Documents/poem-flower-recommend
    source flower_env/bin/activate
    python 02.sample_label_phase2/step2b_bert/download_models.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
from pathlib import Path

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 将 HuggingFace 缓存重定向到项目内，避免系统级权限问题
HF_CACHE = MODELS_DIR / ".hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"]              = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"]   = str(HF_CACHE / "hub")

print(f"📁 模型保存目录 : {MODELS_DIR}")
print(f"📁 HF 缓存目录  : {HF_CACHE}")


def remove_stale_locks(cache_dir: Path):
    """清理可能残留的 .lock 文件"""
    if cache_dir.exists():
        for lock in cache_dir.rglob("*.lock"):
            try:
                lock.unlink()
            except Exception:
                pass


def download_model(model_id: str, save_dir: Path, desc: str) -> bool:
    """从 HuggingFace 下载 tokenizer + model，保存到 save_dir"""
    from transformers import AutoTokenizer, AutoModel

    # 已存在且非空则跳过
    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"  [SKIP] {desc} 已存在：{save_dir}")
        return True

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  ▶ {desc}")
    print(f"    HuggingFace ID : {model_id}")
    print(f"    本地保存路径   : {save_dir}")

    try:
        print("    正在下载 tokenizer ...")
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(HF_CACHE / "hub"),
        )
        tok.save_pretrained(str(save_dir))
        print("    tokenizer 已保存")

        print("    正在下载模型权重（请耐心等待）...")
        mdl = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(HF_CACHE / "hub"),
        )
        mdl.save_pretrained(str(save_dir))

        # 记录隐层维度，供 bert_embed.py 参考
        hidden = getattr(mdl.config, "hidden_size", "unknown")
        print(f"    ✅ {desc} 下载完成  (hidden_size={hidden})")
        return True

    except Exception as e:
        print(f"    ❌ 下载失败：{e}")
        if save_dir.exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        return False


def main():
    remove_stale_locks(HF_CACHE)

    success_all = True

    # ── 模型 1：正文 → BERT-CCPoem 系列 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("模型 1/2：正文用 BERT-CCPoem（古诗词预训练）")
    print("=" * 60)

    bert_ccpoem_dir = MODELS_DIR / "bert_ccpoem"
    ok = download_model(
        model_id="synpjh/BERT_CCPoem_v1-finetuned-poem",
        save_dir=bert_ccpoem_dir,
        desc="BERT-CCPoem（synpjh/BERT_CCPoem_v1-finetuned-poem）",
    )

    if not ok:
        print("\n  [备选] 尝试 GuwenBERT（ethanyt/guwenbert-base）...")
        ok = download_model(
            model_id="ethanyt/guwenbert-base",
            save_dir=bert_ccpoem_dir,
            desc="GuwenBERT-base（古文大规模预训练，768维）",
        )

    if not ok:
        print("\n  [最终备选] 使用 hfl/chinese-roberta-wwm-ext ...")
        ok = download_model(
            model_id="hfl/chinese-roberta-wwm-ext",
            save_dir=bert_ccpoem_dir,
            desc="Chinese-RoBERTa-wwm-ext（通用中文BERT）",
        )

    success_all = success_all and ok

    # ── 模型 2：赏析 → SikuRoBERTa ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("模型 2/2：赏析用 SikuRoBERTa（四库全书预训练）")
    print("=" * 60)

    sikuroberta_dir = MODELS_DIR / "sikuroberta"
    ok = download_model(
        model_id="SIKU-BERT/sikuroberta",
        save_dir=sikuroberta_dir,
        desc="SikuRoBERTa（SIKU-BERT/sikuroberta）",
    )

    if not ok:
        print("\n  [备选] 尝试 SIKU-BERT/sikubert ...")
        ok = download_model(
            model_id="SIKU-BERT/sikubert",
            save_dir=sikuroberta_dir,
            desc="SikuBERT（SIKU-BERT/sikubert）",
        )

    success_all = success_all and ok

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if success_all:
        print("✅ 所有模型下载完成！")
        for name, d in [("BERT-CCPoem (正文)", bert_ccpoem_dir),
                        ("SikuRoBERTa (赏析)", sikuroberta_dir)]:
            files = list(d.iterdir()) if d.exists() else []
            total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
            print(f"  {name:<22} → {d.name}/  ({total_mb:.0f} MB)")
        print("\n▶ 下一步：运行 bert_embed.py 进行特征提取")
    else:
        print("⚠️  部分模型下载失败，请检查网络连接后重试")
        sys.exit(1)


if __name__ == "__main__":
    main()
