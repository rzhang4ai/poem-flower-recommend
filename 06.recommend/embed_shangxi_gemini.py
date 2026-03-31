"""
离线一次性脚本：用 Gemini 嵌入模型（默认 gemini-embedding-2-preview）对 shangxi_clean 文本生成诗词嵌入。

task_type=RETRIEVAL_DOCUMENT（文档侧），与在线 RETRIEVAL_QUERY（查询侧）配对使用。

运行：
  source flower_env/bin/activate
  export GOOGLE_API_KEY="your-key"        # 或写入 05.recommend/.env
  python 05.recommend/embed_shangxi_gemini.py

输出（写入 05.recommend/output/）：
  poems_embed_ids.json    — 每行对应一首诗的 ID（str）
  poems_embeddings.npy    — shape=(N, D) float32，D=3072（默认，已归一化）

可选参数：
  --input    CSV 路径（默认 03.final_labels/poems_structured_shangxi_wip.csv）
  --dim      嵌入维度 768|1536|3072，默认 3072（3072 由 API 自动归一化）
  --batch    每批条数（默认 50，单次 API 限制 100）
  --resume   跳过已存在的输出文件中的诗（按 ID 续跑）
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT_DIR = HERE / "output"


def _poem_embed_text(row: pd.Series) -> str:
    """
    构建用于嵌入的文档文本。

    优先使用 shangxi_clean（经过清洗的现代汉语赏析摘要，语义桥梁）；
    若缺失则用 sxhy_raw_words + ccpoem_top10 拼成简短描述。
    """
    shangxi = str(row.get("shangxi_clean") or "").strip()
    if shangxi and shangxi not in ("nan", "None", ""):
        return shangxi

    # 降级：结构化字段拼接
    parts: list[str] = []
    flower = str(row.get("花名") or "").strip()
    if flower:
        parts.append(f"花卉：{flower}")
    emotion = str(row.get("l3_c3_zh") or "").strip()
    if emotion:
        parts.append(f"情感：{emotion}")
    imagery = str(row.get("confirmed_imagery") or row.get("sxhy_raw_words") or "").strip()
    if imagery:
        parts.append(f"意象：{imagery.replace('|', '、')[:80]}")
    dynasty = str(row.get("朝代") or "").strip()
    if dynasty:
        parts.append(f"朝代：{dynasty}")
    return "　".join(parts) or str(row.get("正文_preview") or "")[:120]


def _load_existing(output_dir: Path) -> tuple[list[str], list[list[float]]]:
    """加载已有的嵌入结果（续跑用）。"""
    ids_path = output_dir / "poems_embed_ids.json"
    emb_path = output_dir / "poems_embeddings.npy"
    if not ids_path.exists() or not emb_path.exists():
        return [], []
    try:
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        embs = np.load(str(emb_path)).tolist()
        logger.info("续跑：已加载 %d 条嵌入", len(ids))
        return ids, embs
    except Exception as e:
        logger.warning("加载已有嵌入失败 (%s)，重新开始", e)
        return [], []


def _save(output_dir: Path, ids: list[str], embs: list[list[float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "poems_embed_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False), encoding="utf-8"
    )
    np.save(str(output_dir / "poems_embeddings.npy"), np.array(embs, dtype=np.float32))
    logger.info("保存 %d 条嵌入 → %s", len(ids), output_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini 离线诗词嵌入")
    ap.add_argument(
        "--input", type=Path,
        default=ROOT / "03.final_labels/poems_structured_shangxi_wip.csv",
    )
    ap.add_argument("--dim",    type=int, default=3072,
                    help="嵌入维度 768|1536|3072，默认 3072")
    ap.add_argument("--batch",  type=int, default=50,
                    help="每批条数，默认 50")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已嵌入的诗 ID，续跑")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    import gemini_client

    if not gemini_client.is_available():
        raise EnvironmentError(
            "GOOGLE_API_KEY 未配置。"
            "请 export GOOGLE_API_KEY=xxx 或写入 05.recommend/.env"
        )

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件: {args.input}")

    df = pd.read_csv(args.input)
    logger.info("读取 CSV：%d 行", len(df))

    # 续跑：跳过已处理 ID
    existing_ids, existing_embs = [], []
    if args.resume:
        existing_ids, existing_embs = _load_existing(args.output)
    skip_ids = set(existing_ids)

    all_ids:  list[str]         = list(existing_ids)
    all_embs: list[list[float]] = list(existing_embs)

    # 按批次处理
    pending = [(str(row["ID"]), _poem_embed_text(row))
               for _, row in df.iterrows()
               if str(row["ID"]) not in skip_ids]

    total = len(pending)
    logger.info("待嵌入：%d 首诗（跳过 %d 首）", total, len(skip_ids))

    # 嵌入维度说明
    dim_arg = args.dim if args.dim != 3072 else None  # 3072 是默认值，不传则 API 自动
    if dim_arg and dim_arg != 3072:
        logger.info("使用 %d 维嵌入（非 3072 维时需自行 L2 归一化）", args.dim)

    checkpoint_every = args.batch * 4  # 每 4 批保存一次

    for batch_start in range(0, total, args.batch):
        batch = pending[batch_start : batch_start + args.batch]
        batch_ids   = [b[0] for b in batch]
        batch_texts = [b[1] for b in batch]

        attempt = 0
        while attempt < 3:
            try:
                vecs = gemini_client.embed(
                    batch_texts,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dim_arg,
                )
                break
            except Exception as e:
                attempt += 1
                wait = 2 ** attempt
                logger.warning("批次 %d 失败 (%s)，%d 秒后重试 (%d/3)",
                               batch_start, e, wait, attempt)
                time.sleep(wait)
        else:
            logger.error("批次 %d 连续失败，跳过本批", batch_start)
            continue

        # 非 3072 维需手动 L2 归一化
        if dim_arg and dim_arg != 3072:
            normed = []
            for v in vecs:
                arr = np.array(v, dtype=np.float32)
                n = np.linalg.norm(arr)
                normed.append((arr / n if n > 1e-12 else arr).tolist())
            vecs = normed

        all_ids.extend(batch_ids)
        all_embs.extend(vecs)

        done = batch_start + len(batch)
        logger.info("进度 %d / %d", done, total)

        # 定期保存
        if done % checkpoint_every == 0 or done == total:
            _save(args.output, all_ids, all_embs)

    # 最终保存
    _save(args.output, all_ids, all_embs)
    logger.info("完成：共 %d 首诗嵌入，维度 %d", len(all_ids), args.dim)


if __name__ == "__main__":
    main()
