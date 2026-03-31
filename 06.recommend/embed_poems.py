"""
方向 B（离线步骤）：为每首诗生成嵌入向量，保存到本地文件。

运行一次即可，之后推荐系统直接加载：
  cd /Users/rzhang/Documents/poem-flower-recommend
  source flower_env/bin/activate
  # 先在 05.recommend/.env 填好 ARK_API_KEY 和 ARK_EMBED_MODEL
  python3 05.recommend/embed_poems.py

输出文件（保存到 05.recommend/output/）：
  poems_embed_ids.json     → list[str]，与向量行顺序一一对应的 ID 列表
  poems_embeddings.npy     → shape (N, dim)，float32 numpy 数组

诗歌文本构造策略（让嵌入模型理解语义，而非只看标题）：
  {花名} / {l3_c3_zh} / {l1_polarity_zh}
  意象：{confirmed_imagery（竖线替换为空格）}
  {sxhy_raw_words 前 10 词}
  {ccpoem_top10 单字}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paths import DATA_CSV, OUTPUT_DIR


def poem_text(row: pd.Series) -> str:
    """将一行诗歌数据转换为嵌入用的文本表示。"""
    parts: list[str] = []

    flower = str(row.get("花名", "") or "")
    l3 = str(row.get("l3_c3_zh", "") or "")
    l1 = str(row.get("l1_polarity_zh", "") or "")
    if flower:
        parts.append(flower)
    if l3 and l3 != "nan":
        parts.append(l3)
    if l1 and l1 != "nan":
        parts.append(l1)

    ci = str(row.get("confirmed_imagery", "") or "")
    if ci and ci != "nan":
        words = " ".join(ci.split("|")[:8])
        parts.append(f"意象 {words}")

    raw = str(row.get("sxhy_raw_words", "") or "")
    if raw and raw != "nan":
        words = " ".join(raw.split("|")[:10])
        parts.append(words)

    top10 = str(row.get("ccpoem_top10", "") or "")
    if top10 and top10 != "nan":
        parts.append(top10[:40])

    # 加上朝代，帮助嵌入模型区分时代风格
    dynasty = str(row.get("朝代", "") or "")
    if dynasty and dynasty != "nan":
        parts.append(dynasty)

    return " ".join(p for p in parts if p)


def main() -> None:
    ap = argparse.ArgumentParser(description="离线生成诗歌嵌入向量")
    ap.add_argument("--input",  type=Path, default=DATA_CSV)
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--batch",  type=int,  default=20, help="每次 API 调用的文本数")
    ap.add_argument("--sleep",  type=float, default=0.5, help="批次间暂停秒数（限速）")
    args = ap.parse_args()

    import llm_client
    if not llm_client.is_available():
        print("错误：未找到 ARK_API_KEY，请先在 05.recommend/.env 中配置。")
        sys.exit(1)

    if not args.input.exists():
        print(f"错误：数据文件不存在：{args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"加载 {len(df)} 首诗")

    args.outdir.mkdir(parents=True, exist_ok=True)
    ids_path  = args.outdir / "poems_embed_ids.json"
    embs_path = args.outdir / "poems_embeddings.npy"

    # ── 构建文本列表 ─────────────────────────────────────────────────────
    texts: list[str] = []
    ids:   list[str] = []
    for _, row in df.iterrows():
        ids.append(str(row["ID"]))
        texts.append(poem_text(row))

    print(f"文本样例（前 2 条）：")
    for t in texts[:2]:
        print(f"  > {t}")

    # ── 分批调用 API ──────────────────────────────────────────────────────
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + args.batch - 1) // args.batch

    for i in range(0, len(texts), args.batch):
        batch = texts[i: i + args.batch]
        batch_no = i // args.batch + 1
        print(f"\r批次 {batch_no}/{total_batches} ({i+len(batch)}/{len(texts)})...", end="", flush=True)
        try:
            vecs = llm_client.embed(batch)
            all_embeddings.extend(vecs)
        except Exception as e:
            print(f"\n批次 {batch_no} 失败: {e}")
            # 补零向量，保持索引对齐
            dim = len(all_embeddings[0]) if all_embeddings else 1024
            for _ in batch:
                all_embeddings.append([0.0] * dim)
        if args.sleep > 0 and i + args.batch < len(texts):
            time.sleep(args.sleep)

    print(f"\n嵌入完成，维度 = {len(all_embeddings[0]) if all_embeddings else 0}")

    # ── 保存 ──────────────────────────────────────────────────────────────
    arr = np.array(all_embeddings, dtype=np.float32)
    np.save(embs_path, arr)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    print(f"已保存：{embs_path}  shape={arr.shape}")
    print(f"已保存：{ids_path}")


if __name__ == "__main__":
    main()
