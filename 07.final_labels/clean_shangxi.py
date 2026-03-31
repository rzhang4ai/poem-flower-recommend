"""
从原书「赏析」生成清洗摘要列 shangxi_clean，写入结构化 CSV。

流程：
  1. 规则过滤：按句删除明显噪音（生平、版本考证、互文比较、署名等）
  2. Gemini：在剩余文本上抽取 2～4 句、≤300 字，仅保留意境/情感/意象/象征；
     提示词强制：不能杜撰和新增，只能用原赏析中的内容。

依赖：
  pip install google-genai pandas
  （须使用 google-genai，以便设置 thinking_budget=0；旧版 google-generativeai 无法关闭「思考」易截断）

环境变量（二选一）：
  GOOGLE_API_KEY  或  GEMINI_API_KEY
  GEMINI_MODEL      可选，默认 gemini-3-flash-preview
  GEMINI_MAX_OUTPUT_TOKENS  可选，默认 8192（输出上限；思考已关闭后主要限制摘要长度）

用法：
  cd 项目根目录 && source flower_env/bin/activate
  export GOOGLE_API_KEY=...
  export GEMINI_MODEL=gemini-3-flash-preview
  python3 03.final_labels/clean_shangxi.py --dry-run --limit 3

  # 长跑建议写到单独文件，并每 20 条自动落盘；中断后可续跑：
  python3 03.final_labels/clean_shangxi.py \\
    --output 03.final_labels/poems_structured_shangxi_wip.csv \\
    --checkpoint-every 20
  python3 03.final_labels/clean_shangxi.py \\
    --output 03.final_labels/poems_structured_shangxi_wip.csv --resume
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

DEFAULT_STRUCT = HERE / "poems_structured_with_dims.csv"
DEFAULT_META = ROOT / "00.poems_dataset/poems_dataset_merged_done.csv"

# ── 规则：整句丢弃（匹配任一则丢弃该句）────────────────────────────────────
_SENTENCE_DROP_RES: list[re.Pattern[str]] = [
    re.compile(r"何逊是|梁天监|驻扬州|建安王|水曹|记室|行参军"),
    re.compile(
        r"是(晚唐|初唐|盛唐|中唐|北宋|南宋|南朝|北朝|元|明|清)诗人[，。]|"
        r"他的诗在当时|诗人。他的诗"
    ),
    re.compile(r"生于|卒于|字[\u4e00-\u9fff]{1,3}[，。、]|号[\u4e00-\u9fff]{1,4}[，。]"),
    re.compile(r"进士|官至|贬为|流寓|谪|靖康|安史|天宝|开元年间"),
    re.compile(r"题为《|误加之题|为据|《初学记》|《艺文类聚》|《全唐诗》"),
    re.compile(r"化自|引自|出自.*诗句|见《[^》]+》卷"),
    re.compile(r"与[^。]{1,20}相比|不如[^。]{1,15}|胜于[^。]{1,15}"),
    re.compile(r"收录于|收录在|版本|刻本|校勘"),
    re.compile(r"美学家|评论家|学者.*谈到|黄震云|署名"),  # 评论者署名行
    re.compile(r"这首诗是[^。]{0,40}写的[。]"),  # 「这首诗是…写的」
    re.compile(r"文人专咏|时代较早|极负盛名[，。]"),  # 常见套话开头可删，但整句匹配
]

# 句末署名（整段末尾）
_TRAILING_BYLINE = re.compile(r"[（(][\u4e00-\u9fff]{2,4}[）)]\s*$")


def _split_sentences(text: str) -> list[str]:
    """按中文句号类切分，保留非空句。"""
    if not text or not str(text).strip():
        return []
    t = str(text).replace("\r", "\n")
    # 按 。！？ 换行 切
    parts = re.split(r"(?<=[。！？])\s*|\n+", t)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 4:
            continue
        out.append(p)
    return out


def rule_filter_shangxi(raw: str | float) -> str:
    """规则过滤后的赏析文本（供 LLM 进一步摘要）。"""
    if pd.isna(raw) or not str(raw).strip():
        return ""
    s = str(raw).strip()
    kept: list[str] = []
    for sent in _split_sentences(s):
        drop = False
        for pat in _SENTENCE_DROP_RES:
            if pat.search(sent):
                drop = True
                break
        if not drop:
            kept.append(sent)
    joined = "".join(kept) if kept else s
    joined = _TRAILING_BYLINE.sub("", joined).strip()
    return joined


def _id_key(x) -> str:
    return str(x).strip()


def _load_resume_cleans(out_path: Path, df: pd.DataFrame) -> list[str | None]:
    """
    从已有输出 CSV 读入已完成的 shangxi_clean（按 ID 对齐）。
    文件中为空或缺失的行对应 None，表示待补跑。
    """
    if not out_path.exists():
        return [None] * len(df)
    try:
        ex = pd.read_csv(out_path)
    except Exception:
        return [None] * len(df)
    if "shangxi_clean" not in ex.columns or "ID" not in ex.columns:
        return [None] * len(df)
    done: dict[str, str] = {}
    for _, r in ex.iterrows():
        v = r.get("shangxi_clean")
        if pd.notna(v) and str(v).strip():
            done[_id_key(r["ID"])] = str(v).strip()
    out: list[str | None] = []
    for _, row in df.iterrows():
        k = _id_key(row["ID"])
        out.append(done.get(k))
    return out


def _write_output_csv(df: pd.DataFrame, cleans: list[str | None], out_path: Path) -> None:
    """写入带 shangxi_clean 的完整表（未完成的行写空字符串）。"""
    df_out = df.copy()
    if "赏析" in df_out.columns:
        df_out = df_out.drop(columns=["赏析"])
    df_out["shangxi_clean"] = [("" if c is None else str(c)) for c in cleans]
    cols = list(df_out.columns)
    if "shangxi_clean" in cols:
        cols.remove("shangxi_clean")
        insert_at = cols.index("正文_preview") + 1 if "正文_preview" in cols else len(cols)
        cols = cols[:insert_at] + ["shangxi_clean"] + cols[insert_at:]
        df_out = df_out[cols]
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")


_GEMINI_MODEL: str | None = None


def _configure_gemini() -> str:
    global _GEMINI_MODEL
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("请设置 GOOGLE_API_KEY 或 GEMINI_API_KEY")
    _GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    try:
        import google.genai  # noqa: F401
    except ImportError as e:
        raise ImportError("请安装: pip install google-genai（需新 SDK 以关闭思考模式，避免输出被截断）") from e
    return _GEMINI_MODEL


def _max_output_tokens() -> int:
    raw = os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "8192").strip()
    try:
        n = int(raw)
        return max(256, min(n, 65536))
    except ValueError:
        return 8192


def llm_extract_shangxi_clean(rule_text: str, raw_fallback: str) -> str:
    """
    用 Gemini 从规则过滤后的文本中抽取摘要。
    若规则文本为空，退回原文截断（仍走 LLM 约束）。
    """
    model_name = _configure_gemini()
    source = rule_text.strip() if rule_text.strip() else str(raw_fallback)[:4000]

    system_instruction = (
        "你是古诗词赏析文本整理助手。下面给出一段「赏析」原文（可能仍含少量背景信息）。\n"
        "任务：从中抽取最能概括该诗意境、情感、核心意象与象征意义的表述，总长度不超过 320 个汉字。\n"
        "严格要求：\n"
        "1. 不能杜撰和新增，只能用原赏析中的内容（可以删减句子、合并相邻句，不得编造事实或诗句）。\n"
        "2. 尽量删去作者生平、创作年份、版本书名、与其他诗人或作品的比较、评论者署名。\n"
        "3. 输出纯中文，不要编号、不要 Markdown、不要引号包裹全文。"
    )

    user_prompt = f"请处理以下赏析原文：\n\n{source}"
    full_prompt = system_instruction + "\n\n" + user_prompt

    # 使用 google-genai：thinking_budget=0 关闭内部「思考」，避免与 max_output_tokens 抢配额导致输出被截断
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=_max_output_tokens(),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        try:
            c = resp.candidates[0]
            parts = getattr(c.content, "parts", None) or []
            text = "".join(getattr(p, "text", "") for p in parts).strip()
        except Exception:
            text = ""
    text = re.sub(r"^[\"'「」]|[\"'「」]$", "", text).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="赏析规则过滤 + Gemini 摘要 → shangxi_clean")
    ap.add_argument("--input", type=Path, default=DEFAULT_STRUCT, help="主结构化 CSV")
    ap.add_argument("--meta", type=Path, default=DEFAULT_META, help="含「赏析」列的元数据 CSV")
    ap.add_argument("--output", type=Path, default=None, help="输出 CSV（默认覆盖 --input）")
    ap.add_argument("--dry-run", action="store_true", help="只打印样例，不写文件、不调 API")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 行（调试）")
    ap.add_argument("--sleep", type=float, default=0.35, help="每条 API 间隔秒数")
    ap.add_argument("--skip-llm", action="store_true", help="只做规则过滤，不调 Gemini")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="从 --output 已有文件中读取已完成的 shangxi_clean，只补空行（续跑）",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="每处理 N 行写入一次 --output（0=仅结束时写入）。建议长跑设为 20。",
    )
    ap.add_argument(
        "--redo-all",
        action="store_true",
        help="忽略 --resume 断点文件，从头逐条重调 LLM（修复截断后全量重跑时用）",
    )
    args = ap.parse_args()

    out_path = args.output or args.input

    if not args.input.exists():
        print(f"错误：找不到 {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.meta.exists():
        print(f"错误：找不到 {args.meta}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    meta = pd.read_csv(args.meta, usecols=lambda c: c in ("ID", "赏析"))
    df = df.merge(meta, on="ID", how="left")

    n = len(df)
    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"调试模式：仅处理前 {len(df)} / {n} 行")

    rule_texts: list[str] = []
    for _, row in df.iterrows():
        rule_texts.append(rule_filter_shangxi(row.get("赏析")))

    if args.dry_run:
        for i in range(min(3, len(df))):
            print(f"\n--- ID={df.iloc[i]['ID']} 规则过滤后（前 400 字）---")
            print(rule_texts[i][:400])
        print("\n[dry-run] 未调用 Gemini，未写文件。")
        return

    if args.skip_llm:
        cleans: list[str | None] = list(rule_texts)
        _write_output_csv(df, cleans, out_path)
        print(f"行数: {len(df)}  已写入: {out_path}")
        print(f"shangxi_clean 非空: {sum(1 for x in cleans if str(x).strip())}/{len(cleans)}")
        return

    cleans = [None] * len(df)
    if args.resume and not args.redo_all:
        cleans = _load_resume_cleans(out_path, df)
        filled = sum(1 for x in cleans if x is not None and str(x).strip())
        print(f"[resume] 从 {out_path} 载入已完成 {filled}/{len(df)} 条，将补全其余行。")
    elif args.redo_all:
        print("[redo-all] 不使用断点，全部重新调用 Gemini（已修复 thinking_budget=0）。")

    _configure_gemini()
    pending = sum(
        1 for i in range(len(df))
        if cleans[i] is None or not str(cleans[i]).strip()
    )
    if pending == 0:
        print("已全部完成，无需再调 API。正在写出最终文件。")
        _write_output_csv(df, cleans, out_path)
        print(f"已写入: {out_path}")
        return

    for i, rt in enumerate(rule_texts):
        if cleans[i] is not None and str(cleans[i]).strip():
            continue
        raw = df.iloc[i].get("赏析", "")
        try:
            c = llm_extract_shangxi_clean(rt, str(raw) if pd.notna(raw) else "")
            cleans[i] = c
        except Exception as e:
            print(f"警告 ID={df.iloc[i]['ID']} LLM 失败: {e}，退回规则文本截断", file=sys.stderr)
            fallback = rt[:500] if rt.strip() else (str(raw)[:500] if pd.notna(raw) else "")
            cleans[i] = fallback
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(df)}", flush=True)
        if args.sleep > 0 and i + 1 < len(df):
            time.sleep(args.sleep)
        if args.checkpoint_every and args.checkpoint_every > 0:
            if (i + 1) % args.checkpoint_every == 0:
                _write_output_csv(df, cleans, out_path)
                print(f"  [checkpoint] 已写入 {out_path}  （进度 {i+1}/{len(df)}）", flush=True)

    _write_output_csv(df, cleans, out_path)
    print(f"行数: {len(df)}  已写入: {out_path}")
    nonempty = sum(1 for x in cleans if x is not None and str(x).strip())
    print(f"shangxi_clean 非空: {nonempty}/{len(cleans)}")


if __name__ == "__main__":
    main()
