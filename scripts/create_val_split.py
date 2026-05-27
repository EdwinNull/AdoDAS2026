#!/usr/bin/env python3
"""
创建 val_select / val_holdout 分层切分文件

基于 ADODAS2026 Stage 0 诚实评估地基要求：
- 在 600 人 val 集内部按 participant level 做二次切分
- 400/200 分割 (val_select / val_holdout)
- 按 DASS-21 总分 + 抑郁/焦虑/压力三个子量表分数分层
- 确定性切分（固定 seed），落盘保存为 splits/val_split_v1.json
"""
from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# DASS-21 子量表条目映射
DEPRESSION_ITEMS = ["d03", "d05", "d10", "d13", "d16", "d17", "d21"]
ANXIETY_ITEMS = ["d02", "d04", "d07", "d09", "d15", "d19", "d20"]
STRESS_ITEMS = ["d01", "d06", "d08", "d11", "d12", "d14", "d18"]
ALL_ITEMS = [f"d{i:02d}" for i in range(1, 22)]


def compute_subscale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个 participant 的 DASS-21 子量表分数和总分"""
    pid_rows = []
    for pid, group in df.groupby("anon_pid"):
        row = group.iloc[0]
        d_score = sum(int(row.get(c, 0)) for c in DEPRESSION_ITEMS)
        a_score = sum(int(row.get(c, 0)) for c in ANXIETY_ITEMS)
        s_score = sum(int(row.get(c, 0)) for c in STRESS_ITEMS)
        total = d_score + a_score + s_score
        pid_rows.append({
            "anon_pid": str(pid),
            "dass_depression": d_score,
            "dass_anxiety": a_score,
            "dass_stress": s_score,
            "dass_total": total,
        })
    return pd.DataFrame(pid_rows)


def create_stratify_label(scores: pd.DataFrame, n_bins: int = 4) -> np.ndarray:
    """
    基于 DASS-21 子量表分数创建分层标签

    策略：将每个子量表分数分箱后拼接成组合标签。
    合并样本数过少的稀有 strata（<2）到 DASS total 分箱，保证 StratifiedShuffleSplit 可运行。
    """
    labels = []
    for col in ["dass_depression", "dass_anxiety", "dass_stress"]:
        binned = pd.qcut(scores[col], q=n_bins, labels=False, duplicates="drop")
        labels.append(binned.astype(str))
    combined_arr = np.array([
        f"{a}_{b}_{c}" for a, b, c in zip(labels[0], labels[1], labels[2])
    ])

    # 合并稀有 strata（成员数 < 2）
    unique, counts = np.unique(combined_arr, return_counts=True)
    rare_strata = set(unique[counts < 2])
    if rare_strata:
        total_binned = pd.qcut(scores["dass_total"], q=max(n_bins, 6), labels=False, duplicates="drop")
        for i in range(len(combined_arr)):
            if combined_arr[i] in rare_strata:
                combined_arr[i] = f"rare_t{total_binned.iloc[i]}"
        print(f"Merged {len(rare_strata)} rare strata (n<2) into total-score fallback bins")

    return combined_arr


def main():
    parser = argparse.ArgumentParser(description="Create val_select/val_holdout split")
    parser.add_argument("--val-csv", default="/data1/AdoDas/Val/val.csv",
                        help="Path to val.csv manifest")
    parser.add_argument("--output", default="splits/val_split_v1.json",
                        help="Output path for split JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--val-select-size", type=int, default=400,
                        help="Number of participants in val_select")
    parser.add_argument("--n-bins", type=int, default=4,
                        help="Number of quantile bins per subscale for stratification")
    args = parser.parse_args()

    val_csv = Path(args.val_csv)
    if not val_csv.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(val_csv)
    scores = compute_subscale_scores(df)
    print(f"Total participants in val: {len(scores)}")

    # 检查 val_holdout 中各类别样本数
    val_holdout_size = len(scores) - args.val_select_size
    if val_holdout_size < 50:
        print(f"WARNING: val_holdout only {val_holdout_size} participants, may be too small")

    # 分层抽样
    stratify_labels = create_stratify_label(scores, n_bins=args.n_bins)
    unique_strata = len(np.unique(stratify_labels))
    print(f"Stratification: {unique_strata} unique strata from "
          f"depression/anxiety/stress subscale quantiles (n_bins={args.n_bins})")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_holdout_size,
        random_state=args.seed,
    )
    select_idx, holdout_idx = next(splitter.split(scores, stratify_labels))

    val_select_pids = scores.iloc[select_idx]["anon_pid"].tolist()
    val_holdout_pids = scores.iloc[holdout_idx]["anon_pid"].tolist()

    # 验证无重叠
    overlap = set(val_select_pids) & set(val_holdout_pids)
    assert len(overlap) == 0, f"Overlap detected: {overlap}"

    # 验证 val_holdout 中类 3 样本数
    holdout_scores = scores[scores["anon_pid"].isin(val_holdout_pids)]
    holdout_df = df[df["anon_pid"].isin(val_holdout_pids)]
    n_class3 = sum(
        (holdout_df[item] == 3).any().sum() for item in ALL_ITEMS
    )
    print(f"val_holdout participants with class-3 labels: {n_class3}")
    if n_class3 < 5:
        print("WARNING: val_holdout has < 5 class-3 samples; "
              "consider adjusting stratification or split ratio")

    # 构建输出
    split_data = {
        "version": "v1",
        "seed": args.seed,
        "created_from": str(val_csv),
        "stratify_by": ["dass_depression", "dass_anxiety", "dass_stress"],
        "stratify_n_bins": args.n_bins,
        "counts": {
            "val_select": len(val_select_pids),
            "val_holdout": len(val_holdout_pids),
        },
        "val_select_pids": sorted(val_select_pids),
        "val_holdout_pids": sorted(val_holdout_pids),
    }

    # 计算 hash 用于版本校验
    payload = json.dumps({
        "val_select_pids": split_data["val_select_pids"],
        "val_holdout_pids": split_data["val_holdout_pids"],
    }, sort_keys=True)
    split_data["content_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:16]

    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"Split saved to {output_path}")
    print(f"  val_select: {len(val_select_pids)} participants")
    print(f"  val_holdout: {len(val_holdout_pids)} participants")
    print(f"  content_hash: {split_data['content_hash']}")

    # 打印子量表分布统计
    for name, pids in [("val_select", val_select_pids), ("val_holdout", val_holdout_pids)]:
        subset = scores[scores["anon_pid"].isin(pids)]
        print(f"\n  {name} subscale distribution (mean ± std):")
        for col in ["dass_depression", "dass_anxiety", "dass_stress", "dass_total"]:
            print(f"    {col}: {subset[col].mean():.1f} ± {subset[col].std():.1f}")


if __name__ == "__main__":
    main()
