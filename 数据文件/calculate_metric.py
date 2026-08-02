"""Calculate the gene-dependency submission score."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["cell_line_id", "perturbation_gene"]
KS = (5, 10, 15)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a gene-dependency submission against the hidden answer CSV."
    )
    parser.add_argument("submission_csv", type=Path, help="User submission CSV.")
    parser.add_argument(
        "answer_csv",
        type=Path,
        help="Answer CSV.",
    )
    return parser.parse_args()


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" in df.columns and not set(KEYS).issubset(df.columns):
        ids = df["id"].astype(str).str.split("|", n=1, expand=True)
        if ids.shape[1] != 2 or ids.isna().any().any():
            raise ValueError("submission id must use format: cell_line_id|perturbation_gene")
        df["cell_line_id"] = ids[0]
        df["perturbation_gene"] = ids[1]

    pred_col = "prediction" if "prediction" in df.columns else "label"
    required = set(KEYS + [pred_col])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"submission missing columns: {missing}")

    df = df[KEYS + [pred_col]].rename(columns={pred_col: "prediction"}).copy()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    if df["prediction"].isna().any():
        count = int(df["prediction"].isna().sum())
        raise ValueError(f"submission contains {count} missing/non-numeric predictions")
    return df


def read_answer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(KEYS + ["label"]) - set(df.columns))
    if missing:
        raise ValueError(f"answer missing columns: {missing}")
    df = df[KEYS + ["label"]].rename(columns={"label": "truth"}).copy()
    df["truth"] = pd.to_numeric(df["truth"], errors="coerce")
    if df["truth"].isna().any():
        count = int(df["truth"].isna().sum())
        raise ValueError(f"answer contains {count} missing/non-numeric labels")
    df["_order"] = np.arange(len(df))
    return df


def validate_pairs(sub: pd.DataFrame, ans: pd.DataFrame) -> None:
    if sub.duplicated(KEYS).any():
        dup = sub.loc[sub.duplicated(KEYS, keep=False), KEYS].head(5).to_dict("records")
        raise ValueError(f"submission contains duplicate pairs, examples: {dup}")
    if ans.duplicated(KEYS).any():
        dup = ans.loc[ans.duplicated(KEYS, keep=False), KEYS].head(5).to_dict("records")
        raise ValueError(f"answer contains duplicate pairs, examples: {dup}")

    sub_keys = pd.MultiIndex.from_frame(sub[KEYS])
    ans_keys = pd.MultiIndex.from_frame(ans[KEYS])
    missing = ans_keys.difference(sub_keys)
    extra = sub_keys.difference(ans_keys)
    if len(missing) or len(extra):
        raise ValueError(
            "submission pairs do not match answer pairs: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if not sub[KEYS].reset_index(drop=True).equals(ans[KEYS].reset_index(drop=True)):
        raise ValueError("submission row order must match the answer/sample submission order")


def pearson_or_zero(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= EPS:
        return 0.0
    return float(np.dot(x, y) / denom)


def metric_by_cell(df: pd.DataFrame) -> dict[str, float]:
    spearman_values = []
    precision_values = {k: [] for k in KS}
    ndcg_values = {k: [] for k in KS}

    for _, group in df.groupby("cell_line_id", sort=False):
        group = group.copy()
        group["pred_rank"] = group["prediction"].rank(ascending=False, method="average")
        group["true_rank"] = group["truth"].rank(ascending=False, method="average")
        spearman_values.append(
            pearson_or_zero(group["pred_rank"].to_numpy(), group["true_rank"].to_numpy())
        )

        pred_sorted = group.sort_values(["prediction", "_order"], ascending=[False, True])
        true_sorted = group.sort_values(["truth", "_order"], ascending=[False, True])

        for k in KS:
            pred_top = pred_sorted.head(k)
            true_top_genes = set(true_sorted.head(k)["perturbation_gene"])
            precision_values[k].append(
                len(set(pred_top["perturbation_gene"]) & true_top_genes) / k
            )

            rel = np.maximum(k - pred_top["true_rank"].to_numpy(dtype=float) + 1.0, 0.0)
            rel[pred_top["true_rank"].to_numpy(dtype=float) > k] = 0.0
            discounts = np.log2(np.arange(2, len(rel) + 2, dtype=float))
            dcg = float(np.sum((np.power(2.0, rel) - 1.0) / discounts))
            ideal_rel = np.arange(k, 0, -1, dtype=float)
            ideal_discounts = np.log2(np.arange(2, k + 2, dtype=float))
            idcg = float(np.sum((np.power(2.0, ideal_rel) - 1.0) / ideal_discounts))
            ndcg_values[k].append(dcg / idcg if idcg > 0 else 0.0)

    rho_macro = float(np.mean(spearman_values))
    precision_macro = {k: float(np.mean(precision_values[k])) for k in KS}
    ndcg_macro = {k: float(np.mean(ndcg_values[k])) for k in KS}

    return {
        "rho_macro": rho_macro,
        "spearman_score": (rho_macro + 1.0) / 2.0,
        "precision_at_5": precision_macro[5],
        "precision_at_10": precision_macro[10],
        "precision_at_15": precision_macro[15],
        "precision_score": float(np.mean(list(precision_macro.values()))),
        "ndcg_at_5": ndcg_macro[5],
        "ndcg_at_10": ndcg_macro[10],
        "ndcg_at_15": ndcg_macro[15],
        "ndcg_score": float(np.mean(list(ndcg_macro.values()))),
    }


def print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(name) for name, _ in rows)
    print()
    print("Gene Dependency Score")
    print("=" * (width + 14))
    for name, value in rows:
        print(f"{name:<{width}} : {value}")
    print("=" * (width + 14))


def main() -> None:
    args = parse_args()
    submission = read_submission(args.submission_csv)
    answer = read_answer(args.answer_csv)
    validate_pairs(submission, answer)

    df = answer.merge(submission, on=KEYS, how="left", validate="one_to_one")
    metrics = metric_by_cell(df)

    rmse = float(np.sqrt(np.mean(np.square(df["prediction"] - df["truth"]))))
    sigma_y = float(df["truth"].std(ddof=0))
    nrmse = rmse / (sigma_y + EPS)
    rmse_score = 1.0 / (1.0 + nrmse)

    final_score = 100.0 * (
        0.30 * metrics["spearman_score"]
        + 0.30 * metrics["ndcg_score"]
        + 0.25 * metrics["precision_score"]
        + 0.15 * rmse_score
    )

    rows = [
        ("Final score", f"{final_score:.6f} / 100"),
        ("SpearmanScore", f"{metrics['spearman_score']:.6f}  (rho_macro={metrics['rho_macro']:.6f})"),
        ("NDCGScore", f"{metrics['ndcg_score']:.6f}"),
        ("  NDCG@5", f"{metrics['ndcg_at_5']:.6f}"),
        ("  NDCG@10", f"{metrics['ndcg_at_10']:.6f}"),
        ("  NDCG@15", f"{metrics['ndcg_at_15']:.6f}"),
        ("PrecisionScore", f"{metrics['precision_score']:.6f}"),
        ("  Precision@5", f"{metrics['precision_at_5']:.6f}"),
        ("  Precision@10", f"{metrics['precision_at_10']:.6f}"),
        ("  Precision@15", f"{metrics['precision_at_15']:.6f}"),
        ("RMSEScore", f"{rmse_score:.6f}"),
        ("  RMSE", f"{rmse:.6f}"),
        ("  NRMSE", f"{nrmse:.6f}"),
        ("Rows", f"{len(df):,}"),
        ("Cell lines", f"{df['cell_line_id'].nunique():,}"),
    ]
    print_table(rows)


if __name__ == "__main__":
    main()
