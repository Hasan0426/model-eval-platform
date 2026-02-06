from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ProfileConfig:
    max_categories: int = 20          # 类别分布最多输出多少个值
    max_unique_for_categorical: int = 1000  # 唯一值太多就不当成 categorical
    sample_rows_for_inference: int = 5000   # 推断用采样上限（防止超大 CSV 卡死）


def infer_schema(df: pd.DataFrame, cfg: ProfileConfig) -> Dict[str, Any]:
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    # 采样用于推断（保持轻量）
    df_inf = df.head(cfg.sample_rows_for_inference) if n_rows > cfg.sample_rows_for_inference else df

    columns: List[Dict[str, Any]] = []
    for col in df_inf.columns:
        s = df_inf[col]
        dtype = str(s.dtype)

        # 基础统计
        missing = int(s.isna().sum())
        missing_ratio = float(missing / max(n_rows, 1))

        # 唯一值
        nunique = int(s.nunique(dropna=True))

        # 粗略判定 categorical：object/string 或者整数但唯一值不大
        is_object_like = dtype in ("object", "string")
        is_categorical = False
        if is_object_like:
            is_categorical = nunique <= cfg.max_unique_for_categorical
        else:
            # 数值列也可能是类别（比如 0/1 或 1..10）
            if nunique > 0 and nunique <= min(cfg.max_unique_for_categorical, 0.05 * max(n_rows, 1) + 5):
                is_categorical = True

        columns.append(
            {
                "name": str(col),
                "dtype": dtype,
                "nunique": nunique,
                "missing": missing,
                "missing_ratio": missing_ratio,
                "is_categorical": bool(is_categorical),
            }
        )

    return {
        "rows": n_rows,
        "cols": n_cols,
        "columns": columns,
    }


def validate_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    最小 schema 校验（Day1 版）：
    - required columns 是否存在
    - 空列名/重复列名检查
    """
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    cols = [str(c) for c in df.columns]

    # 空列名
    empty_names = [c for c in cols if c.strip() == ""]
    if empty_names:
        errors.append({"type": "EMPTY_COLUMN_NAME", "detail": "Found empty column name(s)."})

    # 重复列名
    dup = pd.Series(cols).duplicated(keep=False)
    if dup.any():
        dup_names = pd.Series(cols)[dup].tolist()
        errors.append({"type": "DUPLICATE_COLUMN_NAMES", "detail": {"duplicates": dup_names}})

    if required_columns:
        missing = [c for c in required_columns if c not in cols]
        if missing:
            errors.append({"type": "MISSING_REQUIRED_COLUMNS", "detail": {"missing": missing}})

    # 这里可以加更多规则：label 是否为 0/1、时间列格式等（Day2 再扩展）
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def missing_stats(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = int(df.shape[0])
    per_col = []
    for col in df.columns:
        s = df[col]
        m = int(s.isna().sum())
        per_col.append(
            {"column": str(col), "missing": m, "missing_ratio": float(m / max(n_rows, 1))}
        )
    per_col_sorted = sorted(per_col, key=lambda x: x["missing"], reverse=True)
    total_missing = int(df.isna().sum().sum())
    return {"total_missing": total_missing, "by_column": per_col_sorted}


def categorical_distributions(df: pd.DataFrame, cfg: ProfileConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n_rows = int(df.shape[0])

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nunique = int(s.nunique(dropna=True))

        # 判定是否输出分布
        is_object_like = dtype in ("object", "string")
        should_profile = False
        if is_object_like and nunique <= cfg.max_unique_for_categorical:
            should_profile = True
        # 数值但唯一值很少，也输出（如 label, segment_id）
        if (not is_object_like) and nunique > 0 and nunique <= 50:
            should_profile = True

        if not should_profile:
            continue

        vc = (
            s.astype("string")
            .fillna("<NA>")
            .value_counts(dropna=False)
            .head(cfg.max_categories)
        )

        dist = []
        for k, v in vc.items():
            dist.append({"value": str(k), "count": int(v), "ratio": float(v / max(n_rows, 1))})

        out[str(col)] = {
            "nunique": nunique,
            "top": dist,
            "truncated": nunique > cfg.max_categories,
        }

    return out
