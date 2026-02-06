from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math

import pandas as pd
import numpy as np

@dataclass
class ProfileConfig:
    max_categories: int = 20
    max_unique_for_categorical: int = 1000
    sample_rows_for_inference: int = 5000

# ==========================================
# 🔥 核心修复：彻底清洗函数
# ==========================================
def clean_nan(obj: Any) -> Any:
    """
    递归遍历字典或列表，将所有的 NaN / Infinity / NaT 强制转换为 None。
    这是唯一能 100% 保证 JSON 兼容的方法。
    """
    # 1. 处理字典
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    
    # 2. 处理列表
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    
    # 3. 处理浮点数 (核心逻辑)
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    
    # 4. 处理 Pandas/Numpy 的缺失值对象
    elif obj is pd.NA or obj is np.nan:
        return None
        
    # 5. 处理 Numpy 整数 (转为 Python int)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
        
    return obj

# ==========================================
# 业务逻辑
# ==========================================

def infer_schema(df: pd.DataFrame, cfg: ProfileConfig) -> Dict[str, Any]:
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    df_inf = df.head(cfg.sample_rows_for_inference) if n_rows > cfg.sample_rows_for_inference else df

    columns: List[Dict[str, Any]] = []
    for col in df_inf.columns:
        s = df_inf[col]
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        missing_ratio = float(missing / max(n_rows, 1))
        nunique = int(s.nunique(dropna=True))

        is_object_like = dtype in ("object", "string")
        is_categorical = False
        if is_object_like:
            is_categorical = nunique <= cfg.max_unique_for_categorical
        else:
            if nunique > 0 and nunique <= min(cfg.max_unique_for_categorical, 0.05 * max(n_rows, 1) + 5):
                is_categorical = True

        columns.append({
            "name": str(col),
            "dtype": dtype,
            "nunique": nunique,
            "missing": missing,
            "missing_ratio": missing_ratio,
            "is_categorical": bool(is_categorical),
        })

    # 🔥 这里的返回值也要清洗
    return clean_nan({
        "rows": n_rows,
        "cols": n_cols,
        "columns": columns,
    })


def validate_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    errors = []
    warnings = []
    cols = [str(c) for c in df.columns]

    empty_names = [c for c in cols if c.strip() == ""]
    if empty_names:
        errors.append({"type": "EMPTY_COLUMN_NAME", "detail": "Found empty column name(s)."})

    dup = pd.Series(cols).duplicated(keep=False)
    if dup.any():
        dup_names = pd.Series(cols)[dup].tolist()
        errors.append({"type": "DUPLICATE_COLUMN_NAMES", "detail": {"duplicates": dup_names}})

    if required_columns:
        missing = [c for c in required_columns if c not in cols]
        if missing:
            errors.append({"type": "MISSING_REQUIRED_COLUMNS", "detail": {"missing": missing}})

    return clean_nan({"ok": len(errors) == 0, "errors": errors, "warnings": warnings})


def missing_stats(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = int(df.shape[0])
    per_col = []
    for col in df.columns:
        s = df[col]
        m = int(s.isna().sum())
        per_col.append({"column": str(col), "missing": m, "missing_ratio": float(m / max(n_rows, 1))})
    
    per_col_sorted = sorted(per_col, key=lambda x: x["missing"], reverse=True)
    total_missing = int(df.isna().sum().sum())
    
    return clean_nan({"total_missing": total_missing, "by_column": per_col_sorted})


def categorical_distributions(df: pd.DataFrame, cfg: ProfileConfig) -> Dict[str, Any]:
    out = {}
    n_rows = int(df.shape[0])

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nunique = int(s.nunique(dropna=True))

        is_object_like = dtype in ("object", "string")
        should_profile = False
        if is_object_like and nunique <= cfg.max_unique_for_categorical:
            should_profile = True
        if (not is_object_like) and nunique > 0 and nunique <= 50:
            should_profile = True

        if not should_profile:
            continue

        vc = s.astype("string").fillna("<NA>").value_counts(dropna=False).head(cfg.max_categories)
        dist = []
        for k, v in vc.items():
            dist.append({"value": str(k), "count": int(v), "ratio": float(v / max(n_rows, 1))})

        out[str(col)] = {
            "nunique": nunique,
            "top": dist,
            "truncated": nunique > cfg.max_categories,
        }

    # 🔥 清洗所有类别统计
    return clean_nan(out)


def numerical_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {}
    num_df = df.select_dtypes(include=['number'])
    
    if num_df.empty:
        return stats

    desc = num_df.describe().to_dict()
    for col, metrics in desc.items():
        stats[col] = {
            "mean": metrics.get("mean"),
            "std": metrics.get("std"),
            "min": metrics.get("min"),
            "max": metrics.get("max"),
            "q25": metrics.get("25%"),
            "q50": metrics.get("50%"),
            "q75": metrics.get("75%")
        }
    
    # 🔥 清洗所有数值统计 (这里不需要 safe_float 了，直接由 clean_nan 统一处理)
    return clean_nan(stats)


def get_preview(df: pd.DataFrame, rows: int = 10) -> List[Dict[str, Any]]:
    """
    获取前 N 行预览
    """
    # 1. 截取前 N 行
    sample = df.head(rows)
    
    # 2. 转换为 Python 字典 (此时里面可能还含有 NaN)
    raw_preview = sample.to_dict(orient="records")
    
    # 3. 🔥 在 Python 层面进行彻底清洗
    return clean_nan(raw_preview)