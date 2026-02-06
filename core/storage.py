from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DATASETS_DIR = Path("data/datasets")
METADATA_DIR = Path("data/metadata")


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def new_dataset_id() -> str:
    # 统一用 uuid4，后面接 DB 也好迁移
    return str(uuid.uuid4())


def save_dataset_bytes(dataset_id: str, filename: str, content: bytes) -> Dict[str, Any]:
    ensure_dirs()
    # 只做 CSV（Day1），文件名标准化存储
    raw_path = DATASETS_DIR / f"{dataset_id}__{filename}"
    raw_path.write_bytes(content)

    stat = raw_path.stat()
    return {
        "path": str(raw_path),
        "bytes": int(stat.st_size),
    }


def save_metadata(dataset_id: str, meta: Dict[str, Any]) -> str:
    ensure_dirs()
    meta_path = METADATA_DIR / f"{dataset_id}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return str(meta_path)


def load_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
    meta_path = METADATA_DIR / f"{dataset_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_datasets(limit: int = 50) -> Dict[str, Any]:
    ensure_dirs()
    items = []
    for p in sorted(METADATA_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        items.append(p.stem)
        if len(items) >= limit:
            break
    return {"dataset_ids": items}
