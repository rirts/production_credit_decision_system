"""Versioned atomic training bundle shared by prediction and drift."""
from functools import lru_cache
from pathlib import Path
import os
import joblib

ROOT = Path(__file__).resolve().parents[1]

def artifact_path():
    return Path(os.getenv("CREDIT_MODEL_DIR", str(ROOT / "models"))) / "bundle.joblib"

@lru_cache(maxsize=4)
def _load(path, stamp):
    bundle = joblib.load(path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model bundle schema")
    return bundle

def load_bundle():
    path = artifact_path()
    return _load(str(path), path.stat().st_mtime_ns)
