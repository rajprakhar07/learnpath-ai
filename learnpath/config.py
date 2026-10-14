"""Configuration loaded from environment (.env supported)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(v, default):
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def load_config(overrides=None):
    data_dir = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    if not data_dir.is_absolute():
        data_dir = (BASE_DIR / data_dir).resolve()
    cfg = {
        "DATA_DIR": data_dir,
        "UPLOAD_DIR": data_dir / "uploads",
        "FIGURE_DIR": data_dir / "figures",
        "DB_PATH": data_dir / "learnpath.sqlite3",
        "MAX_CONTENT_LENGTH": int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").rstrip("/"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        "LLM_MODEL": os.getenv("LLM_MODEL", ""),
        "LLM_JSON_MODE": _bool(os.getenv("LLM_JSON_MODE"), True),
        "LLM_TIMEOUT_SECONDS": float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        "TOP_K": int(os.getenv("TOP_K", "4")),
        "MIN_COVERAGE": float(os.getenv("MIN_COVERAGE", "0.6")),
        "MIN_BM25_SCORE": float(os.getenv("MIN_BM25_SCORE", "0.5")),
    }
    if overrides:
        cfg.update(overrides)
        if "DATA_DIR" in overrides:  # derive dependent paths unless given explicitly
            d = Path(overrides["DATA_DIR"])
            cfg["FIGURE_DIR"] = d / "figures" if "FIGURE_DIR" not in overrides else cfg["FIGURE_DIR"]
            if "UPLOAD_DIR" not in overrides:
                cfg["UPLOAD_DIR"] = d / "uploads"
            if "DB_PATH" not in overrides:
                cfg["DB_PATH"] = d / "learnpath.sqlite3"
    return cfg
