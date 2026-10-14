"""LearnPath AI - source-grounded study assistant."""
from flask import Flask

from .config import load_config
from .db import init_db
from .llm import LLMClient
from .store import Retriever


def create_app(overrides: dict | None = None) -> Flask:
    cfg = load_config(overrides)
    app = Flask(__name__)
    app.config.update(cfg)
    cfg["DATA_DIR"].mkdir(parents=True, exist_ok=True)
    cfg["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    cfg["FIGURE_DIR"].mkdir(parents=True, exist_ok=True)
    init_db(cfg["DB_PATH"])
    app.extensions["llm"] = LLMClient(
        cfg["LLM_BASE_URL"], cfg["LLM_API_KEY"], cfg["LLM_MODEL"], cfg["LLM_JSON_MODE"], cfg["LLM_TIMEOUT_SECONDS"]
    )
    app.extensions["retriever"] = Retriever()

    from .web import bp
    app.register_blueprint(bp)
    return app
