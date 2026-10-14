import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from learnpath import create_app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "sample_course.pdf"


@pytest.fixture(scope="session", autouse=True)
def _ensure_sample():
    if not SAMPLE.exists():
        import subprocess, sys
        subprocess.check_call([sys.executable, str(ROOT / "sample_data" / "make_sample_pdf.py")])


@pytest.fixture()
def app(tmp_path):
    return create_app({"DATA_DIR": tmp_path / "data", "LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def loaded(client):
    r = client.post("/api/documents/sample")
    assert r.status_code == 201, r.get_json()
    return client


class FakeLLMServer:
    """Scripted OpenAI-compatible server. Tests push replies; it records every request."""

    def __init__(self):
        self.replies, self.requests = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "body": json.loads(self.rfile.read(n) or b"{}")})
                status, content = outer.replies.pop(0) if outer.replies else (500, "no scripted reply")
                payload = json.dumps({"choices": [{"message": {"content": content}}]}) if status == 200 else content
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode())

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def reply(self, content, status=200):
        self.replies.append((status, content))


@pytest.fixture()
def fake_llm():
    s = FakeLLMServer()
    yield s
    s.httpd.shutdown()


@pytest.fixture()
def llm_client(tmp_path, fake_llm):
    app = create_app({"DATA_DIR": tmp_path / "data", "LLM_BASE_URL": fake_llm.url, "LLM_API_KEY": "sk-secret-test-key",
                      "LLM_MODEL": "fake-model"})
    c = app.test_client()
    assert c.post("/api/documents/sample").status_code == 201
    return c
