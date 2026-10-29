"""Web UI for the Smarket knowledge-graph assistant.

Usage:
    uv run python -m src.server

Then open http://localhost:5000.  Requires OPENROUTER_API_KEY in .env.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

from .ontology_agent import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL,
    ask,
    build_client,
    build_system_prompt,
)

UI_DIR = Path(__file__).parent / "ui"

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="")
_system_prompt = build_system_prompt()
_client: OpenAI | None = None


@app.get("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.get("/api/ontology")
def api_ontology():
    from .ontology_agent import DEFAULT_ONTOLOGY_PATH
    return DEFAULT_ONTOLOGY_PATH.read_text(), 200, {"Content-Type": "text/turtle; charset=utf-8"}


@app.post("/api/ask")
def api_ask():
    global _client

    body     = request.get_json(silent=True) or {}
    messages = body.get("messages", [])

    if not messages or not isinstance(messages, list):
        return jsonify({"error": "messages array is required"}), 400

    last = messages[-1]
    if not isinstance(last, dict) or not last.get("content", "").strip():
        return jsonify({"error": "last message must have non-empty content"}), 400

    if _client is None:
        try:
            _client = build_client()
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 503

    result = ask(messages, _system_prompt, _client, model=DEFAULT_MODEL, data_path=DEFAULT_DATA_PATH)
    return jsonify(result)


def main() -> None:
    app.run(port=5000, debug=False)


if __name__ == "__main__":
    main()
