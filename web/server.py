from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:
    sys.stderr.write(
        "flask is required to run the web server.\n"
        "Install it with:  pip install flask\n"
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from heart.toolface import ToolFace
from heart.tools import register_all


WEB_DIR = Path(__file__).resolve().parent

tf = ToolFace()
register_all(tf)

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.get("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.get("/tools")
def list_tools():
    return jsonify([
        {
            "id": s.id,
            "category": s.category,
            "description": s.description,
            "parameters": [p.__dict__ for p in s.parameters],
        }
        for s in tf.list_schemas()
    ])


def _dispatch(tool_id: str):
    try:
        fn = tf.get_function(tool_id)
    except KeyError:
        fn = None
    if fn is None:
        return jsonify({
            "ok": False,
            "detail": f"unknown tool: {tool_id!r}. "
                      f"This server only exposes the bundled ToolFace tools "
                      f"({len(list(tf.list_schemas()))} total). "
                      f"Call GET /tools to list them."
        }), 404
    body = request.get_json(silent=True) or {}
    args = body.get("arguments", body)
    try:
        result = fn.fn(**args)
        return jsonify({"ok": True, "result": result})
    except TypeError as e:
        return jsonify({"ok": False, "detail": f"argument error: {e}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "detail": f"{type(e).__name__}: {e}"}), 500


@app.post("/tools/airline/<tool_name>")
@app.post("/tools/retail/<tool_name>")
@app.post("/tools/telecom/<tool_name>")
def tau2_route(tool_name):
    return _dispatch(tool_name)


@app.post("/tools/nestful/<tool_name>")
def nestful_route(tool_name):
    return _dispatch(tool_name)


@app.post("/ace/<tool_name>")
def ace_route(tool_name):
    return _dispatch(tool_name)


@app.post("/execute/<tool_id>")
def execute(tool_id):
    return _dispatch(tool_id)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "detail": "route not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"ToolFace web server starting on http://{host}:{port}")
    print(f"  - Serves index.html at /")
    print(f"  - Bundled tools: {len(list(tf.list_schemas()))}")
    print(f"  - Tool dispatch: POST /execute/<tool_id> with body {{\"arguments\": {{...}}}}")
    print(f"  - Also routes: /tools/<domain>/<tool> and /ace/<tool> for HTML compatibility")
    app.run(host=host, port=port, debug=False)
