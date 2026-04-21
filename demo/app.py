"""
Demo Flask web server.

Usage:
    python3 demo/setup_demo.py   # one-time setup
    python3 demo/app.py          # start server on http://localhost:5000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

DEMO_DIR = Path(__file__).parent
DEMO_MODEL_PATH = DEMO_DIR / "demo_lm.pkl"
DEMO_MERGES_PATH = DEMO_DIR / "demo_merges.json"
DEMO_VOCAB_PATH = DEMO_DIR / "demo_vocab.json"

app = Flask(__name__, template_folder="templates")

# Globals loaded once at startup
_lm = None
_merges = None
_vocab = None


def _load_model():
    global _lm, _merges, _vocab
    if _lm is not None:
        return

    if not DEMO_MODEL_PATH.exists():
        raise RuntimeError(
            "Demo model not found. Run: python3 demo/setup_demo.py"
        )

    from model.language_model import LanguageModel
    from tokenizer.bpe import load_tokenizer

    print("Loading demo model…", flush=True)
    _lm = LanguageModel.load(DEMO_MODEL_PATH)
    _merges, _vocab = load_tokenizer(DEMO_MERGES_PATH, DEMO_VOCAB_PATH)
    print(f"  Vocab: {len(_vocab)}  Max order: {_lm.max_order}", flush=True)


@app.before_request
def ensure_model():
    _load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "The meaning of").strip()
    max_tokens = int(data.get("max_tokens", 80))
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.9))
    seed = data.get("seed")
    seed = int(seed) if seed not in (None, "", "null") else None

    from generate.generator import generate as gen_fn

    try:
        text = gen_fn(
            _lm,
            _merges,
            _vocab,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        return jsonify({"output": text, "error": None})
    except Exception as exc:
        return jsonify({"output": "", "error": str(exc)}), 500


@app.route("/stream", methods=["POST"])
def stream():
    """SSE endpoint: streams tokens one at a time."""
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "The meaning of").strip()
    max_tokens = int(data.get("max_tokens", 80))
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.9))
    seed = data.get("seed")
    seed = int(seed) if seed not in (None, "", "null") else None

    from generate.generator import generate_stream as gen_stream_fn

    def event_stream():
        try:
            for token_text in gen_stream_fn(
                _lm, _merges, _vocab, prompt,
                max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, seed=seed,
            ):
                yield f"data: {json.dumps({'token': token_text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/info")
def info():
    """Return model metadata for the UI info bar."""
    return jsonify({
        "vocab_size": len(_vocab) if _vocab else 0,
        "max_order": _lm.max_order if _lm else 0,
    })


# ── Character-level Markov visualizer ──────────────────────────────────────

@app.route("/markov")
def markov():
    return render_template("markov.html")


@app.route("/char_predict", methods=["POST"])
def char_predict():
    """Return next-char distribution + Monte Carlo walks for a prefix."""
    from demo.char_ngrams import (
        compute_entropy,
        get_bigram_matrix,
        get_tables,
        monte_carlo_walks,
        predict as char_predict_fn,
    )

    data = request.get_json(force=True)
    prefix = data.get("prefix", "")
    n_walkers = min(int(data.get("n_walkers", 20)), 50)
    walk_length = min(int(data.get("walk_length", 30)), 100)

    tables = get_tables()
    dist, order_used = char_predict_fn(tables, prefix)

    # Top 30 chars for the bar chart
    sorted_dist = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:30])

    # Top 12 chars for the transition matrix (square matrix, manageable size)
    top_chars = list(sorted_dist.keys())[:12]

    walks = monte_carlo_walks(
        tables, prefix, n_walkers=n_walkers, walk_length=walk_length,
    )

    best_char = max(dist, key=dist.get) if dist else ""
    entropy = round(compute_entropy(dist), 3)
    bigram_matrix = get_bigram_matrix(tables, top_chars)

    # Last character of the prefix (used by the matrix to highlight the active row)
    active_char = prefix[-1].lower() if prefix else ""

    return jsonify({
        "distribution": sorted_dist,
        "walks": walks,
        "best_char": best_char,
        "order_used": order_used,
        "entropy": entropy,
        "bigram_matrix": bigram_matrix,
        "top_chars": top_chars,
        "active_char": active_char,
    })


if __name__ == "__main__":
    _load_model()
    print("\n🌐  Open http://localhost:5000  in your browser\n", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
