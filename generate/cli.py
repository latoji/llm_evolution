"""
Track 4/5 – CLI
Command-line interface for text generation.

Usage:
  python generate/cli.py --prompt "The meaning of" --tokens 100 --temp 0.7
  python generate/cli.py --prompt "In the beginning" --top-k 50 --top-p 0.9 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text with the n-gram language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, required=True, help="Seed text")
    parser.add_argument(
        "--tokens", type=int, default=200, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k filter (0 = disabled)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model", type=Path, default=BASE_DIR / "model" / "lm.pkl",
        help="Path to serialised LanguageModel"
    )
    parser.add_argument(
        "--merges", type=Path, default=BASE_DIR / "tokenizer" / "merges.json"
    )
    parser.add_argument(
        "--vocab", type=Path, default=BASE_DIR / "tokenizer" / "vocab.json"
    )
    args = parser.parse_args()

    # Lazy imports (keep startup fast)
    from generate.generator import generate_stream
    from model.language_model import LanguageModel
    from tokenizer.bpe import load_tokenizer

    print(f"Loading model from {args.model}…", file=sys.stderr)
    lm = LanguageModel.load(args.model)

    print("Loading tokenizer…", file=sys.stderr)
    merges, vocab = load_tokenizer(args.merges, args.vocab)

    # Print the prompt and then stream generated tokens
    print(args.prompt, end="", flush=True)
    for token_text in generate_stream(
        model=lm,
        merges=merges,
        vocab=vocab,
        prompt=args.prompt,
        max_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    ):
        print(token_text, end="", flush=True)

    print()  # newline at end


if __name__ == "__main__":
    main()
