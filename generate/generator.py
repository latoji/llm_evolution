"""
Track 4 – Prediction Engine (extended by Track 5)
Markov chain text generation using the trained language model.
"""
from __future__ import annotations

import random
from typing import Iterator

from generate.sampling import combined_sample
from model.language_model import LanguageModel
from tokenizer.bpe import ENDOFTEXT_ID, decode, encode

DEFAULT_CONTEXT_WINDOW = 3  # fallback; overridden by model.max_order - 1


def _context_window(model: LanguageModel) -> int:
    """Derive context length from the model (n-gram order minus 1)."""
    return max(model.max_order - 1, 1)


def generate(
    model: LanguageModel,
    merges: list[tuple[str, str]],
    vocab: dict[str, int],
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int | None = None,
) -> str:
    """
    Generate text conditioned on prompt using the language model.

    Steps:
      1. Tokenise prompt.
      2. Take last CONTEXT_WINDOW token IDs as context.
      3. Get next-token distribution from model.
      4. Apply temperature, top-k, and top-p; sample.
      5. Append token, slide window.
      6. Stop on <|endoftext|> or max_tokens.

    Args:
        model:       Trained LanguageModel.
        merges:      BPE merges list (from load_tokenizer).
        vocab:       BPE vocab dict (from load_tokenizer).
        prompt:      Input string to condition on.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (> 0).
        top_k:       Keep top-k tokens (0 = disabled).
        top_p:       Nucleus threshold (1.0 = disabled).
        seed:        Random seed for reproducibility.

    Returns:
        Generated string (decoded, not including the original prompt).
    """
    rng = random.Random(seed)
    cw = _context_window(model)

    # Tokenise prompt
    prompt_ids = encode(prompt, merges)
    context = list(prompt_ids[-cw:])
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        ctx_tuple = tuple(context[-cw:])
        dist = model.next_token_distribution(ctx_tuple)

        token = combined_sample(dist, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)

        if token == ENDOFTEXT_ID:
            break

        generated_ids.append(token)
        context.append(token)
        if len(context) > cw:
            context.pop(0)

    return decode(generated_ids, vocab)


def generate_stream(
    model: LanguageModel,
    merges: list[tuple[str, str]],
    vocab: dict[str, int],
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int | None = None,
) -> Iterator[str]:
    """
    Streaming version of generate(): yields one decoded token string at a time.
    Useful for CLI streaming output.
    """
    rng = random.Random(seed)
    cw = _context_window(model)
    prompt_ids = encode(prompt, merges)
    context = list(prompt_ids[-cw:])

    for _ in range(max_tokens):
        ctx_tuple = tuple(context[-cw:])
        dist = model.next_token_distribution(ctx_tuple)

        token = combined_sample(dist, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)

        if token == ENDOFTEXT_ID:
            break

        text = decode([token], vocab)
        yield text

        context.append(token)
        if len(context) > cw:
            context.pop(0)
