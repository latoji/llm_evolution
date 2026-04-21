# Token-Level N-Gram Language Model

A BPE tokenizer combined with a Modified Kneser-Ney smoothed n-gram language model, built from scratch in Python.

## Architecture

```
Data Pipeline → BPE Tokenizer → N-Gram Counts → Smoothed LM → Text Generator
```

- **Tokenizer**: Byte Pair Encoding (GPT-2 style, vocab_size=8000), no external libraries
- **N-gram model**: 4-gram with Modified Kneser-Ney smoothing and backoff
- **Generator**: Markov chain sampling with temperature, top-k, and top-p controls

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download corpus
python data/download_gutenberg.py
python data/download_wikipedia.py

# 2. Clean and split
python data/clean.py
python data/split.py

# 3. Train tokenizer
python tokenizer/train_tokenizer.py

# 4. Count n-grams
python model/count_ngrams.py

# 5. Build language model
python model/build_model.py

# 6. Generate text
python generate/cli.py --prompt "The meaning of" --tokens 100 --temp 0.7
```

## Project Structure

```
├── data/           Corpus acquisition and cleaning
├── tokenizer/      BPE tokenizer (train + encode/decode)
├── model/          N-gram counts, smoothing, language model
├── generate/       Text generation CLI
├── eval/           Perplexity evaluation and ablation experiments
└── tests/          pytest test suite
```

## Known Limitations (by design)

- No long-range coherence (3-token context window)
- No semantic understanding (pure statistical pattern matching)
- Memory-heavy n-gram tables (exponential growth with context length)
- Gutenberg corpus may produce Victorian-sounding output
