"""
Demo setup: download public-domain books from Project Gutenberg and
Wikipedia articles, then train a BPE tokenizer + 5-gram language model
so the web UI works out of the box.

Run once before starting the server:
    python3 demo/setup_demo.py

To rebuild from scratch (e.g. after adding more sources):
    rm demo/demo_lm.pkl && python3 demo/setup_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "demo"
DEMO_MODEL_PATH = DEMO_DIR / "demo_lm.pkl"
DEMO_MERGES_PATH = DEMO_DIR / "demo_merges.json"
DEMO_VOCAB_PATH = DEMO_DIR / "demo_vocab.json"
DEMO_CORPUS_PATH = DEMO_DIR / "demo_corpus.txt"

# ---------------------------------------------------------------------------
# Built-in seed corpus — diverse English sentences covering many styles
# (repeated 8× so the model has enough data even without network access)
# ---------------------------------------------------------------------------

CORPUS = """\
The quick brown fox jumps over the lazy dog near the river bank.
Science is the systematic study of the natural world through observation and experiment.
In the beginning was the word and the word was with God and the word was God.
It was the best of times it was the worst of times it was the age of wisdom.
To be or not to be that is the question whether it is nobler in the mind to suffer.
All happy families are alike each unhappy family is unhappy in its own way.
It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.
Call me Ishmael some years ago never mind how long precisely having little or no money in my purse.
The only way to do great work is to love what you do if you have not found it yet keep looking.
Mathematics is the language in which God wrote the universe and numbers are the alphabet.
The brain is wider than the sky for put them side by side the one the other will contain.
Language is the road map of a culture it tells you where its people come from and where they are going.
The greatest glory in living lies not in never falling but in rising every time we fall.
In three words I can sum up everything I have learned about life it goes on.
Not all those who wander are lost the old that is strong does not wither deep roots are not reached by frost.
The future belongs to those who believe in the beauty of their dreams and work to achieve them.
Knowledge is power information is liberating education is the premise of progress in every society.
Two roads diverged in a yellow wood and I sorry I could not travel both and be one traveler long I stood.
Water is the driving force of all nature and life cannot exist without it.
The mind is everything what you think you become and what you become shapes the world around you.
Time is what we want most but what we use worst and yet we must learn to use it wisely.
Books are a uniquely portable magic that lets you live thousands of lives before your own.
The universe is not required to be in perfect harmony with human ambition and yet we reach for stars.
Art enables us to find ourselves and lose ourselves at the same time in creative expression.
Love is patient love is kind it does not envy it does not boast it is not proud.
The pen is mightier than the sword and ideas written down outlast empires built by force.
Music gives a soul to the universe wings to the mind flight to the imagination and life to everything.
History is written by the victors but remembered by the survivors who carry stories forward.
Every child is an artist the problem is how to remain an artist once we grow up.
The sea never changes and its works for all the talk of men are slight in comparison.
Words are free it is how you use them that may cost you everything in the end.
In science the important thing is to modify and change ones ideas as science advances forward.
The measure of intelligence is the ability to change and adapt when faced with new information.
Philosophy is a battle against the bewitchment of our intelligence by means of our language.
The good life is one inspired by love and guided by knowledge toward wisdom and understanding.
Beauty is in the eye of the beholder and truth is in the mind of the thinker.
Learning never exhausts the mind but fuels it with curiosity wonder and the desire to know more.
A reader lives a thousand lives before he dies the man who never reads lives only one.
Nature is not a place to visit it is home and we are all children of the earth.
The only true wisdom is in knowing you know nothing and always seeking to learn more.
Science and art are not opposed they are both branches of the same great tree of human creativity.
Language shapes thought and thought shapes language in an endless dance of meaning and understanding.
The human brain contains roughly one hundred billion neurons each connected to thousands of others.
Consciousness emerges from the complex interaction of billions of neurons firing in coordinated patterns.
Statistical language models learn the probability distribution of sequences of words or tokens.
Natural language processing enables computers to understand generate and respond to human language.
Deep learning neural networks learn hierarchical representations from large amounts of training data.
The transformer architecture revolutionized natural language processing with attention mechanisms.
Byte pair encoding creates a subword vocabulary by iteratively merging the most frequent pairs.
The Markov assumption states that the future depends only on the present not the past.
N-gram language models assign probabilities to sequences based on the preceding context window.
Kneser-Ney smoothing improves language model estimates by redistributing probability mass cleverly.
The perplexity of a language model measures how well it predicts a sample of text data.
Entropy measures the average amount of information contained in a message or probability distribution.
Cross entropy between two distributions measures how different one distribution is from another.
Sampling temperature controls the randomness of text generation from a language model distribution.
Top-k sampling limits the vocabulary to the k most likely tokens at each generation step.
Nucleus sampling keeps only the smallest set of tokens whose cumulative probability exceeds p.
Tokenization is the process of splitting text into smaller units called tokens for processing.
Words sentences and paragraphs can all be considered units of language at different levels.
Grammar describes the rules by which words are combined to form meaningful sentences in a language.
Syntax refers to the arrangement of words and phrases to create well-formed sentences.
Semantics is the branch of linguistics concerned with meaning in language and logic.
Pragmatics studies the ways in which context contributes to meaning beyond literal words.
The sky is blue because of Rayleigh scattering of sunlight by the atmosphere above us.
Stars are born in nebulae vast clouds of gas and dust scattered across galaxies far away.
The Earth orbits the sun at a distance of about one astronomical unit taking one year.
Light travels at approximately three hundred thousand kilometers per second through empty space.
Gravity is the weakest of the four fundamental forces but acts over infinite distances.
Quantum mechanics describes the behavior of matter and energy at very small scales precisely.
Evolution by natural selection is the central unifying theory of modern biological science.
DNA contains the genetic instructions used in the development and functioning of all living organisms.
The cell is the basic structural and functional unit of all known living organisms on Earth.
Photosynthesis is the process by which plants use sunlight water and carbon dioxide to produce food.
""" * 8  # repeat 8× to give the model enough data for meaningful n-gram statistics


# ---------------------------------------------------------------------------
# Corpus downloaders (imported from sub-modules for modularity)
# ---------------------------------------------------------------------------

from demo.corpus_gutenberg import download_all as _download_all_gutenberg
from demo.corpus_wikipedia import download_all as _download_all_wikipedia


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if DEMO_MODEL_PATH.exists() and DEMO_MERGES_PATH.exists():
        print("Demo model already exists. Delete demo/demo_lm.pkl to retrain.")
        return

    print("=" * 60)
    print("  Building demo model (vocab=4000, 5-gram + Kneser-Ney)")
    print("  Sources: built-in seed + Gutenberg books + Wikipedia articles")
    print("=" * 60)

    # Download Gutenberg books
    print("\n[1/2] Downloading Project Gutenberg books…")
    gutenberg_text = _download_all_gutenberg()
    gutenberg_mb = len(gutenberg_text) / 1_000_000

    # Download Wikipedia articles
    print("\n[2/2] Downloading Wikipedia articles…")
    wikipedia_text = _download_all_wikipedia()
    wikipedia_mb = len(wikipedia_text) / 1_000_000

    # Combine all sources
    full_corpus = (
        CORPUS
        + "\n\n"
        + gutenberg_text
        + "\n\n# ── Wikipedia articles ─────────────────────────────────────────\n\n"
        + wikipedia_text
    )
    DEMO_CORPUS_PATH.write_text(full_corpus, encoding="utf-8")
    total_mb = len(full_corpus) / 1_000_000
    print(
        f"\nCorpus breakdown:"
        f"\n  Gutenberg:  {gutenberg_mb:.1f} MB"
        f"\n  Wikipedia:  {wikipedia_mb:.1f} MB"
        f"\n  Total:      {total_mb:.1f} MB  ({len(full_corpus.splitlines()):,} lines)"
    )

    # Train tokenizer
    from tokenizer.bpe import save_tokenizer, train_bpe
    print("\nTraining BPE tokenizer (vocab=4000)…")
    merges, vocab = train_bpe(DEMO_CORPUS_PATH, vocab_size=4000)
    save_tokenizer(merges, vocab, DEMO_MERGES_PATH, DEMO_VOCAB_PATH)

    # Count n-grams
    from tokenizer.bpe import encode
    from model.ngram_counter import count_ngrams, prune_counts
    print("\nCounting n-grams…")
    ids = encode(full_corpus, merges)
    print(f"  {len(ids):,} tokens")
    counts = count_ngrams(iter(ids), max_n=5)
    counts = prune_counts(counts, min_count=2)
    for n, c in counts.items():
        print(f"  order {n}: {len(c):,} n-grams")

    # Build and save model
    from model.language_model import LanguageModel
    print("\nBuilding language model…")
    lm = LanguageModel.from_counts(counts, len(vocab))
    lm.save(DEMO_MODEL_PATH)

    print("\n✓ Demo model ready.")
    print(f"  Model:  {DEMO_MODEL_PATH}")
    print(f"  Vocab:  {len(vocab)} tokens")


if __name__ == "__main__":
    main()
