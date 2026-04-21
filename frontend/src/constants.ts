/** Static metadata for the 13 language models.
 *
 * Used by ModelCard and StatsPage to display ordered, annotated model cards.
 */

export interface ModelMeta {
  displayName: string;
  architectureNote: string;
  order: number;
}

export const MODEL_META: Record<string, ModelMeta> = {
  char_1gram: {
    displayName: "Char 1-gram",
    architectureNote: "Predicts each character from character frequency alone.",
    order: 1,
  },
  char_2gram: {
    displayName: "Char 2-gram",
    architectureNote: "Predicts each character from the previous 1 character.",
    order: 2,
  },
  char_3gram: {
    displayName: "Char 3-gram",
    architectureNote: "Predicts each character from the previous 2 characters.",
    order: 3,
  },
  char_4gram: {
    displayName: "Char 4-gram",
    architectureNote: "Predicts each character from the previous 3 characters.",
    order: 4,
  },
  char_5gram: {
    displayName: "Char 5-gram",
    architectureNote: "Predicts each character from the previous 4 characters.",
    order: 5,
  },
  word_1gram: {
    displayName: "Word 1-gram",
    architectureNote: "Predicts each word from word frequency alone.",
    order: 6,
  },
  word_2gram: {
    displayName: "Word 2-gram",
    architectureNote: "Predicts each word from the previous 1 word.",
    order: 7,
  },
  word_3gram: {
    displayName: "Word 3-gram",
    architectureNote: "Predicts each word from the previous 2 words.",
    order: 8,
  },
  bpe_1gram: {
    displayName: "BPE 1-gram",
    architectureNote: "Predicts each BPE subword token from token frequency alone.",
    order: 9,
  },
  bpe_2gram: {
    displayName: "BPE 2-gram",
    architectureNote: "Predicts each BPE subword token from the previous 1 token.",
    order: 10,
  },
  bpe_3gram: {
    displayName: "BPE 3-gram",
    architectureNote: "Predicts each BPE subword token from the previous 2 tokens.",
    order: 11,
  },
  feedforward: {
    displayName: "Feedforward NN",
    architectureNote:
      "Predicts the next token using a fixed-size context window and learned embeddings.",
    order: 12,
  },
  transformer: {
    displayName: "Transformer",
    architectureNote:
      "Predicts the next token using causal self-attention over all previous tokens.",
    order: 13,
  },
};

export function getModelMeta(name: string): ModelMeta {
  return (
    MODEL_META[name] ?? {
      displayName: name,
      architectureNote: "Unknown model.",
      order: 99,
    }
  );
}

export function sortedModelNames(
  names: string[],
  sortByAccuracy = false,
  accuracyMap?: Record<string, number>
): string[] {
  return [...names].sort((a, b) => {
    if (sortByAccuracy && accuracyMap) {
      const da = accuracyMap[a] ?? 0;
      const db = accuracyMap[b] ?? 0;
      return db - da;
    }
    return (getModelMeta(a).order ?? 99) - (getModelMeta(b).order ?? 99);
  });
}
