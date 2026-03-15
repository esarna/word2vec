import re
from collections import Counter
import numpy as np


class Dataset:

    def __init__(self, corpus: list[str], min_count: int = 1):
        # tokenizing sentences
        self.raw_words = []
        for sentence in corpus:
            tokens = re.findall(r'[a-zA-Z\']+', sentence.lower())
            self.raw_words.extend(tokens)

        # counting word frequencies
        counter = Counter(self.raw_words)
        self.word2id = {}
        self.id2word = {}
        self.word_counts = {}  # id -> count
        idx = 0
        for word, count in counter.items():
            if count >= min_count:
                self.word2id[word] = idx
                self.id2word[idx] = word
                self.word_counts[idx] = count
                idx += 1

        # filtering raw_words
        self.words = [w for w in self.raw_words if w in self.word2id]
        self.word_ids = [self.word2id[w] for w in self.words]

        self.self_size = len(self.word2id)
        self.total_words = sum(self.word_counts.values())
        self.keep_ids = []

    def __len__(self):
        return self.self_size

    def subsample_words(self, threshold: float = 1e-3) -> list[int]:
        if threshold <= 0:
            return list(self.word_ids)  # keep all words

        total = self.total_words
        for wid in self.word_ids:
            freq = self.word_counts[wid] / total
            # probability of keeping this word
            p_keep = min(1.0, np.sqrt(threshold / freq))
            if np.random.random() < p_keep:
                self.keep_ids.append(wid)

        return self.keep_ids

    def generate_skipgram_pairs(self, context_size: int) -> list[tuple[int, int]]:
        pairs = []
        n = len(self.keep_ids)
        for i in range(n):
            center = self.keep_ids[i]
            # Randomly reduce the effective window (Mikolov 2013a, Section 3.2)
            R = np.random.randint(1, context_size + 1)
            for j in range(max(0, i - R), min(n, i + R + 1)):
                if j != i:
                    pairs.append((center, self.keep_ids[j]))
        return pairs