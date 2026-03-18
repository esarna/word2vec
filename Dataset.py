import re
from collections import Counter
import numpy as np


class Dataset:

    def __init__(self, corpus: list[str], min_count: int = 1):
        # tokenizing sentences
        self.raw_words = []
        for sentence in corpus:
            self.raw_words.extend(re.findall(r'[a-zA-Z\']+', sentence.lower()))

        # counting word frequencies
        counter = Counter(self.raw_words)
        self.word2id = {}
        self.id2word = {}
        self.word_counts = {}  # id -> count

        # indexing
        idx = 0
        for word, count in counter.items():
            if count >= min_count:
                self.word2id[word] = idx
                self.id2word[idx] = word
                self.word_counts[idx] = count
                idx += 1

        # filtering raw_words (which count >= min_count)
        self.words = [w for w in self.raw_words if w in self.word2id]
        self.word_ids = [self.word2id[w] for w in self.words]

        self.self_size = len(self.word2id)
        self.total_words = sum(self.word_counts.values())
        self.keep_ids = []

    def subsample_words(self, threshold: float = 1e-3) -> list[int]:
        if threshold <= 0:
            self.keep_ids = self.word_ids
            return self.keep_ids  # keep all words

        for word_id in self.word_ids:
            freq = self.word_counts[word_id] / self.total_words
            # probability of keeping this word
            p_keep = min(1.0, np.sqrt(threshold / freq))
            if np.random.random() < p_keep:
                self.keep_ids.append(word_id)

        return self.keep_ids

    def generate_skipgram_pairs(self, context_size: int) -> list[tuple[int, int]]:
        pairs = []
        n = len(self.keep_ids)

        for i in range(n):
            center = self.keep_ids[i]
            # randomly reduce the effective window
            R = np.random.randint(1, context_size + 1)
            for j in range(max(0, i - R), min(n, i + R + 1)):
                if j != i:
                    pairs.append((center, self.keep_ids[j]))

        return pairs