import re
from collections import Counter


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

        self.vocab_size = len(self.word2id)
        self.total_words = sum(self.word_counts.values())

    def __len__(self):
        return self.vocab_size