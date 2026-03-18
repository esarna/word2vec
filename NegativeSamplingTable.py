import numpy as np
from word2vec.Dataset import Dataset


class NegativeSamplingTable:

    def __init__(self, data: Dataset, table_size: int = 10_000_000):
        self.table_size = table_size

        # P_n(w) = f(w)^(3/4) / SUM
        power = 0.75
        counts = np.array([data.word_counts[i] for i in range(data.self_size)], dtype=np.float64)
        powered = counts ** power
        # creating normalized distribution
        self.probs = powered / powered.sum()

        # creating sampling table according to prob
        self.table = np.zeros(table_size, dtype=np.int64)
        idx = 0
        cumulative = 0.0
        for wid in range(data.self_size):
            cumulative += self.probs[wid]
            while idx < table_size and idx / table_size < cumulative:
                self.table[idx] = wid
                idx += 1

    def sample(self, count: int, exclude: int) -> np.ndarray:

        neg_samples = []
        while len(neg_samples) < count:
            candidates = self.table[np.random.randint(0, self.table_size, size=count * 2)]
            for c in candidates:
                if c != exclude and len(neg_samples) < count:
                    neg_samples.append(c)
        return np.array(neg_samples, dtype=np.int64)