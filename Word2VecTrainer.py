import time
import numpy as np
from word2vec.Dataset import Dataset
from word2vec.NegativeSamplingTable import NegativeSamplingTable
from word2vec.SkipGramNegSampling import SkipGramNegSampling


class Word2VecTrainer:

    def __init__(self, corpus: list[str], embed_dim: int = 100, context_size: int = 2,
                 num_neg_samples: int = 5, learning_rate: float = 0.025,
                 num_epochs: int = 5, min_count: int = 1,
                 subsample_threshold: float = 1e-3):
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.num_neg_samples = num_neg_samples
        self.initial_lr = learning_rate
        self.num_epochs = num_epochs
        self.subsample_threshold = subsample_threshold

        self.vocab = Dataset(corpus, min_count=min_count)
        print(f"Vocabulary size: {self.vocab.self_size}")
        print(f"Total words in corpus: {self.vocab.total_words}")

        self.neg_table = NegativeSamplingTable(self.vocab)

        self.model = SkipGramNegSampling(self.vocab.self_size, embed_dim)

    def train(self) -> list[float]:
        epoch_losses = []

        for epoch in range(self.num_epochs):
            t0 = time.time()

            self.vocab.subsample_words(self.subsample_threshold)

            pairs = self.vocab.generate_skipgram_pairs(self.context_size)
            np.random.shuffle(pairs)

            total_pairs = len(pairs)
            total_loss = 0.0
            num_pairs = 0
            lr = 0

            for step, (center_id, context_id) in enumerate(pairs):
                progress = (epoch * total_pairs + step) / (self.num_epochs * total_pairs)
                lr = self.initial_lr * max(1.0 - progress, 0.0001)

                neg_ids = self.neg_table.sample(self.num_neg_samples, exclude=context_id)

                loss = self.model.train_pair(center_id, context_id, neg_ids, lr)
                total_loss += loss
                num_pairs += 1

            avg_loss = total_loss / max(num_pairs, 1)
            elapsed = time.time() - t0
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Pairs: {num_pairs} | "
                  f"Time: {elapsed:.1f}s | "
                  f"LR: {lr:.6f}")

        return epoch_losses

    def most_similar(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:

        if word not in self.vocab.word2id:
            print(f"Word '{word}' not in vocabulary")
            return []

        word_id = self.vocab.word2id[word]
        word_vec = self.model.get_embedding(word_id)

        embeddings = self.model.get_all_embeddings()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms

        word_norm = word_vec / max(np.linalg.norm(word_vec), 1e-8)
        similarities = normalized @ word_norm

        top_ids = (-similarities).argsort()
        results = []
        for idx in top_ids:
            if idx != word_id:
                results.append((self.vocab.id2word[idx], float(similarities[idx])))
            if len(results) >= top_n:
                break

        return results

    def analogy(self, a: str, b: str, c: str, top_n: int = 5) -> list[tuple[str, float]]:

        for w in [a, b, c]:
            if w not in self.vocab.word2id:
                print(f"Word '{w}' not in vocabulary")
                return []

        exclude = {self.vocab.word2id[a], self.vocab.word2id[b], self.vocab.word2id[c]}
        target_vec = (
            self.model.get_embedding(self.vocab.word2id[b])
            - self.model.get_embedding(self.vocab.word2id[a])
            + self.model.get_embedding(self.vocab.word2id[c])
        )

        embeddings = self.model.get_all_embeddings()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms

        target_norm = target_vec / max(np.linalg.norm(target_vec), 1e-8)
        similarities = normalized @ target_norm

        top_ids = (-similarities).argsort()
        results = []
        for idx in top_ids:
            if idx not in exclude:
                results.append((self.vocab.id2word[idx], float(similarities[idx])))
            if len(results) >= top_n:
                break

        return results