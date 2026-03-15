import numpy as np


class SkipGramNegSampling:

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.W = (np.random.random((vocab_size, embed_dim)) - 0.5) / embed_dim

        self.W_prime = np.zeros((vocab_size, embed_dim))

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def train_pair(self, center_id: int, context_id: int, neg_ids: np.ndarray, lr: float) -> float:

        h = self.W[center_id].copy()

        v_pos = self.W_prime[context_id].copy()
        u_pos = np.dot(v_pos, h)
        sig_pos = self.sigmoid(u_pos)

        v_negs = self.W_prime[neg_ids].copy()
        u_negs = v_negs @ h
        sig_negs = self.sigmoid(u_negs)

        eps = 1e-7
        loss = -np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_negs + eps))

        e_pos = sig_pos - 1.0
        e_negs = sig_negs

        EH = e_pos * v_pos + np.sum(e_negs[:, np.newaxis] * v_negs, axis=0)

        self.W_prime[context_id] -= lr * e_pos * h

        for k in range(len(neg_ids)):
            self.W_prime[neg_ids[k]] -= lr * e_negs[k] * h

        self.W[center_id] -= lr * EH

        return loss

    def get_embedding(self, word_id: int) -> np.ndarray:
        return self.W[word_id]

    def get_all_embeddings(self) -> np.ndarray:
        return self.W