import numpy as np
from word2vec.Word2VecTrainer import Word2VecTrainer

EMBEDDING_DIM = 100
CONTEXT_SIZE = 2
NUM_NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.025
NUM_EPOCHS = 5
MIN_COUNT = 1
SUBSAMPLING_THRESHOLD = 1e-3


corpus = [
    "we are what we repeatedly do excellence then is not an act but a habit",
    "the only way to do great work is to love what you do",
    "if you can dream it you can do it",
    "do not wait to strike till the iron is hot but make it hot by striking",
    "whether you think you can or you think you cannot you are right",
]

np.random.seed(42)

trainer = Word2VecTrainer(
    corpus=corpus,
    embed_dim=EMBEDDING_DIM,
    context_size=CONTEXT_SIZE,
    num_neg_samples=NUM_NEGATIVE_SAMPLES,
    learning_rate=LEARNING_RATE,
    num_epochs=50,
    min_count=MIN_COUNT,
    subsample_threshold=0,
)

losses = trainer.train()


for query in ["do", "you", "work", "can"]:
    if query in trainer.vocab.word2id:
        similar = trainer.most_similar(query, top_n=5)
        print(f"\nMost similar to '{query}':")
        for word, sim in similar:
            print(f"  {word:15s} {sim:.4f}")