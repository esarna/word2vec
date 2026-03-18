import os
import numpy as np
from word2vec.Word2VecTrainer import Word2VecTrainer


def load_conll_sentences(filepath: str, max_sentences: int = 5000) -> list[str]:
    sentences = []
    current_sentence = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
                    if len(sentences) >= max_sentences:
                        break
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[1]
                    if word not in ('-LRB-', '-RRB-', '-LSB-', '-RSB-'):
                        current_sentence.append(word.lower())
    return sentences

conll_path = os.path.join('data', 'train.conll')
if os.path.exists(conll_path):
    large_corpus = load_conll_sentences(conll_path, max_sentences=3000)
    print(f"Loaded {len(large_corpus)} sentences from CoNLL data")
    total_words = sum(len(s.split()) for s in large_corpus)
    print(f"Total words: {total_words}")
    print(f"Sample: {large_corpus[0][:100]}...")
else:
    exit(0)

np.random.seed(42)

trainer_large = Word2VecTrainer(
    corpus=large_corpus,
    embed_dim=100,
    context_size=5,
    num_neg_samples=5,
    learning_rate=0.025,
    num_epochs=3,
    min_count=3,
    subsample_threshold=1e-3,
)

losses_large = trainer_large.train()

for query in ["the", "said", "would", "new", "time", "people", "first"]:
    if query in trainer_large.vocab.word2id:
        trainer_large.most_similar(query, top_n=5)

trainer_large.analogy("are", "you", "i", top_n=5)
trainer_large.analogy("chicago", "city", "country", top_n=5)
trainer_large.analogy("is", "it", "you", top_n=5)
trainer_large.analogy("luxury", "auto", "gold", top_n=5)