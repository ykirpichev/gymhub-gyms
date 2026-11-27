# Language-model mechanics in NumPy

This track builds the pieces of a language model from their definitions, small enough to write in an interview.

**From counts to vectors.** **TF-IDF** weights a term's count in a document by how rare it is across documents, $\text{tf}(t, d) \cdot \log \frac{N}{\text{df}(t)}$ (libraries add smoothing), so common words matter less. It is still a strong retrieval baseline.

**Attention.** $\operatorname{softmax}(QK^\top / \sqrt{d_k} + M)\,V$. The mask $M$ is 0 where attention is allowed and $-\infty$ where it is not: above the diagonal for **causal** decoding, and on padding positions. Apply it before the softmax, subtract the row max for stability, and decide what a fully masked row returns.

**Tokenization.** **BPE** training counts adjacent symbol pairs across the corpus, merges the most frequent pair everywhere, and repeats. The output is the ordered merge list; encoding new text replays the merges in order. Ties and the order of merges are where implementations differ, so they are spelled out in the exercise.

**Positions.** Sinusoidal encodings give position $p$ the vector with $\sin(p / 10000^{2i/d})$ in even dimensions and $\cos$ in odd ones: each pair of dimensions rotates at its own frequency, and a fixed offset is a linear map.

**Sampling.** From logits: divide by the temperature, then **top-k** keeps the $k$ largest, **top-p** keeps the smallest set of tokens whose probabilities reach $p$ (sorted descending, always keeping the first), renormalize, and sample with the given generator.

**The KV cache.** When decoding token $t$, keys and values of tokens $0 \ldots t-1$ are unchanged, so they are appended to a cache instead of recomputed. Attention for the new token uses one query against the whole cache: $O(t)$ work per step instead of $O(t^2)$.

**Padding and batching.** Sequences of different lengths are padded to a common length with an attention mask that hides the padding. Grouping similar lengths into the same batch (bucketing) wastes less compute on padding.
