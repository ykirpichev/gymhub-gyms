# Generative-AI building blocks

Each exercise in this track is a short formula with a few traps. The formulas:

**LoRA linear layer.** $y = x W^\top + b + \frac{\alpha}{r}\, x A^\top B^\top$, with $W$ frozen, $A$ of shape $r \times d_\text{in}$ and $B$ of shape $d_\text{out} \times r$. Watch the scale $\alpha / r$ and the transposes; with $B = 0$ the output equals the base layer.

**Contrastive loss (InfoNCE).** Normalize a batch of query and key embeddings, compute the similarity matrix $S = QK^\top / \tau$, and apply cross-entropy where the right key for query $i$ is key $i$. The other rows of the batch act as negatives, so bigger batches give harder training signal. CLIP averages the loss over both directions.

**Retrieval for RAG.** Embed the query and the chunks, score by cosine similarity (a dot product after normalization), take the top $k$ with a deterministic tie-break, and often apply a minimum score. In a real system: chunking with overlap, hybrid scores with keyword search (BM25), and a re-ranker on the top results.

**DPO loss.** For a chosen response $y_w$ and a rejected one $y_l$, with summed log-probabilities under the policy and a frozen reference:

$$\mathcal{L} = -\log \sigma\big(\beta\,[(\log \pi(y_w) - \log \pi_\text{ref}(y_w)) - (\log \pi(y_l) - \log \pi_\text{ref}(y_l))]\big).$$

Compute $-\log \sigma(z)$ stably as $\operatorname{softplus}(-z)$, never as `log(1 / (1 + exp(-z)))`.

**Classifier-free guidance.** $\hat{\epsilon} = \epsilon_u + w(\epsilon_c - \epsilon_u)$: one extrapolation, applied per denoising step. $w = 0$ ignores the prompt, $w = 1$ is plain conditional, $w > 1$ amplifies the prompt.

> [!TIP]
> For each of these, interviewers often ask *why* as the follow-up: why LoRA starts from $B = 0$, why in-batch negatives work, why DPO needs a reference model, why guidance trades diversity for prompt adherence. The LLM fundamentals course covers the reasoning.
