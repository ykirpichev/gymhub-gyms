# Inference, decoding and prompting

**Two phases.** **Prefill** processes the whole prompt in one pass: large matmuls, compute-bound, and it sets the time to first token. **Decode** generates one token per step: every step reads all the weights for a little math, so it is memory-bandwidth-bound, and it sets the time per output token. Most serving optimizations attack decode:

- **Batching** many sequences (continuously, admitting and retiring requests every step) reuses each weight read.
- **Quantization** shrinks the bytes: weights to 8 or 4 bits (GPTQ, AWQ), sometimes activations and the KV cache to 8 bits. Weight-only 4-bit speeds up memory-bound decode the most.
- **Speculative decoding**: a small draft model proposes several tokens and the large model checks them in one pass, accepting a prefix with a rule that keeps the output distribution exact.
- **Paged KV cache** and **prefix caching** cut memory waste and reuse a shared system prompt.

**Decoding knobs** turn logits $z$ into a choice:

- **Greedy** takes the argmax: deterministic, prone to repetition.
- **Temperature** samples from $\operatorname{softmax}(z / T)$. $T < 1$ sharpens the distribution, $T > 1$ flattens it, $T \to 0$ is greedy.
- **Top-k** keeps the $k$ most likely tokens; **top-p** (nucleus) keeps the smallest set whose probabilities sum to at least $p$, so its size adapts to the model's confidence.
- **Repetition and frequency penalties** lower the scores of tokens already used.
- **Beam search** keeps the $b$ best partial sequences; good for translation, bland for open-ended text.

**Prompting.** Give the model what a careful human expert would need: the task, the context, the format, and examples (**few-shot**). **Chain of thought** ("think step by step", or a reasoning model) helps multi-step problems. **Retrieval-augmented generation (RAG)** puts fresh or private documents into the prompt at query time. Knowledge that changes often or must be cited belongs in retrieval; behaviour, format and style are better taught by fine-tuning.
