# How to answer a deep-dive question

Deep-dive questions ("explain flash attention", "how would you write a fast softmax") are graded less on facts than on structure. A shape that works:

1. **The problem in one sentence**, with its size. *"Attention writes an N × N score matrix; at 8K tokens that's 256 MB per head in FP32, read and written several times."*
2. **The bottleneck**, with a number. *"So it's memory-bound: the FLOPs are cheap, the HBM traffic isn't."*
3. **The idea that removes it.** *"Tile Q, K and V into on-chip memory and use an online softmax so the scores never leave the SM."*
4. **The details that make it correct**: masking, numerical stability, edge tiles, synchronization.
5. **What you would measure** to show it worked, and what you would try next.

Expect follow-ups that move one step sideways: the backward pass, another precision, another GPU generation, a different shape (small batch, long sequence). "I don't know, but here's how I'd find out" followed by a concrete plan is a good answer; a confident wrong one is not.

> [!TIP]
> Practise with a timer: two minutes for the short version, then stop and let the interviewer (or the rubric) pick what to go deeper on.
