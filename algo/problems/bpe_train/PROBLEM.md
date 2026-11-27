# Train a BPE tokenizer

**Level:** medium   **Teaches:** byte-pair encoding, incremental counting, priority queues with lazy deletion

## Task

Learn the merge list of a byte-pair-encoding (BPE) tokenizer from a text corpus. This is pure Python: no tensors, no GPU.

1. Split the text into words with `text.split()` (any run of whitespace separates words). Identical words are grouped and counted.
2. Each word starts as a sequence of single characters. Merges never cross word boundaries.
3. Repeat up to `num_merges` times: count every adjacent symbol pair inside every word, weighted by how often the word occurs. Pick the pair with the highest count; break ties with the smallest `(a, b)` under Python's tuple ordering of strings. Record it, then replace every occurrence of `a` followed by `b` in every word with the single symbol `a + b`, scanning each word left to right (so `a a a` with pair `(a, a)` becomes `aa a`).
4. Stop early if no word has two or more symbols left.

Return the merges in the order they were learned.

## Arguments

| Name         | Type   | Role                                                            |
| ------------ | ------ | --------------------------------------------------------------- |
| `text`       | `str`  | the corpus                                                      |
| `num_merges` | `int`  | maximum number of merges to learn                               |
| returns      | `list` | merges, each a pair of strings `(a, b)` (tuple or 2-item list)  |

## Formula

```
count(a, b) = sum over distinct words w of  freq(w) * (number of adjacent positions in w holding a, b)
best        = argmax count, ties -> min (a, b)
```

## Example

```
text = "low low lower newest"   num_merges = 4
words: low ×2, lower ×1, newest ×1
step 1: (l,o)=3 (o,w)=3 (w,e)=2 (e,r)=1 (n,e)=1 (e,w)=1 (e,s)=1 (s,t)=1
        tie at 3 -> ('l','o') < ('o','w')        merge ('l','o'):  lo w, lo w e r, n e w e s t
step 2: (lo,w)=3                                 merge ('lo','w')
step 3: all remaining pairs have count 1 -> smallest is ('e','r')
step 4: next smallest with count 1 is ('e','s')
result = [('l','o'), ('lo','w'), ('e','r'), ('e','s')]
```

## How to think about it

The direct approach recounts every pair in every word after every merge. That is fine for the small tests and far too slow for the benchmark: most merges touch only a small fraction of the words. Keep the pair counts across steps, plus an index from each pair to the words that contain it. After a merge, only those words change: subtract their old pairs, rewrite them, add their new pairs. Finding the best pair each step is the other cost; scanning all pairs every time adds up, so consider a heap keyed by `(-count, pair)` where outdated entries are skipped when popped.

## Tests

The example, an empty text, words of one character, a corpus that runs out of pairs before `num_merges`, overlapping runs like `aaaa`, ties, tabs/newlines/non-ASCII characters, and a generated corpus of 3,000 words. Benchmark input: a generated corpus of 300,000 words (about 18,000 distinct) and `num_merges = 2000`, with a 5 s time limit (the reference takes about 0.5 s). Recomputing every count after each merge takes tens of seconds.

## This version and the other one

This is the algorithms version: the point is making each merge cheap on a large corpus (incremental counts and a heap) under a time limit. Words are plain character sequences with no end-of-word marker. ML coding's `byte_pair_encoding` is the interview version: a small corpus, an end-of-word symbol `</w>` (so a pair at the end of a word is a different symbol from the same pair inside one), and it also encodes new words with the learned merges.
