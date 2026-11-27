# Learn and apply byte-pair encoding

Write `solve(corpus, num_merges, words)`.

- `corpus`: a list of words (strings). A word that appears `n` times has frequency `n`.
- `num_merges`: an int; raise `ValueError` if it is negative.
- `words`: a list of words to encode; they may contain characters never seen in the corpus.

**Learning.** Split each distinct corpus word into its characters followed by the end-of-word symbol `"</w>"` (`"low"` → `["l", "o", "w", "</w>"]`). Then repeat `num_merges` times:

1. Count every adjacent symbol pair in every word, adding the word's frequency for each occurrence (overlapping occurrences all count: `a a a` has the pair `(a, a)` twice).
2. Pick the pair with the highest count. On a tie, pick the **smallest pair** by Python tuple comparison of the two strings, `(a, b) < (c, d)`.
3. Record it and, in every word, replace occurrences of that pair with the joined symbol `a + b`, scanning left to right without overlap (`a a a` → `aa a`).

Stop early if no word has two or more symbols left.

**Encoding.** Split each word in `words` the same way, then apply every learned merge **in the order it was learned**, each one with the left-to-right rule above. An empty word encodes as `["</w>"]`.

Return a tuple `(merges, encodings)`: `merges` is the list of learned pairs `(a, b)` in order (tuples or 2-element lists), and `encodings` holds one list of symbols per word in `words`.

```
solve(["low", "low", "lower", "newest", "newest", "widest"], 4, ["lowest", "new", "slow"])
-> ([("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o")],
    [["lo", "w", "est</w>"], ["n", "e", "w", "</w>"], ["s", "lo", "w", "</w>"]])
```

In the first round six pairs tie with count 3, among them `(l, o)`, `(w, e)` and `(e, s)`; `("e", "s")` is the smallest.

**Interview focus:** frequency-weighted pair counts, deterministic tie-breaking, non-overlapping merges, and encoding unseen words by replaying merges in order.

## This version and the other one

This is the interview version: a small corpus, the `</w>` end-of-word symbol, and encoding new words with the merges. The algo gym's `bpe_train` is the algorithms version: no end-of-word marker, a corpus of 300,000 words, and a time limit that needs incremental counts and a heap, so the two produce different merge lists on the same text.

**Follow-ups** to answer out loud once your solution passes:

- Why do production tokenizers work on bytes rather than characters?
- How would you make training fast on a large corpus? (Update only the pair counts a merge touches.)
- What does a larger vocabulary cost the model?
