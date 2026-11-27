# TF-IDF matrix

Write `solve(docs)` for a corpus of `N` already-tokenized documents.

- `docs`: a list of `N` documents; each document is a list of string tokens. `N ≥ 0`, and a document may be empty. Tokens are compared exactly (case-sensitive).

Return a tuple `(vocab, matrix)`:

- `vocab`: a list of the `V` distinct tokens in the corpus, **sorted** in ascending Python string order. Column `j` of the matrix belongs to `vocab[j]`.
- `matrix`: an `N × V` float NumPy array, built as follows:
  - `tf[i, j]` = the raw number of times `vocab[j]` occurs in document `i`.
  - `df[j]` = the number of **documents** that contain `vocab[j]` at least once.
  - `idf[j] = ln((1 + N) / (1 + df[j])) + 1` (smoothed, natural log).
  - Row `i` is `tf[i] * idf`, divided by its L2 norm.

An empty document gives a row of zeros (do not divide by zero). With no tokens at all, `V = 0` and the matrix has shape `(N, 0)`; with no documents it has shape `(0, 0)`.

```
solve([["the", "cat", "sat"], ["the", "dog"], ["cat", "cat"]])
-> (["cat", "dog", "sat", "the"],
    array([[0.5179, 0.    , 0.6809, 0.5179],
           [0.    , 0.796 , 0.    , 0.6053],
           [1.    , 0.    , 0.    , 0.    ]]))
```

Here `N = 3`, `idf(cat) = idf(the) = ln(4/3) + 1 ≈ 1.2877` and `idf(dog) = idf(sat) = ln(2) + 1 ≈ 1.6931`.

**Interview focus:** token counts versus document frequency, IDF smoothing, a deterministic vocabulary order, and normalizing rows without dividing by zero.

**Follow-ups** to answer out loud once your solution passes:

- Why does IDF use a log?
- How would you compute TF-IDF over a corpus that does not fit in memory?
- When does BM25 beat plain TF-IDF?
