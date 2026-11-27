# Multinomial Naive Bayes for text

Write `solve(train_docs, train_labels, test_docs, alpha)` that trains a multinomial Naive Bayes classifier and scores new documents.

- `train_docs`: a list of `N ≥ 1` documents; each document is a list of string tokens (possibly repeated).
- `train_labels`: `N` string class labels.
- `test_docs`: a list of `M` tokenized documents (any may be empty).
- `alpha`: the smoothing constant. If `alpha <= 0`, raise `ValueError`.

**Training.** The classes are the distinct labels, **in sorted order**. `V` is the set of all tokens in the training documents. For class `c`:

```
P(c)     = (number of training documents with label c) / N
P(t | c) = (count of token t in class-c documents + alpha) / (total tokens in class-c documents + alpha · |V|)
```

**Scoring.** For a test document, the score of class `c` is its unnormalized log-posterior:

```
score(c) = log P(c) + Σ log P(t | c)   over every token occurrence t in the document that is in V
```

Tokens not in `V` are ignored. A token that appears twice counts twice. The prediction is the class with the highest score. Scores within `1e-9` of the highest are ties, and a tie goes to the class that sorts first.

Return a tuple `(predictions, scores)`:

- `predictions`: a list of `M` labels.
- `scores`: an `M × K` NumPy array (or list of lists), with columns in sorted class order.

```
train_docs   = [["win", "money", "now"], ["win", "prize"], ["meeting", "today"], ["lunch", "today", "now"]]
train_labels = ["spam", "spam", "ham", "ham"]
solve(train_docs, train_labels, [["win", "now"], ["today"]], 1.0)
-> (["spam", "ham"], array([[-4.969813, -3.871201],
                            [-2.079442, -3.178054]]))
```

The columns are `["ham", "spam"]`, and `|V| = 7`. For example, `P(win | spam) = (2 + 1) / (5 + 7) = 0.25`.

Test documents can have thousands of tokens.

**Interview focus:** working in log space so long documents do not underflow, Laplace smoothing over the training vocabulary, ignoring unseen tokens, the independence assumption behind the sum, and deterministic tie-breaking.

**Follow-ups** to answer out loud once your solution passes:

- Why does naive Bayes classify well even though the independence assumption is false?
- How would you handle a word never seen in training?
- Multinomial or Bernoulli naive Bayes: what is the difference?
