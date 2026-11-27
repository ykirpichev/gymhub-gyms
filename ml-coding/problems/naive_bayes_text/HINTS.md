## Hint 1
A product of thousands of probabilities below 1 underflows to 0.0 in float64, and then every class looks the same. Add logarithms instead: `log P(c) + Σ log P(t | c)`.

## Hint 2
Smoothing uses the vocabulary of the **whole** training set, not of one class, so the denominators for each class are `total_c + alpha · |V|`. A token that is in `V` but never occurs in class `c` still gets probability `alpha / (total_c + alpha·|V|)`.

## Hint 3
Count with one `Counter` per class plus a document count per class. Precompute `log P(t | c)` for every `t` in `V`. To score, start each class at its log prior and add the log-likelihood of every in-vocabulary token occurrence. Walk the classes in sorted order and keep the first whose score is within `1e-9` of the maximum.
