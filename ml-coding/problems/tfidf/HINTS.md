## Hint 1
Fix the column order first: `vocab = sorted(set of all tokens)` and a dict from token to column. Insertion order (first seen) is not the same thing.

## Hint 2
Document frequency counts documents, not occurrences: a token that appears three times in one document adds 1 to its `df`. With a count matrix `tf`, that is `(tf > 0).sum(axis=0)`.

## Hint 3
Multiply `tf` by the `idf` row vector (broadcasting), compute each row's L2 norm with `np.linalg.norm(..., axis=1, keepdims=True)`, replace zero norms by 1, and divide.
