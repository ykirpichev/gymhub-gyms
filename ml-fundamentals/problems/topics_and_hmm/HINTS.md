## Hint 1
LSA is a matrix factorization of counts, not a labeled classifier. An HMM has two tables: one for state-to-state, one for state-to-observation.

## Hint 2
Viterbi keeps a single winning path into each state. The algorithm that adds up every path is a different one.
