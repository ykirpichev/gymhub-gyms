# Diversify results with maximal marginal relevance

Write `solve(relevance, similarity, k, lam)` that picks a diverse top `k`.

- `relevance`: `n` floats, the ranker's score for each candidate. `n ≥ 1`.
- `similarity`: an `n × n` list of lists of floats; `similarity[i][j]` is how alike candidates `i` and `j` are (symmetric).
- `k`: an int `≥ 1`. `lam`: a float in `[0, 1]`.

Build the result greedily. At each step, for every candidate `i` not yet chosen, compute

```
mmr(i) = lam * relevance[i] − (1 − lam) * max(similarity[i][j] for j already chosen)
```

where the max is `0` while nothing is chosen. Append the candidate with the largest `mmr`; on a tie, the lowest index. Stop after `min(k, n)` picks. Return the list of chosen indices in the order they were picked.

```
relevance  = [0.9, 0.85, 0.5]
similarity = [[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]]
solve(relevance, similarity, 2, 0.5)  ->  [0, 2]
```

First pick: candidate 0 (`0.45`). Then candidate 1 scores `0.425 − 0.475 = −0.05` and candidate 2 scores `0.25 − 0.05 = 0.2`, so 2 wins although it is less relevant.

#### What interviewers look for

- Why re-ranking exists: the ranker scores items independently, so it cannot see redundancy within the list.
- A correct greedy loop, and keeping the running max similarity per candidate so each step is `O(n)`, not `O(n·k)`.
- How `lam` would be chosen (online, with diversity and engagement metrics), and what "similar" means (embeddings, same creator, same topic).

#### Follow-ups

- How would you enforce a hard rule such as "at most two items from the same creator" together with MMR?
- Similarity comes from item embeddings with `n = 1,000`. What does this cost per request, and how could you cut it?
- What alternatives to MMR do you know (determinantal point processes, sliding-window rules)?
