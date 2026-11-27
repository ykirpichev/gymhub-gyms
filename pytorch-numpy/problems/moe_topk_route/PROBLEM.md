# Top-k routing for a mixture of experts

A mixture-of-experts layer sends each token to its `k` highest-scoring experts. Write `solve(router_logits, expert_outputs, k)`:

- `router_logits`: `T × E`, one row of expert scores per token.
- `expert_outputs`: `E × T × D`: what each expert would output for each token (in a real layer, only the routed ones are computed; here all are given so you can focus on routing).
- `k`: experts per token, `1 ≤ k ≤ E`.

For each token: take the `k` experts with the largest logits (ties broken by the **lower** expert index), compute the softmax of their `k` logits (the gates), and output the gate-weighted sum of those experts' outputs for that token. Also count how many tokens each expert receives.

Return a tuple `(output, experts, gates, load)`:

- `output`: `T × D` array.
- `experts`: `T × k` int array, each row in descending order of logit (ties: lower index first).
- `gates`: `T × k` array matching `experts`.
- `load`: length-`E` int array of tokens per expert.

```
solve([[2, 1, 0]], [[[1, 0]], [[0, 1]], [[5, 5]]], 2)
-> (array([[0.7311, 0.2689]]), array([[0, 1]]), array([[0.7311, 0.2689]]), array([1, 1, 0]))
```

#### What interviewers look for

- A stable, tie-breaking top-k: `np.argsort(-logits, axis=1, kind="stable")[:, :k]`.
- Gathering with `take_along_axis` and the expert outputs with advanced indexing `expert_outputs[experts, token_index]`.
- Counting load with `bincount`, and saying why load matters (an auxiliary balance loss, capacity limits).

#### Follow-ups

- Softmax over all `E` logits then take the top-k, or top-k then softmax? What changes? (Both are used: Mixtral renormalizes over the top-k.)
- Add a capacity: each expert takes at most `C` tokens; the rest are dropped. In what order would you drop them?
- How is this computed efficiently on a GPU? (Sort tokens by expert, grouped GEMM, then scatter back: see the GPU gym's `moe_matmul`.)
