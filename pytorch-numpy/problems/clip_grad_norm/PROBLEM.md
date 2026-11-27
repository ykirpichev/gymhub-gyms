# Clip by global norm

Write `solve(grads, max_norm)` where `grads` is a list of arrays (one per parameter, any shapes). Compute the **global** L2 norm over all their elements together. If it exceeds `max_norm`, multiply every gradient by `max_norm / (norm + 1e-6)`; otherwise leave them unchanged. Return `(clipped, norm)`: the list of arrays and the norm before clipping, a float.

```
solve([[3.0], [4.0]], 1.0)  ->  ([array([0.6]), array([0.8])], 5.0)
```

#### What interviewers look for

- One norm over all parameters, not one per tensor: clipping each tensor separately changes the direction of the update.
- Returning the pre-clip norm, which is worth logging: spikes in it warn of instability before the loss shows it.

#### Follow-ups

- Why clip in LLM training at all, and what does a gradient-norm spike usually mean?
- With mixed precision and a gradient scaler, when must you unscale before clipping?
- With FSDP, each rank holds a shard of each gradient. How is the global norm computed?
