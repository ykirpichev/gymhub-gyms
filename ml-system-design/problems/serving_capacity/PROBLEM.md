# Plan the capacity of an LLM replica

Write `solve(model, gpu, workload)` that turns the back-of-the-envelope serving estimate into code. It gives a **memory-bound lower bound** on step time: compute, communication and overheads are ignored.

- `model`: `{"params_b", "layers", "kv_heads", "head_dim", "weight_bytes", "kv_bytes"}`: parameters in billions, and bytes per weight and per KV value.
- `gpu`: `{"count", "memory_gb", "bandwidth_tbs", "price_per_hour"}`: GPUs in the replica, each GPU's memory (GB) and bandwidth (TB/s), and price per GPU-hour.
- `workload`: `{"context_tokens", "reserve_gb"}`: tokens of KV cache per sequence, and GB of the replica's memory kept back for activations and the runtime.

Use 1 GB = 10⁹ bytes and 1 TB = 10¹² bytes. Compute:

```
weights_gb      = params_b * weight_bytes
kv_bytes_token  = 2 * layers * kv_heads * head_dim * kv_bytes
kv_gb_per_seq   = kv_bytes_token * context_tokens / 1e9
free_gb         = count * memory_gb - weights_gb - reserve_gb
max_batch       = floor(free_gb / kv_gb_per_seq)
step_ms         = (weights_gb + max_batch * kv_gb_per_seq) / (count * bandwidth_tbs)
tokens_per_s    = max_batch * 1000 / step_ms
cost_per_m      = count * price_per_hour / (tokens_per_s * 3600 / 1e6)
```

(`step_ms` is GB over TB/s, which comes out in milliseconds.) If `max_batch < 1`, the model does not fit with even one sequence: raise `ValueError`.

Return a dict with the keys `"weights_gb"`, `"kv_bytes_token"` (an int), `"kv_gb_per_seq"`, `"max_batch"` (an int), `"step_ms"`, `"tokens_per_s"`, `"user_tokens_per_s"` (`1000 / step_ms`) and `"cost_per_m"`.

```
model    = {"params_b": 70, "layers": 80, "kv_heads": 8, "head_dim": 128, "weight_bytes": 2, "kv_bytes": 2}
gpu      = {"count": 4, "memory_gb": 80, "bandwidth_tbs": 3.35, "price_per_hour": 2.5}
workload = {"context_tokens": 4096, "reserve_gb": 20}
solve(model, gpu, workload)
-> {"weights_gb": 140, "kv_bytes_token": 327680, "kv_gb_per_seq": 1.3422, "max_batch": 119,
    "step_ms": 22.367, "tokens_per_s": 5320.3, "user_tokens_per_s": 44.71, "cost_per_m": 0.5221}
```

#### What interviewers look for

- The order of reasoning: memory first (weights, then KV per sequence, then concurrency), then step time from bandwidth, then throughput and cost.
- Units handled carefully, and the assumptions stated: every sequence at full context, KV read in full every step, no compute or communication time.
- What to change when a number misses the target: a TPOT goal of 50 ms is met here (22 ms), so the batch could grow if memory allowed; FP8 weights would free 70 GB.

#### Follow-ups

- Rerun with FP8 weights (`weight_bytes = 1`) and an FP8 KV cache. Which outputs change, and by how much?
- The average conversation is 1,500 tokens, not 4,096. How does a paged KV cache let you exploit that, and what is the risk?
- Is it cheaper to serve this model on 8 GPUs as one replica or as two 4-GPU replicas? What does the formula leave out that would decide it?
