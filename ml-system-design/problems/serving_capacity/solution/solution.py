import math


def solve(model, gpu, workload):
    weights_gb = model["params_b"] * model["weight_bytes"]
    kv_bytes_token = 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * model["kv_bytes"]
    kv_gb_per_seq = kv_bytes_token * workload["context_tokens"] / 1e9
    free_gb = gpu["count"] * gpu["memory_gb"] - weights_gb - workload["reserve_gb"]
    max_batch = math.floor(free_gb / kv_gb_per_seq)
    if max_batch < 1:
        raise ValueError("the model does not fit with even one sequence")
    step_ms = (weights_gb + max_batch * kv_gb_per_seq) / (gpu["count"] * gpu["bandwidth_tbs"])
    tokens_per_s = max_batch * 1000 / step_ms
    return {
        "weights_gb": weights_gb,
        "kv_bytes_token": int(kv_bytes_token),
        "kv_gb_per_seq": kv_gb_per_seq,
        "max_batch": int(max_batch),
        "step_ms": step_ms,
        "tokens_per_s": tokens_per_s,
        "user_tokens_per_s": 1000 / step_ms,
        "cost_per_m": gpu["count"] * gpu["price_per_hour"] / (tokens_per_s * 3600 / 1e6),
    }
