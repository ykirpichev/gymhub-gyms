import numpy as np


class KVCache:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive int")
        self.capacity = capacity
        self.k = None
        self.v = None

    def prefill(self, K, V):
        if self.k is not None:
            raise ValueError("prefill requires an empty cache")
        keys = np.asarray(K, dtype=np.float64)
        values = np.asarray(V, dtype=np.float64)
        if keys.ndim != 3 or keys.shape != values.shape:
            raise ValueError("K and V must share shape (heads, T, d)")
        heads, steps, dim = keys.shape
        if heads < 1 or dim < 1 or steps < 1 or steps > self.capacity:
            raise ValueError("prefill exceeds capacity or is empty")
        self.k = keys.copy()
        self.v = values.copy()

    def append(self, k, v):
        key = np.asarray(k, dtype=np.float64)
        value = np.asarray(v, dtype=np.float64)
        if key.ndim != 2 or key.shape != value.shape:
            raise ValueError("k and v must share shape (heads, d)")
        heads, dim = key.shape
        if heads < 1 or dim < 1:
            raise ValueError("k and v must have shape (heads, d)")
        if self.k is None:
            self.k = key[:, None, :].copy()
            self.v = value[:, None, :].copy()
            return
        if (heads, dim) != (self.k.shape[0], self.k.shape[2]):
            raise ValueError("k and v must match the cache heads and d")
        if self.k.shape[1] + 1 > self.capacity:
            raise ValueError("capacity exceeded")
        self.k = np.concatenate([self.k, key[:, None, :]], axis=1)
        self.v = np.concatenate([self.v, value[:, None, :]], axis=1)

    def get(self):
        if self.k is None:
            raise ValueError("empty cache")
        return self.k.copy(), self.v.copy()

    def attention(self, query):
        if self.k is None:
            raise ValueError("empty cache")
        q = np.asarray(query, dtype=np.float64)
        heads, _steps, dim = self.k.shape
        if q.shape != (heads, dim):
            raise ValueError("query must have shape (heads, d)")
        scores = np.einsum("hd,htd->ht", q, self.k) / np.sqrt(dim)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return np.einsum("ht,htd->hd", weights, self.v)

    @staticmethod
    def memory_bytes(layers, heads, dim, seq, batch, dtype):
        dims = (layers, heads, dim, seq, batch)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in dims):
            raise ValueError("layers, heads, dim, seq, and batch must be non-negative ints")
        try:
            itemsize = int(np.dtype(dtype).itemsize)
        except TypeError:
            raise ValueError(f"unknown dtype {dtype!r}") from None
        return 2 * layers * heads * dim * seq * batch * itemsize
