# GYM_STARTER
import numpy as np


class KVCache:
    def __init__(self, capacity):
        """Fixed-capacity cache for keys and values of shape (heads, T, d)."""
        pass

    def prefill(self, K, V):
        """Store a prompt. K and V have shape (heads, T, d). The cache must be empty."""
        pass

    def append(self, k, v):
        """Append one time step. k and v have shape (heads, d)."""
        pass

    def get(self):
        """Return (K, V) for the filled prefix, each of shape (heads, T, d)."""
        pass

    def attention(self, query):
        """Attend with one query per head, shape (heads, d). Return shape (heads, d)."""
        pass

    @staticmethod
    def memory_bytes(layers, heads, dim, seq, batch, dtype):
        """Bytes used by stacked key and value caches."""
        pass
