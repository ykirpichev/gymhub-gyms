# GYM_STARTER


class StreamingWeightedMean:
    def __init__(self):
        """Empty aggregator. result() is None until the total weight is positive."""
        pass

    def update(self, values, weights):
        """Add one batch. Raise ValueError on a negative weight or a length mismatch."""
        pass

    def merge(self, other):
        """Add other's count, weight sum and weighted sum into this aggregator."""
        pass

    def result(self):
        """The weighted mean, or None when the total weight is 0."""
        pass
