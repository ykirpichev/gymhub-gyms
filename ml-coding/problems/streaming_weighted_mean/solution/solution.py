class StreamingWeightedMean:
    def __init__(self):
        self.count = 0
        self.weight_sum = 0.0
        self.weighted_sum = 0.0

    def update(self, values, weights):
        if len(values) != len(weights):
            raise ValueError("values and weights must have the same length")
        batch = []
        for value, weight in zip(values, weights):
            value = float(value)
            weight = float(weight)
            if weight < 0:
                raise ValueError("weights must be non-negative")
            batch.append((value, weight))
        for value, weight in batch:
            self.count += 1
            self.weight_sum += weight
            self.weighted_sum += weight * value

    def merge(self, other):
        self.count += other.count
        self.weight_sum += other.weight_sum
        self.weighted_sum += other.weighted_sum

    def result(self):
        if self.weight_sum == 0:
            return None
        return self.weighted_sum / self.weight_sum
