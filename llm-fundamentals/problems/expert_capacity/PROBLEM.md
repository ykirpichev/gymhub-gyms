# Expert capacity

A top-1 mixture-of-experts layer has 8 experts. One batch contains 256 tokens. Each expert's capacity is

`(tokens / experts) × capacity_factor`

with `capacity_factor = 1.25`.

What is one expert's capacity, in tokens? Give the number.
