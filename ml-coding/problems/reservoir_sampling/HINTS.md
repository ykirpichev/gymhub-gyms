## Hint 1
Fill the reservoir with the first k items. After that, item number i (0-based) should end up in the sample with probability k / (i + 1), as it would if the stream ended right after it.

## Hint 2
To keep item i with probability k / (i + 1) and evict a uniformly random old item, draw one integer `j` uniform in `[0, i]` (that is `rng.randrange(i + 1)`). If `j < k`, put the item in slot `j`. Watch the bounds: `randrange(i)` or `randint(0, i + 1)` bias the sample.

## Hint 3
Why it is uniform: an item already in the reservoir survives step i with probability 1 − (k/(i+1))·(1/k) = i/(i+1). Multiply these from its arrival to the end and the product telescopes to k/n.
