## Hint 1
Ask, for each scheme, which index is sharded: the hidden dimension inside a layer, the layer index, or the sequence position.

## Hint 2
Prefill touches every prompt token in parallel. Decode touches one new token and rereads the cache.
