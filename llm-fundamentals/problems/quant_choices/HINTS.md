## Hint 1
Split 'when you quantize' (after training, or during training) from 'which tensor' (weights, or the KV cache).

## Hint 2
GPTQ and AWQ are both post-training weight methods. They do not become QAT by saving a 4-bit file.
