# Beam search step

**Level:** medium   **Teaches:** beam search, top-k over a flattened candidate set, index arithmetic with `//` and `%`

## Task

Beam search keeps the `W` best partial sequences (beams) per input. One step extends every beam by every vocabulary token, scores each candidate by its total log-probability, and keeps the best `W` candidates. Do one step for a batch of `B` inputs at once.

## Arguments

| Name          | Type           | Shape          | Role                                                      |
| ------------- | -------------- | -------------- | --------------------------------------------------------- |
| `beam_scores` | `torch.Tensor` | `[B, W]`       | total log-probability of each current beam, float32, CUDA |
| `logprobs`    | `torch.Tensor` | `[B, W, V]`    | next-token log-probabilities for each beam                |
| returns       | `tuple`        |                | `(scores, parents, tokens)`, each `[B, W]`                 |

`scores` is float32; `parents` and `tokens` are int64.

## Formula

```
candidate[b, w, t] = beam_scores[b, w] + logprobs[b, w, t]
keep the W largest candidates of each b, best first
parents[b, i] = the beam w of the i-th best candidate      tokens[b, i] = its token t
scores[b, i]  = its candidate score
```

## Example

```
B = 1, W = 2, V = 3
beam_scores = [[-1.0, -2.0]]
logprobs    = [[[-0.1, -2.0, -3.0],
                [-0.2, -0.3, -4.0]]]
candidates: beam 0: -1.1, -3.0, -4.0     beam 1: -2.2, -2.3, -6.0
best two: -1.1 (beam 0, token 0), -2.2 (beam 1, token 0)
scores = [[-1.1, -2.2]]   parents = [[0, 1]]   tokens = [[0, 0]]
```

The test inputs are random, so two candidates of one input never tie.

## How to think about it

Add the beam scores to the log-probabilities with broadcasting, flatten each input's `W × V` candidates into one row of length `W·V`, and take `topk(W)` of each row. A flat index `f` came from beam `f // V` and token `f % V`. A Python loop over beams or tokens is far too slow at a 32,000-token vocabulary.

## Tests

`W = 1` (greedy), `V = 1`, one input, beams with very different scores (all survivors come from one beam), and random batches. Benchmark input: `B = 64, W = 8, V = 32,000`.
