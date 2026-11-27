# A checklist for the round

### In the room

1. **Clarify for five minutes.** Users, scale, latency, what success means for the business, constraints (privacy, cost, fairness). Write the assumptions where both of you can see them.
2. **Frame the ML problem.** Inputs, output, label and how you get it, and the metric offline and online. Say what would make ML the wrong tool.
3. **Size it.** Requests per second at peak, items to score, storage, memory and compute. Round hard; show units.
4. **Draw the high-level design before any detail.** Data and training on one side, serving on the other, the feedback loop between them.
5. **Go deep where the interviewer points**, or where the problem's main risk is: data quality, cold start, latency, or cost.
6. **Evaluate, launch, monitor.** Offline metrics, an A/B test with guardrails, staged rollout, drift and freshness monitoring.
7. **Close with trade-offs and next steps.** What you would build first, what you would measure, what could go wrong.

### Numbers worth knowing

| Quantity | Rule of thumb |
|---|---|
| Seconds in a day | about 86,400, round to 10⁵ |
| Peak over average traffic | 2–3× |
| Memory reference / SSD read / cross-region round trip | ~100 ns / ~100 µs / ~100 ms |
| Embedding table | rows × dims × 4 bytes (FP32) |
| Transformer FLOPs per token | about 2 × parameters (forward) |
| KV cache per token | 2 × layers × KV heads × head dim × bytes |
| Decode step lower bound | bytes read per step / memory bandwidth |
| Cost per million tokens | GPU price per hour / (tokens per hour / 10⁶) |

### Habits that score well

- **Lead with the simplest thing that works**, then improve it: a popularity baseline before a two-tower model, a hosted API before your own GPUs.
- **Make trade-offs explicit.** "I'd choose X over Y because of Z; if Z changes, I'd switch."
- **Separate the model from the decision.** The model predicts; thresholds, weights and business rules decide, and they belong to the product.
- **Name failure modes before you are asked**: leakage, training-serving skew, feedback loops, cold start, drift.
- **Keep talking in numbers.** A design with numbers attached can be checked; one without can only be admired.
