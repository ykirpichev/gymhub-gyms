# Model FLOPs utilization

A 7-billion-parameter dense transformer trains on 8 GPUs at a combined 80,000 tokens per second. Each GPU peaks at 1,000 TFLOP/s in BF16. Using $6 \times \text{parameters}$ FLOPs per token for a training step, what is the run's model FLOPs utilization, as a percentage?
