# How these interviews work

Roles that write or tune GPU code (kernel engineer, ML performance engineer, inference engineer, ML infrastructure engineer) usually interview in four or five rounds. The names vary between companies; the shape is stable.

| Round | What you do | What it tests | Practise with |
|---|---|---|---|
| **Coding** | Solve a problem in Python, often NumPy, on a shared editor. | Clean, correct code under time pressure; edge cases; complexity. | The coding exercises in every track here, the ML coding and Algo gyms. |
| **Kernel** | Write or fix a CUDA or Triton kernel on paper or in an editor, usually without running it. | Indexing, bounds, synchronization, memory access patterns, a plan to make it fast. | The fill-in kernels here, then the GPU gym on a real GPU. |
| **Performance reasoning** | Estimate how long something takes, or why it is slow. | Rooflines, bytes moved, FLOPs, the numbers of one real GPU. | *Rooflines and estimates* track. |
| **ML systems design** | Design a training or serving system for a given model and budget. | Parallelism, memory budgets, batching, the trade-offs you name. | *Training at scale* and *Serving LLMs* tracks. |
| **Deep dive** | Walk through a project or a well-known kernel (flash attention, a fused norm) in depth. | Depth, honesty about what you measured, follow-ups. | The essays in *Mock interview questions*. |

#### What interviewers listen for

1. **Correctness first, then speed.** A kernel that is right and slow is a starting point; a fast kernel with a race is a fail. Say where the bounds check goes before you say anything about shared memory.
2. **Numbers.** "This is memory-bound" is half an answer. "It moves 268 MB, so at 3 TB/s it can't beat 90 µs, and we're at 300 µs" is the whole answer. Learn the numbers of one GPU well (bandwidth, peak FLOP/s, shared memory per SM, registers per SM).
3. **A plan, in order.** For a slow kernel: measure, compare with the roofline, find the limiter, fix the biggest one, measure again. Interviewers ask *what would you try next* more than *what is the answer*.
4. **Trade-offs said out loud.** More occupancy versus more registers per thread; bigger tiles versus fewer blocks; batching for throughput versus latency.

#### How to use this gym

- Each exercise has a **minutes** estimate. Time yourself, and treat going far over as a sign to read the explanation and come back.
- In coding exercises, read **What interviewers look for** before you start and the **Follow-ups** after you pass: answer each follow-up out loud in a sentence or two. That is the part of the round where people lose points.
- Write the essays as you would speak them: short, ordered, with numbers.
- When a track has a GPU gym counterpart (named in the exercise), write that kernel on a GPU after you finish the track.

> [!TIP]
> One GPU to memorize for this gym: the *interview GPU*, numbers rounded from a current data-center part. **3 TB/s** of HBM bandwidth, **1,000 TFLOP/s** of dense BF16 on Tensor Cores, **60 TFLOP/s** of FP32 without them, **132 SMs**, each with **64K 32-bit registers**, at most **2,048 resident threads** and up to **228 KB** of shared memory. Exercises that need other numbers give them.
