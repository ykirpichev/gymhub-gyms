# Sum reduction

**Level:** medium   **Teaches:** tree reductions in shared memory, atomics, warp shuffles

## Task

Add up every element of `A` and write the total to `out[0]`.

## Arguments

| Name  | Type           | Shape | Role   |
| ----- | -------------- | ----- | ------ |
| `A`   | `const float*` | `N`   | input  |
| `out` | `float*`       | `1`   | output |
| `N`   | `int`          |       | size   |

`N >= 1`. `out` holds a single float. It is not zeroed before your call.

## Formula

```
out[0] = A[0] + A[1] + ... + A[N - 1]
```

Float32 addition is not associative, so a parallel sum can differ slightly from a sequential one. The checker allows `atol = 1e-1` and `rtol = 1e-2`. That still rejects a result that drops part of the vector.

## Example

```
A = [1, 2, 3, 4]
pairwise:  (1 + 2) + (3 + 4) = 3 + 7 = 10
out = [10]
```

## How to think about it

Every element contributes to one shared result, so threads must combine their values. This is the first exercise where they cannot just work alone.

Addition can be regrouped. Instead of a chain of `N - 1` dependent adds, add pairs, then pairs of pairs, as a tree. Each level halves the number of values, so the depth is `log2(N)`.

<figure class="diagram"><svg viewBox="0 0 680 212" width="680" role="img" aria-label="Tree reduction in shared memory with sequential addressing">
<text x="10" y="38" class="d-muted" text-anchor="start">shared memory</text>
<rect x="170" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="192" y="38" class="d-text" text-anchor="middle">3</text>
<rect x="228" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="250" y="38" class="d-text" text-anchor="middle">1</text>
<rect x="286" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="308" y="38" class="d-text" text-anchor="middle">4</text>
<rect x="344" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="366" y="38" class="d-text" text-anchor="middle">1</text>
<rect x="402" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="424" y="38" class="d-text" text-anchor="middle">5</text>
<rect x="460" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="482" y="38" class="d-text" text-anchor="middle">9</text>
<rect x="518" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="540" y="38" class="d-text" text-anchor="middle">2</text>
<rect x="576" y="18" width="44" height="30" rx="4" class="d-box"/>
<text x="598" y="38" class="d-text" text-anchor="middle">6</text>
<text x="10" y="90" class="d-muted" text-anchor="start">stride 4: threads 0–3</text>
<rect x="170" y="70" width="44" height="30" rx="4" class="d-a"/>
<text x="192" y="90" class="d-text" text-anchor="middle">8</text>
<line x1="192" y1="48" x2="192" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="424" y1="48" x2="192" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="228" y="70" width="44" height="30" rx="4" class="d-a"/>
<text x="250" y="90" class="d-text" text-anchor="middle">10</text>
<line x1="250" y1="48" x2="250" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="482" y1="48" x2="250" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="286" y="70" width="44" height="30" rx="4" class="d-a"/>
<text x="308" y="90" class="d-text" text-anchor="middle">6</text>
<line x1="308" y1="48" x2="308" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="540" y1="48" x2="308" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="344" y="70" width="44" height="30" rx="4" class="d-a"/>
<text x="366" y="90" class="d-text" text-anchor="middle">7</text>
<line x1="366" y1="48" x2="366" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="598" y1="48" x2="366" y2="69" class="d-line" marker-end="url(#d-arrow)"/>
<text x="10" y="142" class="d-muted" text-anchor="start">stride 2: threads 0–1</text>
<rect x="170" y="122" width="44" height="30" rx="4" class="d-a"/>
<text x="192" y="142" class="d-text" text-anchor="middle">14</text>
<line x1="192" y1="100" x2="192" y2="121" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="308" y1="100" x2="192" y2="121" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="228" y="122" width="44" height="30" rx="4" class="d-a"/>
<text x="250" y="142" class="d-text" text-anchor="middle">17</text>
<line x1="250" y1="100" x2="250" y2="121" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="366" y1="100" x2="250" y2="121" class="d-line" marker-end="url(#d-arrow)"/>
<text x="10" y="194" class="d-muted" text-anchor="start">stride 1: thread 0</text>
<rect x="170" y="174" width="44" height="30" rx="4" class="d-c"/>
<text x="192" y="194" class="d-text" text-anchor="middle">31</text>
<line x1="192" y1="152" x2="192" y2="173" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="250" y1="152" x2="192" y2="173" class="d-line" marker-end="url(#d-arrow)"/>
</svg>
<figcaption>A block's tree reduction with sequential addressing: at each step thread <code>t</code> adds element <code>t + stride</code> into element <code>t</code>, the stride halves, and half the threads go idle. Active threads stay contiguous, so whole warps retire instead of diverging.</figcaption></figure>

On a GPU this happens in stages. Each thread sums several elements on its own, the threads of a block combine their sums through shared memory, and the blocks combine their results with one atomic add each or with a second small kernel.

The naive version has every thread do `atomicAdd(out, A[i])`. It is correct, but millions of atomics on one address run one after another. The kernel is memory-bound when done right: it reads `4N` bytes and writes 4.

## In an interview

Probably the most-asked kernel. Interviewers walk you through the optimization ladder and want the reasons for each step.

Follow-ups to answer out loud once your kernel passes:

- Name the steps from the naive tree reduction to a near-peak one, and the bottleneck each fixes.
- Why must `__syncthreads()` be outside any branch that not all threads take?
- How would you finish across blocks: a second launch, atomics, or a last-block-done pattern? Which is deterministic?
