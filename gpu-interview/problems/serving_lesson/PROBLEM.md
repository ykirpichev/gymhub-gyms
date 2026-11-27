# LLM inference in one page

Generating a reply has two phases:

- **Prefill** runs the whole prompt through the model at once. Its matmuls have as many rows as prompt tokens, so it is **compute-bound**. It sets the **time to first token (TTFT)**.
- **Decode** produces one token per step per sequence. At batch 1 every step reads all weights to do a sliver of math, so it is **memory-bound**. It sets the **time per output token (TPOT)**, and its inverse, tokens per second per user.

**Batching** many sequences in decode reuses each weight read for every sequence: throughput grows almost linearly with batch size until the step becomes compute-bound or memory runs out. What runs out first is usually the **KV cache**: each layer keeps the keys and values of every past token of every sequence, $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes}$ per token.

Three ideas make modern servers (vLLM, SGLang, TensorRT-LLM) fast:

1. **Continuous (in-flight) batching**: the scheduler admits new requests and retires finished ones *every step*, instead of waiting for a whole batch to finish. Short replies stop waiting for long ones.
2. **Paged KV cache**: KV memory is split into fixed-size blocks, and each sequence has a block table, like virtual memory. No up-front reservation for the maximum length, almost no fragmentation, and prompts with a shared prefix can share blocks.
3. **Speculative decoding**: a small draft model proposes `k` tokens; the big model checks all of them in one forward pass (which costs about the same as one decode step, since decode is memory-bound) and keeps the longest acceptable prefix. The output distribution is exactly the big model's.

<figure class="diagram"><svg viewBox="0 0 680 180" width="680" role="img" aria-label="Paged KV cache: block tables map each sequence to physical blocks">
<text x="10" y="18" class="d-text d-bold" text-anchor="start">Block tables (logical → physical)</text>
<text x="380" y="18" class="d-text d-bold" text-anchor="start">Physical KV blocks (16 tokens each)</text>
<text x="10" y="60" class="d-text" text-anchor="start">seq A</text>
<rect x="70" y="40" width="46" height="30" rx="3" class="d-c"/>
<text x="93" y="60" class="d-text" text-anchor="middle">7</text>
<rect x="120" y="40" width="46" height="30" rx="3" class="d-a"/>
<text x="143" y="60" class="d-text" text-anchor="middle">1</text>
<rect x="170" y="40" width="46" height="30" rx="3" class="d-a"/>
<text x="193" y="60" class="d-text" text-anchor="middle">3</text>
<text x="10" y="120" class="d-text" text-anchor="start">seq B</text>
<rect x="70" y="100" width="46" height="30" rx="3" class="d-c"/>
<text x="93" y="120" class="d-text" text-anchor="middle">7</text>
<rect x="120" y="100" width="46" height="30" rx="3" class="d-b"/>
<text x="143" y="120" class="d-text" text-anchor="middle">4</text>
<rect x="380" y="34" width="50" height="36" rx="3" class="d-box"/>
<text x="405" y="57" class="d-text" text-anchor="middle">0</text>
<rect x="438" y="34" width="50" height="36" rx="3" class="d-a"/>
<text x="463" y="57" class="d-text" text-anchor="middle">1</text>
<rect x="496" y="34" width="50" height="36" rx="3" class="d-box"/>
<text x="521" y="57" class="d-text" text-anchor="middle">2</text>
<rect x="554" y="34" width="50" height="36" rx="3" class="d-a"/>
<text x="579" y="57" class="d-text" text-anchor="middle">3</text>
<rect x="612" y="34" width="50" height="36" rx="3" class="d-b"/>
<text x="637" y="57" class="d-text" text-anchor="middle">4</text>
<rect x="380" y="84" width="50" height="36" rx="3" class="d-box"/>
<text x="405" y="107" class="d-text" text-anchor="middle">5</text>
<rect x="438" y="84" width="50" height="36" rx="3" class="d-box"/>
<text x="463" y="107" class="d-text" text-anchor="middle">6</text>
<rect x="496" y="84" width="50" height="36" rx="3" class="d-c"/>
<text x="521" y="107" class="d-text" text-anchor="middle">7</text>
<rect x="554" y="84" width="50" height="36" rx="3" class="d-box"/>
<text x="579" y="107" class="d-text" text-anchor="middle">8</text>
<rect x="612" y="84" width="50" height="36" rx="3" class="d-box"/>
<text x="637" y="107" class="d-text" text-anchor="middle">9</text>
<text x="10" y="152" class="d-muted" text-anchor="start">Blue: seq A's blocks. Green: seq B's. Orange: the shared prompt prefix, stored once.</text>
<text x="10" y="168" class="d-muted" text-anchor="start">Unshaded blocks are free and go to whichever sequence grows next, in any order.</text>
</svg>
<figcaption>A paged KV cache. Each sequence's block table maps its logical blocks to physical ones anywhere in memory, so nothing is reserved up front and two sequences with the same prompt prefix share its block.</figcaption></figure>

> [!NOTE]
> Throughput and latency pull in different directions: bigger batches raise tokens per second for the server and slow each user's TPOT. A serving design answer names the SLO (for example TTFT under 500 ms, TPOT under 50 ms) and picks the batch size and hardware that meet it at the lowest cost.
