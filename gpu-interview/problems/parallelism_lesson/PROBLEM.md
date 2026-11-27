# Parallelism in one page

**Memory per parameter.** Mixed-precision training with Adam keeps, for each parameter: BF16 weights (2 bytes), BF16 gradients (2), an FP32 master copy (4) and Adam's two FP32 moments (4 + 4). **16 bytes per parameter** before activations. A 7B model needs 112 GB of state: more than one 80 GB GPU.

**Activations** are the other half: every layer saves its inputs for the backward pass, and they grow with batch × sequence length. *Activation checkpointing* stores only some of them and recomputes the rest in the backward pass, trading about 30% more compute for much less memory.

<figure class="diagram"><svg viewBox="0 0 680 236" width="680" role="img" aria-label="Data, tensor and pipeline parallelism on two GPUs">
<text x="112" y="20" class="d-text d-bold" text-anchor="middle">Data parallel</text>
<rect x="10" y="34" width="92" height="140" rx="6" class="d-fill"/>
<text x="56" y="190" class="d-muted" text-anchor="middle">GPU 0</text>
<rect x="20" y="46" width="72" height="22" rx="3" class="d-a"/>
<text x="56" y="61" class="d-muted" text-anchor="middle">layer 1</text>
<rect x="20" y="76" width="72" height="22" rx="3" class="d-a"/>
<text x="56" y="91" class="d-muted" text-anchor="middle">layer 2</text>
<rect x="20" y="106" width="72" height="22" rx="3" class="d-a"/>
<text x="56" y="121" class="d-muted" text-anchor="middle">layer 3</text>
<rect x="20" y="136" width="72" height="22" rx="3" class="d-a"/>
<text x="56" y="151" class="d-muted" text-anchor="middle">layer 4</text>
<rect x="116" y="34" width="92" height="140" rx="6" class="d-fill"/>
<text x="162" y="190" class="d-muted" text-anchor="middle">GPU 1</text>
<rect x="126" y="46" width="72" height="22" rx="3" class="d-a"/>
<text x="162" y="61" class="d-muted" text-anchor="middle">layer 1</text>
<rect x="126" y="76" width="72" height="22" rx="3" class="d-a"/>
<text x="162" y="91" class="d-muted" text-anchor="middle">layer 2</text>
<rect x="126" y="106" width="72" height="22" rx="3" class="d-a"/>
<text x="162" y="121" class="d-muted" text-anchor="middle">layer 3</text>
<rect x="126" y="136" width="72" height="22" rx="3" class="d-a"/>
<text x="162" y="151" class="d-muted" text-anchor="middle">layer 4</text>
<line x1="104" y1="210" x2="114" y2="210" class="d-line"/>
<path d="M98,210 L130,210" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M130,210 L98,210" class="d-line" marker-end="url(#d-arrow)"/>
<text x="114" y="228" class="d-muted" text-anchor="middle">all-reduce gradients</text>
<text x="337" y="20" class="d-text d-bold" text-anchor="middle">Tensor parallel</text>
<rect x="235" y="34" width="92" height="140" rx="6" class="d-fill"/>
<text x="281" y="190" class="d-muted" text-anchor="middle">GPU 0</text>
<rect x="245" y="46" width="72" height="22" rx="3" class="d-b"/>
<text x="281" y="61" class="d-muted" text-anchor="middle">½ of layer 1</text>
<rect x="245" y="76" width="72" height="22" rx="3" class="d-b"/>
<text x="281" y="91" class="d-muted" text-anchor="middle">½ of layer 2</text>
<rect x="245" y="106" width="72" height="22" rx="3" class="d-b"/>
<text x="281" y="121" class="d-muted" text-anchor="middle">½ of layer 3</text>
<rect x="245" y="136" width="72" height="22" rx="3" class="d-b"/>
<text x="281" y="151" class="d-muted" text-anchor="middle">½ of layer 4</text>
<rect x="341" y="34" width="92" height="140" rx="6" class="d-fill"/>
<text x="387" y="190" class="d-muted" text-anchor="middle">GPU 1</text>
<rect x="351" y="46" width="72" height="22" rx="3" class="d-b"/>
<text x="387" y="61" class="d-muted" text-anchor="middle">½ of layer 1</text>
<rect x="351" y="76" width="72" height="22" rx="3" class="d-b"/>
<text x="387" y="91" class="d-muted" text-anchor="middle">½ of layer 2</text>
<rect x="351" y="106" width="72" height="22" rx="3" class="d-b"/>
<text x="387" y="121" class="d-muted" text-anchor="middle">½ of layer 3</text>
<rect x="351" y="136" width="72" height="22" rx="3" class="d-b"/>
<text x="387" y="151" class="d-muted" text-anchor="middle">½ of layer 4</text>
<line x1="329" y1="210" x2="339" y2="210" class="d-line"/>
<path d="M323,210 L355,210" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M355,210 L323,210" class="d-line" marker-end="url(#d-arrow)"/>
<text x="339" y="228" class="d-muted" text-anchor="middle">all-reduce activations</text>
<text x="562" y="20" class="d-text d-bold" text-anchor="middle">Pipeline parallel</text>
<rect x="460" y="34" width="92" height="140" rx="6" class="d-fill"/>
<text x="506" y="190" class="d-muted" text-anchor="middle">GPU 0</text>
<rect x="470" y="46" width="72" height="22" rx="3" class="d-c"/>
<text x="506" y="61" class="d-muted" text-anchor="middle">layer 1</text>
<rect x="470" y="76" width="72" height="22" rx="3" class="d-c"/>
<text x="506" y="91" class="d-muted" text-anchor="middle">layer 2</text>
<rect x="470" y="106" width="72" height="22" rx="3" class="d-box d-dash"/>
<rect x="470" y="136" width="72" height="22" rx="3" class="d-box d-dash"/>
<rect x="566" y="34" width="92" height="140" rx="6" class="d-fill"/>
<text x="612" y="190" class="d-muted" text-anchor="middle">GPU 1</text>
<rect x="576" y="46" width="72" height="22" rx="3" class="d-box d-dash"/>
<rect x="576" y="76" width="72" height="22" rx="3" class="d-box d-dash"/>
<rect x="576" y="106" width="72" height="22" rx="3" class="d-c"/>
<text x="612" y="121" class="d-muted" text-anchor="middle">layer 3</text>
<rect x="576" y="136" width="72" height="22" rx="3" class="d-c"/>
<text x="612" y="151" class="d-muted" text-anchor="middle">layer 4</text>
<line x1="554" y1="210" x2="564" y2="210" class="d-line"/>
<path d="M548,210 L580,210" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M580,210 L548,210" class="d-line" marker-end="url(#d-arrow)"/>
<text x="564" y="228" class="d-muted" text-anchor="middle">send activations</text>
<text x="56" y="206" class="d-muted" text-anchor="middle">batch ½</text>
<text x="162" y="206" class="d-muted" text-anchor="middle">batch ½</text>
</svg>
<figcaption>Three ways to split a four-layer model over two GPUs. Data parallel copies the model and splits the batch; tensor parallel splits every weight matrix; pipeline parallel gives each GPU a run of layers. Real runs combine them.</figcaption></figure>

| Strategy | What is split | What is communicated | Cost |
|---|---|---|---|
| **Data parallel (DDP)** | The batch | All-reduce of gradients each step | Every GPU holds the full model and optimizer |
| **ZeRO / FSDP** | Optimizer state (stage 1), + gradients (2), + weights (3) across data-parallel ranks | Reduce-scatter of gradients, all-gather of weights before each layer | More communication, much less memory |
| **Tensor parallel** | Each matmul's weight matrix, by rows or columns | All-reduce (or reduce-scatter + all-gather) of activations, twice per transformer layer | Needs fast links: stays inside one NVLink node |
| **Pipeline parallel** | Layers, into stages | Activations between neighbouring stages | Idle *bubble* at the start and end of each batch |
| **Sequence / context parallel** | The sequence dimension | Keys and values (ring attention) or activations | For very long contexts |
| **Expert parallel** | The experts of a MoE layer | All-to-all of tokens to their experts and back | Load imbalance between experts |

**Ring all-reduce** of `S` bytes over `N` GPUs runs a *reduce-scatter* then an *all-gather*, each `N − 1` steps of `S / N` bytes. Each GPU sends $2 \frac{N-1}{N} S$ bytes in total, almost independent of `N`, which is why it scales.

<figure class="diagram"><svg viewBox="0 0 680 210" width="680" role="img" aria-label="Ring all-reduce over four GPUs">
<rect x="70" y="16" width="100" height="48" rx="8" class="d-fill"/>
<text x="120" y="32" class="d-text" text-anchor="middle">GPU 0</text>
<rect x="80" y="40" width="18" height="14" rx="2" class="d-box"/>
<rect x="100" y="40" width="18" height="14" rx="2" class="d-c"/>
<rect x="120" y="40" width="18" height="14" rx="2" class="d-box"/>
<rect x="140" y="40" width="18" height="14" rx="2" class="d-box"/>
<rect x="250" y="16" width="100" height="48" rx="8" class="d-fill"/>
<text x="300" y="32" class="d-text" text-anchor="middle">GPU 1</text>
<rect x="260" y="40" width="18" height="14" rx="2" class="d-box"/>
<rect x="280" y="40" width="18" height="14" rx="2" class="d-box"/>
<rect x="300" y="40" width="18" height="14" rx="2" class="d-c"/>
<rect x="320" y="40" width="18" height="14" rx="2" class="d-box"/>
<rect x="250" y="136" width="100" height="48" rx="8" class="d-fill"/>
<text x="300" y="152" class="d-text" text-anchor="middle">GPU 2</text>
<rect x="260" y="160" width="18" height="14" rx="2" class="d-box"/>
<rect x="280" y="160" width="18" height="14" rx="2" class="d-box"/>
<rect x="300" y="160" width="18" height="14" rx="2" class="d-box"/>
<rect x="320" y="160" width="18" height="14" rx="2" class="d-c"/>
<rect x="70" y="136" width="100" height="48" rx="8" class="d-fill"/>
<text x="120" y="152" class="d-text" text-anchor="middle">GPU 3</text>
<rect x="80" y="160" width="18" height="14" rx="2" class="d-c"/>
<rect x="100" y="160" width="18" height="14" rx="2" class="d-box"/>
<rect x="120" y="160" width="18" height="14" rx="2" class="d-box"/>
<rect x="140" y="160" width="18" height="14" rx="2" class="d-box"/>
<path d="M174,40 L246,40" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M300,68 L300,132" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M246,160 L174,160" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M120,132 L120,68" class="d-line" marker-end="url(#d-arrow)"/>
<text x="390" y="30" class="d-text" text-anchor="start">Each GPU splits its S bytes into N chunks.</text>
<text x="390" y="49" class="d-muted" text-anchor="start">Reduce-scatter: N − 1 steps; each step every</text>
<text x="390" y="68" class="d-muted" text-anchor="start">GPU sends one chunk to its neighbour, which</text>
<text x="390" y="87" class="d-muted" text-anchor="start">adds it to its own. Afterwards each GPU holds</text>
<text x="390" y="106" class="d-muted" text-anchor="start">one fully summed chunk (orange).</text>
<text x="390" y="125" class="d-muted" text-anchor="start">All-gather: N − 1 more steps pass the summed</text>
<text x="390" y="144" class="d-muted" text-anchor="start">chunks around the ring.</text>
<text x="390" y="163" class="d-text" text-anchor="start">Bytes sent per GPU: 2 (N − 1) / N × S,</text>
<text x="390" y="182" class="d-text" text-anchor="start">almost independent of N.</text>
</svg>
<figcaption>Ring all-reduce over 4 GPUs, shown after the reduce-scatter half. Every link is busy at every step, so the time is set by the slowest link, not by the number of GPUs.</figcaption></figure>

**Pipeline bubble.** With `p` stages and `m` micro-batches in a GPipe schedule, the idle fraction is $\frac{p-1}{m+p-1}$. More micro-batches shrink it.

<figure class="diagram"><svg viewBox="0 0 680 200" width="680" role="img" aria-label="GPipe schedule with 4 stages and 4 micro-batches">
<text x="70" y="49" class="d-muted" text-anchor="end">GPU 0</text>
<rect x="81" y="30" width="36" height="28" rx="3" class="d-a"/>
<text x="99.0" y="49" class="d-text" text-anchor="middle">F1</text>
<rect x="119" y="30" width="36" height="28" rx="3" class="d-a"/>
<text x="137.0" y="49" class="d-text" text-anchor="middle">F2</text>
<rect x="157" y="30" width="36" height="28" rx="3" class="d-a"/>
<text x="175.0" y="49" class="d-text" text-anchor="middle">F3</text>
<rect x="195" y="30" width="36" height="28" rx="3" class="d-a"/>
<text x="213.0" y="49" class="d-text" text-anchor="middle">F4</text>
<rect x="233" y="30" width="36" height="28" rx="3" class="d-fill"/>
<rect x="271" y="30" width="36" height="28" rx="3" class="d-fill"/>
<rect x="309" y="30" width="36" height="28" rx="3" class="d-fill"/>
<rect x="347" y="30" width="36" height="28" rx="3" class="d-fill"/>
<rect x="385" y="30" width="36" height="28" rx="3" class="d-fill"/>
<rect x="423" y="30" width="36" height="28" rx="3" class="d-fill"/>
<rect x="461" y="30" width="36" height="28" rx="3" class="d-b"/>
<text x="479.0" y="49" class="d-text" text-anchor="middle">B4</text>
<rect x="499" y="30" width="36" height="28" rx="3" class="d-b"/>
<text x="517.0" y="49" class="d-text" text-anchor="middle">B3</text>
<rect x="537" y="30" width="36" height="28" rx="3" class="d-b"/>
<text x="555.0" y="49" class="d-text" text-anchor="middle">B2</text>
<rect x="575" y="30" width="36" height="28" rx="3" class="d-b"/>
<text x="593.0" y="49" class="d-text" text-anchor="middle">B1</text>
<text x="70" y="83" class="d-muted" text-anchor="end">GPU 1</text>
<rect x="81" y="64" width="36" height="28" rx="3" class="d-fill"/>
<rect x="119" y="64" width="36" height="28" rx="3" class="d-a"/>
<text x="137.0" y="83" class="d-text" text-anchor="middle">F1</text>
<rect x="157" y="64" width="36" height="28" rx="3" class="d-a"/>
<text x="175.0" y="83" class="d-text" text-anchor="middle">F2</text>
<rect x="195" y="64" width="36" height="28" rx="3" class="d-a"/>
<text x="213.0" y="83" class="d-text" text-anchor="middle">F3</text>
<rect x="233" y="64" width="36" height="28" rx="3" class="d-a"/>
<text x="251.0" y="83" class="d-text" text-anchor="middle">F4</text>
<rect x="271" y="64" width="36" height="28" rx="3" class="d-fill"/>
<rect x="309" y="64" width="36" height="28" rx="3" class="d-fill"/>
<rect x="347" y="64" width="36" height="28" rx="3" class="d-fill"/>
<rect x="385" y="64" width="36" height="28" rx="3" class="d-fill"/>
<rect x="423" y="64" width="36" height="28" rx="3" class="d-b"/>
<text x="441.0" y="83" class="d-text" text-anchor="middle">B4</text>
<rect x="461" y="64" width="36" height="28" rx="3" class="d-b"/>
<text x="479.0" y="83" class="d-text" text-anchor="middle">B3</text>
<rect x="499" y="64" width="36" height="28" rx="3" class="d-b"/>
<text x="517.0" y="83" class="d-text" text-anchor="middle">B2</text>
<rect x="537" y="64" width="36" height="28" rx="3" class="d-b"/>
<text x="555.0" y="83" class="d-text" text-anchor="middle">B1</text>
<rect x="575" y="64" width="36" height="28" rx="3" class="d-fill"/>
<text x="70" y="117" class="d-muted" text-anchor="end">GPU 2</text>
<rect x="81" y="98" width="36" height="28" rx="3" class="d-fill"/>
<rect x="119" y="98" width="36" height="28" rx="3" class="d-fill"/>
<rect x="157" y="98" width="36" height="28" rx="3" class="d-a"/>
<text x="175.0" y="117" class="d-text" text-anchor="middle">F1</text>
<rect x="195" y="98" width="36" height="28" rx="3" class="d-a"/>
<text x="213.0" y="117" class="d-text" text-anchor="middle">F2</text>
<rect x="233" y="98" width="36" height="28" rx="3" class="d-a"/>
<text x="251.0" y="117" class="d-text" text-anchor="middle">F3</text>
<rect x="271" y="98" width="36" height="28" rx="3" class="d-a"/>
<text x="289.0" y="117" class="d-text" text-anchor="middle">F4</text>
<rect x="309" y="98" width="36" height="28" rx="3" class="d-fill"/>
<rect x="347" y="98" width="36" height="28" rx="3" class="d-fill"/>
<rect x="385" y="98" width="36" height="28" rx="3" class="d-b"/>
<text x="403.0" y="117" class="d-text" text-anchor="middle">B4</text>
<rect x="423" y="98" width="36" height="28" rx="3" class="d-b"/>
<text x="441.0" y="117" class="d-text" text-anchor="middle">B3</text>
<rect x="461" y="98" width="36" height="28" rx="3" class="d-b"/>
<text x="479.0" y="117" class="d-text" text-anchor="middle">B2</text>
<rect x="499" y="98" width="36" height="28" rx="3" class="d-b"/>
<text x="517.0" y="117" class="d-text" text-anchor="middle">B1</text>
<rect x="537" y="98" width="36" height="28" rx="3" class="d-fill"/>
<rect x="575" y="98" width="36" height="28" rx="3" class="d-fill"/>
<text x="70" y="151" class="d-muted" text-anchor="end">GPU 3</text>
<rect x="81" y="132" width="36" height="28" rx="3" class="d-fill"/>
<rect x="119" y="132" width="36" height="28" rx="3" class="d-fill"/>
<rect x="157" y="132" width="36" height="28" rx="3" class="d-fill"/>
<rect x="195" y="132" width="36" height="28" rx="3" class="d-a"/>
<text x="213.0" y="151" class="d-text" text-anchor="middle">F1</text>
<rect x="233" y="132" width="36" height="28" rx="3" class="d-a"/>
<text x="251.0" y="151" class="d-text" text-anchor="middle">F2</text>
<rect x="271" y="132" width="36" height="28" rx="3" class="d-a"/>
<text x="289.0" y="151" class="d-text" text-anchor="middle">F3</text>
<rect x="309" y="132" width="36" height="28" rx="3" class="d-a"/>
<text x="327.0" y="151" class="d-text" text-anchor="middle">F4</text>
<rect x="347" y="132" width="36" height="28" rx="3" class="d-b"/>
<text x="365.0" y="151" class="d-text" text-anchor="middle">B4</text>
<rect x="385" y="132" width="36" height="28" rx="3" class="d-b"/>
<text x="403.0" y="151" class="d-text" text-anchor="middle">B3</text>
<rect x="423" y="132" width="36" height="28" rx="3" class="d-b"/>
<text x="441.0" y="151" class="d-text" text-anchor="middle">B2</text>
<rect x="461" y="132" width="36" height="28" rx="3" class="d-b"/>
<text x="479.0" y="151" class="d-text" text-anchor="middle">B1</text>
<rect x="499" y="132" width="36" height="28" rx="3" class="d-fill"/>
<rect x="537" y="132" width="36" height="28" rx="3" class="d-fill"/>
<rect x="575" y="132" width="36" height="28" rx="3" class="d-fill"/>
<path d="M80,172 L612,172" class="d-line" marker-end="url(#d-arrow)"/>
<text x="612" y="190" class="d-muted" text-anchor="end">time</text>
<text x="80" y="18" class="d-muted" text-anchor="start">F = forward of micro-batch, B = backward. Grey slots are the bubble.</text>
</svg>
<figcaption>A GPipe schedule with p = 4 stages and m = 4 micro-batches. Each GPU idles for p − 1 of its m + p − 1 forward slots, and the same in the backward pass: a bubble of 3/7 here. Eight micro-batches would cut it to 3/11.</figcaption></figure>

**MFU** (model FLOPs utilization): the FLOPs the model needs, about $6 \times \text{parameters} \times \text{tokens}$ for a training step of a dense transformer, divided by the time and the GPUs' peak FLOP/s. Good large runs report 35–55%.
