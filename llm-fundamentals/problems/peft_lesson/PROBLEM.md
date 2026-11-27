# Fine-tuning without updating every weight

Full fine-tuning updates every weight and needs optimizer state for all of them: about 16 bytes per parameter with mixed-precision Adam. **Parameter-efficient fine-tuning (PEFT)** freezes the base model and trains a small number of new parameters.

**LoRA** adds a low-rank update to chosen weight matrices. For a frozen $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

$$W' = W + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times d_\text{in}},\; B \in \mathbb{R}^{d_\text{out} \times r},\; r \ll d.$$

- $B$ starts at zero, so training starts exactly from the base model.
- Trainable parameters per matrix: $r(d_\text{in} + d_\text{out})$ instead of $d_\text{in} d_\text{out}$. For a $4096 \times 4096$ matrix with $r = 16$, that is 131K instead of 16.8M.
- After training, $BA$ can be merged into $W$, so inference costs nothing extra; or kept separate, so one base model can serve many adapters.
- Usually applied to the attention projections, often also to the MLP.

<figure class="diagram"><svg viewBox="0 0 720 200" width="720" role="img" aria-label="A LoRA adapter beside a frozen weight matrix">
<text x="30" y="104" class="d-text d-bold" text-anchor="middle">x</text>
<line x1="40" y1="100" x2="80" y2="100" class="d-line"/>
<path d="M80,100 L80,48 L118,48" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M80,100 L80,152 L118,152" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="120" y="8" width="160" height="80" rx="4" class="d-fill"/>
<text x="200" y="44" class="d-text d-bold" text-anchor="middle">W (frozen)</text>
<text x="200" y="62" class="d-muted" text-anchor="middle">d_out × d_in</text>
<rect x="120" y="120" width="120" height="26" rx="4" class="d-a"/>
<text x="180" y="138" class="d-text" text-anchor="middle">A: r × d_in</text>
<rect x="256" y="104" width="26" height="70" rx="4" class="d-b"/>
<text x="269" y="190" class="d-muted" text-anchor="middle">B: d_out × r</text>
<path d="M240,133 L254,133" class="d-line" marker-end="url(#d-arrow)"/>
<text x="306" y="145" class="d-muted" text-anchor="start">× α/r</text>
<path d="M280,48 L380,48 L380,86" class="d-line" marker-end="url(#d-arrow)"/>
<path d="M282,140 L300,140 M350,140 L380,140 L380,114" class="d-line" marker-end="url(#d-arrow)"/>
<circle cx="380" cy="100" r="13" class="d-box"/>
<text x="380" y="105" class="d-text d-bold" text-anchor="middle">+</text>
<line x1="393" y1="100" x2="440" y2="100" class="d-line" marker-end="url(#d-arrow)"/>
<text x="452" y="104" class="d-text" text-anchor="start">h = W x + (α/r) B A x</text>
<text x="452" y="130" class="d-muted" text-anchor="start">A starts random, B starts at zero,</text>
<text x="452" y="146" class="d-muted" text-anchor="start">so training begins at the base model.</text>
<text x="452" y="170" class="d-muted" text-anchor="start">Trainable: r (d_in + d_out) instead of</text>
<text x="452" y="186" class="d-muted" text-anchor="start">d_in · d_out; r = 16 on 4096 × 4096 is 0.8%.</text>
</svg>
<figcaption>LoRA runs a narrow two-matrix path beside the frozen weight. After training, B A can be added into W, so the merged model is as fast as the original.</figcaption></figure>

**QLoRA** keeps the frozen base model in 4-bit (NF4) and trains LoRA adapters in 16-bit on top, which fits a 65B model's fine-tuning on one 48 GB GPU.

**Other PEFT methods:**

| Method | What is trained |
|---|---|
| Adapters | small bottleneck MLPs inserted after sublayers (adds inference latency) |
| Prefix / prompt tuning | learned vectors prepended to the keys and values or to the input embeddings |
| IA³ | per-channel scaling vectors |
| DoRA | splits weights into magnitude and direction and applies LoRA to the direction |

**When is full fine-tuning worth it?** When the new domain or task is far from pretraining (a new language, heavy continued pretraining), when you have lots of data and compute, or when the last bit of quality matters. PEFT wins when you need many task-specific variants, have little data, or have limited memory. LoRA also tends to forget less of the base model's abilities.
