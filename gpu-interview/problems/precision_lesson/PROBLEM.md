# Number formats in one page

A floating-point number stores a sign, an exponent (the range) and a mantissa (the precision):

| Format | Exponent bits | Mantissa bits | Largest finite value | Relative step near 1 | Typical use |
|---|---|---|---|---|---|
| FP32 | 8 | 23 | ≈ 3.4 × 10³⁸ | ≈ 1.2 × 10⁻⁷ | accumulators, master weights, optimizer state |
| BF16 | 8 | 7 | ≈ 3.4 × 10³⁸ | ≈ 7.8 × 10⁻³ | training and inference weights and activations |
| FP16 | 5 | 10 | 65,504 | ≈ 9.8 × 10⁻⁴ | inference; training with loss scaling |
| FP8 E4M3 | 4 | 3 | 448 | 0.125 | matmul inputs with per-tensor or per-block scales |
| FP8 E5M2 | 5 | 2 | 57,344 | 0.25 | gradients, where range matters more |

<figure class="diagram"><svg viewBox="0 0 700 222" width="700" role="img" aria-label="Bit layouts of FP32, BF16, FP16 and two FP8 formats">
<text x="10" y="39" class="d-text d-bold" text-anchor="start">FP32</text>
<rect x="100" y="22" width="15" height="24" rx="0" class="d-c"/>
<rect x="115" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="130" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="145" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="160" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="175" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="190" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="205" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="220" y="22" width="15" height="24" rx="0" class="d-a"/>
<rect x="235" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="250" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="265" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="280" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="295" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="310" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="325" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="340" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="355" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="370" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="385" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="400" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="415" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="430" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="445" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="460" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="475" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="490" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="505" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="520" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="535" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="550" y="22" width="15" height="24" rx="0" class="d-b"/>
<rect x="565" y="22" width="15" height="24" rx="0" class="d-b"/>
<text x="588" y="39" class="d-muted" text-anchor="start">1 + 8 + 23 bits</text>
<text x="10" y="73" class="d-text d-bold" text-anchor="start">BF16</text>
<rect x="100" y="56" width="15" height="24" rx="0" class="d-c"/>
<rect x="115" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="130" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="145" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="160" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="175" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="190" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="205" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="220" y="56" width="15" height="24" rx="0" class="d-a"/>
<rect x="235" y="56" width="15" height="24" rx="0" class="d-b"/>
<rect x="250" y="56" width="15" height="24" rx="0" class="d-b"/>
<rect x="265" y="56" width="15" height="24" rx="0" class="d-b"/>
<rect x="280" y="56" width="15" height="24" rx="0" class="d-b"/>
<rect x="295" y="56" width="15" height="24" rx="0" class="d-b"/>
<rect x="310" y="56" width="15" height="24" rx="0" class="d-b"/>
<rect x="325" y="56" width="15" height="24" rx="0" class="d-b"/>
<text x="348" y="73" class="d-muted" text-anchor="start">1 + 8 + 7 bits</text>
<text x="10" y="107" class="d-text d-bold" text-anchor="start">FP16</text>
<rect x="100" y="90" width="15" height="24" rx="0" class="d-c"/>
<rect x="115" y="90" width="15" height="24" rx="0" class="d-a"/>
<rect x="130" y="90" width="15" height="24" rx="0" class="d-a"/>
<rect x="145" y="90" width="15" height="24" rx="0" class="d-a"/>
<rect x="160" y="90" width="15" height="24" rx="0" class="d-a"/>
<rect x="175" y="90" width="15" height="24" rx="0" class="d-a"/>
<rect x="190" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="205" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="220" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="235" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="250" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="265" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="280" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="295" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="310" y="90" width="15" height="24" rx="0" class="d-b"/>
<rect x="325" y="90" width="15" height="24" rx="0" class="d-b"/>
<text x="348" y="107" class="d-muted" text-anchor="start">1 + 5 + 10 bits</text>
<text x="10" y="141" class="d-text d-bold" text-anchor="start">FP8 E4M3</text>
<rect x="100" y="124" width="15" height="24" rx="0" class="d-c"/>
<rect x="115" y="124" width="15" height="24" rx="0" class="d-a"/>
<rect x="130" y="124" width="15" height="24" rx="0" class="d-a"/>
<rect x="145" y="124" width="15" height="24" rx="0" class="d-a"/>
<rect x="160" y="124" width="15" height="24" rx="0" class="d-a"/>
<rect x="175" y="124" width="15" height="24" rx="0" class="d-b"/>
<rect x="190" y="124" width="15" height="24" rx="0" class="d-b"/>
<rect x="205" y="124" width="15" height="24" rx="0" class="d-b"/>
<text x="228" y="141" class="d-muted" text-anchor="start">1 + 4 + 3 bits</text>
<text x="10" y="175" class="d-text d-bold" text-anchor="start">FP8 E5M2</text>
<rect x="100" y="158" width="15" height="24" rx="0" class="d-c"/>
<rect x="115" y="158" width="15" height="24" rx="0" class="d-a"/>
<rect x="130" y="158" width="15" height="24" rx="0" class="d-a"/>
<rect x="145" y="158" width="15" height="24" rx="0" class="d-a"/>
<rect x="160" y="158" width="15" height="24" rx="0" class="d-a"/>
<rect x="175" y="158" width="15" height="24" rx="0" class="d-a"/>
<rect x="190" y="158" width="15" height="24" rx="0" class="d-b"/>
<rect x="205" y="158" width="15" height="24" rx="0" class="d-b"/>
<text x="228" y="175" class="d-muted" text-anchor="start">1 + 5 + 2 bits</text>
<rect x="100" y="200" width="14" height="14" rx="0" class="d-c"/>
<text x="120" y="212" class="d-muted" text-anchor="start">sign</text>
<rect x="170" y="200" width="14" height="14" rx="0" class="d-a"/>
<text x="190" y="212" class="d-muted" text-anchor="start">exponent: range</text>
<rect x="300" y="200" width="14" height="14" rx="0" class="d-b"/>
<text x="320" y="212" class="d-muted" text-anchor="start">mantissa: precision</text>
</svg>
<figcaption>BF16 is FP32 with the mantissa cut to 7 bits: same exponent, same range. FP16 spends its bits on mantissa instead, so it is more precise near 1 and overflows at 65,504.</figcaption></figure>

**BF16 keeps FP32's range** and gives up precision, so values that fit in FP32 fit in BF16, and training rarely overflows. **FP16 keeps more precision** but overflows above 65,504 and flushes small gradients to zero, which is why FP16 training needs *loss scaling*.

**Tensor Cores** multiply low-precision inputs and **accumulate in FP32**. Summing thousands of products in BF16 would lose most of their low bits: once the running sum is 256, adding 1 in BF16 does nothing at all.

**Integer quantization** stores `q = round(x / scale)` in int8 (−127 to 127) or int4 (−8 to 7), with a scale per tensor, per row (channel) or per group of, say, 128 values. Smaller groups mean smaller error and more scales to store. An int8 matmul multiplies the integers, accumulates in int32, then multiplies by both scales once at the end.

> [!TIP]
> Why quantize? For batch-1 decoding, which is memory-bound, int4 weights move a quarter of the bytes of BF16, so the bound on tokens per second is about four times higher. For large-batch prefill, which is compute-bound, the gain comes from faster integer or FP8 Tensor Core math instead.
