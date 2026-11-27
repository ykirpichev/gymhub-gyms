# The roofline in one page

Every kernel has two lower bounds on its time:

$$t \ge \frac{\text{bytes moved}}{\text{bandwidth}} \qquad t \ge \frac{\text{FLOPs}}{\text{peak FLOP/s}}$$

The larger one wins. Their ratio is the kernel's **arithmetic intensity**, FLOPs per byte moved to and from DRAM. The GPU has its own ratio, the **ridge point**, peak FLOP/s over bandwidth. A kernel whose intensity is below the ridge point is **memory-bound**: no amount of faster math helps, only moving fewer bytes. Above it, it is **compute-bound**.

<figure class="diagram"><svg viewBox="0 0 680 290" width="680" role="img" aria-label="Roofline of a GPU with 3 TB/s and 1,000 TFLOP/s">
<rect x="70" y="20" width="560" height="220" rx="0" class="d-fill"/>
<path d="M118.8,240.0 L492.1,35.4 L630.0,35.4" class="d-line" style="stroke:var(--text);stroke-width:2.5"/>
<line x1="70.0" y1="240" x2="70.0" y2="244" class="d-line"/>
<text x="70.0" y="257" class="d-muted" text-anchor="middle">0.01</text>
<line x1="163.33333333333331" y1="240" x2="163.33333333333331" y2="244" class="d-line"/>
<text x="163.33333333333331" y="257" class="d-muted" text-anchor="middle">0.1</text>
<line x1="256.66666666666663" y1="240" x2="256.66666666666663" y2="244" class="d-line"/>
<text x="256.66666666666663" y="257" class="d-muted" text-anchor="middle">1</text>
<line x1="350.0" y1="240" x2="350.0" y2="244" class="d-line"/>
<text x="350.0" y="257" class="d-muted" text-anchor="middle">10</text>
<line x1="443.3333333333333" y1="240" x2="443.3333333333333" y2="244" class="d-line"/>
<text x="443.3333333333333" y="257" class="d-muted" text-anchor="middle">100</text>
<line x1="536.6666666666667" y1="240" x2="536.6666666666667" y2="244" class="d-line"/>
<text x="536.6666666666667" y="257" class="d-muted" text-anchor="middle">1000</text>
<line x1="630.0" y1="240" x2="630.0" y2="244" class="d-line"/>
<text x="630.0" y="257" class="d-muted" text-anchor="middle">10000</text>
<line x1="66" y1="188.84946158901712" x2="70" y2="188.84946158901712" class="d-line"/>
<text x="62" y="192.84946158901712" class="d-muted" text-anchor="end">1</text>
<line x1="66" y1="137.69892317803425" x2="70" y2="137.69892317803425" class="d-line"/>
<text x="62" y="141.69892317803425" class="d-muted" text-anchor="end">10</text>
<line x1="66" y1="86.54838476705135" x2="70" y2="86.54838476705135" class="d-line"/>
<text x="62" y="90.54838476705135" class="d-muted" text-anchor="end">100</text>
<line x1="66" y1="35.39784635606847" x2="70" y2="35.39784635606847" class="d-line"/>
<text x="62" y="39.39784635606847" class="d-muted" text-anchor="end">1000</text>
<text x="350.0" y="274" class="d-muted" text-anchor="middle">arithmetic intensity (FLOPs per byte of DRAM traffic), log scale</text>
<text x="18" y="130.0" class="d-muted" text-anchor="middle" transform="rotate(-90 18 130.0)">TFLOP/s, log scale</text>
<line x1="492.13534955949814" y1="35.39784635606847" x2="492.13534955949814" y2="240" class="d-line d-dash"/>
<text x="498.13534955949814" y="232" class="d-muted" text-anchor="start">ridge ≈ 333</text>
<text x="312.85893252394317" y="123.64875981064569" class="d-muted" text-anchor="middle" transform="rotate(-28.7 313 124)">memory roof: 3 TB/s × intensity</text>
<text x="573.8077341427236" y="27.39784635606847" class="d-muted" text-anchor="middle">compute roof: 1,000 TFLOP/s</text>
<circle cx="172.4" cy="210.6" r="6" class="d-a"/>
<text x="182.37826788075193" y="228.63799159098812" class="d-text" text-anchor="start">y = x + 1 (FP32)</text>
<circle cx="256.7" cy="164.4" r="6" class="d-b"/>
<text x="266.66666666666663" y="182.44445252278268" class="d-text" text-anchor="start">RMSNorm (BF16)</text>
<circle cx="549.3" cy="35.4" r="6" class="d-c"/>
<text x="500.13534955949814" y="61.39784635606847" class="d-text" text-anchor="start">4096³ GEMM (BF16)</text>
</svg>
<figcaption>The roofline of the interview GPU (BF16 Tensor Cores). Left of the ridge a kernel is memory-bound and its speed is bandwidth times intensity; right of it, compute-bound.</figcaption></figure>

Worked example on the interview GPU (3 TB/s, 1,000 TFLOP/s BF16 on Tensor Cores, 60 TFLOP/s FP32):

- `y = x + 1` on FP32: one FLOP per element, 8 bytes moved (read 4, write 4). Intensity 1/8. Far below any ridge point: memory-bound. For 1 GiB of input the bound is 2 GiB / 3 TB/s ≈ 0.72 ms.
- An `M = N = K = 4096` BF16 matmul: $2MNK \approx 1.37 \times 10^{11}$ FLOPs over at least $3 \times 4096^2 \times 2$ bytes ≈ 100 MB. Intensity ≈ 1,365, above the BF16 ridge point of 333: compute-bound, ≥ 137 µs.

> [!NOTE]
> The bytes in the formula are the *minimum*: each input read once, each output written once. A real kernel moves more (re-reads, spills, poor caching), and the gap between your measured time and the bound is what you are trying to close. "We're at 80% of the memory roofline" is a sentence interviewers want to hear.

Three habits that make estimates fast:

1. Count bytes per element first: dtype size times (inputs read + outputs written).
2. Write FLOPs of a matmul as $2MNK$ (a multiply and an add).
3. Round the GPU's numbers (3 TB/s, 1 PFLOP/s) and the answer to two significant figures. Interviewers care about the order of magnitude and the reasoning.
