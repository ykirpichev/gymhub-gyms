# Coalescing and banks in one page

**Global memory** is read in **32-byte sectors** (L2 and DRAM work in sectors; L1 lines are 128 bytes, four sectors). When the 32 lanes of a warp load, the hardware gathers the distinct sectors their addresses fall in and fetches each once. That is **coalescing**:

- `x[i]` with consecutive `i` across lanes, 4-byte floats: 128 bytes, **4 sectors**, every byte used. 100% efficiency.
- `x[32 * i]`: each lane in its own sector, **32 sectors**, 4 useful bytes out of each 32. 12.5% efficiency: the kernel needs 8 times the bandwidth it seems to.
- `x[i + 1]`: misaligned by 4 bytes, so 5 sectors instead of 4. Not free, but close.

<figure class="diagram"><svg viewBox="0 0 700 266" width="700" role="img" aria-label="Coalesced versus strided loads of one warp">
<text x="10" y="18" class="d-text d-bold" text-anchor="start">x[i]: lane i reads float i</text>
<text x="10" y="41" class="d-muted" text-anchor="start">lanes</text>
<text x="10" y="83" class="d-muted" text-anchor="start">memory</text>
<rect x="60" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="60" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="67.5" y1="44" x2="67.5" y2="70" class="d-line"/>
<rect x="75" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="75" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="82.5" y1="44" x2="82.5" y2="70" class="d-line"/>
<rect x="90" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="90" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="97.5" y1="44" x2="97.5" y2="70" class="d-line"/>
<rect x="105" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="105" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="112.5" y1="44" x2="112.5" y2="70" class="d-line"/>
<rect x="120" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="120" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="127.5" y1="44" x2="127.5" y2="70" class="d-line"/>
<rect x="135" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="135" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="142.5" y1="44" x2="142.5" y2="70" class="d-line"/>
<rect x="150" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="150" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="157.5" y1="44" x2="157.5" y2="70" class="d-line"/>
<rect x="165" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="165" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="172.5" y1="44" x2="172.5" y2="70" class="d-line"/>
<rect x="180" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="180" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="187.5" y1="44" x2="187.5" y2="70" class="d-line"/>
<rect x="195" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="195" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="202.5" y1="44" x2="202.5" y2="70" class="d-line"/>
<rect x="210" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="210" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="217.5" y1="44" x2="217.5" y2="70" class="d-line"/>
<rect x="225" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="225" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="232.5" y1="44" x2="232.5" y2="70" class="d-line"/>
<rect x="240" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="240" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="247.5" y1="44" x2="247.5" y2="70" class="d-line"/>
<rect x="255" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="255" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="262.5" y1="44" x2="262.5" y2="70" class="d-line"/>
<rect x="270" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="270" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="277.5" y1="44" x2="277.5" y2="70" class="d-line"/>
<rect x="285" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="285" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="292.5" y1="44" x2="292.5" y2="70" class="d-line"/>
<rect x="300" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="300" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="307.5" y1="44" x2="307.5" y2="70" class="d-line"/>
<rect x="315" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="315" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="322.5" y1="44" x2="322.5" y2="70" class="d-line"/>
<rect x="330" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="330" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="337.5" y1="44" x2="337.5" y2="70" class="d-line"/>
<rect x="345" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="345" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="352.5" y1="44" x2="352.5" y2="70" class="d-line"/>
<rect x="360" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="360" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="367.5" y1="44" x2="367.5" y2="70" class="d-line"/>
<rect x="375" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="375" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="382.5" y1="44" x2="382.5" y2="70" class="d-line"/>
<rect x="390" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="390" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="397.5" y1="44" x2="397.5" y2="70" class="d-line"/>
<rect x="405" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="405" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="412.5" y1="44" x2="412.5" y2="70" class="d-line"/>
<rect x="420" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="420" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="427.5" y1="44" x2="427.5" y2="70" class="d-line"/>
<rect x="435" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="435" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="442.5" y1="44" x2="442.5" y2="70" class="d-line"/>
<rect x="450" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="450" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="457.5" y1="44" x2="457.5" y2="70" class="d-line"/>
<rect x="465" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="465" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="472.5" y1="44" x2="472.5" y2="70" class="d-line"/>
<rect x="480" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="480" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="487.5" y1="44" x2="487.5" y2="70" class="d-line"/>
<rect x="495" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="495" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="502.5" y1="44" x2="502.5" y2="70" class="d-line"/>
<rect x="510" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="510" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="517.5" y1="44" x2="517.5" y2="70" class="d-line"/>
<rect x="525" y="28" width="15" height="16" rx="0" class="d-a"/>
<rect x="525" y="70" width="15" height="16" rx="0" class="d-b"/>
<line x1="532.5" y1="44" x2="532.5" y2="70" class="d-line"/>
<rect x="60" y="70" width="120" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<text x="120" y="102" class="d-muted" text-anchor="middle">sector 0</text>
<rect x="180" y="70" width="120" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<text x="240" y="102" class="d-muted" text-anchor="middle">sector 1</text>
<rect x="300" y="70" width="120" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<text x="360" y="102" class="d-muted" text-anchor="middle">sector 2</text>
<rect x="420" y="70" width="120" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<text x="480" y="102" class="d-muted" text-anchor="middle">sector 3</text>
<text x="560" y="74" class="d-text" text-anchor="start">4 sectors fetched</text>
<text x="560" y="90" class="d-muted" text-anchor="start">every byte used: 100%</text>
<text x="10" y="140" class="d-text d-bold" text-anchor="start">x[32 * i]: lane i reads float 32·i</text>
<text x="10" y="163" class="d-muted" text-anchor="start">lanes</text>
<text x="10" y="233" class="d-muted" text-anchor="start">memory</text>
<rect x="60" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="75" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="90" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="105" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="120" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="135" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="150" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="165" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="180" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="195" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="210" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="225" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="240" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="255" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="270" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="285" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="300" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="315" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="330" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="345" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="360" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="375" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="390" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="405" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="420" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="435" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="450" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="465" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="480" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="495" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="510" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="525" y="150" width="15" height="16" rx="0" class="d-a"/>
<rect x="60" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="60" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="67.5" y1="166" x2="63" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="60" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="120" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="120" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="82.5" y1="166" x2="123" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="120" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="180" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="180" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="97.5" y1="166" x2="183" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="180" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="240" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="240" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="112.5" y1="166" x2="243" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="240" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="300" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="300" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="127.5" y1="166" x2="303" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="300" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="360" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="360" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="142.5" y1="166" x2="363" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="360" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="420" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="420" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="157.5" y1="166" x2="423" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="420" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<rect x="480" y="220" width="52" height="16" rx="0" class="d-fill"/>
<rect x="480" y="220" width="6.5" height="16" rx="0" class="d-b"/>
<line x1="172.5" y1="166" x2="483" y2="220" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="480" y="220" width="52" height="16" class="d-line" style="stroke:var(--text);stroke-width:2"/>
<text x="300" y="256" class="d-muted" text-anchor="middle">… lanes 8–31 the same: one 32-byte sector each</text>
<text x="560" y="224" class="d-text" text-anchor="start">32 sectors fetched</text>
<text x="560" y="240" class="d-muted" text-anchor="start">4 of 32 bytes used: 12.5%</text>
</svg>
<figcaption>One warp, two access patterns. Consecutive addresses fall in 4 sectors of 32 bytes; a stride of 32 floats puts every lane in its own sector, so the warp moves 8 times the bytes it uses.</figcaption></figure>

A row-major matrix read along a row coalesces; read down a column it does not. That is the whole problem `transpose` solves: read rows, write rows, and turn the tile around in shared memory.

**Shared memory** is split into **32 banks**, 4 bytes wide; word `w` (byte address / 4) lives in bank `w % 32`. A warp's shared-memory access is served in one go when every lane hits a different bank, or several lanes read the *same word* (a broadcast). When `k` lanes hit *different words in the same bank*, the access is replayed `k` times: a **k-way bank conflict**.

The textbook case is a 32 × 32 float tile read down a column: lane `t` reads `tile[t][c]`, word `32t + c`, and all 32 lanes are in bank `c`. A 32-way conflict. Declaring the tile `[32][33]` shifts each row by one bank: word `33t + c` is in bank `(t + c) % 32`, all different. One unused column, no conflicts.

<figure class="diagram"><svg viewBox="0 0 680 236" width="680" role="img" aria-label="Bank numbers of a shared-memory tile, without and with padding">
<text x="30" y="22" class="d-text d-bold" text-anchor="start">float tile[8][8]</text>
<rect x="30" y="34" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="49" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="49" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="49" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="49" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="49" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="49" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="49" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="49" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="56" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="71" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="71" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="71" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="71" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="71" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="71" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="71" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="71" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="78" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="93" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="93" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="93" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="93" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="93" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="93" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="93" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="93" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="100" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="115" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="115" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="115" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="115" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="115" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="115" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="115" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="115" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="122" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="137" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="137" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="137" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="137" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="137" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="137" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="137" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="137" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="144" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="159" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="159" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="159" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="159" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="159" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="159" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="159" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="159" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="166" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="181" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="181" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="181" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="181" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="181" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="181" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="181" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="181" class="d-muted" text-anchor="middle">7</text>
<rect x="30" y="188" width="22" height="22" rx="0" class="d-c"/>
<text x="41" y="203" class="d-muted" text-anchor="middle">0</text>
<rect x="52" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="63" y="203" class="d-muted" text-anchor="middle">1</text>
<rect x="74" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="85" y="203" class="d-muted" text-anchor="middle">2</text>
<rect x="96" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="107" y="203" class="d-muted" text-anchor="middle">3</text>
<rect x="118" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="129" y="203" class="d-muted" text-anchor="middle">4</text>
<rect x="140" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="151" y="203" class="d-muted" text-anchor="middle">5</text>
<rect x="162" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="173" y="203" class="d-muted" text-anchor="middle">6</text>
<rect x="184" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="195" y="203" class="d-muted" text-anchor="middle">7</text>
<text x="30" y="226" class="d-muted" text-anchor="start">column 0: lanes 0–7 all hit bank 0 → 8 replays</text>
<text x="370" y="22" class="d-text d-bold" text-anchor="start">float tile[8][9] (one pad column)</text>
<rect x="370" y="34" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="49" class="d-muted" text-anchor="middle">0</text>
<rect x="392" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="49" class="d-muted" text-anchor="middle">1</text>
<rect x="414" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="49" class="d-muted" text-anchor="middle">2</text>
<rect x="436" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="49" class="d-muted" text-anchor="middle">3</text>
<rect x="458" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="49" class="d-muted" text-anchor="middle">4</text>
<rect x="480" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="49" class="d-muted" text-anchor="middle">5</text>
<rect x="502" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="49" class="d-muted" text-anchor="middle">6</text>
<rect x="524" y="34" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="49" class="d-muted" text-anchor="middle">7</text>
<rect x="546" y="34" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="49" class="d-muted" text-anchor="middle">0</text>
<rect x="370" y="56" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="71" class="d-muted" text-anchor="middle">1</text>
<rect x="392" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="71" class="d-muted" text-anchor="middle">2</text>
<rect x="414" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="71" class="d-muted" text-anchor="middle">3</text>
<rect x="436" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="71" class="d-muted" text-anchor="middle">4</text>
<rect x="458" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="71" class="d-muted" text-anchor="middle">5</text>
<rect x="480" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="71" class="d-muted" text-anchor="middle">6</text>
<rect x="502" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="71" class="d-muted" text-anchor="middle">7</text>
<rect x="524" y="56" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="71" class="d-muted" text-anchor="middle">0</text>
<rect x="546" y="56" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="71" class="d-muted" text-anchor="middle">1</text>
<rect x="370" y="78" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="93" class="d-muted" text-anchor="middle">2</text>
<rect x="392" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="93" class="d-muted" text-anchor="middle">3</text>
<rect x="414" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="93" class="d-muted" text-anchor="middle">4</text>
<rect x="436" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="93" class="d-muted" text-anchor="middle">5</text>
<rect x="458" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="93" class="d-muted" text-anchor="middle">6</text>
<rect x="480" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="93" class="d-muted" text-anchor="middle">7</text>
<rect x="502" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="93" class="d-muted" text-anchor="middle">0</text>
<rect x="524" y="78" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="93" class="d-muted" text-anchor="middle">1</text>
<rect x="546" y="78" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="93" class="d-muted" text-anchor="middle">2</text>
<rect x="370" y="100" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="115" class="d-muted" text-anchor="middle">3</text>
<rect x="392" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="115" class="d-muted" text-anchor="middle">4</text>
<rect x="414" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="115" class="d-muted" text-anchor="middle">5</text>
<rect x="436" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="115" class="d-muted" text-anchor="middle">6</text>
<rect x="458" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="115" class="d-muted" text-anchor="middle">7</text>
<rect x="480" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="115" class="d-muted" text-anchor="middle">0</text>
<rect x="502" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="115" class="d-muted" text-anchor="middle">1</text>
<rect x="524" y="100" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="115" class="d-muted" text-anchor="middle">2</text>
<rect x="546" y="100" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="115" class="d-muted" text-anchor="middle">3</text>
<rect x="370" y="122" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="137" class="d-muted" text-anchor="middle">4</text>
<rect x="392" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="137" class="d-muted" text-anchor="middle">5</text>
<rect x="414" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="137" class="d-muted" text-anchor="middle">6</text>
<rect x="436" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="137" class="d-muted" text-anchor="middle">7</text>
<rect x="458" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="137" class="d-muted" text-anchor="middle">0</text>
<rect x="480" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="137" class="d-muted" text-anchor="middle">1</text>
<rect x="502" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="137" class="d-muted" text-anchor="middle">2</text>
<rect x="524" y="122" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="137" class="d-muted" text-anchor="middle">3</text>
<rect x="546" y="122" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="137" class="d-muted" text-anchor="middle">4</text>
<rect x="370" y="144" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="159" class="d-muted" text-anchor="middle">5</text>
<rect x="392" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="159" class="d-muted" text-anchor="middle">6</text>
<rect x="414" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="159" class="d-muted" text-anchor="middle">7</text>
<rect x="436" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="159" class="d-muted" text-anchor="middle">0</text>
<rect x="458" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="159" class="d-muted" text-anchor="middle">1</text>
<rect x="480" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="159" class="d-muted" text-anchor="middle">2</text>
<rect x="502" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="159" class="d-muted" text-anchor="middle">3</text>
<rect x="524" y="144" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="159" class="d-muted" text-anchor="middle">4</text>
<rect x="546" y="144" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="159" class="d-muted" text-anchor="middle">5</text>
<rect x="370" y="166" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="181" class="d-muted" text-anchor="middle">6</text>
<rect x="392" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="181" class="d-muted" text-anchor="middle">7</text>
<rect x="414" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="181" class="d-muted" text-anchor="middle">0</text>
<rect x="436" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="181" class="d-muted" text-anchor="middle">1</text>
<rect x="458" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="181" class="d-muted" text-anchor="middle">2</text>
<rect x="480" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="181" class="d-muted" text-anchor="middle">3</text>
<rect x="502" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="181" class="d-muted" text-anchor="middle">4</text>
<rect x="524" y="166" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="181" class="d-muted" text-anchor="middle">5</text>
<rect x="546" y="166" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="181" class="d-muted" text-anchor="middle">6</text>
<rect x="370" y="188" width="22" height="22" rx="0" class="d-c"/>
<text x="381" y="203" class="d-muted" text-anchor="middle">7</text>
<rect x="392" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="403" y="203" class="d-muted" text-anchor="middle">0</text>
<rect x="414" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="425" y="203" class="d-muted" text-anchor="middle">1</text>
<rect x="436" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="447" y="203" class="d-muted" text-anchor="middle">2</text>
<rect x="458" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="469" y="203" class="d-muted" text-anchor="middle">3</text>
<rect x="480" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="491" y="203" class="d-muted" text-anchor="middle">4</text>
<rect x="502" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="513" y="203" class="d-muted" text-anchor="middle">5</text>
<rect x="524" y="188" width="22" height="22" rx="0" class="d-box"/>
<text x="535" y="203" class="d-muted" text-anchor="middle">6</text>
<rect x="546" y="188" width="22" height="22" rx="0" class="d-fill"/>
<text x="557" y="203" class="d-muted" text-anchor="middle">7</text>
<text x="370" y="226" class="d-muted" text-anchor="start">column 0: banks 0,1,…,7 → one pass</text>
<text x="250" y="120" class="d-muted" text-anchor="start">lane t reads</text>
<text x="250" y="136" class="d-muted" text-anchor="start">tile[t][0]</text>
</svg>
<figcaption>Each cell shows the bank its float lives in. Drawn with 8 banks to fit; real GPUs have 32, so the fix is <code>tile[32][33]</code>. Reading a column of the unpadded tile is a 32-way conflict.</figcaption></figure>

> [!TIP]
> In the coding exercises below you count sectors and conflicts yourself. On a real GPU, Nsight Compute reports both: *sectors per request* for global loads and *shared memory bank conflicts* per instruction.
