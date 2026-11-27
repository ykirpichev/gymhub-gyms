# Positions, attention variants and the KV cache

Attention by itself is **permutation-invariant**: shuffle the tokens and each output just moves with its input. Order has to be injected.

| Method | How | Notes |
|---|---|---|
| Learned absolute | add a trained vector per position | can't go past the trained length |
| Sinusoidal | add fixed sines and cosines of many frequencies | the original transformer |
| **RoPE** | rotate pairs of query and key dimensions by an angle proportional to the position | the score then depends on the *relative* offset; used by Llama, Qwen, Mistral; extended to longer contexts by rescaling its frequencies |
| ALiBi | subtract a per-head penalty proportional to distance from the scores | extrapolates to longer inputs |

**The KV cache.** When generating, each new token attends to all earlier ones. Their keys and values don't change, so they are stored rather than recomputed. Per token, the cache holds

$$2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value}.$$

For a 32-layer model with 32 KV heads of width 128 in FP16, that is $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes, half a megabyte per token. A batch of 32 conversations with 8K tokens each then needs 128 GiB, more than the model's weights. The KV cache, not the weights, usually limits how many requests a server can batch.

**Sharing keys and values across heads** shrinks it:

- **MHA**: every query head has its own key and value head.
- **GQA** (grouped-query): query heads are split into groups that share one KV head. Llama-style models use 8 KV heads for 32 or 64 query heads: 4 to 8 times less cache, with little quality loss.
- **MQA** (multi-query): one KV head for all query heads. The smallest cache, but a larger quality cost.
- **MLA** (multi-head latent attention, DeepSeek): caches a small compressed latent per token and projects keys and values out of it.

<figure class="diagram"><svg viewBox="0 0 680 180" width="680" role="img" aria-label="Query heads sharing key and value heads in MHA, GQA and MQA">
<text x="110" y="18" class="d-text d-bold" text-anchor="middle">MHA: 8 KV heads</text>
<rect x="14" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="39" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="64" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="89" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="114" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="139" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="164" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="189" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="16.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="24" y1="54" x2="26.5" y2="106" class="d-line"/>
<rect x="41.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="49" y1="54" x2="51.5" y2="106" class="d-line"/>
<rect x="66.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="74" y1="54" x2="76.5" y2="106" class="d-line"/>
<rect x="91.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="99" y1="54" x2="101.5" y2="106" class="d-line"/>
<rect x="116.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="124" y1="54" x2="126.5" y2="106" class="d-line"/>
<rect x="141.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="149" y1="54" x2="151.5" y2="106" class="d-line"/>
<rect x="166.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="174" y1="54" x2="176.5" y2="106" class="d-line"/>
<rect x="191.5" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="199" y1="54" x2="201.5" y2="106" class="d-line"/>
<text x="110" y="146" class="d-muted" text-anchor="middle">cache: 8 × K, V per token</text>
<text x="335" y="18" class="d-text d-bold" text-anchor="middle">GQA: 2 KV heads</text>
<rect x="239" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="264" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="289" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="314" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="339" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="364" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="389" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="414" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="279.0" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="249" y1="54" x2="289.0" y2="106" class="d-line"/>
<line x1="274" y1="54" x2="289.0" y2="106" class="d-line"/>
<line x1="299" y1="54" x2="289.0" y2="106" class="d-line"/>
<line x1="324" y1="54" x2="289.0" y2="106" class="d-line"/>
<rect x="379.0" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="349" y1="54" x2="389.0" y2="106" class="d-line"/>
<line x1="374" y1="54" x2="389.0" y2="106" class="d-line"/>
<line x1="399" y1="54" x2="389.0" y2="106" class="d-line"/>
<line x1="424" y1="54" x2="389.0" y2="106" class="d-line"/>
<text x="335" y="146" class="d-muted" text-anchor="middle">cache: 2 × K, V per token (¼)</text>
<text x="560" y="18" class="d-text d-bold" text-anchor="middle">MQA: 1 KV head</text>
<rect x="464" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="489" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="514" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="539" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="564" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="589" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="614" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="639" y="34" width="20" height="20" rx="3" class="d-a"/>
<rect x="554.0" y="106" width="20" height="20" rx="3" class="d-b"/>
<line x1="474" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="499" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="524" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="549" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="574" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="599" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="624" y1="54" x2="564.0" y2="106" class="d-line"/>
<line x1="649" y1="54" x2="564.0" y2="106" class="d-line"/>
<text x="560" y="146" class="d-muted" text-anchor="middle">cache: 1 × K, V per token (⅛)</text>
<text x="10" y="74" class="d-muted" text-anchor="start"></text>
<rect x="470" y="160" width="12" height="12" rx="2" class="d-a"/>
<text x="488" y="170" class="d-muted" text-anchor="start">query head</text>
<rect x="570" y="160" width="12" height="12" rx="2" class="d-b"/>
<text x="588" y="170" class="d-muted" text-anchor="start">key/value head</text>
</svg>
<figcaption>Eight query heads and the key/value heads they read. The KV cache stores only the green heads, so grouped-query attention cuts it by the group size while every query head keeps its own projection.</figcaption></figure>

Other ways to cut cache memory: quantize the cache to FP8 or int8, use sliding-window attention on some layers, and page the cache in blocks so memory isn't reserved for the maximum length (PagedAttention).
