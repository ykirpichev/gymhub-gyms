# Unpack int4 weights

4-bit weights are stored two per byte. Value `2i` sits in the **low** nibble (bits 0–3) of byte `i` and value `2i + 1` in the **high** nibble (bits 4–7). Each nibble is a signed two's-complement integer from −8 to 7: `0x0`–`0x7` are 0 to 7, `0x8`–`0xF` are −8 to −1.

Write `solve(packed)` that takes a list of byte values (ints from 0 to 255) and returns the list of `2 × len(packed)` signed values.

```
solve([0x21])   ->  [1, 2]
solve([0xF8])   ->  [-8, -1]
```

#### What interviewers look for

- Masks and shifts (`b & 0xF`, `b >> 4`) and the sign extension of a nibble: subtract 16 when the value is 8 or more.
- Keeping the low-nibble-first order straight.

#### Follow-ups

- Write the packer. What should it do with a value outside −8 to 7?
- On a GPU, threads unpack eight int4 values from one 32-bit load. How would you lay out the bytes so that each thread's eight values are the ones it needs, and how do you convert them to FP16 quickly?
- Why might a kernel store weights in an interleaved, not row-major, order?
