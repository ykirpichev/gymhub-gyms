# Arrays in memory

A NumPy array or a PyTorch tensor is four things: a flat **buffer** of bytes, a **dtype**, a **shape** and **strides**. The strides say how far to jump in the buffer to move one step along each axis. A C-ordered (row-major) `float32` array of shape `(3, 4)` has strides `(16, 4)` in bytes in NumPy, or `(4, 1)` in elements in PyTorch: one row is 4 floats = 16 bytes.

```python
a = np.arange(12, dtype=np.float32).reshape(3, 4)
a.strides          # (16, 4)
a.T.strides        # (4, 16): same buffer, axes swapped
a[:, 1].strides    # (16,): a column, every 16 bytes
a[::2].strides     # (32, 4): every other row
```

<figure class="diagram"><svg viewBox="0 0 680 206" width="680" role="img" aria-label="A 2 × 3 array and its transpose as two views of one buffer">
<text x="10" y="18" class="d-text d-bold" text-anchor="start">a = arange(6).reshape(2, 3)</text>
<text x="360" y="18" class="d-text d-bold" text-anchor="start">a.T: shape (3, 2), same buffer</text>
<rect x="20" y="30" width="36" height="30" rx="3" class="d-a"/>
<text x="38" y="50" class="d-text" text-anchor="middle">0</text>
<rect x="60" y="30" width="36" height="30" rx="3" class="d-a"/>
<text x="78" y="50" class="d-text" text-anchor="middle">1</text>
<rect x="100" y="30" width="36" height="30" rx="3" class="d-a"/>
<text x="118" y="50" class="d-text" text-anchor="middle">2</text>
<rect x="20" y="64" width="36" height="30" rx="3" class="d-a"/>
<text x="38" y="84" class="d-text" text-anchor="middle">3</text>
<rect x="60" y="64" width="36" height="30" rx="3" class="d-a"/>
<text x="78" y="84" class="d-text" text-anchor="middle">4</text>
<rect x="100" y="64" width="36" height="30" rx="3" class="d-a"/>
<text x="118" y="84" class="d-text" text-anchor="middle">5</text>
<text x="20" y="118" class="d-muted" text-anchor="start">strides (12, 4) bytes: next row +12,</text>
<text x="20" y="134" class="d-muted" text-anchor="start">next column +4</text>
<rect x="380" y="30" width="36" height="30" rx="3" class="d-b"/>
<text x="398" y="50" class="d-text" text-anchor="middle">0</text>
<rect x="420" y="30" width="36" height="30" rx="3" class="d-b"/>
<text x="438" y="50" class="d-text" text-anchor="middle">3</text>
<rect x="380" y="64" width="36" height="30" rx="3" class="d-b"/>
<text x="398" y="84" class="d-text" text-anchor="middle">1</text>
<rect x="420" y="64" width="36" height="30" rx="3" class="d-b"/>
<text x="438" y="84" class="d-text" text-anchor="middle">4</text>
<rect x="380" y="98" width="36" height="30" rx="3" class="d-b"/>
<text x="398" y="118" class="d-text" text-anchor="middle">2</text>
<rect x="420" y="98" width="36" height="30" rx="3" class="d-b"/>
<text x="438" y="118" class="d-text" text-anchor="middle">5</text>
<text x="480" y="54" class="d-muted" text-anchor="start">strides (4, 12):</text>
<text x="480" y="70" class="d-muted" text-anchor="start">next row +4,</text>
<text x="480" y="86" class="d-muted" text-anchor="start">next column +12.</text>
<text x="480" y="102" class="d-muted" text-anchor="start">Not contiguous.</text>
<text x="10" y="172" class="d-muted" text-anchor="start">buffer (float32)</text>
<rect x="120" y="156" width="46" height="26" rx="3" class="d-fill"/>
<text x="143" y="174" class="d-text" text-anchor="middle">0</text>
<text x="143" y="198" class="d-muted" text-anchor="middle">+0</text>
<rect x="170" y="156" width="46" height="26" rx="3" class="d-fill"/>
<text x="193" y="174" class="d-text" text-anchor="middle">1</text>
<text x="193" y="198" class="d-muted" text-anchor="middle">+4</text>
<rect x="220" y="156" width="46" height="26" rx="3" class="d-fill"/>
<text x="243" y="174" class="d-text" text-anchor="middle">2</text>
<text x="243" y="198" class="d-muted" text-anchor="middle">+8</text>
<rect x="270" y="156" width="46" height="26" rx="3" class="d-fill"/>
<text x="293" y="174" class="d-text" text-anchor="middle">3</text>
<text x="293" y="198" class="d-muted" text-anchor="middle">+12</text>
<rect x="320" y="156" width="46" height="26" rx="3" class="d-fill"/>
<text x="343" y="174" class="d-text" text-anchor="middle">4</text>
<text x="343" y="198" class="d-muted" text-anchor="middle">+16</text>
<rect x="370" y="156" width="46" height="26" rx="3" class="d-fill"/>
<text x="393" y="174" class="d-text" text-anchor="middle">5</text>
<text x="393" y="198" class="d-muted" text-anchor="middle">+20</text>
</svg>
<figcaption>Both arrays read the same six floats. Only the strides differ, so <code>a.T</code> costs nothing to make, and code that needs contiguous rows has to copy it first.</figcaption></figure>

Operations that only change shape and strides return a **view** that shares the buffer: slicing with `start:stop:step`, transposes and `permute`, `reshape` *when possible*, `view`, `expand` and `broadcast_to`, `squeeze` and `unsqueeze`. Writing through a view changes the original.

Operations that pick arbitrary elements make a **copy**: indexing with an integer array or a boolean mask (*advanced indexing*), `np.concatenate` and `torch.cat`, arithmetic, and `.copy()` or `.clone()`.

An array is **contiguous** when its elements sit in the buffer in row-major order with no gaps. A transpose is not. Some operations need contiguity: PyTorch's `view` refuses a non-contiguous tensor (use `reshape`, which copies when it must, or `.contiguous().view(...)`), and GPU kernels written for row-major input read the wrong elements if you pass a non-contiguous tensor's data pointer.

> [!WARNING]
> `reshape` returns a view *if it can* and a copy otherwise, so code that writes through the result of `reshape` works on one input and silently does nothing on another. If you mean to write into the original, use a view operation you know is a view, or write with indexing.
