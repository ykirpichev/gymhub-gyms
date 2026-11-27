# Neural networks from the bottom up

A **feed-forward network** stacks layers $h_{l+1} = \phi(W_l h_l + b_l)$. A dense layer from $n$ inputs to $m$ outputs has $nm + m$ parameters. Without the nonlinearity $\phi$, any stack of layers collapses into one linear map.

**Backpropagation** is the chain rule applied from the loss backward. Each layer receives the gradient of the loss with respect to its output, $\delta$, and passes back $W^\top (\delta \odot \phi')$ while accumulating $\partial L / \partial W = (\delta \odot \phi') h^\top$. One backward pass costs about twice a forward pass.

<figure class="diagram"><svg viewBox="0 0 720 236" width="720" role="img" aria-label="Forward and backward pass through a two-layer classifier">
<rect x="10" y="26" width="108" height="34" rx="5" class="d-box"/>
<text x="64" y="48" class="d-text" text-anchor="middle">x</text>
<rect x="128" y="26" width="108" height="34" rx="5" class="d-a"/>
<text x="182" y="48" class="d-text" text-anchor="middle">z₁ = W₁x + b₁</text>
<line x1="118" y1="43" x2="127" y2="43" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="246" y="26" width="108" height="34" rx="5" class="d-box"/>
<text x="300" y="48" class="d-text" text-anchor="middle">h = ReLU(z₁)</text>
<line x1="236" y1="43" x2="245" y2="43" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="364" y="26" width="108" height="34" rx="5" class="d-a"/>
<text x="418" y="48" class="d-text" text-anchor="middle">z₂ = W₂h + b₂</text>
<line x1="354" y1="43" x2="363" y2="43" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="482" y="26" width="108" height="34" rx="5" class="d-box"/>
<text x="536" y="48" class="d-text" text-anchor="middle">p = softmax(z₂)</text>
<line x1="472" y1="43" x2="481" y2="43" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="600" y="26" width="108" height="34" rx="5" class="d-box"/>
<text x="654" y="48" class="d-text" text-anchor="middle">L = −log p_y</text>
<line x1="590" y1="43" x2="599" y2="43" class="d-line" marker-end="url(#d-arrow)"/>
<text x="10" y="16" class="d-muted" text-anchor="start">forward: save each layer's input</text>
<path d="M708,78 L14,78" class="d-line d-dash" marker-end="url(#d-arrow)"/>
<text x="708" y="96" class="d-muted" text-anchor="end">backward: one local derivative per step</text>
<text x="20" y="122" class="d-muted" text-anchor="start">1.</text>
<text x="40" y="122" class="d-text" text-anchor="start">∂L/∂z₂ = p − onehot(y)</text>
<text x="300" y="122" class="d-muted" text-anchor="start">softmax and cross-entropy together</text>
<text x="20" y="156" class="d-muted" text-anchor="start">2.</text>
<text x="40" y="156" class="d-text" text-anchor="start">∂L/∂W₂ = ∂L/∂z₂ · hᵀ</text>
<text x="300" y="156" class="d-muted" text-anchor="start">∂L/∂h = W₂ᵀ · ∂L/∂z₂</text>
<text x="20" y="190" class="d-muted" text-anchor="start">3.</text>
<text x="40" y="190" class="d-text" text-anchor="start">∂L/∂z₁ = ∂L/∂h ⊙ 1[z₁ > 0]</text>
<text x="300" y="190" class="d-muted" text-anchor="start">ReLU passes or blocks</text>
<text x="20" y="224" class="d-muted" text-anchor="start">4.</text>
<text x="40" y="224" class="d-text" text-anchor="start">∂L/∂W₁ = ∂L/∂z₁ · xᵀ</text>
<text x="300" y="224" class="d-muted" text-anchor="start">∂L/∂b₁ = ∂L/∂z₁</text>
</svg>
<figcaption>The forward pass (top) keeps each layer's input; the backward pass walks the same chain from the loss, multiplying the incoming gradient by one local derivative per step. Every weight gradient is that gradient times the saved input, transposed.</figcaption></figure>

**Activations:**

| Activation | Use |
|---|---|
| ReLU, GELU, SiLU | hidden layers; non-saturating for positive inputs, so gradients flow |
| sigmoid | a probability per output (binary or multi-label) |
| softmax | one distribution over mutually exclusive classes |
| tanh | bounded outputs; inside LSTM gates |

**Training deep networks** needs care with initialization (He or Xavier scaling keeps activations from exploding or vanishing), normalization (batch norm in CNNs, layer norm in sequence models), residual connections (gradients flow through the identity path) and regularization (dropout, weight decay).

**Before transformers:**

- **CNNs** slide small shared filters over an image. Weight sharing and local connectivity give translation equivariance with few parameters; pooling or strides grow the receptive field.
- **RNNs** carry a hidden state through a sequence, $h_t = \phi(W h_{t-1} + U x_t)$. Repeated multiplication by $W$ makes gradients vanish or explode over long sequences. **LSTMs** and **GRUs** add gates and an additive cell state so information and gradients survive longer.
- **Attention** lets every position read directly from every other, replacing recurrence; stacked with MLPs and residual connections it becomes the transformer.

**Transfer learning:** start from a network pretrained on a large dataset, replace its head, and fine-tune (all layers with a small learning rate, or only the top layers if data is scarce). Early layers learn general features (edges, textures), later layers task-specific ones.
