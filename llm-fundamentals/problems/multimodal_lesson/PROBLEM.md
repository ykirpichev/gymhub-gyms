# Vision-language models and diffusion

**CLIP** trains an image encoder and a text encoder together so matching image-caption pairs have similar embeddings. For a batch of $N$ pairs, it computes all $N \times N$ similarities and applies a cross-entropy loss in both directions, so each image must pick its own caption among $N$ (and vice versa). The result: zero-shot classification by comparing an image with prompts like "a photo of a {class}", and embeddings for search.

**Vision-language models (VLMs)** let an LLM read images. A common recipe: a pretrained vision encoder (often CLIP-like) turns the image into patch embeddings; a **projector** (a linear layer or a small MLP) maps them into the LLM's token-embedding space; the LLM reads them as if they were tokens, followed by the text. Training usually first aligns the projector on image-caption pairs, then instruction-tunes the whole stack. Alternatives use cross-attention layers into the LLM.

**Vision-language-action models (VLAs)** extend this to robots: the model reads camera images and an instruction and outputs **actions** (discretized into tokens, or produced by a separate action head), trained on robot demonstrations on top of a VLM's web-scale knowledge.

**Diffusion models** generate by learning to reverse a noising process. Training adds Gaussian noise at a random level $t$ to a clean sample and teaches a network to predict that noise (or the clean sample). Generating starts from pure noise and denoises step by step. Compared with autoregressive generation, diffusion refines the whole output in parallel, which suits images, but needs many network evaluations per sample. Ways to need fewer: better samplers (DDIM, DPM-Solver), distillation into few-step models, and flow-matching formulations. **Latent diffusion** runs the process in a compressed autoencoder latent space instead of pixels.

**Classifier-free guidance** trains one network both with and without the conditioning (the prompt is dropped some of the time). At sampling time it extrapolates from the unconditional prediction toward the conditional one:

$$\hat{\epsilon} = \epsilon_\text{uncond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond}).$$

$w = 1$ is plain conditional sampling; larger $w$ follows the prompt more closely at the cost of diversity and, if too large, image quality.
