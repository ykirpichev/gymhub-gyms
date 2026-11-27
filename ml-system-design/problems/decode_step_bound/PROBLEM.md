# The fastest a decode step can be

The 70B model's weights take **140 GB** in BF16, split across **4** GPUs with **3.35 TB/s** of memory bandwidth each. At batch size 1, every decode step must read all the weights once. Ignoring the KV cache and all overheads, what is the shortest possible decode step, in **milliseconds**?
