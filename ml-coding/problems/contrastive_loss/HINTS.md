## Hint 1
After normalizing, `img_n @ txt_nᵀ` holds every pairwise cosine similarity. The diagonal holds the positives; the other entries in each row are that row's in-batch negatives. So the targets are simply `0, 1, ..., N-1`.

## Hint 2
Cross-entropy with target `i` on row `i` is `logsumexp(row) - row[i]`. Compute `logsumexp` by subtracting the row max first; with `temperature = 1e-4` a plain `exp` overflows.

## Hint 3
Text → image is the same computation on `logits.T`: it normalizes over the images for each text, which is a different softmax from the image → text one. Average the two mean losses.
