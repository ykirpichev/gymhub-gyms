# Where the norm sits

You are training a 32-layer decoder. With the original post-norm block, `Norm(x + SubLayer(x))`, the loss spikes unless the warmup is long. The same depth trains with a short warmup after you switch to a pre-norm block, `x + SubLayer(Norm(x))`.

What is the main reason pre-norm is easier to train this deep?
