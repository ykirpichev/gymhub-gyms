# What the guidance scale does

A latent diffusion model is trained with the text condition dropped on some steps, so the same network can denoise with a caption and without one. At sampling time the implementer sets

`guided = uncond + s × (cond − uncond)`

with scale `s = 5`.

What does a scale greater than 1 do?
