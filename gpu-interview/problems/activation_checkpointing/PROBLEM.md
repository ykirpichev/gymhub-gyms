# Recompute instead of store

A model trains fine at micro-batch size 1 but runs out of memory at 4, and the profile shows activations take most of the memory. Your colleague enables activation checkpointing on every transformer block. What should you expect?
