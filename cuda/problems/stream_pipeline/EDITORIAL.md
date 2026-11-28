#### Idea
It is a three-stage pipeline. Each engine processes chunks in order; a stage starts when both its input (the previous
stage of the same chunk) and its engine are ready. The finish time tends to `longest stage + (sum of the other stages) /
chunks` as the chunk count grows.

#### In real code
Pinned host memory, `cudaMemcpyAsync`, and a kernel per chunk in its own stream. More chunks overlap better but each
launch and copy has a fixed cost (microseconds), so very small chunks lose. Consumer GPUs may have a single copy engine,
which serializes copies in and out.

#### Pitfalls
- Letting a later chunk's copy in jump ahead of an earlier one: engines are FIFO per issue order here.
- Starting a kernel when the previous chunk's kernel finished but this chunk's data has not arrived.
