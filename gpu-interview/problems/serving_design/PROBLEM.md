# Design an LLM serving deployment

An interviewer says: *"We want to serve a 70B-parameter chat model. Users expect the first token in under a second and then at least 20 tokens per second. Traffic peaks at 200 concurrent conversations with up to 8K tokens of context. We have 80 GB GPUs. Walk me through the design."*

Answer as you would out loud, with rough numbers.
