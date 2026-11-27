# Pick the cheapest fix that meets the goal

A 70B model serves a document-summarisation API. Requests send **6,000**-token documents and get **150**-token summaries. Measurements show GPUs spend most of their time in prefill, decode batches are small, and the prefix-cache hit rate is near zero because every document is different. The team must cut cost by half. Which change is most likely to get there?
