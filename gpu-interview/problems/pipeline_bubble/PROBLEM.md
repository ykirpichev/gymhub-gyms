# The pipeline bubble

A model is split into `p = 4` pipeline stages and each batch into `m = 12` micro-batches, run with a GPipe schedule (all forwards, then all backwards). What percentage of each GPU's time is the idle bubble?
