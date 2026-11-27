# When one expert gets every token

A mixture-of-experts layer has 8 experts and sends each token to its top expert. After a few thousand steps one expert receives almost every token and the other seven receive almost none. Throughput falls, and the idle experts barely change.

What failed, and what is the usual fix in the training loss?
