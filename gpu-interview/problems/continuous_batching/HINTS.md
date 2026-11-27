## Hint 1

Keep a queue of waiting indices sorted by `(arrival, index)` and a dict of running requests to tokens generated.

## Hint 2

If nothing is running and the next request has not arrived, jump `t` to its arrival.
