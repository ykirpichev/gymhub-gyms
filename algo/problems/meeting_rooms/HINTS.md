# Hints for Meeting rooms

## Hint 1

Count, for every moment, how many meetings are running. The answer is the maximum of that count.

## Hint 2

Sort the start times and the end times separately. Walk the starts in order with a pointer into the ends: before counting a start, release every meeting whose end is `<=` that start.

## Hint 3

Another way: sort meetings by start and keep a min-heap of end times of the rooms in use. If the earliest end is `<=` the new start, reuse that room (pop it). Push the new end. The heap's largest size is the answer.
