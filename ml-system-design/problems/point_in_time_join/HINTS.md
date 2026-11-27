## Hint 1

Group feature rows by entity, then sort each group by `ts` with a stable sort so equal timestamps keep their input order.

## Hint 2

`bisect.bisect_left(times, event_ts)` is the number of rows strictly before the event; the row just before that position is the one you want.
