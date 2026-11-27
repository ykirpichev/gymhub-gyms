# Point-in-time feature join

Write `solve(events, features, max_age)` that builds training features without leaking the future.

- `events`: a list of `[entity, ts]`: an entity id (a string) and the event time (an int, seconds).
- `features`: a list of `[entity, ts, value]`: a feature value for that entity, computed at time `ts`. Not sorted.
- `max_age`: an int `≥ 0`. A value older than this is stale.

For each event, find the feature rows of the same entity with `ts < event_ts` (**strictly** before: a value computed in the same second may already include the event). Take the one with the largest `ts`; if several share that `ts`, take the one that appears **last** in `features`. Use its `value` if `event_ts − ts ≤ max_age`, otherwise `None`. An entity with no earlier row also gets `None`.

Return a list with one value per event, in the order of `events`.

```
events   = [["u1", 10], ["u1", 25], ["u2", 5], ["u3", 7]]
features = [["u1", 20, "b"], ["u1", 3, "a"], ["u2", 5, "x"], ["u1", 25, "c"]]
solve(events, features, 15)  ->  ["a", "b", None, None]
```

`u1` at 10 sees only the row at 3. `u1` at 25 sees 3 and 20 (the row at 25 is not strictly before). `u2` at 5 has only a row at the same second, and `u3` has none.

#### What interviewers look for

- Saying why the join must be "as of" the event: a plain join on the entity leaks future values into training.
- An efficient approach: group by entity, sort each group by time once, and binary-search each event, `O((E + F) log F)`, rather than scanning all features per event.
- Handling the edges deliberately: strict `<`, ties at the same timestamp, staleness, and unknown entities.

#### Follow-ups

- The features arrive with a processing delay: a value with timestamp `ts` is only queryable at `ts + 30` seconds. How do you model that in the join?
- The offline store holds billions of rows. How would you run this join in Spark or a data warehouse?
- How do you check that the online store serves the same values your training join produced?
