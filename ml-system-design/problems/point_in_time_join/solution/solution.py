import bisect
from collections import defaultdict


def solve(events, features, max_age):
    rows = defaultdict(list)
    for order, (entity, ts, value) in enumerate(features):
        rows[entity].append((ts, order, value))
    index = {}
    for entity, items in rows.items():
        items.sort()  # by time, then by input order: the last of equal times comes last
        index[entity] = ([ts for ts, _, _ in items], [value for _, _, value in items])
    out = []
    for entity, ts in events:
        if entity not in index:
            out.append(None)
            continue
        times, values = index[entity]
        i = bisect.bisect_left(times, ts)  # rows strictly before ts
        if i == 0 or ts - times[i - 1] > max_age:
            out.append(None)
        else:
            out.append(values[i - 1])
    return out
