import heapq


def solve(block_size, num_blocks, ops):
    free = list(range(num_blocks))
    heapq.heapify(free)
    tables, lengths, rejected = {}, {}, []
    for index, op in enumerate(ops):
        if op[0] == "append":
            seq, n = op[1], op[2]
            new_length = lengths.get(seq, 0) + n
            needed = -(-new_length // block_size) - len(tables.get(seq, []))
            if needed > len(free):
                rejected.append(index)
                continue
            table = tables.setdefault(seq, [])
            table.extend(heapq.heappop(free) for _ in range(needed))
            lengths[seq] = new_length
        elif op[0] == "free":
            seq = op[1]
            if seq not in tables:
                raise ValueError(f"unknown sequence {seq!r}")
            for block in tables.pop(seq):
                heapq.heappush(free, block)
            del lengths[seq]
        else:
            raise ValueError(f"unknown op {op[0]!r}")
    return {"tables": tables, "lengths": lengths, "free": len(free), "rejected": rejected}
