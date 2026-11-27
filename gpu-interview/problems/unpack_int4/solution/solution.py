def solve(packed):
    out = []
    for byte in packed:
        for nibble in (byte & 0xF, (byte >> 4) & 0xF):
            out.append(nibble - 16 if nibble >= 8 else nibble)
    return out
