#### Idea
Field `f` of particle `i` sits at `aos[i * fields + f]`, so a slice with step `fields` starting at `f` collects it.

#### Why it matters on a GPU
In AoS, consecutive threads reading `x` are `fields` elements apart: a stride of 4 floats wastes 3/4 of every sector.
In SoA they are neighbours. When a kernel reads all fields together, AoS with a vector load (`float4` per particle) is
also fine; the bad case is reading one field of a wide struct.
