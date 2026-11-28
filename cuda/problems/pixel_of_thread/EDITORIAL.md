#### Idea
`x` is the column and `y` the row, exactly as in the kernel; then check both against the image and flatten row-major.

#### Pitfalls
- Swapping width and height in the flattening (`col * height + row` is column-major).
- Checking only one of the two bounds: the right edge and the bottom edge both overhang with a rounded-up grid.
- Making `x` the row: neighbouring threads (consecutive `threadIdx.x`) then touch addresses `width` apart, and the
  loads stop coalescing.
