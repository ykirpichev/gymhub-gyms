## Hint 1
Squared distance is enough: `(X[:, None, :] - centroids[None, :, :]) ** 2`, summed over the feature axis. `np.argmin` along the centroid axis returns the lowest index when distances tie.

## Hint 2
Update cluster `k` with `X[labels == k].mean(axis=0)`. If that slice is empty, copy the previous centroid. Do not move it to the origin and do not drop it.

## Hint 3
Loop at most `max_iter` times and break when `np.max(np.abs(updated - centroids)) <= 1e-8`. `max_iter = 0` skips the loop. Compute the labels from the final centroids, not from the centroids you started the last update with, unless those are already the ones you return.
