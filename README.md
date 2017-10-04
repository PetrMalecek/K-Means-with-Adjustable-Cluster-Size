An approximation algorithm for k-means clustering allowing for adjustable cluster size.

This algorithm is relatively fast and converges quickly; it is based on Lloyd's algorithm for calculating k-means. It runs as follows:

1. Initialize the centers with k-means.
2. For each center, calculate distance weights according to the relative number of items in each cluster from an "ideal" value. This value can be determined by the user, in the default example, it is set as an equal number of items in all clusters. These weights are applied to distances between centers and items, so that all items are more "attracted" to centers serving low number of items, and "repelled" by centers serving high number of items.
3. Recalculate the number of items in each cluster given weights obtained in indent 2. The objective function is the sum of squared distances from each center, as in the standard k-means.
4. Iterate between indents 2. and 3. until convergence is reached.
5. Shift the centroids to their final positions.

This method converges quite rapidly, and may be suitable for a large number of items. However, it is still an approximation algorithm to the global optimum, which can be defined as a sum of squared distances from each center conditional upon given cluster size.
