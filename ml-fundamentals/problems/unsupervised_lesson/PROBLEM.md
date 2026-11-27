# Learning without labels

Without labels, a model can only find structure: groups (**clustering**) or a smaller description (**dimensionality reduction**).

**Clustering families** differ in what they think a cluster is:

| Method | A cluster is… | Watch out for |
|---|---|---|
| k-means | points near a centroid; minimizes within-cluster squared distance | you choose $k$; assumes round clusters of similar size; sensitive to scale and initialization (use k-means++) |
| Gaussian mixture | a Gaussian blob, fit with expectation-maximization | soft assignments; can model elongated clusters |
| Hierarchical (agglomerative) | a merge tree; cut it at any level | quadratic memory; the linkage rule shapes the result |
| DBSCAN | a dense region; sparse points are noise | finds any shape, needs a density radius |

Don't confuse **k-means** (unsupervised: $k$ is a number of clusters) with **k-nearest neighbours** (supervised: $k$ is a number of labelled neighbours that vote).

**PCA** finds orthogonal directions of largest variance. Center the data; the principal directions are the top eigenvectors of the covariance matrix (equivalently, the right singular vectors of the centered data). Projecting onto the top $k$ keeps the most variance any $k$-dimensional linear projection can, and the eigenvalues say how much each direction explains. PCA is linear, sensitive to feature scale, and unsupervised: the directions of largest variance need not be the ones that separate your classes (that is **LDA**, which uses labels).

**t-SNE** and **UMAP** are nonlinear embeddings for *visualizing* neighbourhoods. Distances between far-apart clusters and cluster sizes in their plots mean little.

**Topic models** and **hidden Markov models** are probabilistic latent-variable models: LDA (latent Dirichlet allocation, a different LDA) explains documents as mixtures of topics, and an HMM explains a sequence through hidden states that follow a Markov chain.
