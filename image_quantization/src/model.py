import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        tol: float = 1e-2,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _initialize_centers(self, X: np.ndarray):
        centers = self.rng.choice(X, size=self.k, replace=False)
        self.cluster_centers_ = np.float64(centers)

    def _update_labels(self, X: np.ndarray):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_, axis=2)
        self.labels_ = np.argmin(distances, axis=1)

    def _update_centers(self, X):
        for j in range(len(self.cluster_centers_)):
            cluster = X[np.where(self.labels_ == j)]
            if len(cluster) == 0:
                return
            self.cluster_centers_[j] = np.mean(cluster, axis=0)

    def fit(self, X: np.ndarray):
        self._initialize_centers(X)
        self.labels_ = np.full(X.shape[0], -1)
        self.iterations = 0
        prev_centers = np.copy(self.cluster_centers_)
        for _ in range(self.max_iter):
            self.iterations += 1
            self._update_labels(X)
            self._update_centers(X)
            if np.linalg.norm(self.cluster_centers_ - prev_centers) < self.tol:
                break
            prev_centers = np.copy(self.cluster_centers_)
        return self
