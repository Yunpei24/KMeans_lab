import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """
    K-Means clustering algorithm implemented from scratch.

    Parameters
    ----------
    k : int, optional (default=3)
        The number of clusters to form.
    max_iter : int, optional (default=100)
        Maximum number of iterations of the k-means algorithm for a
        single run.
    plot_during_fitting : bool, optional (default=False)
        If True, plots clusters at each iteration. Useful for visualization
        but can be slow.
    random_seed : int, optional (default=None)
        Seed for random number generation for centroid initialization.
        Use an int to make the randomness deterministic.
    """
    def __init__(self, k=3, max_iter=100, plot_during_fitting=False, random_seed=None):
        if k <= 0:
            raise ValueError("Number of clusters k must be positive.")
        self.k = k
        self.max_iter = max_iter
        self.plot_during_fitting = plot_during_fitting
        self.random_seed = random_seed
        self.centroids = None
        self.clusters = None
        self.labels = None 
        self.history_centroids = []
        self.losses = []
        self.convergence_iteration = None

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def _initialize_centroid(self, X):
        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError(f"Number of clusters k ({self.k}) cannot be greater than "
                             f"the number of samples ({n_samples}).")
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        self.history_centroids.append(self.centroids.copy())

    def euclidean_distance(self, x1, x2):
        """Computes the Euclidean distance between two points or arrays of points."""
        return np.linalg.norm(x1 - x2)
    
    def _plot_current_state(self, X, iteration_num):
        """Plots the current state of clusters and centroids."""
        if self.centroids is None or self.clusters is None:
            print("Warning: Centroids or clusters not initialized for plotting.")
            return
        
        plt.figure(figsize=(8, 6))
        colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for cluster_idx, point_indices_in_cluster in enumerate(self.clusters):
            if point_indices_in_cluster:
                points = X[point_indices_in_cluster]
                color = colors_palette[cluster_idx % len(colors_palette)]
                plt.scatter(points[:, 0], points[:, 1], color=color, label=f'Cluster {cluster_idx}', alpha=0.7)
        
        if self.centroids is not None and self.centroids.shape[0] > 0:
             plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='X', s=120, label='Centroids', zorder=5)
        
        plt.title(f'K-Means Clustering (Iteration: {iteration_num})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        if self.k > 0 and any(self.clusters): plt.legend()
        plt.show()

    def _closest_centroid(self, x, centroids):
        """Finds the index of the closest centroid to a single point x."""
        distances = [self.euclidean_distance(x, centroid) for centroid in centroids]
        return np.argmin(distances)
    
    def _create_clusters(self, X, current_centroids):
        """Assigns each data point in X to the closest centroid."""
        self.clusters = [[] for _ in range(self.k)]
        for idx, x_sample in enumerate(X):
            centroid_idx = self._closest_centroid(x_sample, current_centroids)
            self.clusters[centroid_idx].append(idx)
    
    def _calculate_new_centroids(self, X, old_centroids_for_empty):
        """
        Calculates new centroids as the mean of the points in each cluster.
        Handles empty clusters by keeping their previous centroid position.
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster:
                cluster_mean = np.mean(X[cluster], axis=0)
                new_centroids[cluster_idx] = cluster_mean
            else:
                new_centroids[cluster_idx] = old_centroids_for_empty[cluster_idx]
        return new_centroids
    
    def _is_converged(self, new_centroids, old_centroids, tol=1e-4):
        """Checks if the centroids have converged (moved less than tol)."""
        distances = [self.euclidean_distance(new_centroids[i], old_centroids[i]) for i in range(self.k)]
        return sum(distances) < tol
    
    def _get_cluster_labels(self, X):
        """Assigns a label to each data point based on its final cluster."""
        self.labels = np.empty(X.shape[0], dtype=int)
        for cluster_idx, point_indices_in_cluster in enumerate(self.clusters):
            for sample_idx in point_indices_in_cluster:
                self.labels[sample_idx] = cluster_idx
    
    def _get_wcss_loss(self, X):
        """Calculates the Within-Cluster Sum of Squares (WCSS)."""
        loss = 0
        if self.clusters is None or self.centroids is None:
            return np.inf # Or handle as an error

        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster:
                centroid_for_cluster = self.centroids[cluster_idx]
                for sample_idx in cluster:
                    loss += self.euclidean_distance(X[sample_idx], centroid_for_cluster)**2
        return loss
    
    def predict(self, X):
        """
        Performs K-Means clustering on the data X.

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each point in X.
        centroids : np.ndarray
            Final centroid locations.
        clusters : list
            List of lists, where each inner list contains the indices of points 
            belonging to that cluster.
        history_centroids : list
            History of centroid locations at each iteration.
        losses : list
            WCSS loss at each iteration.
        """
        if not isinstance(X, np.ndarray): X = np.array(X)
        if X.ndim != 2: raise ValueError("Input data X must be 2-dimensional (samples, features).")

        self._initialize_centroid(X)
        
        converged = False

        for i in range(self.max_iter):
            old_centroids_for_iter = self.centroids.copy() 
            
            # Assignment step: Assign points to the current centroids
            self._create_clusters(X, self.centroids) 

            if self.plot_during_fitting:
                self._plot_current_state(X, iteration_num=i)
            
            # Pass old_centroids_for_iter to handle empty clusters gracefully
            self.centroids = self._calculate_new_centroids(X, old_centroids_for_iter) # Update self.centroids
            
            self.history_centroids.append(self.centroids.copy()) # Append updated centroids
            
            current_wcss = self._get_wcss_loss(X)
            self.losses.append(current_wcss)

            if self._is_converged(self.centroids, old_centroids_for_iter):
                self.convergence_iteration = i + 1
                # print(f"{self.__class__.__name__} converged at iteration {i+1}.")
                converged = True
                break
        
        # Ensure final cluster assignments and labels are based on the very last centroids
        self._create_clusters(X, self.centroids)
        self._get_cluster_labels(X)
        
        # If loop finished due to max_iter, the last loss might not have been recorded if convergence check was last
        if not converged and len(self.losses) == self.max_iter - 1 :
             final_wcss = self._get_wcss_loss(X)
             if not self.losses or (self.losses and abs(self.losses[-1] - final_wcss) > 1e-9 ) :
                 self.losses.append(final_wcss)


        return self.labels, self.centroids, self.clusters, self.history_centroids, self.losses
    
    def _check_clusters_balanced(self):
        """Checks if clusters are somewhat balanced (difference in size <= 1)."""
        if not self.clusters:
            # print("Clusters not yet formed to check balance.")
            return False # Or raise error
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        if not cluster_sizes: return True # No clusters, vacuously balanced
        return max(cluster_sizes) - min(cluster_sizes) <= 1


class KMeansPlusPlus(KMeans):
    """
    K-Means clustering algorithm with K-Means++ initialization.
    Inherits from KMeans and overrides the centroid initialization method.
    """
    def __init__(self, k=3, max_iter=100, plot_during_fitting=False, random_seed=None):
        super().__init__(k, max_iter, plot_during_fitting, random_seed)

    def _initialize_centroid(self, X):
        n_samples, _ = X.shape
        if self.k > n_samples:
             raise ValueError(f"Number of clusters k ({self.k}) cannot be greater than "
                              f"the number of samples ({n_samples}).")

        if self.random_seed is not None: 
            np.random.seed(self.random_seed)

        centroids_list = []
        first_centroid_idx = np.random.choice(n_samples)
        centroids_list.append(X[first_centroid_idx])
        min_sq_distances = np.full(n_samples, np.inf, dtype=float)

        for i in range(1, self.k):
            last_added_centroid = centroids_list[-1]
            dist_sq_to_last = np.sum((X - last_added_centroid)**2, axis=1)
            min_sq_distances = np.minimum(min_sq_distances, dist_sq_to_last)
            
            sum_min_sq_distances = np.sum(min_sq_distances)

            if abs(sum_min_sq_distances) < 1e-9: 
                chosen_centroids_tuples = {tuple(c) for c in centroids_list}
                available_indices_for_fallback = [
                    idx for idx, point in enumerate(X) 
                    if tuple(point) not in chosen_centroids_tuples
                ]
                if available_indices_for_fallback:
                    next_centroid_idx = np.random.choice(available_indices_for_fallback)
                else:
                    next_centroid_idx = np.random.choice(n_samples) 
            else:
                probabilities = min_sq_distances / sum_min_sq_distances
                probabilities_sum = np.sum(probabilities)
                if abs(probabilities_sum - 1.0) > 1e-9 : 
                    probabilities = probabilities / probabilities_sum
                next_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids_list.append(X[next_centroid_idx])

        self.centroids = np.array(centroids_list)
        self.history_centroids.append(self.centroids.copy())

class KMeansFarthestFirst(KMeans):
    """
    K-Means clustering algorithm with Farthest First (Maximin) initialization.
    Inherits from KMeans and overrides the centroid initialization method.
    """
    def __init__(self, k=3, max_iter=100, plot_during_fitting=False, random_seed=None):
        super().__init__(k, max_iter, plot_during_fitting, random_seed)

    def _initialize_centroid(self, X):
        """
        Initializes centroids using the Farthest First (Maximin) strategy.
        """
        n_samples, n_features = X.shape
        if self.k > n_samples:
             raise ValueError(f"Number of clusters k ({self.k}) cannot be greater than "
                              f"the number of samples ({n_samples}).")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        centroids_list = []

        # 1. Choose the first centroid randomly
        first_centroid_idx = np.random.choice(n_samples)
        centroids_list.append(X[first_centroid_idx])

        for _ in range(1, self.k):
            # Calculate the distance from each point to its closest already chosen centroid
            min_distances_to_centroids = np.full(n_samples, np.inf)
            for centroid in centroids_list:
                distances_to_current_centroid = np.linalg.norm(X - centroid, axis=1)
                min_distances_to_centroids = np.minimum(min_distances_to_centroids, distances_to_current_centroid)

            # Select the point that is farthest from any chosen centroid
            next_centroid_idx = np.argmax(min_distances_to_centroids)
            centroids_list.append(X[next_centroid_idx])

        self.centroids = np.array(centroids_list)
        self.history_centroids = []
        self.history_centroids.append(self.centroids.copy())