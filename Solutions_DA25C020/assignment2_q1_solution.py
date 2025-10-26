import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class BernoulliMixtureEM:
    """
    Bernoulli Mixture Model with EM algorithm
    """
    def __init__(self, n_components=4, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pi = None
        self.mu = None
        
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        n_samples, n_features = X.shape
        
        # Initialize mixing weights (uniform)
        self.pi = np.ones(self.n_components) / self.n_components
        
        # Initialize Bernoulli parameters (random between 0.25 and 0.75)
        self.mu = np.random.uniform(0.25, 0.75, (self.n_components, n_features))
        
        return self.pi, self.mu
    
    def _e_step(self, X):
        """E-step: compute responsibilities using log-sum-exp trick"""
        n_samples = X.shape[0]
        
        # Compute log probabilities for each component
        # log P(x|k) = sum_d [x_d * log(mu_kd) + (1-x_d) * log(1-mu_kd)]
        log_prob = (X[:, None, :] * np.log(self.mu[None, :, :] + 1e-10) + 
                    (1 - X[:, None, :]) * np.log(1 - self.mu[None, :, :] + 1e-10))
        log_prob = np.sum(log_prob, axis=2)  # Shape: (n_samples, n_components)
        
        # Add log of mixing weights
        log_prob += np.log(self.pi + 1e-10)
        
        # Use log-sum-exp trick for numerical stability
        log_sum = np.logaddexp.reduce(log_prob, axis=1, keepdims=True)
        log_resp = log_prob - log_sum
        responsibilities = np.exp(log_resp)
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """M-step: update parameters"""
        n_samples = X.shape[0]
        
        # Update mixing weights
        Nk = responsibilities.sum(axis=0)
        self.pi = Nk / n_samples
        
        # Update Bernoulli parameters
        self.mu = (responsibilities.T @ X) / (Nk[:, None] + 1e-10)
        
        # Clip to valid range
        self.mu = np.clip(self.mu, 1e-10, 1 - 1e-10)
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood"""
        # Compute log probabilities
        log_prob = (X[:, None, :] * np.log(self.mu[None, :, :] + 1e-10) + 
                    (1 - X[:, None, :]) * np.log(1 - self.mu[None, :, :] + 1e-10))
        log_prob = np.sum(log_prob, axis=2)
        
        # Add mixing weights
        log_prob += np.log(self.pi + 1e-10)
        
        # Sum over components using log-sum-exp
        log_sum = np.logaddexp.reduce(log_prob, axis=1)
        
        # Sum over all samples
        return np.sum(log_sum)
    
    def fit(self, X):
        """Fit the model using EM algorithm"""
        self._initialize_parameters(X)
        log_likelihoods = []
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                break
        
        return log_likelihoods


class GaussianMixtureEM:
    """
    Gaussian Mixture Model with EM algorithm
    """
    def __init__(self, n_components=4, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pi = None
        self.mu = None
        self.sigma = None
        
    def _initialize_parameters(self, X):
        """Initialize parameters"""
        n_samples, n_features = X.shape
        
        # Initialize mixing weights (uniform)
        self.pi = np.ones(self.n_components) / self.n_components
        
        # Initialize means (random from data)
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.mu = X[indices].copy()
        
        # Initialize covariances (small diagonal matrix for binary data)
        self.sigma = np.array([np.eye(n_features) * 0.1 for _ in range(self.n_components)])
        
        return self.pi, self.mu, self.sigma
    
    def _multivariate_normal_log_pdf(self, X, mean, cov):
        """Compute log of multivariate normal PDF (vectorized)"""
        n_features = len(mean)
        
        # Add regularization to avoid singular matrix
        cov_reg = cov + np.eye(n_features) * 1e-6
        
        try:
            # Use Cholesky decomposition for stability
            L = np.linalg.cholesky(cov_reg)
            log_det = 2 * np.sum(np.log(np.diagonal(L)))
            
            # Solve for inv(cov) @ diff efficiently
            diff = X - mean
            y = np.linalg.solve(L, diff.T)
            mahalanobis = np.sum(y * y, axis=0)
            
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            det = np.linalg.det(cov_reg)
            log_det = np.log(det + 1e-10)
            inv = np.linalg.inv(cov_reg)
            diff = X - mean
            mahalanobis = np.sum(diff @ inv * diff, axis=1)
        
        # Compute log PDF
        log_norm_const = -0.5 * (n_features * np.log(2 * np.pi) + log_det)
        log_prob = log_norm_const - 0.5 * mahalanobis
        
        return log_prob
    
    def _e_step(self, X):
        """E-step: compute responsibilities using log probabilities"""
        n_samples = X.shape[0]
        log_responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute log responsibilities for each component
        for k in range(self.n_components):
            log_responsibilities[:, k] = (np.log(self.pi[k] + 1e-10) + 
                                          self._multivariate_normal_log_pdf(X, self.mu[k], self.sigma[k]))
        
        # Normalize using log-sum-exp trick
        log_sum = np.logaddexp.reduce(log_responsibilities, axis=1, keepdims=True)
        log_responsibilities -= log_sum
        responsibilities = np.exp(log_responsibilities)
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """M-step: update parameters"""
        n_samples, n_features = X.shape
        
        # Effective number of points in each cluster
        Nk = responsibilities.sum(axis=0) + 1e-10
        
        # Update mixing weights
        self.pi = Nk / n_samples
        
        # Update means
        for k in range(self.n_components):
            self.mu[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.mu[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.sigma[k] = (weighted_diff.T @ diff) / Nk[k]
            
            # Add regularization (important for binary data)
            self.sigma[k] += np.eye(n_features) * 0.01
    
    def _compute_log_likelihood(self, X):
        """Compute total log-likelihood (not per sample)"""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_prob[:, k] = (np.log(self.pi[k] + 1e-10) + 
                              self._multivariate_normal_log_pdf(X, self.mu[k], self.sigma[k]))
        
        # Sum over components using log-sum-exp, then sum over samples
        log_likelihood = np.sum(np.logaddexp.reduce(log_prob, axis=1))
        
        return log_likelihood  # Return TOTAL, not per-sample
    
    def fit(self, X):
        """Fit the model using EM algorithm"""
        self._initialize_parameters(X)
        log_likelihoods = []
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                break
        
        return log_likelihoods


class KMeans:
    """
    K-Means clustering algorithm
    """
    def __init__(self, n_clusters=4, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        
    def _initialize_centroids(self, X):
        """Initialize centroids randomly from data"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        # Vectorized distance computation
        distances = np.sum((X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        return assignments
    
    def _update_centroids(self, X, assignments):
        """Update centroids based on current assignments"""
        for k in range(self.n_clusters):
            mask = assignments == k
            if np.sum(mask) > 0:
                self.centroids[k] = np.mean(X[mask], axis=0)
    
    def _compute_objective(self, X, assignments):
        """Compute K-means objective (sum of squared distances)"""
        objective = 0
        for k in range(self.n_clusters):
            mask = assignments == k
            if np.sum(mask) > 0:
                distances = np.sum((X[mask] - self.centroids[k]) ** 2)
                objective += distances
        return objective
    
    def fit(self, X):
        """Fit K-means algorithm"""
        self.centroids = self._initialize_centroids(X)
        objectives = []
        
        for iteration in range(self.max_iter):
            # Assign points to clusters
            assignments = self._assign_clusters(X)
            
            # Compute objective
            objective = self._compute_objective(X, assignments)
            objectives.append(objective)
            
            # Update centroids
            old_centroids = self.centroids.copy()
            self._update_centroids(X, assignments)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
        
        return objectives


def load_data():
    """Load the A2Q1.csv dataset"""
    data = pd.read_csv('datasets/A2Q1.csv', header=None)
    return data.values.astype(np.float64)


def run_experiments():
    """Run all experiments and generate plots"""
    # Load data
    X = load_data()
    print(f"Data shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Unique values: {np.unique(X)}")
    print(f"Data is binary: {np.all(np.isin(X, [0, 1]))}")
    
    # Number of random initializations
    n_runs = 100
    max_iter = 40
    
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)
    print(f"Number of runs: {n_runs}")
    print(f"Max iterations per run: {max_iter}")
    
    # Storage for results
    bernoulli_lls = []
    gaussian_lls = []
    kmeans_objs = []
    
    print("\nRunning Bernoulli Mixture EM...")
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")
        
        model = BernoulliMixtureEM(n_components=4, max_iter=max_iter)
        ll = model.fit(X)
        bernoulli_lls.append(ll)
    
    print("\nRunning Gaussian Mixture EM...")
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")
        
        model = GaussianMixtureEM(n_components=4, max_iter=max_iter)
        ll = model.fit(X)
        gaussian_lls.append(ll)
    
    print("\nRunning K-means...")
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")
        
        model = KMeans(n_clusters=4, max_iter=max_iter)
        obj = model.fit(X)
        kmeans_objs.append(obj)
    
    # Compute averages (handle variable lengths)
    max_len_b = max(len(ll) for ll in bernoulli_lls)
    max_len_g = max(len(ll) for ll in gaussian_lls)
    max_len_k = max(len(obj) for obj in kmeans_objs)
    
    bernoulli_padded = np.zeros((n_runs, max_len_b))
    gaussian_padded = np.zeros((n_runs, max_len_g))
    kmeans_padded = np.zeros((n_runs, max_len_k))
    
    for i in range(n_runs):
        ll = bernoulli_lls[i]
        bernoulli_padded[i, :len(ll)] = ll
        if len(ll) < max_len_b:
            bernoulli_padded[i, len(ll):] = ll[-1]
        
        ll = gaussian_lls[i]
        gaussian_padded[i, :len(ll)] = ll
        if len(ll) < max_len_g:
            gaussian_padded[i, len(ll):] = ll[-1]
        
        obj = kmeans_objs[i]
        kmeans_padded[i, :len(obj)] = obj
        if len(obj) < max_len_k:
            kmeans_padded[i, len(obj):] = obj[-1]
    
    avg_bernoulli_ll = np.mean(bernoulli_padded, axis=0)
    avg_gaussian_ll = np.mean(gaussian_padded, axis=0)
    avg_kmeans_obj = np.mean(kmeans_padded, axis=0)
    
    # Create results folder
    import os
    results_folder = 'assignment2_q1_results'
    os.makedirs(results_folder, exist_ok=True)
    
    # Create individual plots for each algorithm
    
    # Plot 1: Bernoulli Mixture EM
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(avg_bernoulli_ll) + 1), avg_bernoulli_ll, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Log-Likelihood', fontsize=12)
    plt.title('Bernoulli Mixture EM (K=4)\nAveraged over 100 runs', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_folder}/bernoulli_em.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Gaussian Mixture EM
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(avg_gaussian_ll) + 1), avg_gaussian_ll, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Log-Likelihood', fontsize=12)
    plt.title('Gaussian Mixture EM (K=4)\nAveraged over 100 runs', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_folder}/gaussian_em.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: K-means
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(avg_kmeans_obj) + 1), avg_kmeans_obj, 'g-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective (SSE)', fontsize=12)
    plt.title('K-means (K=4)\nAveraged over 100 runs', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_folder}/kmeans.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create combined plot for reference
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Bernoulli Mixture EM
    ax = axes[0]
    ax.plot(range(1, len(avg_bernoulli_ll) + 1), avg_bernoulli_ll, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Log-Likelihood', fontsize=12)
    ax.set_title('Bernoulli Mixture EM (K=4)\nAveraged over 100 runs', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gaussian Mixture EM
    ax = axes[1]
    ax.plot(range(1, len(avg_gaussian_ll) + 1), avg_gaussian_ll, 'r-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Log-Likelihood', fontsize=12)
    ax.set_title('Gaussian Mixture EM (K=4)\nAveraged over 100 runs', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: K-means
    ax = axes[2]
    ax.plot(range(1, len(avg_kmeans_obj) + 1), avg_kmeans_obj, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective (SSE)', fontsize=12)
    ax.set_title('K-means (K=4)\nAveraged over 100 runs', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_folder}/combined_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Individual plots saved in '{results_folder}/' folder:")
    print(f"  - {results_folder}/bernoulli_em.png")
    print(f"  - {results_folder}/gaussian_em.png") 
    print(f"  - {results_folder}/kmeans.png")
    print(f"  - {results_folder}/combined_results.png")
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Bernoulli Mixture EM:")
    print(f"  Initial Log-Likelihood: {avg_bernoulli_ll[0]:.2f}")
    print(f"  Final Log-Likelihood: {avg_bernoulli_ll[-1]:.2f}")
    print(f"  Improvement: {avg_bernoulli_ll[-1] - avg_bernoulli_ll[0]:.2f}")
    
    print(f"\nGaussian Mixture EM:")
    print(f"  Initial Log-Likelihood: {avg_gaussian_ll[0]:.2f}")
    print(f"  Final Log-Likelihood: {avg_gaussian_ll[-1]:.2f}")
    print(f"  Improvement: {avg_gaussian_ll[-1] - avg_gaussian_ll[0]:.2f}")
    
    print(f"\nK-means:")
    print(f"  Initial Objective: {avg_kmeans_obj[0]:.2f}")
    print(f"  Final Objective: {avg_kmeans_obj[-1]:.2f}")
    print(f"  Reduction: {avg_kmeans_obj[0] - avg_kmeans_obj[-1]:.2f}")
    
    return {
        'bernoulli_ll': avg_bernoulli_ll,
        'gaussian_ll': avg_gaussian_ll,
        'kmeans_obj': avg_kmeans_obj
    }

if __name__ == "__main__":
    results = run_experiments()