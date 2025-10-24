import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import logsumexp
from sklearn.cluster import KMeans
EPS = 1e-6

class GaussianHMM(BaseEstimator):
    """
    Gaussian Hidden Markov Model for Digit Speech Recognition
    """
    def __init__(self, 
                 n_components=5,
                 n_iter=100,
                 covariance_type='diag', # only placeholder, diag for default
                 tol=1e-4,
                 verbose=False):
        
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.pi = np.ones(shape=(n_components, )) / n_components
        self.A = np.ones(shape=(n_components, n_components)) / n_components
        self.means_, self.covars_ = None, None
    
    def _initialize_parameters(self, X):
        """Better initialization using K-means"""
        n_samples, n_features = X.shape
        
        # Use K-means for better initialization
        if n_samples >= self.n_components:
            kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            self.means_ = kmeans.cluster_centers_
            self.covars_ = np.zeros((self.n_components, n_features))
            
            for i in range(self.n_components):
                mask = labels == i
                if np.sum(mask) > 0:
                    self.covars_[i] = np.var(X[mask], axis=0) + EPS
                else:
                    self.covars_[i] = np.var(X, axis=0) + EPS
        else:
            # Fallback: add random perturbations
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0) + EPS
            
            self.means_ = mean + np.random.randn(self.n_components, n_features) * np.sqrt(var) * 0.1
            self.covars_ = np.tile(var, (self.n_components, 1))
        
        # Initialize log probabilities
        self.log_pi = np.log(self.pi + EPS)
        self.log_A = np.log(self.A + EPS)
            
    def fit(self, X, lengths=None):
        """
        Estimate model parameters with Baum-Welch algorithm.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self
        """
        if lengths is None:
            lengths = [X.shape[0]]
        
        # Better initialization
        self._initialize_parameters(X)
        
        prev_log_prob = -np.inf
        
        for iteration in range(self.n_iter): 
            total_log_prob = 0
            pos = 0
            
            # Accumulate statistics for M-step
            gamma_sum = np.zeros(self.n_components)
            gamma_init_sum = np.zeros(self.n_components)
            xi_sum = np.zeros((self.n_components, self.n_components))
            
            means_num = np.zeros((self.n_components, X.shape[1]))
            covars_num = np.zeros((self.n_components, X.shape[1]))
            
            for length in lengths:
                obs_seq = X[pos:pos + length]
                
                # Calculate log emission probabilities
                log_B = self._compute_log_emission_prob(obs_seq)
                
                # Forward-Backward in log space
                log_alpha = self._forward_log(log_B)
                log_beta = self._backward_log(log_B)
                
                # Calculate gamma (posterior state probabilities)
                log_gamma = log_alpha + log_beta
                log_gamma -= logsumexp(log_gamma, axis=0, keepdims=True)
                gamma = np.exp(log_gamma)
                
                # Calculate xi (posterior transition probabilities)
                log_xi = self._compute_log_xi(log_alpha, log_beta, log_B)
                xi = np.exp(log_xi)
                
                # Accumulate statistics
                gamma_sum += np.sum(gamma, axis=1)
                gamma_init_sum += gamma[:, 0]
                xi_sum += np.sum(xi, axis=2)
                
                # Accumulate for Gaussian parameter updates
                for i in range(self.n_components):
                    means_num[i] += np.sum(gamma[i, :, None] * obs_seq, axis=0)
                
                # Calculate sequence log probability
                total_log_prob += logsumexp(log_alpha[:, -1])
                
                pos += length
            
            # M-step: Update parameters
            
            # Update pi (initial state distribution)
            self.pi = gamma_init_sum 
            self.log_pi = np.log(self.pi + EPS)
            
            # Update A (transition matrix)
            for i in range(self.n_components):
                self.A[i] = xi_sum[i] / (np.sum(xi_sum[i]) + EPS)
            self.log_A = np.log(self.A + EPS)
            
            # Update Gaussian means
            for i in range(self.n_components):
                self.means_[i] = means_num[i] / (gamma_sum[i] + EPS)
            
            # Update Gaussian covariances using new means
            pos = 0
            for length in lengths:
                obs_seq = X[pos:pos + length]
                log_B = self._compute_log_emission_prob(obs_seq)
                log_alpha = self._forward_log(log_B)
                log_beta = self._backward_log(log_B)
                log_gamma = log_alpha + log_beta
                log_gamma -= logsumexp(log_gamma, axis=0, keepdims=True)
                gamma = np.exp(log_gamma)
                
                for i in range(self.n_components):
                    diff = obs_seq - self.means_[i]
                    covars_num[i] += np.sum(gamma[i, :, None] * diff**2, axis=0)
                
                pos += length
            
            for i in range(self.n_components):
                self.covars_[i] = covars_num[i] / (gamma_sum[i] + EPS)
                self.covars_[i] = np.maximum(self.covars_[i], EPS)
            
            # Check convergence
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Log Likelihood: {total_log_prob:.2f}")
            
            if abs(total_log_prob - prev_log_prob) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_log_prob = total_log_prob
                
        return self       
    
    def score(self, X, lengths=None):
        """
        Compute the log probability under the model using Forward algorithm.
        
        IMPORTANT: Use forward algorithm, NOT Viterbi, for classification!

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        log_prob : float
            Log likelihood of ``X``.
            
        """
        if lengths is None:
            lengths = [X.shape[0]]
            
        log_prob = 0
        pos = 0
        
        for length in lengths:
            obs_seq = X[pos:pos + length]
            log_B = self._compute_log_emission_prob(obs_seq)
            
            # Use FORWARD algorithm to compute log probability
            log_alpha = self._forward_log(log_B)
            log_prob += logsumexp(log_alpha[:, -1])  # CORRECT: sum over all paths
            
            pos += length
        
        return log_prob

    def _compute_log_emission_prob(self, obs_seq):
        """Compute log emission probabilities"""
        T, d = obs_seq.shape
        log_B = np.zeros((self.n_components, T))
        
        for i in range(self.n_components):
            var = self.covars_[i]
            mean = self.means_[i]
            
            # Log normalization constant
            log_norm = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var + EPS)))
            
            for t in range(T):
                diff = obs_seq[t] - mean
                log_exp = -0.5 * np.sum((diff ** 2) / (var + EPS))
                log_B[i, t] = log_norm + log_exp
        
        return log_B
    
    def _forward_log(self, log_B):
        """Forward algorithm in log space"""
        obs_len = log_B.shape[1]
        log_alpha = np.zeros((self.n_components, obs_len))
        
        # Initial state
        log_alpha[:, 0] = self.log_pi + log_B[:, 0]
        
        for t in range(1, obs_len):
            for j in range(self.n_components):
                log_alpha[j, t] = logsumexp(log_alpha[:, t-1] + self.log_A[:, j]) + log_B[j, t]
        
        return log_alpha

    def _backward_log(self, log_B):
        """Backward algorithm in log space"""
        obs_len = log_B.shape[1]
        log_beta = np.zeros((self.n_components, obs_len))
        
        # log(1) = 0
        log_beta[:, -1] = 0
        
        for t in range(obs_len - 2, -1, -1):
            for i in range(self.n_components):
                log_beta[i, t] = logsumexp(
                    self.log_A[i, :] + log_B[:, t+1] + log_beta[:, t+1]
                )
        
        return log_beta
    
    def _compute_log_xi(self, log_alpha, log_beta, log_B):
        """Compute log xi in log space"""
        T = log_B.shape[1]
        log_xi = np.zeros((self.n_components, self.n_components, T - 1))
        
        for t in range(T - 1):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    log_xi[i, j, t] = (log_alpha[i, t] + 
                                       self.log_A[i, j] + 
                                       log_B[j, t+1] + 
                                       log_beta[j, t+1])
            
            # Normalize
            log_xi[:, :, t] -= logsumexp(log_xi[:, :, t])
        
        return log_xi