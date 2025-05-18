import numpy as np
from abc import ABC, abstractmethod

class UncertaintyEstimator(ABC):
    """
    Base class defining methods for total, epistemic, and aleatoric uncertainty.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def epistemic_uncertainty(self):
        """Model uncertainty due to parameters."""
        pass

    @abstractmethod
    def aleatoric_uncertainty(self):
        """Data uncertainty (noise inherent to observations)."""
        pass

    @abstractmethod
    def total_uncertainty(self):
        """Combined uncertainty."""
        pass

class RegressionUncertaintyEstimator(UncertaintyEstimator):
    """
    Computes uncertainties for regression tasks using MC samples.

    Args:
        samples_mean: array-like, shape (T, ...) predictive means from T MC samples
        samples_var : array-like, shape (T, ...) predictive variances from T MC samples
    """
    def __init__(self, samples_mean: np.ndarray, samples_var: np.ndarray):
        self.means = np.atleast_2d(samples_mean)
        self.vars = np.atleast_2d(samples_var)

    def epistemic_uncertainty(self) -> np.ndarray:
        """Variance of the predictive means across MC runs."""
        return np.var(self.means, axis=0)

    def aleatoric_uncertainty(self) -> np.ndarray:
        """Mean of predictive variances across MC runs."""
        return np.mean(self.vars, axis=0)

    def total_uncertainty(self) -> np.ndarray:
        """Sum of epistemic and aleatoric uncertainties."""
        return self.epistemic_uncertainty() + self.aleatoric_uncertainty()


class ClassificationUncertaintyEstimator(UncertaintyEstimator):
    """
    Computes uncertainties for classification tasks using MC dropout.

    Args:
        samples_probs: array-like, shape (T, C) softmax probabilities from T MC samples
        eps          : float, stability constant for log computations
    """
    def __init__(self, samples_probs: np.ndarray, eps: float = 1e-12):
        self.P = np.atleast_2d(samples_probs)
        self.eps = eps

    def _entropy(self, probs: np.ndarray) -> np.ndarray:
        """Compute Shannon entropy of probability distributions."""
        return -np.sum(probs * np.log(probs + self.eps), axis=-1)

    def total_uncertainty(self) -> float:
        """Entropy of the mean predictive distribution."""
        p_mean = np.mean(self.P, axis=0)
        return float(self._entropy(p_mean))

    def aleatoric_uncertainty(self) -> float:
        """Mean entropy of each predictive distribution."""
        ent = self._entropy(self.P)
        return float(np.mean(ent))

    def epistemic_uncertainty(self) -> float:
        """Mutual information between predictions and model posterior."""
        # MI = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]
        return self.total_uncertainty() - self.aleatoric_uncertainty()




# Example usage with test values:
if __name__ == "__main__":
    # Create small test arrays for regression (scalar outputs)
    means_array = np.array([1.0, 1.2, 0.8, 1.1, 0.9])    # T=5 MC predictive means
    vars_array  = np.array([0.1, 0.2, 0.15, 0.05, 0.12]) # T=5 MC predictive variances

    reg = RegressionUncertaintyEstimator(means_array, vars_array)
    print("Regression Uncertainties:")
    print("Epistemic:", reg.epistemic_uncertainty())
    print("Aleatoric:", reg.aleatoric_uncertainty())
    print("Total:", reg.total_uncertainty())

    # For vector outputs, e.g., 2-dimensional predictions:
    means_vec = np.array([[1.0, 2.0], [1.1, 1.9], [0.9, 2.1]])
    vars_vec  = np.array([[0.05, 0.1], [0.06, 0.08], [0.04, 0.12]])

    reg_vec = RegressionUncertaintyEstimator(means_vec, vars_vec)
    print("Vector Regression Uncertainties:")
    print("Epistemic:", reg_vec.epistemic_uncertainty())
    print("Aleatoric:", reg_vec.aleatoric_uncertainty())
    print("Total:", reg_vec.total_uncertainty())
