import numpy as np

class CustomRidgeRegression:
    def __init__(self, lambda_reg: float = 0.1, lr: float = 0.01, max_iters: int = 1000):
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.max_iters = max_iters
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains Ridge Regression (L2 Regularization) using Gradient Descent.
        Loss = (1/2m) * ||Xθ - y||^2 + (λ/2) * ||θ||^2
        Gradient = (1/m) * X^T (Xθ - y) + λ * θ
        """
        m, n = X.shape
        self.theta = np.zeros(n)
        
        for _ in range(self.max_iters):
            predictions = np.dot(X, self.theta)
            errors = predictions - y
            # Gradient descent step with L2 penalty term
            grad = (1.0 / m) * np.dot(X.T, errors) + self.lambda_reg * self.theta
            self.theta -= self.lr * grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.theta)


class CustomLassoRegression:
    def __init__(self, lambda_reg: float = 0.1, lr: float = 0.01, max_iters: int = 1000):
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.max_iters = max_iters
        self.theta = None

    def _soft_threshold(self, rho: float, kappa: float) -> float:
        """Soft-thresholding operator for L1 (Lasso) proximal gradient descent."""
        if rho > kappa:
            return rho - kappa
        elif rho < -kappa:
            return rho + kappa
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains Lasso Regression (L1 Regularization) using Proximal Gradient Descent.
        Enforces feature sparsity by driving insignificant spatial coefficients exactly to zero.
        """
        m, n = X.shape
        self.theta = np.zeros(n)
        
        for _ in range(self.max_iters):
            predictions = np.dot(X, self.theta)
            errors = predictions - y
            
            # Standard gradient step for the smooth least-squares loss component
            grad = (1.0 / m) * np.dot(X.T, errors)
            temp_theta = self.theta - self.lr * grad
            
            # Apply proximal operator (soft-thresholding) for L1 penalty
            kappa = self.lr * self.lambda_reg
            for j in range(n):
                self.theta[j] = self._soft_threshold(temp_theta[j], kappa)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.theta)