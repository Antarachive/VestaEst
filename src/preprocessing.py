import numpy as np

class SpatialPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X: np.ndarray):
        """Computes mean and standard deviation for Z-score feature scaling across spatial properties."""
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        # Prevent division by zero for constant or invariant feature columns
        self.stds[self.stds == 0.0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies standardization: x_scaled = (x - μ) / σ."""
        if self.means is None or self.stds is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Reverts standardized scaling back to original physical real estate dimensions."""
        if self.means is None or self.stds is None:
            raise ValueError("Preprocessor not fitted.")
        return X_scaled * self.stds + self.means