import numpy as np

def inverse_optimization_solver(theta: np.ndarray, x_base: np.ndarray, y_target: float) -> np.ndarray:
    """
    Solves the Inverse Optimization / Goal-Seeking Projection Problem:
    Minimize    ||x - x_base||^2
    Subject to  θ^T x = y_target
    
    Derived analytically via Lagrange Multipliers:
    x_opt = x_base + ((y_target - θ^T x_base) / ||θ||^2) * θ
    """
    theta_norm_sq = np.dot(theta, theta)
    
    # Safeguard against zero vector weights
    if theta_norm_sq == 0.0:
        return x_base.copy()
        
    current_price = np.dot(theta, x_base)
    residual = y_target - current_price
    
    # Analytical projection formula solution
    optimal_features = x_base + (residual / theta_norm_sq) * theta
    return optimal_features