import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class LinearRegression:
    """
    Linear Regression implementation with multiple optimization methods
    """
    
    def __init__(self):
        self.w = None
        self.w_history = []
    
    def analytical_solution(self, X, y):
        """
        Analytical solution: w = (X^T X)^(-1) X^T y
        X should already have bias column
        """
        y = y.reshape(-1, 1)
        try:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self.w
    
    def gradient_descent(self, X, y, alpha=0.001, epochs=1000):
        """
        Batch gradient descent
        X should already have bias column
        """
        m, n = X.shape
        y = y.reshape(-1, 1)
        
        self.w = np.zeros((n, 1))
        self.w_history = []
        
        for epoch in range(epochs):
            # Compute gradient
            grad = (X.T @ (X @ self.w - y)) / m
            
            # Update weights
            self.w -= alpha * grad
            
            # Store history
            self.w_history.append(self.w.copy())
        
        return self.w, self.w_history
    
    def stochastic_gradient_descent(self, X, y, alpha=0.001, epochs=1000, batch_size=100):
        """
        Mini-batch stochastic gradient descent implementation
        X should already have bias column
        """
        m, n = X.shape
        y = y.reshape(-1, 1)
        
        self.w = np.zeros((n, 1))
        self.w_history = []
        
        rng = np.random.default_rng()
        
        for epoch in range(epochs):
            # Shuffle data
            indices = rng.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for start in range(0, m, batch_size):
                end = start + batch_size
                Xi = X_shuffled[start:end]
                yi = y_shuffled[start:end]
                
                # Compute gradient
                grad = Xi.T @ (Xi @ self.w - yi) / Xi.shape[0]
                
                # Update weights
                self.w -= alpha * grad
                
                # Store history
                self.w_history.append(self.w.copy())
        
        return self.w, self.w_history
    
    def ridge_regression(self, X, y, lambda_reg):
        """
        Ridge regression analytical solution
        X should already have bias column
        """
        y = y.reshape(-1, 1)
        n = X.shape[1]
        
        # Don't regularize bias term
        reg_matrix = lambda_reg * np.eye(n)
        reg_matrix[0, 0] = 0
        
        try:
            self.w = np.linalg.inv(X.T @ X + reg_matrix) @ X.T @ y
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(X.T @ X + reg_matrix) @ X.T @ y
        
        return self.w
    
    def ridge_gradient_descent(self, X, y, lambda_reg, alpha=0.001, epochs=5000):
        """
        Ridge regression using gradient descent
        X should already have bias column
        """
        m, n = X.shape
        y = y.reshape(-1, 1)
        
        self.w = np.zeros((n, 1))
        
        for epoch in range(epochs):
            # Gradient with regularization
            grad = (X.T @ (X @ self.w - y)) / m + lambda_reg * self.w
            
            # Don't regularize bias
            grad[0, 0] = np.sum(X[:, 0:1] * (X @ self.w - y)) / m
            
            # Update
            self.w -= alpha * grad
        
        return self.w
    
    def predict(self, X):
        """Make predictions"""
        return (X @ self.w).flatten()
    
    def mse(self, X, y):
        """Compute MSE"""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)


def load_data():
    """Load training data"""
    data = pd.read_csv('datasets/A2Q2Data_train.csv', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def load_test_data():
    """Load test data"""
    try:
        data = pd.read_csv('datasets/A2Q2Data_test.csv', header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    except:
        return None, None

def cross_validate_ridge(X, y, lambdas, k=5, alpha=0.001, epochs=5000):
    """
    K-fold cross-validation for ridge regression
    """
    N = X.shape[0]
    fold_size = N // k
    errors = []
    
    for lambda_reg in lambdas:
        fold_errors = []
        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            X_val, y_val = X[start:end], y[start:end]
            X_train = np.vstack((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            
            model = LinearRegression()
            model.ridge_gradient_descent(X_train, y_train, lambda_reg, alpha, epochs)
            
            y_pred = model.predict(X_val)
            fold_errors.append(np.mean((y_val - y_pred) ** 2))
        
        errors.append(np.mean(fold_errors))
    
    return errors

def run_experiments():
    """Run all experiments"""
    
    # Load data
    X, y = load_data()
    print(f"Training data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Add bias column
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    print("\n" + "="*60)
    print("PART (i): ANALYTICAL LEAST SQUARES SOLUTION")
    print("="*60)
    
    # Analytical solution
    model_ml = LinearRegression()
    w_ml = model_ml.analytical_solution(X_with_bias, y)
    print(f"w_ML shape: {w_ml.shape}")
    print(f"w_ML norm: {np.linalg.norm(w_ml):.6f}")
    
    print("\n" + "="*60)
    print("PART (ii): GRADIENT DESCENT")
    print("="*60)
    
    # Gradient descent implementation
    model_gd = LinearRegression()
    w_gd, history_gd = model_gd.gradient_descent(X_with_bias, y, alpha=0.0001, epochs=10000)
    
    # Compute convergence to analytical solution
    norm_gd = [np.linalg.norm(w_ml - wi) for wi in history_gd]
    
    print(f"GD completed: {len(history_gd)} iterations")
    print(f"Initial distance from w_ML: {norm_gd[0]:.6e}")
    print(f"Final distance from w_ML: {norm_gd[-1]:.6e}")
    print(f"Distance between w_GD and w_ML: {np.linalg.norm(w_gd - w_ml):.6e}")
    
    print("\n" + "="*60)
    print("PART (iii): STOCHASTIC GRADIENT DESCENT")
    print("="*60)
    
    # Stochastic gradient descent implementation
    model_sgd = LinearRegression()
    w_sgd, history_sgd = model_sgd.stochastic_gradient_descent(
        X_with_bias, y, alpha=0.001, epochs=1000, batch_size=100
    )
    
    # Compute convergence to analytical solution
    norm_sgd = [np.linalg.norm(w_ml - wi) for wi in history_sgd]
    
    print(f"SGD completed: {len(history_sgd)} updates")
    print(f"Initial distance from w_ML: {norm_sgd[0]:.6e}")
    print(f"Final distance from w_ML: {norm_sgd[-1]:.6e}")
    print(f"Distance between w_SGD and w_ML: {np.linalg.norm(w_sgd - w_ml):.6e}")
    
    print("\n" + "="*60)
    print("PART (iv): RIDGE REGRESSION WITH CROSS-VALIDATION")
    print("="*60)
    
    # Cross-validation
    lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
    print(f"Testing lambda values: {lambdas}")
    
    cv_errors = cross_validate_ridge(X_with_bias, y, lambdas, k=5, alpha=0.001, epochs=5000)
    
    best_lambda = lambdas[np.argmin(cv_errors)]
    print(f"Best lambda: {best_lambda}")
    print(f"Best CV error: {cv_errors[np.argmin(cv_errors)]:.6f}")
    
    # Train ridge regression with best lambda
    model_ridge = LinearRegression()
    w_ridge = model_ridge.ridge_gradient_descent(X_with_bias, y, best_lambda, alpha=0.001, epochs=20000)
    
    print(f"Ridge solution w_R shape: {w_ridge.shape}")
    print(f"w_R norm: {np.linalg.norm(w_ridge):.6f}")
    
    # Test evaluation
    X_test, y_test = load_test_data()
    
    if X_test is not None:
        X_test_with_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        
        mse_ml = model_ml.mse(X_test_with_bias, y_test)
        mse_ridge = model_ridge.mse(X_test_with_bias, y_test)
        
        print(f"\nTest MSE (w_ML): {mse_ml:.6f}")
        print(f"Test MSE (w_R): {mse_ridge:.6f}")
        print(f"Improvement: {((mse_ml - mse_ridge) / mse_ml * 100):.2f}%")
    
    # Create results folder
    import os
    results_folder = 'assignment2_q2_results'
    os.makedirs(results_folder, exist_ok=True)
    
    # Create individual plots for each algorithm
    
    # Plot 1: Gradient Descent Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(norm_gd, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('||w_ML - w_t||₂', fontsize=12)
    plt.title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_folder}/gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Stochastic GD Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(norm_sgd, 'r-', linewidth=2)
    plt.xlabel('Update', fontsize=12)
    plt.ylabel('||w_ML - w_t||₂', fontsize=12)
    plt.title('Stochastic GD Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_folder}/stochastic_gd.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Cross-Validation Error vs λ
    plt.figure(figsize=(8, 6))
    plt.semilogx(lambdas, cv_errors, 'g-o', linewidth=2, markersize=8)
    plt.axvline(best_lambda, color='red', linestyle='--', linewidth=2, 
               label=f'Best λ={best_lambda}')
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Validation MSE', fontsize=12)
    plt.title('Cross-Validation Error vs λ', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_folder}/cross_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Convergence Comparison
    plt.figure(figsize=(8, 6))
    # Sample SGD to align with GD length for comparison
    sgd_sampled = norm_sgd[::max(1, len(norm_sgd)//len(norm_gd))][:len(norm_gd)]
    plt.plot(norm_gd, 'b-', linewidth=2, label='Batch GD')
    plt.plot(sgd_sampled, 'r-', linewidth=2, label='Mini-batch SGD', alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('||w_ML - w_t||₂', fontsize=12)
    plt.title('Convergence Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_folder}/convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Ridge Coefficients at Different λ
    plt.figure(figsize=(10, 6))
    w_ml_flat = w_ml.flatten()
    
    # Compute Ridge coefficients for different lambda values
    lambda_values = [0, 0.01, 0.1, 1, 10]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (lam, color) in enumerate(zip(lambda_values, colors)):
        model_temp = LinearRegression()
        w_temp = model_temp.ridge_gradient_descent(X_with_bias, y, lam, alpha=0.001, epochs=20000)
        w_temp_flat = w_temp.flatten()
        plt.plot(w_temp_flat[1:], color=color, linewidth=2, label=f'λ={lam}', alpha=0.8)
    
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.title('Ridge Coefficients at Different λ', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_folder}/ridge_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Ridge Shrinkage Effect
    plt.figure(figsize=(8, 6))
    max_coeffs = []
    lambda_range = np.logspace(-3, 2, 50)  # More lambda values for smooth curve
    
    for lam in lambda_range:
        model_temp = LinearRegression()
        w_temp = model_temp.ridge_gradient_descent(X_with_bias, y, lam, alpha=0.001, epochs=20000)
        max_coeff = np.max(np.abs(w_temp.flatten()[1:]))  # Exclude bias term
        max_coeffs.append(max_coeff)
    
    plt.semilogx(lambda_range, max_coeffs, 'g-o', linewidth=2, markersize=4)
    plt.axvline(best_lambda, color='red', linestyle='--', linewidth=2, 
               label=f'Best λ={best_lambda}')
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Max |Coefficient|', fontsize=12)
    plt.title('Ridge Shrinkage Effect', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_folder}/ridge_shrinkage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 7: Ridge vs OLS Comparison
    plt.figure(figsize=(8, 6))
    w_ridge_flat = w_ridge.flatten()
    plt.plot(w_ml_flat[1:], 'b-', linewidth=2, label='w_ML (OLS)', alpha=0.7)
    plt.plot(w_ridge_flat[1:], 'r-', linewidth=2, label=f'w_R (λ={best_lambda})', alpha=0.7)
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.title('Coefficients: Ridge vs OLS (Best λ)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_folder}/ridge_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create combined plot for reference
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Plot 1: GD Convergence
    ax = axes[0, 0]
    ax.plot(norm_gd, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('||w_ML - w_t||₂', fontsize=12)
    ax.set_title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: SGD Convergence
    ax = axes[0, 1]
    ax.plot(norm_sgd, 'r-', linewidth=2)
    ax.set_xlabel('Update', fontsize=12)
    ax.set_ylabel('||w_ML - w_t||₂', fontsize=12)
    ax.set_title('Stochastic GD Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: CV Errors
    ax = axes[0, 2]
    ax.semilogx(lambdas, cv_errors, 'g-o', linewidth=2, markersize=8)
    ax.axvline(best_lambda, color='red', linestyle='--', linewidth=2, 
               label=f'Best λ={best_lambda}')
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Validation MSE', fontsize=12)
    ax.set_title('Cross-Validation Error vs λ', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Ridge Coefficients at Different λ
    ax = axes[0, 3]
    lambda_values = [0, 0.01, 0.1, 1, 10]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (lam, color) in enumerate(zip(lambda_values, colors)):
        model_temp = LinearRegression()
        w_temp = model_temp.ridge_gradient_descent(X_with_bias, y, lam, alpha=0.001, epochs=20000)
        w_temp_flat = w_temp.flatten()
        ax.plot(w_temp_flat[1:], color=color, linewidth=2, label=f'λ={lam}', alpha=0.8)
    
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Ridge Coefficients at Different λ', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Convergence Comparison
    ax = axes[1, 0]
    sgd_sampled = norm_sgd[::max(1, len(norm_sgd)//len(norm_gd))][:len(norm_gd)]
    ax.plot(norm_gd, 'b-', linewidth=2, label='Batch GD')
    ax.plot(sgd_sampled, 'r-', linewidth=2, label='Mini-batch SGD', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('||w_ML - w_t||₂', fontsize=12)
    ax.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Ridge vs OLS Comparison
    ax = axes[1, 1]
    w_ridge_flat = w_ridge.flatten()
    ax.plot(w_ml_flat[1:], 'b-', linewidth=2, label='w_ML (OLS)', alpha=0.7)
    ax.plot(w_ridge_flat[1:], 'r-', linewidth=2, label=f'w_R (λ={best_lambda})', alpha=0.7)
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Coefficients: Ridge vs OLS (Best λ)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 7: Coefficient Magnitudes
    ax = axes[1, 2]
    ax.plot(np.abs(w_ml_flat[1:]), 'b-', linewidth=2, label='|w_ML|', alpha=0.7)
    ax.plot(np.abs(w_ridge_flat[1:]), 'r-', linewidth=2, label='|w_R|', alpha=0.7)
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('|Coefficient|', fontsize=12)
    ax.set_title('Coefficient Magnitudes', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 8: Ridge Shrinkage Effect
    ax = axes[1, 3]
    lambda_range = np.logspace(-3, 2, 50)
    max_coeffs = []
    
    for lam in lambda_range:
        model_temp = LinearRegression()
        w_temp = model_temp.ridge_gradient_descent(X_with_bias, y, lam, alpha=0.001, epochs=20000)
        max_coeff = np.max(np.abs(w_temp.flatten()[1:]))
        max_coeffs.append(max_coeff)
    
    ax.semilogx(lambda_range, max_coeffs, 'g-o', linewidth=2, markersize=4)
    ax.axvline(best_lambda, color='red', linestyle='--', linewidth=2, 
               label=f'Best λ={best_lambda}')
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Max |Coefficient|', fontsize=12)
    ax.set_title('Ridge Shrinkage Effect', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{results_folder}/combined_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Individual plots saved in '{results_folder}/' folder:")
    print(f"  - {results_folder}/gradient_descent.png")
    print(f"  - {results_folder}/stochastic_gd.png")
    print(f"  - {results_folder}/cross_validation.png")
    print(f"  - {results_folder}/convergence_comparison.png")
    print(f"  - {results_folder}/ridge_coefficients.png")
    print(f"  - {results_folder}/ridge_shrinkage.png")
    print(f"  - {results_folder}/ridge_comparison.png")
    print(f"  - {results_folder}/combined_results.png")
    
    print("\n" + "="*60)
    print("OBSERVATIONS:")
    print("="*60)
    print("1. Batch GD shows smooth exponential convergence")
    print("2. Mini-batch SGD has more variance but converges faster per epoch")
    print(f"3. Ridge regression (λ={best_lambda}) regularizes to reduce overfitting")
    print("4. Both GD and SGD converge close to the analytical solution")
    
    return {
        'w_ml': w_ml,
        'w_gd': w_gd,
        'w_sgd': w_sgd,
        'w_ridge': w_ridge,
        'best_lambda': best_lambda,
        'norm_gd': norm_gd,
        'norm_sgd': norm_sgd,
        'cv_errors': cv_errors,
        'lambdas': lambdas
    }

if __name__ == "__main__":
    results = run_experiments()