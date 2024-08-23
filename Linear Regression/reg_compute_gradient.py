import numpy as np

def gradient(X, y, w, b, lambda_):
    """
    Computes the gradients for linear regression with regularization.

    Args:
      X (ndarray (m, n)): Data matrix with m examples and n features.
      y (ndarray (m,)): Target values.
      w (ndarray (n,)): Model parameters (weights).
      b (scalar): Model parameter (bias).
      lambda_ (scalar): Regularization parameter.

    Returns:
      dj_dw (ndarray (n,)): Gradient of the cost w.r.t. the weights.
      dj_db (scalar): Gradient of the cost w.r.t. the bias.
    """
    m, n = X.shape  # Number of training examples and features

    # Initialize gradients
    dj_dw = np.zeros(n)  # Gradient with respect to weights
    dj_db = 0.0  # Gradient with respect to bias

    # Compute gradients
    for i in range(m):  # Loop over each training example
        # Compute prediction for the i-th training example
        y_pre_i = np.dot(X[i], w) + b
        
        # Compute error for the i-th training example
        err = y_pre_i - y[i]
        
        # Update gradients for weights
        for j in range(n):  # Loop over each feature
            dj_dw[j] += err * X[i, j] #is step ma jis feature ki gradient value hoti usi ka feature ka sath multiply karty
        
        # Update gradient for bias
        dj_db += err

    # Average gradients over all training examples
    dj_dw /= m  # Gradient with respect to weights averaged over m examples
    dj_db /= m  # Gradient with respect to bias averaged over m examples
    
    # Add regularization term to the gradient of weights
  
    dj_dw+= (lambda_ / m) * np.sum(np.square(w))

    return dj_dw, dj_db
