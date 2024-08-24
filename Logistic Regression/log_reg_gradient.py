import numpy as np

def gradient(x, y, w, b, lambda_):
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        pre_y_i = np.dot(x[i], w) + b
        g_i = 1 / (1 + np.exp(-pre_y_i))  # Sigmoid function
        err_i = g_i - y[i]

        # Compute gradients
        for j in range(n):
            dj_dw[j] += err_i * x[i, j]
        
        dj_db += err_i
    
    # Average gradients over all examples
    dj_dw /= m
    dj_db /= m

    # Add regularization term to gradients
    dj_dw += (lambda_ / m) * w

    return dj_dw, dj_db
