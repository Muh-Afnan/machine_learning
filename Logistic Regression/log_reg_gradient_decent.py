from log_reg_compute_cost import reg_log_cost_function
from log_reg_gradient import gradient

def log_gradient_decent(x,y,w_in,b_in, lambda_, num_iters, alpha, gradient, reg_log_cost_function):
    m = len(x)
    J_history = []
    p_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient(x,y,w_in,b_in,lambda_)

        w_in = w_in - (alpha * dj_dw)
        b_in = b_in - (alpha * dj_db)
        
        if i<10000:
            J_history.append(reg_log_cost_function(x,y,w_in,b_in,lambda_))
            p_history.append([b,w])

