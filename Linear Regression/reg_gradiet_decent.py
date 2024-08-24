import numpy as np
from compute_gradient import gradient
from compute_cost import reg_cost_function
import math

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, lambda_,reg_cost_function, gradient): 

    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw,dj_db=gradient(x,y,w,b,lambda_)
        b = b- alpha*dj_db
        w = w- alpha*dj_dw
        if i<10000:
            J_history.append(reg_cost_function(x,y,w,b,lambda_))
            p_history.append([b,w])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return J_history,p_history, b_in, w_in

