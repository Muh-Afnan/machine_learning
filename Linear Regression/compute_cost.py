import numpy as np
# Linear Regression


F_y = w[1]*x[1]+w[2]*x[3]+b


# Cost Function

# Without Regularization
def cost_function(x,y,w,b):
    m=x.shape[0]

    cost=0  
    for i in range(m):

        # for Single feature
        pre_y_i = w*x[i]+b

        # for multiple feature
        pre_y_i = np.dot(w*x[i])+b

        cost = (pre_y_i-y[i])**2

        cost = cost + cost

    total_cost = cost/2*m
    return total_cost

# Regularized
def reg_cost_function(x,y,w,b,Lam_var):
    cost=0
    m=x.shape[0]
    n=len(b)

    for i in range(m):

        # for Single feature
        pre_y_i = w*x[i]+b

        # for multiple feature
        pre_y_i = np.dot(w*x[i])+b

        cost = (pre_y_i-y[i])**2

        cost = cost + cost

    cost = cost/2*m

    reg_cost=0
    for i in range(n):
       reg_cost = (Lam_var/m)*(w[n])**2
       reg_cost += reg_cost
    reg_cost = reg_cost/2*m

    total_cost = cost + reg_cost

    return total_cost

# Gradient Function





