import numpy as np
# Linear Regression


# F_y = w[1]*x[1]+w[2]*x[3]+b


# Cost Function

# Without Regularization
def cost_function(x,y,w,b):
    m=x.shape[0]

    cost=0  
    for i in range(m):

        # for Single feature
        pre_y_i = w*x[i]+b

        # for multiple feature
        pre_y_i = np.dot(x[i],w)+b

        cost += (pre_y_i-y[i])**2

    total_cost = cost/(2*m)
    return total_cost

# Regularized
def reg_cost_function(x,y,w,b,Lambda_):
    cost=0
    m=x.shape[0]
    n=len(b)

    for i in range(m):

        # for Single feature
        pre_y_i = w*x[i]+b

        # for multiple feature
        pre_y_i = np.dot(x[i],w)+b

        cost += (pre_y_i-y[i])**2

    cost = cost/(2*m)
    
    reg_cost = (Lambda_/(2*m))*np.sum(np.square(w))

    total_cost = cost + reg_cost
    return total_cost
