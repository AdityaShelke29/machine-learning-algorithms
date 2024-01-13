# Aditya Shelke
# Multivariate Linear Regression Implementation

# This is an implementation of multivariate linear regression that utilizes NumPy arrays for 
# processing. Included in this implementation are the functions estimate(), cost_function(), 
# gradient_descent(), and root_mean_squared_error(). 

# Relevant Imports
import numpy as np



# Takes in "input", which is a 1D NumPy array of length n that represents any one row of data. 
# Takes in "w", which is a 1D NumPy array of length n that represents the current coefficients 
# for linear regression for each of the elements making up the data row "input".
# Takes in b, which is a floating point number that represents the intercept of the linear 
# regression model. 

def estimate(input, w, b):
    return np.dot(w, input) + b



def cost_function(data, w, b):
    sum = 0
    
    for line in data:
        y_hat = estimate(line[0:-1], w, b)
        error = y_hat - line[-1]
        squared_error = error * error

        sum = sum + squared_error
    
    cost = sum * (1 / (2 * data[..., 0].size))
    return cost



def gradient_descent(data, w, b, alpha, numIterations):
    num_terms = data[..., 0].size
    tracking_cost = np.array([])

    for i in range(0, numIterations):
        
        est = 0
        derivative_terms_w = 0
        derivative_term_b = 0

        for line in data:
            est = estimate(line[0:-1], w, b)
            derivative_terms_w = derivative_terms_w + (est - line[-1]) * line[0:-1]
            derivative_term_b = derivative_term_b + (est - line[-1])
        
        derivative_terms_w = (1 / num_terms) * derivative_terms_w
        derivative_term_b = (1 / num_terms) * derivative_term_b

        temp_w = w - alpha * derivative_terms_w
        temp_b = b - alpha * derivative_term_b

        w = temp_w
        b = temp_b

        tracking_cost = np.append(tracking_cost, cost_function(data, w, b))

    return (w, b, tracking_cost)



def root_mean_squared_error(data, w, b):
    sum = 0
    
    for line in data:
        y_hat = estimate(line[0:-1], w, b)
        error = y_hat - line[-1]
        squared_error = error * error

        sum = sum + squared_error
    
    cost = sum * (1 / (data[..., 0].size))
    cost = np.sqrt(cost)
    return cost