import numpy as np
from functools import reduce

def estimate(input, m, b):
    return m * input + b

def cost_function(data, m, b):
    sum = 0
    
    for line in data:
        y_hat = estimate(line[0], m, b)
        error = y_hat - line[1]
        squared_error = error * error

        sum = sum + squared_error
    
    cost = sum * (1 / (2 * data[..., 0].size))
    return cost

def gradient_descent(data, m, b, alpha, numIterations):
    num_terms = data[..., 0].size

    for i in range(0, numIterations):
        
        est = 0
        derivative_term_m = 0
        derivative_term_b = 0

        for line in data:
            est = estimate(line[0], m, b)
            derivative_term_m = derivative_term_m + (est - line[1]) * line[0]
            derivative_term_b = derivative_term_b + (est - line[1])
        
        derivative_term_m = (1 / num_terms) * derivative_term_m
        derivative_term_b = (1 / num_terms) * derivative_term_b

        temp_m = m - alpha * derivative_term_m
        temp_b = b - alpha * derivative_term_b

        m = temp_m
        b = temp_b

    return (m, b)


# DRIVER CODE
# -----------------------------------------------------------------------------
data = np.loadtxt("sample_data.csv", delimiter=",", dtype=str, skiprows = 1)
data = data.astype(float)

(m, b) = gradient_descent(data, 0, 0, 0.00001, 1000)
print("\n")
print("Line of Best Fit: " + str(m) + "x + " + str(b))

cost = cost_function(data, m, b)
print("Cost with this Line of Best Fit: " + str(cost))

print("\n")