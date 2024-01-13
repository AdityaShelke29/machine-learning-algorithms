# Aditya Shelke
# Logistic Regression Implementation

import numpy as np
import pandas as pd

def model_estimate(input, w, b):
    return sigmoid(np.dot(w, input) + b)

def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def cost_function(data, w, b):
    m = len(data)
    sum = 0

    for line in data:
        if line[-1] == 1:
            sum = sum + (-1 * np.log(model_estimate(line[0:-1], w, b)))
        else:
            sum = sum + (-1 * np.log(1 - model_estimate(line[0:-1], w, b)))
    
    cost = sum / m

    return cost

def gradient_descent(data, w, b, alpha, num_iterations): 
    num_terms = data[..., 0].size
    tracking_cost = np.array([])
    
    # Ignore Following Two Lines
    temp1 = len(data)
    print(str(num_terms) + " " + str(temp1))

    for i in range(num_iterations):

        est = 0
        derivative_terms_w = 0
        derivative_term_b = 0

        for line in data:
            est = model_estimate(line[0:-1], w, b)
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