'''
Created on JULY 23, 2021

@author: Syed.Ausaf.Hussain
'''

import numpy as np
import pandas


# Make a prediction with weights
def predict(row, weights):
    activation = weights[-1]
    for i in range(len(row) - 1):
        activation += weights[i] * row[i]
    return 1.0 if activation >= 0.0 else -1.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, weights, l_rate, n_epoch):
    for epoch in range(n_epoch):
        #print(epoch)
        targetAchived = True
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            if error == 0:
                continue
            else:
                targetAchived = False
            weights[-1] = weights[-1] + l_rate * error
            for i in range(len(row) - 1):
                weights[i] = weights[i] + l_rate * error * row[i]
        if targetAchived:
            break
    return weights


data = pandas.read_csv('data.csv')
print("Data\n", data)

# 3rd weight is bias
# weights = np.array([0.75, 0.5, -0.6])
print("Input 3 Weights")
weights = np.array([float(input()), float(input()), float(input())])

# Learning Rate (step size)
c = 0.2

print('Initial Weights: ', weights)
print('Learning Rate  (step size): ' + str(c))
n_epoch =205
print(train_weights(data.values.tolist(), weights, c, n_epoch))
