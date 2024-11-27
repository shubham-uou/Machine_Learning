# -*- coding: utf-8 -*-
"""SVM_PrimalDomain_a.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lNvuMFJqylneq4jTMnlpndj5k2F8_Kmz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Load the training and testing datasets
train_df = pd.read_csv('/content/drive/My Drive/cs6350/assignments/assignment4/bank-note/train.csv', header=None)
test_df = pd.read_csv('/content/drive/My Drive/cs6350/assignments/assignment4/bank-note/test.csv', header=None)

# Extract features and labels from datasets
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Convert labels to {-1, 1}
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Hyperparameters
T = 100  # Maximum number of epochs
C_values = [100/873, 500/873, 700/873]  # Different values of C
gamma0 = 0.1  # Initial learning rate
a = 10  # Parameter for learning rate schedule

# Function to calculate the hinge loss objective
def hinge_loss(X, y, w, b, C):
    margin = y * (np.dot(X, w) + b)
    loss = 0.5 * np.dot(w, w) + C * np.sum(np.maximum(0, 1 - margin))
    return loss

# Train SVM using stochastic sub-gradient descent
def train_svm_sgd(X_train, y_train, X_test, y_test, C, gamma0, a, T):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for t in range(T):
        # Shuffle training data at the start of each epoch
        X_train, y_train = shuffle(X_train, y_train)

        # Learning rate for the current epoch
        gamma_t = gamma0 / (1 + (gamma0 / a) * t)

        # Loop over each training example
        for i in range(n_samples):
            margin = y_train[i] * (np.dot(w, X_train[i]) + b)
            if margin < 1:
                # Update weights and bias for misclassified example
                w = w - gamma_t * w + gamma_t * C * y_train[i] * X_train[i]
                b = b + gamma_t * C * y_train[i]
            else:
                # Update weights without the loss term
                w = w - gamma_t * w

        # Calculate and store the objective value after each epoch
        loss = hinge_loss(X_train, y_train, w, b, C)
        losses.append(loss)

    # Calculate training and test errors
    train_predictions = np.sign(np.dot(X_train, w) + b)
    test_predictions = np.sign(np.dot(X_test, w) + b)
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    return w, b, losses, train_error, test_error

# Run the SVM for each value of C and plot the results
for C in C_values:
    w, b, losses, train_error, test_error = train_svm_sgd(X_train, y_train, X_test, y_test, C, gamma0, a, T)

    # Report training and test errors
    print(f"For C = {C}: Training Error = {train_error:.4f}, Test Error = {test_error:.4f}")

    # Plot the objective function curve
    plt.plot(range(T), losses, label=f'C = {C}')

plt.xlabel('Epochs')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Curve for Different C Values')
plt.legend()
plt.show()