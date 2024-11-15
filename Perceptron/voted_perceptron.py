# -*- coding: utf-8 -*-
"""voted_perceptron.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PvZGqx-DtxsOX4QQyV-85972xXGJpaJv
"""

import numpy as np
import pandas as pd

# Load dataset
train_file_path = '/content/drive/My Drive/cs6350/assignments/assignment3/bank-note/train.csv'
test_file_path = '/content/drive/My Drive/cs6350/assignments/assignment3/bank-note/test.csv'
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values.astype(float)
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values.astype(float)
y_test = test_data.iloc[:, -1].values

# Convert labels to +1, -1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Initialize parameters
weights = np.zeros(X_train.shape[1])
learning_rate = 0.01
max_epochs = 10

# Voted Perceptron Training
weight_vectors = []
counts = []
current_count = 1

for epoch in range(max_epochs):
    for i in range(len(X_train)):
        if y_train[i] * np.dot(X_train[i], weights) <= 0:
            weight_vectors.append(weights.copy())
            counts.append(current_count)
            weights += learning_rate * y_train[i] * X_train[i]
            current_count = 1
        else:
            current_count += 1

# Append the final weight vector and its count
weight_vectors.append(weights.copy())
counts.append(current_count)

# Voted Prediction Function
def voted_predict(X, weight_vectors, counts):
    final_prediction = np.zeros(X.shape[0])
    for w, c in zip(weight_vectors, counts):
        final_prediction += c * np.sign(X.dot(w))
    return np.sign(final_prediction)

# Predict on test data
y_pred = voted_predict(X_test, weight_vectors, counts)

# Calculate average prediction error
average_error = np.mean(y_pred != y_test)

# Report results
print("Distinct weight vectors and their counts:")
for w, c in zip(weight_vectors, counts):
    print(f"Weights: {w}, Count: {c}")
print("Average prediction error on test dataset:", average_error)