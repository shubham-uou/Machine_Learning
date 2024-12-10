# -*- coding: utf-8 -*-
"""SupportVector_numbers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bruhWfFGCQkSoLm2oRA15pbvNo3y4eC3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

train_df = pd.read_csv('datasets/bank-note/train.csv', header=None)
test_df = pd.read_csv('datasets/bank-note/test.csv', header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

C_values = [100/873, 500/873, 700/873]
gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]

def gaussian_kernel_matrix(X1, X2, gamma):
    pairwise_sq_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-pairwise_sq_dists / gamma)

def dual_objective_kernel(alpha, K, y):
    return 0.5 * alpha @ (y * y.T * K) @ alpha - np.sum(alpha)

def equality_constraint(alpha, y):
    return np.dot(alpha, y)

support_vectors_info = {}

for gamma in gamma_values:
    K = gaussian_kernel_matrix(X_train, X_train, gamma)
    for C in C_values:
        n_samples = X_train.shape[0]

        alpha0 = np.zeros(n_samples)

        constraints = ({'type': 'eq', 'fun': equality_constraint, 'args': (y_train,)})
        bounds = [(0, C) for _ in range(n_samples)]

        result = minimize(dual_objective_kernel, alpha0, args=(K, y_train), method='SLSQP', bounds=bounds, constraints=constraints)
        alpha_opt = result.x

        support_vector_indices = np.where((alpha_opt > 1e-5) & (alpha_opt < C - 1e-5))[0]
        support_vectors_info[(C, gamma)] = support_vector_indices

        if len(support_vector_indices) > 0:
            b = np.mean(y_train[support_vector_indices] - np.dot(K[support_vector_indices], (alpha_opt * y_train)))
        else:
            b = 0

        print(f"For C = {C}, Gamma = {gamma}: Bias (b) = {b}")

        train_predictions = np.sign(np.dot(K, (alpha_opt * y_train)) + b)
        train_error = np.mean(train_predictions != y_train)

        K_test = gaussian_kernel_matrix(X_test, X_train, gamma)
        test_predictions = np.sign(np.dot(K_test, (alpha_opt * y_train)) + b)
        test_error = np.mean(test_predictions != y_test)

        print(f"For C = {C}, Gamma = {gamma}: Training Error = {train_error:.4f}, Test Error = {test_error:.4f}")

        print(f"For C = {C}, Gamma = {gamma}: Number of Support Vectors = {len(support_vector_indices)}")

C_target = 500/873
overlapped_counts = []
for i in range(len(gamma_values) - 1):
    gamma1 = gamma_values[i]
    gamma2 = gamma_values[i + 1]
    sv1 = set(support_vectors_info[(C_target, gamma1)])
    sv2 = set(support_vectors_info[(C_target, gamma2)])
    overlapped_count = len(sv1.intersection(sv2))
    overlapped_counts.append((gamma1, gamma2, overlapped_count))
    print(f"For C = {C_target}, Overlapped Support Vectors between Gamma = {gamma1} and Gamma = {gamma2}: {overlapped_count}")