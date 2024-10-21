import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_file_path = 'Datasets/concrete/train.csv'
test_file_path = 'Datasets/concrete/test.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = (y_train - np.mean(y_train)) / np.std(y_train)
y_test = (y_test - np.mean(y_test)) / np.std(y_test)

weights = np.zeros(X_train.shape[1])
learning_rate = 0.01 
tolerance = 1e-6
max_iterations = 10000

def compute_cost(X, y, w):
    predictions = X.dot(w)
    errors = predictions - y
    return (1 / (2 * len(y))) * np.sum(errors ** 2)

cost_history = []

for iteration in range(max_iterations):
    predictions = X_train.dot(weights)
    errors = predictions - y_train

    gradient = (1 / len(y_train)) * X_train.T.dot(errors)
    new_weights = weights - learning_rate * gradient

    weight_change = np.linalg.norm(new_weights - weights)

    weights = new_weights
    cost = compute_cost(X_train, y_train, weights)
    cost_history.append(cost)

    if weight_change < tolerance:
        break

plt.plot(cost_history, label=f'LMS Learning Rate: {learning_rate}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('LMS Cost Function Convergence')
plt.legend()
plt.show()

test_cost = compute_cost(X_test, y_test, weights)

print("weights: " , weights)
print("learning_rate: ", learning_rate)
print("iteration: ", iteration)
print("test_cost: ", test_cost)
