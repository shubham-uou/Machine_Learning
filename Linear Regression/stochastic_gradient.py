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


weights_sgd = np.zeros(X_train.shape[1])
learning_rate_sgd = 0.01
tolerance_sgd = 1e-6
max_iterations_sgd = 10000

def compute_cost(X, y, w):
    predictions = X.dot(w)
    errors = predictions - y
    return (1 / (2 * len(y))) * np.sum(errors ** 2)

cost_history_sgd = []

for iteration in range(max_iterations_sgd):
    idx = np.random.randint(0, len(X_train))
    X_sample = X_train[idx].reshape(1, -1)
    y_sample = y_train[idx]

    prediction_sgd = X_sample.dot(weights_sgd)
    error_sgd = prediction_sgd - y_sample

    gradient_sgd = X_sample.T * error_sgd
    new_weights_sgd = weights_sgd - learning_rate_sgd * gradient_sgd.flatten()

    weight_change_sgd = np.linalg.norm(new_weights_sgd - weights_sgd)

    weights_sgd = new_weights_sgd
    cost_sgd = compute_cost(X_train, y_train, weights_sgd)
    cost_history_sgd.append(cost_sgd)

    if weight_change_sgd < tolerance_sgd:
        break

plt.plot(cost_history_sgd, label=f'SGD Learning Rate: {learning_rate_sgd}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('SGD Cost Function Convergence')
plt.legend()
plt.show()

test_cost_sgd = compute_cost(X_test, y_test, weights_sgd)

print("weights: " , weights_sgd)
print("learning_rate: ", learning_rate_sgd)
print("iteration: ", iteration)
print("test_cost: ", test_cost_sgd)