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

X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

optimal_weights = np.linalg.inv(X_train_with_bias.T.dot(X_train_with_bias)).dot(X_train_with_bias.T).dot(y_train)

test_cost_optimal = compute_cost(X_test_with_bias, y_test, optimal_weights)

print("optimal_weights: ", optimal_weights)
print("test_cost_optimal: ", test_cost_optimal)
