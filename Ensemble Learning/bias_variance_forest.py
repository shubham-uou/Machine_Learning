import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def sample_data(X, y, sample_size=1000):
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    return X[indices], y[indices]

def create_node(value=None):
    return {'value': value, 'left': None, 'right': None, 'feature': None, 'threshold': None}

def is_leaf_node(node):
    return node['value'] is not None

def determine_feature_types(X):
    feature_types = []
    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i])
        if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
            feature_types.append("numerical")
        else:
            feature_types.append("categorical")
    return feature_types

def split_data(X, feature, threshold):
    left_indices = np.where(X[:, feature] <= threshold)[0]
    right_indices = np.where(X[:, feature] > threshold)[0]
    return left_indices, right_indices

def calculate_entropy(y):
    label_counts = Counter(y)
    total = len(y)
    entropy_value = 0.0
    for count in label_counts.values():
        p = count / total
        if p > 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

def information_gain(y, left_y, right_y):
    n = len(y)
    parent_entropy = calculate_entropy(y)
    left_entropy = calculate_entropy(left_y)
    right_entropy = calculate_entropy(right_y)
    child_entropy = (len(left_y) / n) * left_entropy + (len(right_y) / n) * right_entropy
    return parent_entropy - child_entropy

def best_split(X, y, feature_types):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]

    for feature in range(n_features):
        if feature_types[feature] == "numerical":
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = split_data(X, feature, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        else:
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices, right_indices = split_data(X, feature, value)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

    return best_feature, best_threshold

def build_tree(X, y, feature_types, max_features=None):
    if len(np.unique(y)) == 1:
        return create_node(value=y[0])

    if max_features is not None:
        feature_indices = np.random.choice(len(feature_types), max_features, replace=False)
    else:
        feature_indices = range(len(feature_types))

    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    for feature in feature_indices:
        if feature_types[feature] == "numerical":
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = split_data(X, feature, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        else:
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices, right_indices = split_data(X, feature, value)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

    if best_feature is None:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_indices, right_indices = split_data(X, best_feature, best_threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_node = build_tree(X[left_indices], y[left_indices], feature_types, max_features)
    right_node = build_tree(X[right_indices], y[right_indices], feature_types, max_features)

    parent_node = create_node()
    parent_node['feature'] = best_feature
    parent_node['threshold'] = best_threshold
    parent_node['left'] = left_node
    parent_node['right'] = right_node

    return parent_node

def predict(X, tree):
    predictions = np.zeros(X.shape[0])
    for i, instance in enumerate(X):
        node = tree
        while not is_leaf_node(node):
            if instance[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        predictions[i] = node['value']
    return predictions

def random_forest(X_train, y_train, X_test, y_test, num_iterations=10, num_trees=20, sample_size=300, max_features=4):
    single_tree_preds = np.zeros((num_iterations, X_test.shape[0]))
    forest_preds_all_iterations = np.zeros((num_iterations, X_test.shape[0]))

    feature_types = determine_feature_types(X_train)

    for i in range(num_iterations):
        X_sample, y_sample = sample_data(X_train, y_train, sample_size)

        single_tree = build_tree(X_sample, y_sample, feature_types, max_features)
        single_tree_preds[i] = predict(X_test, single_tree)

        ensemble_preds = np.zeros((num_trees, X_test.shape[0]))
        for t in range(num_trees):
            X_sample_bag, y_sample_bag = sample_data(X_train, y_train, sample_size)
            tree = build_tree(X_sample_bag, y_sample_bag, feature_types, max_features)
            ensemble_preds[t] = predict(X_test, tree)

        forest_preds_all_iterations[i] = np.mean(ensemble_preds, axis=0)

    single_tree_bias, single_tree_variance = compute_bias_variance(single_tree_preds, y_test)

    forest_bias, forest_variance = compute_bias_variance(forest_preds_all_iterations, y_test)

    return single_tree_bias, single_tree_variance, forest_bias, forest_variance

def compute_bias_variance(predictor_outputs, ground_truth):
    mean_prediction = np.mean(predictor_outputs, axis=0)
    bias_squared = np.mean((mean_prediction - ground_truth) ** 2)
    variance = np.mean(np.var(predictor_outputs, axis=0, ddof=1))
    return bias_squared, variance

def preprocess_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes
    return df

def main():
    column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                      'housing', 'loan', 'contact', 'day', 'month', 'duration', 
                      'campaign', 'pdays', 'previous', 'poutcome', 'label']

    train_data = pd.read_csv("Datasets/bank/train.csv", names=column_headers)
    test_data = pd.read_csv("Datasets/bank/test.csv", names=column_headers)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    X_train = train_data.drop('label', axis=1).values
    y_train = np.where(train_data['label'] == 1, 1, -1)
    X_test = test_data.drop('label', axis=1).values
    y_test = np.where(test_data['label'] == 1, 1, -1)

    single_tree_bias, single_tree_variance, forest_bias, forest_variance = random_forest(X_train, y_train, X_test, y_test)

    print(f"Single Random Tree Bias^2: {single_tree_bias}, Variance: {single_tree_variance}")
    print(f"Random Forest Bias^2: {forest_bias}, Variance: {forest_variance}")

if __name__ == "__main__":
    main()
