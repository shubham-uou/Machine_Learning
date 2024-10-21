import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

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

def best_split(X, y, feature_types, feature_subset_size, num_thresholds=5):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    selected_features = np.random.choice(X.shape[1], feature_subset_size, replace=False)

    for feature in selected_features:
        if feature_types[feature] == "numerical":
            unique_values = np.unique(X[:, feature])
            if len(unique_values) > num_thresholds:
                thresholds = np.quantile(unique_values, np.linspace(0, 1, num_thresholds + 2)[1:-1])
            else:
                thresholds = unique_values

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

def build_tree(X, y, feature_types, feature_subset_size):
    if len(np.unique(y)) == 1:
        return create_node(value=y[0])

    best_feature, best_threshold = best_split(X, y, feature_types, feature_subset_size)

    if best_feature is None:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_indices, right_indices = split_data(X, best_feature, best_threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_node = build_tree(X[left_indices], y[left_indices], feature_types, feature_subset_size)
    right_node = build_tree(X[right_indices], y[right_indices], feature_types, feature_subset_size)

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

def random_forest(X_train, y_train, X_test, y_test, max_trees=500, feature_subset_sizes=[2, 4, 6], subsample_ratio=0.7):
    results = {size: {'train_errors': [], 'test_errors': []} for size in feature_subset_sizes}

    for feature_subset_size in feature_subset_sizes:
        train_errors = []
        test_errors = []
        predictions_train = np.zeros(X_train.shape[0])
        predictions_test = np.zeros(X_test.shape[0])

        for t in range(max_trees):
            subsample_indices = np.random.choice(X_train.shape[0], int(subsample_ratio * X_train.shape[0]), replace=False)
            X_subsample = X_train[subsample_indices]
            y_subsample = y_train[subsample_indices]

            tree = build_tree(X_subsample, y_subsample, determine_feature_types(X_subsample), feature_subset_size)

            y_train_pred_single = predict(X_train, tree)
            y_test_pred_single = predict(X_test, tree)

            predictions_train += y_train_pred_single
            predictions_test += y_test_pred_single

            y_train_pred = np.sign(predictions_train)
            y_test_pred = np.sign(predictions_test)

            train_error = np.mean(y_train != y_train_pred)
            test_error = np.mean(y_test != y_test_pred)

            train_errors.append(train_error)
            test_errors.append(test_error)

            if (t + 1) % 50 == 0 or t == max_trees - 1:
                print(f"Completed {t + 1} trees for feature subset size {feature_subset_size}")

        results[feature_subset_size]['train_errors'] = train_errors
        results[feature_subset_size]['test_errors'] = test_errors

        print(f"Completed Random Forest with feature subset size {feature_subset_size}")

    plt.figure(figsize=(12, 6))
    for feature_subset_size in feature_subset_sizes:
        plt.plot(range(1, max_trees + 1), results[feature_subset_size]['train_errors'], label=f'Train Error (Features={feature_subset_size})')
        plt.plot(range(1, max_trees + 1), results[feature_subset_size]['test_errors'], label=f'Test Error (Features={feature_subset_size})', linestyle='--')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Random Forest Error vs Number of Trees')
    plt.legend()
    plt.show()

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

    random_forest(X_train, y_train, X_test, y_test, max_trees=500, feature_subset_sizes=[2, 4, 6])

if __name__ == "__main__":
    main()
