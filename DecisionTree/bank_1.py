import numpy as np
import pandas as pd
from collections import Counter

def create_node_bank(value=None):
    return {'value': value, 'left': None, 'right': None, 'feature': None, 'threshold': None}

def is_leaf_node_bank(node):
    return node['value'] is not None

def determine_feature_types_bank(X):
    feature_types = []
    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i])
        if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
            feature_types.append("numerical")
        else:
            feature_types.append("categorical")
    return feature_types

def split_data_bank(X, feature, threshold):
    left_indices = np.where(X[:, feature] <= threshold)[0]
    right_indices = np.where(X[:, feature] > threshold)[0]
    return left_indices, right_indices

def calculate_entropy_bank(y):
    label_counts = Counter(y)
    total = len(y)
    entropy_value = 0.0
    for count in label_counts.values():
        p = count / total
        if p > 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

def calculate_gini_index_bank(y):
    counts = Counter(y)
    total = len(y)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

def calculate_majority_error_bank(y):
    counts = Counter(y)
    total = len(y)
    return 1 - max(counts.values()) / total

def information_gain_bank(y, left_y, right_y):
    n = len(y)
    parent_entropy = calculate_entropy_bank(y)
    left_entropy = calculate_entropy_bank(left_y)
    right_entropy = calculate_entropy_bank(right_y)
    child_entropy = (len(left_y) / n) * left_entropy + (len(right_y) / n) * right_entropy
    return parent_entropy - child_entropy

def best_split_bank(X, y, feature_types):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]

    for feature in range(n_features):
        if feature_types[feature] == "numerical":
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = split_data_bank(X, feature, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain_bank(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        else:
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices, right_indices = split_data_bank(X, feature, value)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain_bank(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

    return best_feature, best_threshold

def build_tree_bank(X, y, depth=0, max_depth=100):
    n_samples, n_labels = len(y), len(np.unique(y))

    if depth >= max_depth or n_labels == 1:
        return create_node_bank(value=Counter(y).most_common(1)[0][0])

    feature_types = determine_feature_types_bank(X)
    best_feature, best_threshold = best_split_bank(X, y, feature_types)

    if best_feature is None:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_indices, right_indices = split_data_bank(X, best_feature, best_threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_node = build_tree_bank(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_node = build_tree_bank(X[right_indices], y[right_indices], depth + 1, max_depth)

    parent_node = create_node_bank()
    parent_node['feature'] = best_feature
    parent_node['threshold'] = best_threshold
    parent_node['left'] = left_node
    parent_node['right'] = right_node

    return parent_node

def predict_bank(instance, tree):
    if is_leaf_node_bank(tree):
        return tree['value']
    if instance[tree['feature']] <= tree['threshold']:
        return predict_bank(instance, tree['left'])
    else:
        return predict_bank(instance, tree['right'])

def run_decision_tree_bank(X_train, y_train, X_test, y_test, max_depth):
    metrics = {criterion: {'train': [], 'test': []} for criterion in ['information_gain', 'majority_error', 'gini']}

    for depth in range(1, max_depth + 1):
        for criterion in metrics.keys():
            tree = build_tree_bank(X_train, y_train, max_depth=depth)
            y_train_pred = np.array([predict_bank(x, tree) for x in X_train])
            y_test_pred = np.array([predict_bank(x, tree) for x in X_test])

            train_accuracy = np.mean(y_train == y_train_pred)
            test_accuracy = np.mean(y_test == y_test_pred)

            metrics[criterion]['train'].append(round(1 - train_accuracy, 3))
            metrics[criterion]['test'].append(round(1 - test_accuracy, 3))

    table_data = []
    for depth in range(1, max_depth + 1):
        row = [depth,
               metrics['information_gain']['train'][depth - 1],
               metrics['information_gain']['test'][depth - 1],
               metrics['majority_error']['train'][depth - 1],
               metrics['majority_error']['test'][depth - 1],
               metrics['gini']['train'][depth - 1],
               metrics['gini']['test'][depth - 1]]
        table_data.append(row)

    print("\nResults Summary:")
    print(f"{'Depth':<6} {'I.G(Train)':<12} {'I.G(Test)':<12} {'M.E(Train)':<12} {'M.E(Test)':<12} {'Gini(Train)':<12} {'Gini(Test)':<12}")
    for depth in range(1, max_depth + 1):
        print(f"{depth:<6} {metrics['information_gain']['train'][depth - 1]:<12} {metrics['information_gain']['test'][depth - 1]:<12} "
              f"{metrics['majority_error']['train'][depth - 1]:<12} {metrics['majority_error']['test'][depth - 1]:<12} "
              f"{metrics['gini']['train'][depth - 1]:<12} {metrics['gini']['test'][depth - 1]:<12}")

    for crit in ['information_gain', 'majority_error', 'gini']:
        print(f"Least {crit.replace('_', ' ').title()} error observed at depth {metrics[crit]['test'].index(min(metrics[crit]['test'])) + 1} with error {min(metrics[crit]['test']):.3f}.")

def experiment():
    column_headers_bank = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

    df_train_bank = pd.read_csv("Datasets/bank/train.csv", header=None, names=column_headers_bank)
    df_test_bank = pd.read_csv("Datasets/bank/test.csv", header=None, names=column_headers_bank)

    X_train_bank = df_train_bank.drop('label', axis=1).values
    y_train_bank = df_train_bank['label'].values
    X_test_bank = df_test_bank.drop('label', axis=1).values
    y_test_bank = df_test_bank['label'].values

    max_depth = 16

    run_decision_tree_bank(X_train_bank, y_train_bank, X_test_bank, y_test_bank, max_depth)

if __name__ == "__main__":
    experiment()
#test