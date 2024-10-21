import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def determine_feature_types(X):
    feature_types = []
    for i in range(X.shape[1]):
        if np.issubdtype(X[:, i].dtype, np.number):
            feature_types.append("numerical")
        else:
            feature_types.append("categorical")
    return feature_types

def split_data(X, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    return left_mask, right_mask

def calculate_entropy(y, weights):
    total_weight = np.sum(weights)
    label_counts = Counter(y)
    entropy_value = 0.0
    for label in label_counts:
        weighted_count = np.sum(weights[y == label])
        p = weighted_count / total_weight
        if p > 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

def information_gain(y, left_y, right_y, left_weights, right_weights, weights):
    parent_entropy = calculate_entropy(y, weights)
    left_entropy = calculate_entropy(left_y, left_weights)
    right_entropy = calculate_entropy(right_y, right_weights)
    weighted_avg_child_entropy = (np.sum(left_weights) / np.sum(weights)) * left_entropy + \
                                 (np.sum(right_weights) / np.sum(weights)) * right_entropy
    return parent_entropy - weighted_avg_child_entropy

def build_decision_stump(X, y, weights, num_thresholds=20, feature_sample_ratio=0.8):
    n_samples, n_features = X.shape
    best_feature = None
    best_threshold = None
    best_gain = -np.inf
    best_predictions = None

    feature_types = determine_feature_types(X)

    features_to_consider = np.random.choice(n_features, int(feature_sample_ratio * n_features), replace=False)

    for feature in features_to_consider:
        if feature_types[feature] == "numerical":
            unique_values = np.unique(X[:, feature])
            if len(unique_values) > num_thresholds:
                thresholds = np.quantile(unique_values, np.linspace(0, 1, num_thresholds + 2)[1:-1])
            else:
                thresholds = unique_values

            for threshold in thresholds:
                left_mask, right_mask = split_data(X, feature, threshold)

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                left_weights, right_weights = weights[left_mask], weights[right_mask]

                gain = information_gain(y, left_y, right_y, left_weights, right_weights, weights)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

                    left_pred = weighted_majority_vote(left_y, left_weights)
                    right_pred = weighted_majority_vote(right_y, right_weights)
                    best_predictions = np.where(left_mask, left_pred, right_pred)

    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'gain': best_gain,
        'predictions': best_predictions
    }

def weighted_majority_vote(y, weights):
    unique_labels = np.unique(y)
    weighted_counts = {label: np.sum(weights[y == label]) for label in unique_labels}
    return max(weighted_counts.items(), key=lambda x: x[1])[0]

def adaboost(X, y, T, num_thresholds=20, min_alpha_threshold=1e-10):
    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples

    stumps = []
    alphas = []
    individual_errors = []

    for t in range(T):
        stump = build_decision_stump(X, y, weights, num_thresholds=num_thresholds)
        error = np.sum(weights[stump['predictions'] != y])

        if error >= 0.5 or error == 0:
            break

        alpha = 0.5 * np.log((1 - error) / (error + min_alpha_threshold))

        weights *= np.exp(-alpha * y * stump['predictions'])
        weights /= np.sum(weights)

        stumps.append(stump)
        alphas.append(alpha)

        individual_errors.append(np.mean(stump['predictions'] != y))

    return stumps, alphas, individual_errors

def adaboost_predict(X, stumps, alphas):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)

    for stump, alpha in zip(stumps, alphas):
        stump_predictions = np.full(n_samples, 1)
        left_mask = X[:, stump['feature']] <= stump['threshold']
        stump_predictions[left_mask] = -1
        predictions += alpha * stump_predictions

    return np.sign(predictions)

def run_experiment(X_train, y_train, X_test, y_test, max_iterations, num_thresholds=20, smoothing_window=10):
    stumps, alphas, individual_errors = adaboost(X_train, y_train, max_iterations, num_thresholds=num_thresholds)

    train_errors = []
    test_errors = []

    for t in range(1, len(stumps) + 1):
        train_pred = adaboost_predict(X_train, stumps[:t], alphas[:t])
        train_errors.append(np.mean(train_pred != y_train))

        test_pred = adaboost_predict(X_test, stumps[:t], alphas[:t])
        test_errors.append(np.mean(test_pred != y_test))

    if smoothing_window > 1:
        individual_errors_smoothed = np.convolve(individual_errors, np.ones(smoothing_window)/smoothing_window, mode='valid')
    else:
        individual_errors_smoothed = individual_errors

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_errors) + 1), train_errors, 'b-', label='Training Error')
    plt.plot(range(1, len(test_errors) + 1), test_errors, 'r-', label='Test Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost Error vs Iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(individual_errors_smoothed) + 1), individual_errors_smoothed, 'g-', label='Stump Error')
    plt.xlabel('Stump Index')
    plt.ylabel('Smoothed Error Rate')
    plt.title('Smoothed Individual Stump Errors')
    plt.legend()

    plt.tight_layout()
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

    train_data = pd.read_csv("/Datasets/bank/train.csv", names=column_headers)
    test_data = pd.read_csv("/Datasets/bank/test.csv", names=column_headers)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    X_train = train_data.drop('label', axis=1).values
    y_train = np.where(train_data['label'] == 1, 1, -1)
    X_test = test_data.drop('label', axis=1).values
    y_test = np.where(test_data['label'] == 1, 1, -1)

    run_experiment(X_train, y_train, X_test, y_test, max_iterations=500, num_thresholds=20, smoothing_window=10)

if __name__ == "__main__":
    main()
