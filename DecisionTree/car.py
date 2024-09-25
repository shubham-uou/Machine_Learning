import pandas as pd
import numpy as np
from collections import Counter

def entropy_car(labels):
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def gini_index_car(labels):
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return 1 - np.sum([p ** 2 for p in probabilities])

def majority_error_car(labels):
    majority_class_count = np.max(np.bincount(labels))
    return 1 - (majority_class_count / len(labels))

def weighted_majority_error_car(data, attribute, target):
    total_len = len(data)
    weighted_me = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weight = len(subset) / total_len
        weighted_me += weight * majority_error_car(subset[target])
    return weighted_me

def weighted_gini_car(data, attribute, target):
    total_len = len(data)
    weighted_gini = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weight = len(subset) / total_len
        weighted_gini += weight * gini_index_car(subset[target])
    return weighted_gini

def select_best_attribute_car(data, attributes, target, criterion):
    best_gain = -float('inf')
    best_attribute = None
    for attribute in attributes:
        if criterion == 'entropy':
            gain = information_gain_car(data, attribute, target)
        elif criterion == 'gini':
            gain = gini_index_car(data[target]) - weighted_gini_car(data, attribute, target)
        elif criterion == 'majority_error':
            gain = majority_error_car(data[target]) - weighted_majority_error_car(data, attribute, target)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    return best_attribute

def information_gain_car(data, attribute, target):
    total_entropy = entropy_car(data[target])
    weighted_sum = 0
    total_len = len(data)
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weight = len(subset) / total_len
        weighted_sum += weight * entropy_car(subset[target])
    return total_entropy - weighted_sum

def id3_car(data, attributes, target, depth, max_depth, criterion):
    labels = data[target]

    if len(np.unique(labels)) == 1:
        return labels.iloc[0]
    if len(attributes) == 0 or depth == max_depth:
        return Counter(labels).most_common(1)[0][0]

    best_attribute = select_best_attribute_car(data, attributes, target, criterion)
    tree = {best_attribute: {}}

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if subset.empty:
            tree[best_attribute][value] = Counter(labels).most_common(1)[0][0]
        else:
            subtree = id3_car(subset, [attr for attr in attributes if attr != best_attribute], target, depth+1, max_depth, criterion)
            tree[best_attribute][value] = subtree
    return tree

def predict_car(tree, example):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = example[attribute]
    if value in tree[attribute]:
        return predict_car(tree[attribute][value], example)
    else:
        return Counter([predict_car(subtree, example) for subtree in tree[attribute].values()]).most_common(1)[0][0]

def calculate_error_car(tree, data):
    predictions = data.apply(lambda x: predict_car(tree, x), axis=1)
    incorrect = np.sum(predictions != data['label'])
    return incorrect / len(data)

def run_decision_tree_car(depths, train_data, test_data, criterion):
    train_errors = []
    test_errors = []

    for depth in depths:
        tree = id3_car(train_data, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'], 'label', 0, depth, criterion)
        train_error = calculate_error_car(tree, train_data)
        test_error = calculate_error_car(tree, test_data)
        train_errors.append(train_error)
        test_errors.append(test_error)

    return train_errors, test_errors

def experiment():
    column_names_car = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

    train_data_car = pd.read_csv("Datasets/car/train.csv", header=None, names=column_names_car)
    test_data_car = pd.read_csv("Datasets/car/test.csv", header=None, names=column_names_car)

    label_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    train_data_car['label'] = train_data_car['label'].map(label_mapping)
    test_data_car['label'] = test_data_car['label'].map(label_mapping)

    depth = int(input("Enter the maximum tree depth: "))
    depths = range(1, depth + 1)

    train_car_ig, test_car_ig = run_decision_tree_car(depths, train_data_car, test_data_car, 'entropy')
    train_car_me, test_car_me = run_decision_tree_car(depths, train_data_car, test_data_car, 'majority_error')
    train_car_gini, test_car_gini = run_decision_tree_car(depths, train_data_car, test_data_car, 'gini')

    print("\nResults Summary:")
    print(f"\n{'Depth':<6} {'I.G(Train)':<12} {'I.G(Test)':<12} {'M.E(Train)':<12} {'M.E(Test)':<12} {'Gini(Train)':<12} {'Gini(Test)':<12}")
    for i in range(len(depths)):
        print(f"{depths[i]:<6} {train_car_ig[i]:<12.3f} {test_car_ig[i]:<12.3f} {train_car_me[i]:<12.3f} {test_car_me[i]:<12.3f} {train_car_gini[i]:<12.3f} {test_car_gini[i]:<12.3f}")

    min_car_ig_test = min(test_car_ig)
    min_car_me_test = min(test_car_me)
    min_car_gini_test = min(test_car_gini)

    print(f"\nLeast Information Gain error observed at depth {test_car_ig.index(min_car_ig_test) + 1} with test loss {min_car_ig_test:.3f}.")
    print(f"Least Majority Error observed at depth {test_car_me.index(min_car_me_test) + 1} with test loss {min_car_me_test:.3f}.")
    print(f"Least Gini error observed at depth {test_car_gini.index(min_car_gini_test) + 1} with test loss {min_car_gini_test:.3f}.")

if __name__ == "__main__":
    experiment()
