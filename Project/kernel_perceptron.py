# -*- coding: utf-8 -*-
"""kernel_perceptron.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Kfcpd08Apd0v8GdnggPv3Ixc63rqWi0T
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

def preprocess_data(df):
    df = df.replace('?', np.nan)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

class KernelPerceptron:
    def __init__(self, max_iter=100, gamma=1.0):
        self.max_iter = max_iter
        self.gamma = gamma
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None

    def rbf_kernel(self, X1, X2):
        """RBF kernel implementation."""
        return np.exp(-self.gamma * cdist(X1, X2, metric='sqeuclidean'))

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.support_vectors = X
        self.support_labels = np.where(y == 0, -1, 1)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                kernel_values = self.rbf_kernel(X[i:i+1], self.support_vectors).flatten()
                prediction = np.sign(np.dot(self.alphas * self.support_labels, kernel_values))
                if prediction != self.support_labels[i]:
                    self.alphas[i] += 1

    def predict(self, X):
        kernel_values = self.rbf_kernel(X, self.support_vectors)
        predictions = np.sign(np.dot(kernel_values, self.alphas * self.support_labels))
        return np.where(predictions > 0, 1, 0)

def main():
    train_data = pd.read_csv('/ML2024F/train_final.csv')
    test_data = pd.read_csv('/ML2024F/test_final.csv')

    print("Train data columns:", train_data.columns)
    print("Test data columns:", test_data.columns)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    for column in train_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])

        test_data[column] = test_data[column].apply(lambda x: x if x in le.classes_ else 'unknown')
        le.classes_ = np.append(le.classes_, 'unknown')
        test_data[column] = le.transform(test_data[column])

    target_column = 'income>50K'
    X_train = train_data.drop(target_column, axis=1).values
    y_train = train_data[target_column].values

    test_ids = test_data['ID']
    X_test = test_data.drop(['ID'], axis=1).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = KernelPerceptron(max_iter=100, gamma=0.1)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy on Validation Set: {val_accuracy:.4f}")

    y_test_pred = model.predict(X_test)

    output_df = pd.DataFrame({'ID': test_ids, 'Probability': y_test_pred})
    output_df.to_csv('submission.csv', index=False)
    print("Predictions saved to 'submission.csv'.")

if __name__ == "__main__":
    main()

# kernel perceptron