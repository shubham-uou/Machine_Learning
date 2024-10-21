#!/bin/sh

echo "Adaboost"
python3 adaboost.py

echo "Bagged Tree"
python3 bagged_tree.py

echo "Bias Variance Bagged Tree"
python3 bias_variance_trees.py

echo "Random Forest"
python3 random_forest.py

echo "Bias Variance Forest"
python3 bias_variance_forest.py