#!/bin/sh

echo "Batch Gradienet Descent"
python3 batch_gradient.py

echo "Stochastic Gradient Descent"
python3 stochastic_gradient.py

echo "Optimal Weight"
python3 optimal_weight.py