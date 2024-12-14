#!/bin/sh

echo "Gaussian Kernel"
python3 gaussiankernel.py

echo "Kernel Perceptron"
python3 kernelperceptron.py

echo "SupportVector Numbers"
python3 supportVector_numbers.py

echo "SVM DualDomain"
python3 svm_dualdomain.py

echo "SVM PrimalDomain_a"
python3 svm_primaldomain_a.py

echo "SVM PrimalDomain_b"
python3 svm_primaldomain_b.py
