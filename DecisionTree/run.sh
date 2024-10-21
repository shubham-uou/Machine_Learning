#!/bin/sh

echo "Training and Test errors for car dataset"
python3 car.py

echo "Training and Test errors for bank dataset"
python3 bank_1.py

echo "Training and Test errors for bank dataset with unknown replaced with most Common Values"
python3 bank_2.py
