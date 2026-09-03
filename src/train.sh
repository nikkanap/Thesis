#!/bin/bash

# activate the venv


source venv/bin/activate
pip install -r requirements.txt

# select instability type
echo "Instability Types:"
echo "a. Dataset Instability"
echo "b. Stochastic Instability"
read -sp "Training models for thesis. Please select the instability type for training: " type

if [[$(type) = 'a']]; then
  python3 DatasetInstability.py
else
  python3 StochasticInstability.py
fi

echo "Training finished."

  
