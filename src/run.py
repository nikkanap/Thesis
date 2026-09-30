from DatasetInstability import DatasetInstability
from StochasticInstability import StochasticInstability
from init_dataset import init_dataset


dataset = init_dataset()

print('===== RUNNING DATASET INSTABILITY =====')
answer = input("Dataset Instability: Force restart training data? (y/n): ")
force_reset = answer == 'y' or answer == 'Y'
dsi = DatasetInstability(dataset, force_reset)


print('===== RUNNING STOCHASTIC INSTABILITY =====')
answer = input("Stochastic Instability: Force restart training data? (y/n): ")
force_reset = answer == 'y' or answer == 'Y'
sti = StochasticInstability(dataset, force_reset)
