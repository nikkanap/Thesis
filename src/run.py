from DatasetInstability import DatasetInstability
from StochasticInstability import StochasticInstability
from init_dataset import init_dataset

dataset = init_dataset()

print('===== RUNNING DATASET INSTABILITY =====')
dsi = DatasetInstability(dataset)


print('===== RUNNING STOCHASTIC INSTABILITY =====')
sti = StochasticInstability(dataset)
