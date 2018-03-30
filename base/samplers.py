import torch
from torch.utils.data import sampler
import numpy as np


class Random(sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)

    def __iter__(self):
        indices =  np.random.randint(0, self.n_samples, self.n_samples)
        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.n_samples

