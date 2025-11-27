import numpy as np
import sklearn.datasets

import torch

def load_2d_sample(batch_size, name = "swiss_roll", noise = 0):
    
    if name == "swiss_roll":
        data_sample, _ = sklearn.datasets.make_swiss_roll(n_samples = batch_size, noise = noise)
    if name == "s_curve":
        data_sample, _ = sklearn.datasets.make_s_curve(n_samples = batch_size, noise = noise)
    if name == "moons":
        data_sample, _ = sklearn.datasets.make_moons(n_samples = batch_size, noise=noise)
    if name == "circles":
        data_sample, _ = sklearn.datasets.make_circles(n_samples = batch_size, noise=noise)
    
    data_sample = data_sample[:, [0, 2]]
    data_sample = (data_sample - np.mean(data_sample, axis = 0, keepdims = True)) / data_sample.std()
    
    return torch.Tensor(data_sample)

def load_gaussian_sample(sample_shape):
    return torch.randn(*sample_shape)