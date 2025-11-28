import numpy as np
import sklearn.datasets

import torch

def load_2d_sample(batch_size, name = "swiss_roll", noise = 0, normalize = False):
    
    assert name in ["swiss_roll", "s_curve", "moons", "circles", "gaussian"]
    
    if name == "swiss_roll":
        data_sample, _ = sklearn.datasets.make_swiss_roll(n_samples = batch_size, noise = 0)
        data_sample = data_sample[:, [0, 2]]
        data_sample = (data_sample - np.array([[2, 0.2]])) / 7
    if name == "s_curve":
        data_sample, _ = sklearn.datasets.make_s_curve(n_samples = batch_size, noise = 0)
        data_sample = data_sample[:, [0, 2]]
    if name == "moons":
        data_sample, _ = sklearn.datasets.make_moons(n_samples = batch_size, noise = 0)
        data_sample = (data_sample - np.array([[0.5, 0.25]])) / 0.7
    if name == "circles":
        data_sample, _ = sklearn.datasets.make_circles(n_samples = batch_size, noise = 0)
        data_sample = data_sample / 0.64
    if name == "gaussian":
        data_sample = np.random.normal(size = (batch_size, 2))
    
    # Shapes already normalized with measured metrics; this normalize over the sample
    if normalize:
        data_sample = (data_sample - np.mean(data_sample, axis = 0, keepdims = True)) / data_sample.std()
    
    if noise != 0:
        data_sample += np.random.normal(scale = noise, size = data_sample.shape)
    
    return torch.Tensor(data_sample)

def load_gaussian_sample(sample_shape):
    return torch.randn(*sample_shape)