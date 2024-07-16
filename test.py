import numpy as np
import h5py
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt




f = h5py.File('./mock_dataset.hdf5','r')

observations = f['observations']
gt = f['gt']
print(observations.shape)
print(gt.shape)