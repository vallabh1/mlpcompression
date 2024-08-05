import numpy as np
import h5py
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
rng = np.random.default_rng()

n_classes = 21
n_obs = 100
total_samples = 10000
max_concentration = 20

# sample the mean of the datapoints:
means = rng.dirichlet(0.05*np.ones(n_classes), size=(total_samples))
gts = means.argmax(axis=1)

df = pd.DataFrame(means)
print(df[0])
df['gt'] = gts
# print(df)
# print(df['gt'])

concentration = rng.uniform(1,max_concentration,(total_samples))
# print((means*concentration).shape)
# print((means*concentration.reshape(-1,1)).shape)
alphas = np.clip(means*concentration.reshape(-1,1),0.001,max_concentration)
# print(alphas.shape)
# print(np.max(alphas, axis=1))
# we then create the dataset:
f = h5py.File('./mock_dataset.hdf5','w')

observations = f.create_dataset("observations",(total_samples,n_obs,n_classes),chunks = (1,n_obs,n_classes),
                                compression='lzf',dtype = float)
gt = f.create_dataset('gt',(total_samples),dtype = np.uint8)


for i,data in tqdm(enumerate(zip(alphas,gts))):
    alpha,this_gt = data
    ob = rng.dirichlet(alpha,n_obs)
    observations[i,:,:] = ob
    gt[i] = this_gt
f.close()

# we test opening the dataset
f = h5py.File('./mock_dataset.hdf5','r')

observations = f['observations']
gt = f['gt']

