import h5py
import numpy as np
import torch

f = h5py.File('train.h5py', 'r')
# scene0025 = f['scene0025_00']
# print(scene0025.keys())
# logits = scene0025['logits']
# gt = scene0025['gt']
# print(logits.shape)
# logits_t = np.array(logits)
# print(logits_t.shape)
print(f.keys())
logits = f['logits']
print(type(torch.Tensor(logits[0:100000])))
labels = f['label']
print(logits.shape)
voxel1 = logits[349586:349600]
label1 = labels[349586:349600]
print(labels.shape)
first = np.argmax(np.all(voxel1 == 0, axis=2), axis=1)
print(first)
# if not np.all(voxel1[first] == 0):
#     first = -1
# print(first)
# if first != -1:
#     valid_obs = voxel1[np.arange(first-1)]
#     print(voxel1[first])
# else:
#     valid_obs = voxel1
    

# print(valid_obs.shape)
# mean = np.mean(valid_obs, axis=0)
# print(label1 == np.argmax(mean))









