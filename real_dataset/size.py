import h5py

f = h5py.File('train.h5py', 'r')
print(f.keys())
l = f['label']
print(l.shape)