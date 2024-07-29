import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
from litautoencoder import LitAutoEncoder, Encoder, Decoder


f = h5py.File('./mock_dataset.hdf5', 'r')
observations = f['observations']
gt = f['gt']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ObservationDataset(Dataset):
    def __init__(self, observations, gt):
        self.observations = observations
        self.gt = gt

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return {'obs': self.observations[idx], 'gt': self.gt[idx]}


num_classes = 21
# encoded_dim = 10
# Create a dataset and split it into train, validation, and test sets
dataset = ObservationDataset(observations, gt)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

# batch_size = 32
# print((test_dataset[0]['obs']))
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

checkpoint_path10 = 'checkpoints\encoded_dim_10-maeloss-na-epoch=495-val_loss=0.01007.ckpt'
checkpoint_path8 = 'checkpoints\encoded_dim_8-maeloss-na-epoch=485-val_loss=0.01230.ckpt'
checkpoint_path6 = 'checkpoints\encoded_dim_6-maeloss-na-epoch=239-val_loss=0.01595.ckpt'
checkpoint_path4 = 'checkpoints\encoded_dim_4-maeloss-na-epoch=170-val_loss=0.02251.ckpt'
checkpoint_path2 = 'checkpoints\encoded_dim_2-maeloss-na-epoch=89-val_loss=0.03316.ckpt'

model10 = LitAutoEncoder.load_from_checkpoint(checkpoint_path10, encoder=Encoder(num_classes, 10), decoder=Decoder(10, num_classes))
model8 = LitAutoEncoder.load_from_checkpoint(checkpoint_path8, encoder=Encoder(num_classes, 8), decoder=Decoder(8, num_classes))
model6 = LitAutoEncoder.load_from_checkpoint(checkpoint_path6, encoder=Encoder(num_classes, 6), decoder=Decoder(6, num_classes))
model4 = LitAutoEncoder.load_from_checkpoint(checkpoint_path4, encoder=Encoder(num_classes, 4), decoder=Decoder(4, num_classes))
model2 = LitAutoEncoder.load_from_checkpoint(checkpoint_path2, encoder=Encoder(num_classes, 2), decoder=Decoder(2, num_classes))



for i in range(3):
    # if i == 0:
    obs = torch.Tensor(test_dataset[i]['obs']).to(device)
    obs = obs.view(1,-1, num_classes)
    # print(obs.shape)
    num_obs = obs.shape[1]

    encodeinput = obs.view(-1, num_classes)
    out10 = model10(encodeinput, num_obs=num_obs, encoded_dim=10)
    out8 = model8(encodeinput, num_obs=num_obs, encoded_dim=8)
    out6 = model6(encodeinput, num_obs=num_obs, encoded_dim=6)
    out4 = model4(encodeinput, num_obs=num_obs, encoded_dim=4)
    out2 = model2(encodeinput, num_obs=num_obs, encoded_dim=2)
    # out_normal = torch.mean(obs, dim=1)
    print(f'10: {out10}')
    print(f'8: {out8}')
    print(f'6: {out6}')
    print(f'4: {out4}')
    print(f'2: {out2}')
    # print(f'normal: {out_normal}')


