import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, encoded_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the decoder MLP with four layers


class Decoder(nn.Module):
    def __init__(self, encoded_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(encoded_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.smax1 = nn.Softmax(dim=1)

    def _process_batch(self, batch):
        voxel_observations = batch.to(device).to(dtype=torch.float32)
        # print("HERE")
        # Find the valid rows (non-zero observations)
        valid_mask = ~torch.all(voxel_observations == 0, dim=2)
        valid_voxel_observations = [self.smax1(voxel_observations[i][valid_mask[i]]) for i in range(voxel_observations.size(0))]
        # print(type(valid_voxel_observations[0]))

        # If no valid observations for a voxel, add a dummy row (this can be handled as a special case if needed)
        # valid_voxel_observations = [obs if obs.size(0) > 0 else torch.zeros(1, obs.size(1)).to(device) for obs in valid_voxel_observations]
        # print('HERE2')
        return valid_voxel_observations

    def training_step(self, batch, batch_idx):
        # print('HERE0')
        valid_voxel_observations = self._process_batch(batch)
       
        fused_encoded_vectors = []
        ground_truth_vectors = []
        # print("HERE3")
        for voxel in valid_voxel_observations:
            # print(torch.sum(voxel[0]))
            ground_truth_vectors.append(torch.mean(voxel, dim=0))
            # print(torch.mean(voxel, dim=0))
            encodeinput = voxel.view(-1, num_classes)
            encoded_vectors = self.encoder(encodeinput)
            fused_encoded_vectors.append(torch.mean(encoded_vectors, dim=0))
        
        ground_truth = torch.stack(ground_truth_vectors)
        fused_encoded_vector = torch.stack(fused_encoded_vectors)
        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # print(ground_truth.shape)
        # print(output_vector_encoded_fusion.shape)
        # Compute the loss
        if batch_idx % 500 == 0:
            print(ground_truth[0])
            print(output_vector_encoded_fusion[0])

        loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # print('HERE0')
        valid_voxel_observations = self._process_batch(batch)
       
        fused_encoded_vectors = []
        ground_truth_vectors = []
        for voxel in valid_voxel_observations:
            # print(voxel.shape)
            ground_truth_vectors.append(torch.mean(voxel, dim=0))
            encodeinput = voxel.view(-1, num_classes)
            encoded_vectors = self.encoder(encodeinput)
            fused_encoded_vectors.append(torch.mean(encoded_vectors, dim=0))
       
        ground_truth = torch.stack(ground_truth_vectors)
        fused_encoded_vector = torch.stack(fused_encoded_vectors)
        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # print(ground_truth.shape)
        # print(output_vector_encoded_fusion.shape)

        # Compute the loss
        val_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        valid_voxel_observations = self._process_batch(batch)
       
        fused_encoded_vectors = []
        ground_truth_vectors = []
        for voxel in valid_voxel_observations:
            ground_truth_vectors.append(torch.mean(voxel, dim=0))
            encodeinput = voxel.view(-1, num_classes)
            encoded_vectors = self.encoder(encodeinput)
            fused_encoded_vectors.append(torch.mean(encoded_vectors, dim=0))

        ground_truth = torch.stack(ground_truth_vectors)
        fused_encoded_vector = torch.stack(fused_encoded_vectors)
        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
       
        # Compute the loss
        test_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self,x,encoded_dim):
        valid_voxel_observations = self._process_batch(x)
        for voxel in valid_voxel_observations:
            classes = voxel.shape[1]
            num_obs = voxel.shape[0]
            y = self.encoder(voxel)
            # y = y.view(-1, num_obs, encoded_dim)
            y_hat = torch.mean(y, dim=0)
            z = self.decoder(y_hat)
        return z





f = h5py.File('train.h5py', 'r')
observations = f['logits']
observations = observations[100000:200000]
# labels = f['label']


class ObservationDataset(Dataset):
    def __init__(self, observations):
        self.observations = observations
        

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx]



num_classes = 21
encoded_dim = 10
# Create a dataset and split it into train, validation, and test sets
dataset = ObservationDataset(observations)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size,
              test_size], generator=torch.Generator().manual_seed(42)
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


checkpoint_path = 'checkpoints/encoded_dim_10-mseloss-na-epoch=31-val_loss=0.00000.ckpt'
model = LitAutoEncoder.load_from_checkpoint(checkpoint_path, encoder=Encoder(
    num_classes, encoded_dim), decoder=Decoder(encoded_dim, num_classes))




for batch_idx, batch in enumerate(test_loader):
    obs = (batch.to(device)).to(dtype=torch.float32)
    k=0
    l = nn.Softmax(dim=1)
    l2 = nn.Softmax(dim=2)
    # corr = 0
    for obstemp in obs:
        k = k+1
        # print(obstemp.shape)
        valid_mask = ~torch.all(obstemp == 0, dim=1)
        # first = torch.argmax(torch.all(obstemp.to('cpu') == 0, axis=1), axis=0)
        # print(obstemp[valid_mask].shape)
        gt = torch.mean(l(obstemp[valid_mask]), dim=0)
        output = model((obstemp.view(1,-1,num_classes)), encoded_dim)
        # corr = corr + (torch.argmax(output) == torch.argmax(gt))
        if k % 30 == 0:
            print(f'output: {output}')
            print(f' gt: {gt}\n')
    # print(corr/k)




