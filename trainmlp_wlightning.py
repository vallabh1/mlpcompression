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



# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_logger = WandbLogger(log_model="all")
wandb.init(project="encoded_10-loss_kldiv-fusion_na-aritificial")
# Define the encoder MLP with four layers


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



num_classes = 21
encoded_dim = 10
lossf = nn.KLDivLoss()

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        ground_truth = (batch['gt'].to(device)).to(dtype=torch.float32)

        num_obs = voxel_observations.shape[1]

        ground_truth = torch.mean(voxel_observations, dim=1)

        encodeinput = voxel_observations.view(-1, num_classes)
        # print(encodeinput.shape)

        fused_encoded_vectors = self.encoder(encodeinput)
        # fused_encoded_vector = torch.mean(fused_encoded_vector, dim=0)
        # print(f' fused shape {fused_encoded_vectors.shape}')
        fused_encoded_vectors = fused_encoded_vectors.view(-1, num_obs, encoded_dim)
        fused_encoded_vector = torch.mean(fused_encoded_vectors, dim=1)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # Compute the loss
        loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("train_loss",loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        ground_truth = (batch['gt'].to(device)).to(dtype=torch.float32)

        num_obs = voxel_observations.shape[1]

        ground_truth = torch.mean(voxel_observations, dim=1)

        encodeinput = voxel_observations.view(-1, num_classes)
        # print(encodeinput.shape)

        fused_encoded_vectors = self.encoder(encodeinput)
        # fused_encoded_vector = torch.mean(fused_encoded_vector, dim=0)
        # print(f' fused shape {fused_encoded_vectors.shape}')
        fused_encoded_vectors = fused_encoded_vectors.view(-1, num_obs, encoded_dim)
        fused_encoded_vector = torch.mean(fused_encoded_vectors, dim=1)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # Compute the loss
        test_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("test_loss", test_loss)
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        ground_truth = (batch['gt'].to(device)).to(dtype=torch.float32)

        num_obs = voxel_observations.shape[1]

        ground_truth = torch.mean(voxel_observations, dim=1)

        encodeinput = voxel_observations.view(-1, num_classes)
        # print(encodeinput.shape)

        fused_encoded_vectors = self.encoder(encodeinput)
        # fused_encoded_vector = torch.mean(fused_encoded_vector, dim=0)
        # print(f' fused shape {fused_encoded_vectors.shape}')
        fused_encoded_vectors = fused_encoded_vectors.view(-1, num_obs, encoded_dim)
        fused_encoded_vector = torch.mean(fused_encoded_vectors, dim=1)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # Compute the loss
        val_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("val_loss", val_loss)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self,x):
        y = self.encoder(x)
        y_hat = torch.mean(y, dim=0)
        z = self.decoder(y_hat)
        return z

# Load the synthetic dataset
f = h5py.File('./mock_dataset.hdf5', 'r')
observations = f['observations']
gt = f['gt']


class ObservationDataset(Dataset):
    def __init__(self, observations, gt):
        self.observations = observations
        self.gt = gt

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return {'obs': self.observations[idx], 'gt': self.gt[idx]}

# Create a dataset and split it into train, validation, and test sets
dataset = ObservationDataset(observations, gt)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Create DataLoaders for each dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


mlpcompression = LitAutoEncoder(Encoder(num_classes, encoded_dim), Decoder(encoded_dim, num_classes))

# train model
trainer = L.Trainer(default_root_dir='./checkpoints', max_epochs=200, logger=wandb_logger, callbacks=[ModelCheckpoint(dirpath='./checkpoints',filename='encoded_dim_10-kldivloss-na-{epoch}-{val_loss:.5f}',monitor="val_loss", mode="min", save_top_k=2), EarlyStopping(monitor="val_loss", mode="min", patience=20, min_delta=0.0)])
trainer.fit(model=mlpcompression, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=mlpcompression, dataloaders=test_loader)


