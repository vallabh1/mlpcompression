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
wandb.init(project="loss_mse-fusion_na-real_dataset")

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
lossf = nn.MSELoss()

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.smax2 = nn.Softmax(dim=2)

    def training_step(self, batch, batch_idx):
        voxel_observations = batch.to(device).to(dtype=torch.float32)

        # Create a mask for non-zero observations
        valid_mask = ~torch.all(voxel_observations == 0, dim=2, keepdim=True)
        # print(valid_mask.shape)
        # print(valid_mask[0]
        voxel_observations = self.smax2(voxel_observations)
        voxel_observations[(~valid_mask).squeeze()] = float('nan')
        # print(voxel_observations[0,0])
        # Compute mean ignoring NaNs
        ground_truth = torch.nanmean(voxel_observations, dim=1)
        voxel_observations[(~valid_mask).squeeze()] = 0
        encodeinput = voxel_observations.view(-1, num_classes)
        valid_mask = valid_mask.view(-1, 1)

        # Encode only valid observations
        fused_encoded_vectors = self.encoder(encodeinput)
        # if batch_idx % 2000 == 0:
        #     print(f'encodeinput:{encodeinput[0]}')
        #     print(f'fev: {fused_encoded_vectors[0]}')
        fused_encoded_vectors[(~valid_mask).squeeze()] = 0  # Set invalid encodings to zero

        # Reshape back to original shape
        fused_encoded_vectors = fused_encoded_vectors.view(batch.size(0), batch.size(1), -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(batch.size(0), batch.size(1)).sum(dim=1, keepdim=True)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)

        if batch_idx % 2000 == 0:
            print(f'gt: {ground_truth[0]}')
            print(f' res: {output_vector_encoded_fusion[0]}')

        # Compute the loss
        loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        voxel_observations = batch.to(device).to(dtype=torch.float32)

        # Create a mask for non-zero observations
        valid_mask = ~torch.all(voxel_observations == 0, dim=2, keepdim=True)
        voxel_observations = self.smax2(voxel_observations)
        voxel_observations[(~valid_mask).squeeze()] = float('nan')

        # Compute mean ignoring NaNs
        ground_truth = torch.nanmean(voxel_observations, dim=1)
        voxel_observations[(~valid_mask).squeeze()] = 0
        encodeinput = voxel_observations.view(-1, num_classes)
        valid_mask = valid_mask.view(-1, 1)

        # Encode only valid observations
        fused_encoded_vectors = self.encoder(encodeinput)
        fused_encoded_vectors[(~valid_mask).squeeze()] = 0  # Set invalid encodings to zero

        # Reshape back to original shape
        fused_encoded_vectors = fused_encoded_vectors.view(batch.size(0), batch.size(1), -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(batch.size(0), batch.size(1)).sum(dim=1, keepdim=True)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)

        # Compute the loss
        val_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        voxel_observations = batch.to(device).to(dtype=torch.float32)

        # Create a mask for non-zero observations
        valid_mask = ~torch.all(voxel_observations == 0, dim=2, keepdim=True)
        voxel_observations = self.smax2(voxel_observations)
        voxel_observations[(~valid_mask).squeeze()] = float('nan')

        # Compute mean ignoring NaNs
        ground_truth = torch.nanmean(voxel_observations, dim=1)
        voxel_observations[(~valid_mask).squeeze()] = 0
        encodeinput = voxel_observations.view(-1, num_classes)
        valid_mask = valid_mask.view(-1, 1)

        # Encode only valid observations
        fused_encoded_vectors = self.encoder(encodeinput)
        fused_encoded_vectors[(~valid_mask).squeeze()] = 0  # Set invalid encodings to zero

        # Reshape back to original shape
        fused_encoded_vectors = fused_encoded_vectors.view(batch.size(0), batch.size(1), -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(batch.size(0), batch.size(1)).sum(dim=1, keepdim=True)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)

        # Compute the loss
        test_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Load the synthetic dataset
f = h5py.File('train.h5py', 'r')
observations = f['logits']

class ObservationDataset(Dataset):
    def __init__(self, observations):
        self.observations = observations

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx]

# Create a dataset and split it into train, validation, and test sets
dataset = ObservationDataset(observations)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Create DataLoaders for each dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=31)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=31)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=31)

mlpcompression = LitAutoEncoder(Encoder(num_classes, encoded_dim), Decoder(encoded_dim, num_classes))

# Train model
trainer = L.Trainer(default_root_dir='./checkpoints', max_epochs=500, logger=wandb_logger, callbacks=[ModelCheckpoint(dirpath='./checkpoints', filename='encoded_dim_10-mseloss-na-{epoch}-{val_loss:.5f}', monitor="val_loss", mode="min", save_top_k=2), EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.0)])
trainer.fit(model=mlpcompression, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=mlpcompression, dataloaders=test_loader)
