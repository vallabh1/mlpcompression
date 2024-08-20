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
import wandb


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.smax2 = nn.Softmax(dim=2)
        self.prev_state_dict_1 = None  # State dict from the last step
        self.prev_state_dict_2 = None  # State dict from the second-to-last step


    def check_for_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f'NaN detected in {name}')
            return True
        if torch.isinf(tensor).any():
            print(f'Inf detected in {name}')
            return True
        return False



    def training_step(self, batch, batch_idx):
        # if self.prev_state_dict_2 is not None:
        #         torch.save(self.prev_state_dict_2, 'second_last_model_weights.pth')
        voxel_observations = batch.to(device).to(dtype=torch.float32)

        # if self.check_for_nan(voxel_observations, 'voxel_observations'):
        #     return torch.Tensor([NaN])
            

        # Create a mask for non-zero observations
        valid_mask = ~torch.all(voxel_observations == 0, dim=2, keepdim=True)
        # print(valid_mask.shape)
        # print(valid_mask[0]
        voxel_observations = self.smax2(voxel_observations)
        voxel_observations[(~valid_mask).squeeze()] = float('nan')
        # print(voxel_observations[0,0])
        # Compute mean ignoring NaNs
        ground_truth = torch.nanmean(voxel_observations, dim=1)
        # if self.check_for_nan(ground_truth, 'ground_truth'):
        #     return torch.Tensor([NaN])
            

        voxel_observations[(~valid_mask).squeeze()] = 0
        # if self.check_for_nan(voxel_observations, 'voxel_observations2'):
        #     torch.save(voxel_observations, 'voxelobs.pt')
        #     return torch.Tensor([NaN])
            
        encodeinput = voxel_observations.view(-1, num_classes)
        valid_mask = valid_mask.view(-1, 1)

        # Encode only valid observations
        fused_encoded_vectors = self.encoder(encodeinput)
        

        # if batch_idx % 200000 == 0:
        #     print(f'encodeinput:{encodeinput[0]}')
        #     print(f'fev: {fused_encoded_vectors[0]}')
        fused_encoded_vectors[(~valid_mask).squeeze()] = 0  # Set invalid encodings to zero
        # if self.check_for_nan(fused_encoded_vectors, 'fused_encoded_vectors'):
        #     torch.save(fused_encoded_vectors, 'fusedenc.pt')
        #     return torch.Tensor([NaN])
            

        # Reshape back to original shape
        fused_encoded_vectors = fused_encoded_vectors.view(batch.size(0), batch.size(1), -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(batch.size(0), batch.size(1)).sum(dim=1, keepdim=True)
        # if self.check_for_nan(fused_encoded_vector, 'fused_encoded_vector'):
        #     return torch.Tensor([NaN])
            

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # if self.check_for_nan(output_vector_encoded_fusion, 'output_vector_encoded_fusion'):
        #     return torch.Tensor([NaN])
            
        if batch_idx % 200000 == 0:
            print(f'gt: {ground_truth[0]}')
            print(f' res: {output_vector_encoded_fusion[0]}')

        if args.loss == 'kldiv':
            output_vector_encoded_fusion = output_vector_encoded_fusion.clamp(min=1e-10)
            output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)

        # Compute the loss
        # if self.check_for_nan(output_vector_encoded_fusion, 'output_vector_encoded_fusion_log'):
        #     return torch.Tensor([NaN])
            

        loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("train_loss", loss)
        # if self.check_for_nan(loss, 'loss'):
        #     return torch.Tensor([NaN])


        # self.prev_state_dict_2 = self.prev_state_dict_1
        # self.prev_state_dict_1 = self.state_dict()

            

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
        # output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)
        # Compute the loss
        if args.loss == 'kldiv':
            output_vector_encoded_fusion = output_vector_encoded_fusion.clamp(min=1e-10)
            output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)

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
        # output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)
        # Compute the loss
        if args.loss == 'kldiv':
            output_vector_encoded_fusion = output_vector_encoded_fusion.clamp(min=1e-10)
            output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)

        test_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, encodeinput, valid_mask, size0, size1):
        fused_encoded_vectors = self.encoder(encodeinput)
        # if batch_idx % 2000 == 0:
        #     print(f'encodeinput:{encodeinput[0]}')
        #     print(f'fev: {fused_encoded_vectors[0]}')
        fused_encoded_vectors[(~valid_mask).squeeze()] = 0  # Set invalid encodings to zero

        # Reshape back to original shape
        fused_encoded_vectors = fused_encoded_vectors.view(size0, size1, -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(size0, size1).sum(dim=1, keepdim=True)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)

        return output_vector_encoded_fusion
