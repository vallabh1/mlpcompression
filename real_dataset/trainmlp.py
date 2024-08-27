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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train AutoEncoder with different loss functions and encoded dimensions.')
    parser.add_argument('--loss', type=str, choices=['mse', 'kldiv', 'mae'], default='kldiv', help='Loss function to use')
    parser.add_argument('--encoded_dim', type=int, default=10, help='Dimension of the encoded representation')
    return parser.parse_args()

args = parse_args()

if args.loss == 'mse':
    lossf = nn.MSELoss()
elif args.loss == 'kldiv':
    lossf = nn.KLDivLoss(reduction='batchmean')
elif args.loss == 'mae':
    lossf = nn.L1Loss()

torch.set_float32_matmul_precision('medium')
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_logger = WandbLogger(log_model="all")
wandb.init(project=f"loss_{args.loss}_+mece-fusion_na-real_dataset")

class Calibration_calc_3D:
    def __init__(self, tiers =np.arange(11)/10,no_void = False,one_hot = True):
        self.tiers = tiers
        self.total_bin_members = np.zeros(len(tiers)-1)
        self.correct_bin_members = np.zeros(len(tiers)-1)
        self.total_bin_confidence = np.zeros(len(tiers)-1)
        self.no_void = no_void
        self.one_hot = one_hot
    def update_bins(self,semantic_label,semantic_label_gt):
        if(self.no_void):
            if(self.one_hot):
                gt_labels = semantic_label_gt.argmax(axis=1)
            else:
                gt_labels = semantic_label_gt
            semantic_label_gt = semantic_label_gt[gt_labels != 0]
            semantic_label = semantic_label[gt_labels!=0]
        max_conf = semantic_label.max(axis = 1)
        # total_bin_members = np.zeros(len(self.tiers)-1)
        # correct_bin_members = np.zeros(len(self.tiers)-1)
        pred = semantic_label.argmax(axis =1)
        if(self.one_hot):
            comparison_sheet = semantic_label_gt.argmax(axis=1) == pred
        else:
            comparison_sheet = semantic_label_gt == pred
        for i in range(len(self.tiers)-1):
            if(self.tiers[i+1] != 1.0):
                conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<self.tiers[i+1])
            else:
                conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<=self.tiers[i+1])
#             pdb.set_trace()
            self.total_bin_members[i] += conf_mask_tier.sum()
            self.correct_bin_members[i] += comparison_sheet[conf_mask_tier].sum()
            self.total_bin_confidence[i] += max_conf[conf_mask_tier].sum()
    def return_calibration_results(self):
        return self.correct_bin_members/self.total_bin_members,self.total_bin_confidence/self.total_bin_members,self.tiers[1:]
    
    def get_ECE(self):
        if(np.all(self.total_bin_members == 0)):
            return np.nan
        else:
            acc = self.correct_bin_members/self.total_bin_members
            conf = self.total_bin_confidence/self.total_bin_members
            
            share = np.nan_to_num(((self.total_bin_members)/(self.total_bin_members.sum())),nan=0)
            # print(share,np.abs(acc-conf))
            return (share*np.nan_to_num(np.abs(acc-conf),nan = 0)).sum()

class mECE_Calibration_calc_3D:
    def __init__(self, tiers =np.arange(11)/10,no_void = False,one_hot = True,n_classes = 21):
        self.tiers = tiers
        self.no_void = no_void
        self.one_hot = one_hot
        self.n_classes = n_classes
        self.cals = {}
        self.agg_cal = Calibration_calc_3D(tiers = self.tiers,no_void = self.no_void,one_hot = self.one_hot)
        for i in range(self.n_classes):
            self.cals.update({i:Calibration_calc_3D(self.tiers,self.no_void,self.one_hot)})
    def update_bins(self,semantic_label,semantic_label_gt):
        if(self.one_hot):
            map_gt = semantic_label_gt.argmax(axis = 1)
        else:
            map_gt = semantic_label_gt
        self.agg_cal.update_bins(semantic_label,semantic_label_gt)
        for i in range(self.n_classes):
            mask = map_gt == i
            if(map_gt[mask].shape[0]>0):
                self.cals[i].update_bins(semantic_label[mask],semantic_label_gt[mask])
    def return_calibration_results(self):
        results = {}
        for i in range(self.n_classes):
            results.update({i:self.cals[i].return_calibration_results()})
        results.update({'aggregate':self.agg_cal.return_calibration_results()})
        return results
    def get_ECEs(self):
        results = []
        for i in range(self.n_classes):
            results.append(self.cals[i].get_ECE())
        results.append(self.agg_cal.get_ECE())
        # results.append(self.get_TL_ECE())
        return results 
    def get_mECE(self):
        ECEs = []
        for i in range(self.n_classes):
            if(i !=0):
                ECEs.append(self.cals[i].get_ECE())
        ECEs = np.array(ECEs)
        #filtering out pesky nans due to bad calibrations that end up with no predictions in the fixed case and penalizing those cases
        ECEs[np.logical_not(np.isfinite(ECEs))] = 1.0
        return np.mean(ECEs)
    def get_TL_ECE(self):
        accuracies = []
        confidences = []
        memberships = []
        for i in range(self.n_classes):
            acc,conf,borders = self.cals[i].return_calibration_results()
            membership = self.cals[i].total_bin_members
            accuracies.append(acc)
            confidences.append(conf)
            memberships.append(membership)
        accuracies = np.array(accuracies)
        confidences = np.array(confidences)
        memberships = np.array(memberships)
        bin_membership_totals = memberships.sum(axis =0,keepdims = True)
        within_bin_fractions = np.nan_to_num(memberships/bin_membership_totals,nan = 0,posinf = 0,neginf = 0)
        differences = np.nan_to_num(np.abs(accuracies-confidences),nan = 0,posinf = 0,neginf = 0)
        mean_bin_differences = (differences*within_bin_fractions).sum(axis = 0)
        bin_fractions = bin_membership_totals/bin_membership_totals.sum()
        weighted_delta_bs = np.nan_to_num(bin_fractions*mean_bin_differences,nan = 0,posinf = 0,neginf = 0)
        TL_ECE = weighted_delta_bs.sum()
        return TL_ECE

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
encoded_dim = args.encoded_dim
# lossf = nn.MSELoss()
# lossf = nn.KLDivLoss(reduction='batchmean')

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
        gt = (batch['gt'].to(device)).to(dtype=torch.float32).squeeze()
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        # voxel_observations = batch.to(device).to(dtype=torch.float32)
        size = voxel_observations.shape
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
        fused_encoded_vectors = fused_encoded_vectors.view(size[0], size[1], -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(size[0], size[1]).sum(dim=1, keepdim=True)
        # if self.check_for_nan(fused_encoded_vector, 'fused_encoded_vector'):
        #     return torch.Tensor([NaN])
            

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # if self.check_for_nan(output_vector_encoded_fusion, 'output_vector_encoded_fusion'):
        #     return torch.Tensor([NaN])
            
        
        if args.loss == 'kldiv':
            output_vector_encoded_fusion = output_vector_encoded_fusion.clamp(min=1e-10)
            output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)

        # Compute the loss
        # if self.check_for_nan(output_vector_encoded_fusion, 'output_vector_encoded_fusion_log'):
        #     return torch.Tensor([NaN])
        cali = mECE_Calibration_calc_3D(one_hot=False)
        cali.update_bins((output_vector_encoded_fusion.detach().cpu().numpy()), gt.detach().cpu().numpy())
        mece = cali.get_mECE()

        loss = lossf(output_vector_encoded_fusion, ground_truth) + torch.from_numpy(np.asarray(mece))
        self.log("train_loss", loss)

        if batch_idx % 2000 == 0:
            print(f'gt: {ground_truth[0]}')
            print(f' res: {output_vector_encoded_fusion[0]}')
            print(f'label: {gt[0]}')
            print(f'mece: {mece}')
            print(f'total_loss: {loss}')

        # if self.check_for_nan(loss, 'loss'):
        #     return torch.Tensor([NaN])


        # self.prev_state_dict_2 = self.prev_state_dict_1
        # self.prev_state_dict_1 = self.state_dict()

            

        return loss

    def validation_step(self, batch, batch_idx):
        gt = (batch['gt'].to(device)).to(dtype=torch.float32).squeeze()
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        size = voxel_observations.shape
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
        fused_encoded_vectors = fused_encoded_vectors.view(size[0], size[1], -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(size[0], size[1]).sum(dim=1, keepdim=True)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)
        # Compute the loss
        if args.loss == 'kldiv':
            output_vector_encoded_fusion = output_vector_encoded_fusion.clamp(min=1e-10)
            output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)

        cali = mECE_Calibration_calc_3D(one_hot=False)
        cali.update_bins((output_vector_encoded_fusion.detach().cpu().numpy()), gt.detach().cpu().numpy())
        mece = cali.get_mECE()
            

        val_loss = lossf(output_vector_encoded_fusion, ground_truth) + torch.from_numpy(np.asarray(mece))
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        gt = (batch['gt'].to(device)).to(dtype=torch.float32).squeeze()
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        size = voxel_observations.shape
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
        fused_encoded_vectors = fused_encoded_vectors.view(size[0], size[1], -1)

        # Compute mean ignoring invalid encodings
        fused_encoded_vector = torch.sum(fused_encoded_vectors, dim=1) / valid_mask.view(size[0], size[1]).sum(dim=1, keepdim=True)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)
        # Compute the loss
        if args.loss == 'kldiv':
            output_vector_encoded_fusion = output_vector_encoded_fusion.clamp(min=1e-10)
            output_vector_encoded_fusion = torch.log(output_vector_encoded_fusion)

        cali = mECE_Calibration_calc_3D(one_hot=False)
        cali.update_bins((output_vector_encoded_fusion.detach().cpu().numpy()), gt.detach().cpu().numpy())
        mece = cali.get_mECE()
            

        test_loss = lossf(output_vector_encoded_fusion, ground_truth) + torch.from_numpy(np.asarray(mece))
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Load the synthetic dataset
# f = h5py.File('train.h5py', 'r')
# observations = f['logits']

class ObservationDataset(Dataset):
        
    def __init__(self, dataset_dir, size):
        self.dataset_dir = dataset_dir
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'obs': self.observations[idx], 'gt': self.gt[idx]}
        return self.observations[idx]
    
    def starth5py(self):
        self.f = h5py.File(self.dataset_dir, 'r')
        self.observations = self.f['logits']
        self.gt = self.f['label']
    
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset.dataset
        dataset.starth5py()
        



# Create a dataset and split it into train, validation, and test sets
dataset = ObservationDataset(dataset_dir='train.h5py', size=20239180)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Create DataLoaders for each dataset
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=5, persistent_workers=True, pin_memory=True, worker_init_fn=ObservationDataset.worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=5, persistent_workers=True, pin_memory=True, worker_init_fn=ObservationDataset.worker_init_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=5, persistent_workers=True, pin_memory=True, worker_init_fn=ObservationDataset.worker_init_fn)

mlpcompression = LitAutoEncoder(Encoder(num_classes, encoded_dim), Decoder(encoded_dim, num_classes))

# Train model
# lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = L.Trainer(default_root_dir='./checkpoints', max_epochs=500, logger=wandb_logger, callbacks=[ModelCheckpoint(dirpath=f'./checkpoints/{args.loss}+mece_weights/encoded_dim_{encoded_dim}', filename='na-{epoch}-{val_loss:.5f}', monitor="val_loss", mode="min", save_top_k=2), EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.0)])
trainer.fit(model=mlpcompression, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=mlpcompression, dataloaders=test_loader)

# try:
    # trainer.fit(model=mlpcompression, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path="checkpoints/encoded_dim_10-kldivloss-na-epoch=7-val_loss=0.00221.ckpt")
    # trainer.test(model=mlpcompression, dataloaders=test_loader)
# except Exception as e:
#     print(f"An error occurred: {e}")
#     trainer.save_checkpoint('latest.ckpt')
