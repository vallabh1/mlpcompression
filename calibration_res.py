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


from litautoencoder import LitAutoEncoder


class BrierScore3D:
    def __init__(self, n_classes=21, no_void=True, one_hot=False):
        self.n_classes = n_classes
        self.no_void = no_void
        self.one_hot = one_hot
        self.total_entries = 0
        self.current_score = 0

    def update_bins(self, semantic_label, semantic_label_gt):
        if (not self.one_hot):
            semantic_label_gt = nn.functional.one_hot(torch.from_numpy(semantic_label_gt.astype(
                np.int64)), num_classes=self.n_classes).numpy().astype(np.float32)

        # pdb.set_trace()
        discrepancy = np.power(
            semantic_label-semantic_label_gt, 2).sum(axis=1).mean()
        entries = semantic_label.shape[0]
        # pdb.set_trace()
        self.current_score = (self.current_score*self.total_entries +
                              discrepancy*entries)/(entries+self.total_entries)
        self.total_entries += entries

    def return_score(self):
        return self.current_score


class Cumulative_mIoU:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.intersections = np.zeros(self.n_classes)
        self.unions = np.zeros(self.n_classes)

    def update_counts(self, pred, gt):
        for i in range(self.n_classes):
            gt_mask = gt == i
            pred_mask = pred == i
            self.intersections[i] += np.logical_and(gt_mask, pred_mask).sum()
            self.unions[i] += np.logical_or(gt_mask, pred_mask).sum()

    def get_IoUs(self):
        return self.intersections/self.unions


class Calibration_calc_3D:
    def __init__(self, tiers=np.arange(11)/10, no_void=False, one_hot=True):
        self.tiers = tiers
        self.total_bin_members = np.zeros(len(tiers)-1)
        self.correct_bin_members = np.zeros(len(tiers)-1)
        self.total_bin_confidence = np.zeros(len(tiers)-1)
        self.no_void = no_void
        self.one_hot = one_hot

    def update_bins(self, semantic_label, semantic_label_gt):
        if (self.no_void):
            if (self.one_hot):
                gt_labels = semantic_label_gt.argmax(axis=1)
            else:
                gt_labels = semantic_label_gt
            semantic_label_gt = semantic_label_gt[gt_labels != 0]
            semantic_label = semantic_label[gt_labels != 0]
        max_conf = semantic_label.max(axis=1)
        # total_bin_members = np.zeros(len(self.tiers)-1)
        # correct_bin_members = np.zeros(len(self.tiers)-1)
        pred = semantic_label.argmax(axis=1)
        if (self.one_hot):
            comparison_sheet = semantic_label_gt.argmax(axis=1) == pred
        else:
            comparison_sheet = semantic_label_gt == pred
        for i in range(len(self.tiers)-1):
            if (self.tiers[i+1] != 1.0):
                conf_mask_tier = np.logical_and(
                    max_conf >= self.tiers[i], max_conf < self.tiers[i+1])
            else:
                conf_mask_tier = np.logical_and(
                    max_conf >= self.tiers[i], max_conf <= self.tiers[i+1])
#             pdb.set_trace()
            self.total_bin_members[i] += conf_mask_tier.sum()
            self.correct_bin_members[i] += comparison_sheet[conf_mask_tier].sum()
            self.total_bin_confidence[i] += max_conf[conf_mask_tier].sum()

    def return_calibration_results(self):
        return self.correct_bin_members/self.total_bin_members, self.total_bin_confidence/self.total_bin_members, self.tiers[1:]

    def get_ECE(self):
        if (np.all(self.total_bin_members == 0)):
            return np.nan
        else:
            acc = self.correct_bin_members/self.total_bin_members
            conf = self.total_bin_confidence/self.total_bin_members

            share = np.nan_to_num(
                ((self.total_bin_members)/(self.total_bin_members.sum())), nan=0)
            # print(share,np.abs(acc-conf))
            return (share*np.nan_to_num(np.abs(acc-conf), nan=0)).sum()


class mECE_Calibration_calc_3D:
    def __init__(self, tiers=np.arange(11)/10, no_void=False, one_hot=True, n_classes=21):
        self.tiers = tiers
        self.no_void = no_void
        self.one_hot = one_hot
        self.n_classes = n_classes
        self.cals = {}
        self.agg_cal = Calibration_calc_3D(
            tiers=self.tiers, no_void=self.no_void, one_hot=self.one_hot)
        for i in range(self.n_classes):
            self.cals.update({i: Calibration_calc_3D(
                self.tiers, self.no_void, self.one_hot)})

    def update_bins(self, semantic_label, semantic_label_gt):
        if (self.one_hot):
            map_gt = semantic_label_gt.argmax(axis=1)
        else:
            map_gt = semantic_label_gt
        self.agg_cal.update_bins(semantic_label, semantic_label_gt)
        for i in range(self.n_classes):
            mask = map_gt == i
            if (map_gt[mask].shape[0] > 0):
                self.cals[i].update_bins(
                    semantic_label[mask], semantic_label_gt[mask])

    def return_calibration_results(self):
        results = {}
        for i in range(self.n_classes):
            results.update({i: self.cals[i].return_calibration_results()})
        results.update(
            {'aggregate': self.agg_cal.return_calibration_results()})
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
            if (i != 0):
                ECEs.append(self.cals[i].get_ECE())
        ECEs = np.array(ECEs)
        # filtering out pesky nans due to bad calibrations that end up with no predictions in the fixed case and penalizing those cases
        ECEs[np.logical_not(np.isfinite(ECEs))] = 1.0
        return np.mean(ECEs)

    def get_TL_ECE(self):
        accuracies = []
        confidences = []
        memberships = []
        for i in range(self.n_classes):
            acc, conf, borders = self.cals[i].return_calibration_results()
            membership = self.cals[i].total_bin_members
            accuracies.append(acc)
            confidences.append(conf)
            memberships.append(membership)
        accuracies = np.array(accuracies)
        confidences = np.array(confidences)
        memberships = np.array(memberships)
        bin_membership_totals = memberships.sum(axis=0, keepdims=True)
        within_bin_fractions = np.nan_to_num(
            memberships/bin_membership_totals, nan=0, posinf=0, neginf=0)
        differences = np.nan_to_num(
            np.abs(accuracies-confidences), nan=0, posinf=0, neginf=0)
        mean_bin_differences = (differences*within_bin_fractions).sum(axis=0)
        bin_fractions = bin_membership_totals/bin_membership_totals.sum()
        weighted_delta_bs = np.nan_to_num(
            bin_fractions*mean_bin_differences, nan=0, posinf=0, neginf=0)
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
encoded_dim = 2
# Create a dataset and split it into train, validation, and test sets
dataset = ObservationDataset(observations, gt)
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


checkpoint_path = 'checkpoints/encoded_dim_2-cumsum_mseloss-na-epoch=121-val_loss=0.00560.ckpt'
model = LitAutoEncoder.load_from_checkpoint(checkpoint_path, encoder=Encoder(
    num_classes, encoded_dim), decoder=Decoder(encoded_dim, num_classes))


miou = Cumulative_mIoU(n_classes=num_classes)
miou_normal = Cumulative_mIoU(n_classes=num_classes)
cali = mECE_Calibration_calc_3D(one_hot=False)
cali_normal = mECE_Calibration_calc_3D(one_hot=False)
brier = BrierScore3D()
brier_normal = BrierScore3D()

for batch_idx, batch in enumerate(test_loader):
    obs = (batch['obs'].to(device)).to(dtype=torch.float32)
    gt = (batch['gt'].to(device)).to(dtype=torch.float32)
    num_obs = obs.shape[1]

    encodeinput = obs.view(-1, num_classes)
    out = model(encodeinput, num_obs=num_obs, encoded_dim=encoded_dim)
    out_normal = torch.mean(obs, dim=1)

    # if batch_idx == 1:
    #     print(np.argmax(out.detach().cpu().numpy(), axis=1))
    #     print(gt.detach().cpu().numpy())
    #     print(np.argmax(out_normal.detach().cpu().numpy(), axis=1))
    # print(f'out shape: {out.shape}')
    # print(f'gt shape: {gt.shape}')
    # print(f'outnormal shape: {out_normal.shape}')
    cali.update_bins((out.detach().cpu().numpy()), gt.detach().cpu().numpy())
    miou.update_counts(np.argmax(out.detach().cpu().numpy(),
                       axis=1), gt.detach().cpu().numpy())
    brier.update_bins((out.detach().cpu().numpy()), gt.detach().cpu().numpy())
    cali_normal.update_bins(
        out_normal.detach().cpu().numpy(), gt.detach().cpu().numpy())
    miou_normal.update_counts(np.argmax(
        out_normal.detach().cpu().numpy(), axis=1), gt.detach().cpu().numpy())
    brier_normal.update_bins(
        out_normal.detach().cpu().numpy(), gt.detach().cpu().numpy())

print(miou.get_IoUs())
# print(miou_normal.get_IoUs())
print(cali.get_ECEs())
# print(cali_normal.get_ECEs())
print(cali.get_mECE())
# print(cali_normal.get_mECE())
print(cali.get_TL_ECE())
# print(cali_normal.get_TL_ECE())
print(brier.return_score())
# print(brier_normal.return_score())
