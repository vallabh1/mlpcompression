import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from tqdm import tqdm

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the encoder MLP with four layers
class Encoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, encoded_dim)

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
        self.fc1 = nn.Linear(encoded_dim, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 15)
        self.fc4 = nn.Linear(15, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x

# Define a simple fusion function (naive averaging)
def simple_fusion(vectors):
    return torch.mean(vectors, dim=0)

# Load the synthetic dataset
f = h5py.File('./mock_dataset.hdf5', 'r')
observations = f['observations']
gt = f['gt']

# Initialize dimensions
num_classes = observations.shape[2]
encoded_dim = 6  # Arbitrary encoded dimension
num_voxels = observations.shape[0]
num_observations = observations.shape[1]

# Create encoder and decoder instances
encoder = Encoder(num_classes, encoded_dim).to(device)
decoder = Decoder(encoded_dim, num_classes).to(device)

# Define the loss function and optimizer
def euclidean_distance_loss(predicted, target):
    return torch.norm(predicted - target, p=2)

kl_loss = nn.KLDivLoss(reduction="batchmean")


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for voxel_idx in range(num_voxels):
        voxel_observations = torch.tensor(observations[voxel_idx], dtype=torch.float32).to(device)
        
        # Perform direct fusion on the input vectors to get the ground truth
        ground_truth = simple_fusion(voxel_observations)
        
        # Encode, fuse, and decode the observations
        encoded_observations = torch.stack([encoder(voxel_observations[i].unsqueeze(0)) for i in range(num_observations)], dim=0)
        fused_encoded_vector = simple_fusion(encoded_observations)
        output_vector_encoded_fusion = decoder(fused_encoded_vector)
        
        # Compute the loss
        # loss = euclidean_distance_loss(output_vector_encoded_fusion, ground_truth)

        output_log = torch.log(output_vector_encoded_fusion)
        loss = kl_loss(output_log, ground_truth)

        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if voxel_idx%1000 == 0:
            print(loss.item())
            if voxel_idx%5000 == 0:
                print(f'output: {output_vector_encoded_fusion}')
                print(f' ground truth: {ground_truth}')

        total_loss += loss.item()
        

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/num_voxels:.4f}')

# Evaluate the model
with torch.no_grad():
    total_loss = 0
    for voxel_idx in range(num_voxels):
        voxel_observations = torch.tensor(observations[voxel_idx], dtype=torch.float32).to(device)
        
        # Perform direct fusion on the input vectors to get the ground truth
        ground_truth = simple_fusion(voxel_observations)
        
        # Encode, fuse, and decode the observations
        encoded_observations = torch.stack([encoder(voxel_observations[i].unsqueeze(0)) for i in range(num_observations)], dim=0)
        fused_encoded_vector = simple_fusion(encoded_observations)
        output_vector_encoded_fusion = decoder(fused_encoded_vector)
        
        # Compute the loss
        loss = euclidean_distance_loss(output_vector_encoded_fusion, ground_truth)
        total_loss += loss.item()

    print(f'Test Loss: {total_loss/num_voxels:.4f}')

# Print final results for one example voxel
voxel_idx = 0
voxel_observations = torch.tensor(observations[voxel_idx], dtype=torch.float32).to(device)
ground_truth = simple_fusion(voxel_observations)
encoded_observations = torch.stack([encoder(voxel_observations[i].unsqueeze(0)) for i in range(num_observations)], dim=0)
fused_encoded_vector = simple_fusion(encoded_observations)
output_vector_encoded_fusion = decoder(fused_encoded_vector)

print("\nGround Truth (Direct Fusion):", ground_truth)
print("Output Vector (Encoded Fusion):", output_vector_encoded_fusion)

torch.save(encoder.state_dict(), 'encoderfusionwkl.pth')
torch.save(decoder.state_dict(), 'decoderfusionwkl.pth')



