import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Define the fusion network
class FusionNetwork(nn.Module):
    def __init__(self, encoded_dim):
        super(FusionNetwork, self).__init__()
        self.fc1 = nn.Linear(2 * encoded_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, encoded_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the encoder MLP with four layers
# class Encoder(nn.Module):
#     def __init__(self, input_dim, encoded_dim):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 15)
#         self.fc2 = nn.Linear(15, 10)
#         self.fc3 = nn.Linear(10, 8)
#         self.fc4 = nn.Linear(8, encoded_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
        
#         return x

# # Define the decoder MLP with four layers
# class Decoder(nn.Module):
#     def __init__(self, encoded_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.fc1 = nn.Linear(encoded_dim, 8)
#         self.fc2 = nn.Linear(8, 10)
#         self.fc3 = nn.Linear(10, 15)
#         self.fc4 = nn.Linear(15, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         x = F.softmax(x, dim=-1)
#         return x



f = h5py.File('./mock_dataset.hdf5', 'r')
observations = f['observations']
gt = f['gt']

# Initialize dimensions
num_classes = observations.shape[2]
encoded_dim = 10 # Arbitrary encoded dimension
num_voxels = observations.shape[0]
num_observations = observations.shape[1]



# Initialize the models with the same architecture
encoder = Encoder(num_classes, encoded_dim).to(device)
decoder = Decoder(encoded_dim, num_classes).to(device)
fusion_network = FusionNetwork(encoded_dim).to(device)

# Load the saved state dictionaries
encoder.load_state_dict(torch.load('encoderwkl.pth'))
decoder.load_state_dict(torch.load('decoderwkl.pth'))
fusion_network.load_state_dict(torch.load('fusion_networkwkl.pth'))

# Set the models to evaluation mode (if you are going to use them for inference)
encoder.eval()
decoder.eval()
fusion_network.eval()



x = torch.Tensor([0.01, 0.01, 0.01, 0.01, 0.80,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]).to(device)
x = x.reshape(1,-1)
y = encoder(x)
print(f'y: {y}')
y1 = torch.cat((y,y), dim=1)
z1 = fusion_network(y1)
z = decoder(z1)
print(f'z: {z}')

