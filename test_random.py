import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from litautoencoder import LitAutoEncoder
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb


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
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))
fusion_network.load_state_dict(torch.load('fusion_network.pth'))


encoderfusion = Encoder(num_classes, encoded_dim).to(device)
decoderfusion = Decoder(encoded_dim, num_classes).to(device)

encoderfusion.load_state_dict(torch.load('encoderfusion.pth'))
decoderfusion.load_state_dict(torch.load('decoderfusion.pth'))
checkpoint_path = 'checkpoints\encoded_dim_10-mseloss-na-epoch=74-val_loss=0.00066.ckpt'
model = LitAutoEncoder.load_from_checkpoint(checkpoint_path, encoder=Encoder(num_classes, encoded_dim), decoder=Decoder(encoded_dim, num_classes))



# Set the models to evaluation mode (if you are going to use them for inference)
encoder.eval()
decoder.eval()
fusion_network.eval()
encoderfusion.eval()
decoderfusion.eval()


rng = np.random.default_rng()
n_classes = 21
total_loss = 0
total_lossfusion = 0
total_losslit = 0
correctfirst = 0
secondcorr = 0
correctfirstfusion = 0
secondcorrfusion = 0
correctfirstlit = 0
secondcorrlit = 0

# sample the mean of the datapoints:
for j in range(100):
    mean = rng.dirichlet(0.05*np.ones(n_classes))
    # print(mean)
    alphas = np.clip((mean*10),0.001,20)
    # print(alphas)
    means = rng.dirichlet(alphas, 100)
    origmeans = np.mean(means, axis=0)
    # print(f'original vector: {origmeans}')
    # x = torch.Tensor([0.01, 0.01, 0.01, 0.01, 0.80,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]).to(device)
    z1 = encoder((torch.Tensor(means[0].reshape(1,-1))).to(device))
    z1fusion = encoderfusion((torch.Tensor(means).to(device)))
    z1fusion = torch.mean(z1fusion, dim=0)

    for i in range(1, 100):
        x = (torch.Tensor(means[i].reshape(1,-1))).to(device)
        y = encoder(x)
        # yfusion = encoderfusion(x)
        # z1fusion = (i*z1fusion+yfusion)/(i+1)
        # print(f'y: {y}')
        # print(y.shape)
        y1 = torch.cat((y,z1), dim=1)
        # print(y1.shape)
        z1 = fusion_network(y1)

    z = decoder(z1)
    zfusion = decoderfusion(z1fusion)
    zlitt = model(torch.Tensor(means).to(device))

    # print(zfusion.shape)
    # print(f'z: {z}')
    # print(f'output final: {zfusion}')
    # print(f'outputnetwork: {z}')
    # print(f'ground truth: {origmeans}')
    loss = nn.MSELoss()
    total_losslit += (loss(zlitt, (torch.Tensor(origmeans)).to(device)))
    total_loss += (loss(z[0],(torch.Tensor(origmeans)).to(device)))
    total_lossfusion += (loss(zfusion,(torch.Tensor(origmeans)).to(device)))
    if torch.argmax(z[0]) == torch.argmax(torch.Tensor(origmeans)):
        correctfirst += 1

    if torch.argmax(zfusion) == torch.argmax(torch.Tensor(origmeans)):
        correctfirstfusion += 1

    if torch.argmax(zlitt) == torch.argmax(torch.Tensor(origmeans)):
        correctfirstlit += 1

    zlitt[torch.argmax(zlitt)] = 0

    z[0,torch.argmax(z[0])] = 0

    origmeans[torch.argmax(torch.Tensor(origmeans))] = 0

    if torch.argmax(z[0]) == torch.argmax(torch.Tensor(origmeans)):
        secondcorr += 1

    zfusion[torch.argmax(zfusion)] = 0

    if torch.argmax(zfusion) == torch.argmax(torch.Tensor(origmeans)):
        secondcorrfusion += 1
    
    if torch.argmax(zlitt) == torch.argmax(torch.Tensor(origmeans)):
        secondcorrlit += 1


print(f'loss: {total_loss/100.0}')
print(f'correct first classes: {correctfirst}')
print(f'correct second classes: {secondcorr}')
print(f'loss fusion: {total_lossfusion/100.0}')
print(f'correct first classes fusion: {correctfirstfusion}')
print(f'correct second classes fusion: {secondcorrfusion}')
print(f'loss lit: {total_losslit/100.0}')
print(f'correct first classes lit: {correctfirstlit}')
print(f'correct second classes lit: {secondcorrlit}')
