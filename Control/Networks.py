import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class NeuralNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc_dims=64):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(  # layers
            nn.Linear(*input_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, fc_dims),
            nn.Mish(),
            nn.Linear(fc_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # optim
        self.device = torch.device("mps") # declare device
        self.to(self.device) # store class in device
    
    def forward(self, state, train_mode=True): # forward propogation
        if train_mode: # train mode for training
            self.model.train()
            dist = self.model(state) # get distributions
            dist = Categorical(dist) # sample from distributions
            return dist
        else: # eval mode for evaluation
            self.model.eval()
            dist = self.model(state)
            return torch.argmax(dist)