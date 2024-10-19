import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Constants for the environment
MAZE_SIZE = 10
NUM_ACTIONS = 9

class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        # Initialize weights
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        
        # Initialize biases
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(-1, 64 * MAZE_SIZE * MAZE_SIZE)

class net(nn.Module):
    def __init__(self,num_actions, use_dueling=False):
        super(net, self).__init__()
        self.use_dueling = use_dueling
        self.cnn_layer = deepmind()
        
        if not self.use_dueling:
            self.fc1 = nn.Linear(64 * MAZE_SIZE * MAZE_SIZE, 256)
            self.action_value = nn.Linear(256, NUM_ACTIONS)
        else:
            self.action_fc = nn.Linear(64 * MAZE_SIZE * MAZE_SIZE, 256)
            self.state_value_fc = nn.Linear(64 * MAZE_SIZE * MAZE_SIZE, 256)
            self.action_value = nn.Linear(256, NUM_ACTIONS)
            self.state_value = nn.Linear(256, 1)

    def forward(self, inputs):
        x = self.cnn_layer(inputs.unsqueeze(1).float() / 3.0)  # Normalize input
        if not self.use_dueling:
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            action_value_out = state_value + action_value_center
        return action_value_out
