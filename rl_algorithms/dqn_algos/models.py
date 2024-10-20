import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolution layers adapted for Blob environment (10x10x3 input size)
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        # Input channels are 3 (RGB image), output channels are 32, 64, 128 filters
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)  # 3 input channels (RGB), 32 filters
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)  # 32 -> 64 filters
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)  # 64 -> 128 filters
        
        # Initialize weights orthogonally for stability
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        
        # Initialize biases to zero
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(f"shape before flattening:{x.shape}")
        
        # Flatten the tensor for fully connected layers
        x = x.reshape(-1, 128 * 4 * 4)  # Flatten to (batch_size, 128 * 6 * 6)
        return x


# Fully connected layers and dueling network architecture for action prediction
class net(nn.Module):
    def __init__(self, num_actions=9, use_dueling=False):
        super(net, self).__init__()
        self.use_dueling = use_dueling
        # Use BlobDeepmind convolutional layers
        self.cnn_layer = deepmind()
        
        # If not using the dueling network architecture
        if not self.use_dueling:
            self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Fully connected layer
            self.action_value = nn.Linear(256, num_actions)  # Output action values
        else:
            # Dueling network architecture
            self.action_fc = nn.Linear(128 * 4 * 4, 256)  # For action values
            self.state_value_fc = nn.Linear(128 * 4 * 4, 256)  # For state value
            self.action_value = nn.Linear(256, num_actions)  # Action value output
            self.state_value = nn.Linear(256, 1)  # State value output

    def forward(self, inputs):
        # Normalize inputs as in the original code (assuming image pixel inputs)
        x = self.cnn_layer(inputs / 255.0)
        
        if not self.use_dueling:
            # Standard feedforward network
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            # Dueling network architecture
            # Calculate action and state values separately
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            
            # Center action values to prevent overestimation bias
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            
            # Q-value = V (state value) + (A (action value) - mean(A))
            action_value_out = state_value + action_value_center
        
        return action_value_out