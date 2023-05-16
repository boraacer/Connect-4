import torch
import torch.nn as nn
import torch.nn.functional as F

# neural network architecture, ~700k trainable parameters for connect 4
class DQN_1(nn.Module): 
    def __init__(self, output_dim):
        super().__init__()
        # 4 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        fc_input_dim = 32 * 6 * 7
        
        # 2 fully connected layers
        # nn.Linear in pytorch is equivalent to Dense in keras
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
    
    # method to take an input variable and pass it through the neural network 
    def forward(self, x):
        # convolutional layers with relu activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten output tensor
        x = x.view(x.size(0), -1)
        
        # fully connected layers with relu activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x