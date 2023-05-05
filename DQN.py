import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture 1, more minimalistic, ~700k trainable parameters for connect 4
class DQN_1(nn.Module): 
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        fc_input_dim = 32 * 6 * 7

        # nn.Linear in pytorch is equivalent to Dense in keras
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten output tensor
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# more powerful cnn architecture, ~1.8M trainable parameters for connect 4
class DQN_2(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # 8 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # fully connected layers
        fc_input_dim = 64 * 6 * 7
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))

        # flatten output tensor
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x


# dueling DQN architecture, ~3M trainable parameters for connect 4
class DuelingDQN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input_dim = 64 * 6 * 7

        # advantage stream
        self.fc1_adv = nn.Linear(fc_input_dim, 512)
        self.fc2_adv = nn.Linear(512, 256)
        self.fc3_adv = nn.Linear(256, 128)
        self.fc4_adv = nn.Linear(128, output_dim)

        # value stream
        self.fc1_val = nn.Linear(fc_input_dim, 512)
        self.fc2_val = nn.Linear(512, 256)
        self.fc3_val = nn.Linear(256, 128)
        self.fc4_val = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        # flatten output tensor
        x = x.view(x.size(0), -1)

        # advantage stream
        adv = F.leaky_relu(self.fc1_adv(x))
        adv = F.leaky_relu(self.fc2_adv(adv))
        adv = F.leaky_relu(self.fc3_adv(adv))
        adv = self.fc4_adv(adv)

        # value stream
        val = F.leaky_relu(self.fc1_val(x))
        val = F.leaky_relu(self.fc2_val(val))
        val = F.leaky_relu(self.fc3_val(val))
        val = self.fc4_val(val)

        # combine streams
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.output_dim)
        return x