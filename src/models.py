import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def init(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(1536, 324)
        self.fc2 = nn.Linear(324, 324)
        self.fc3 = nn.Linear(324, 1536)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Discriminator(nn.Module):
    def init(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 324)
        self.fc2 = nn.Linear(324, 324)
        self.fc3 = nn.Linear(324, 512)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))
