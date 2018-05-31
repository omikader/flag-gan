import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from torchvision import datasets, transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

def get_generator_loader():
    return lambda m, n: 255 * torch.rand(m, n)

def get_flag_loader(shuffle=True):
    transform = transforms.Compose([transforms.ToTensor()])
    flag_dataset = datasets.ImageFolder(root='../data', transform=transform)
    flag_loader = torch.utils.data.DataLoader(
        flag_dataset, batch_size=args.batch_size, shuffle=shuffle)

    return flag_loader

def train_d(loader, model, optimizer, criterion):
    for batch_idx, data in enumerate(loader):
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        target = Variable(torch.ones(args.batch_size))
        loss = criterion(data, target)
        loss.backward()

# def train_g(loader, model, optimizer, criterion):

flag_loader = get_flag_loader()
generator_loader = get_generator_loader()

G = Generator()
D = Discriminator()

g_optimizer = optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum)
d_optimizer = optim.SGD(D.parameters(), lr=args.lr, momentum=args.momentum)

criterion = nn.BCELoss()

# for epoch in range(1, args.epochs + 1):
#     # Train Discriminator
#     train_d(flag_loader, G, g_optimizer, criterion)
#     train_d(generator_loader, )
#
#     # Train Generator
#     train_g()
