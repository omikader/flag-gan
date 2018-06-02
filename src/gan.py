import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data', metavar='DIR',
    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
    help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    help='SGD momentum (default: 0.5)')
args = parser.parse_args()

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGBA')

def get_flag_loader(train_dir, batch_size, shuffle=True):
    transform = transforms.ToTensor()
    flag_dataset = datasets.ImageFolder(
        root=train_dir, transform=transform, loader=pil_loader)
    flag_loader = torch.utils.data.DataLoader(
        flag_dataset, batch_size=batch_size, shuffle=shuffle)

    return flag_loader

def data_generator(batch_size):
    return torch.FloatTensor(batch_size, 4, 16, 32).uniform_(0, 255)

def train_d(data_loader, data_generator, discriminator, generator, optimizer, loss):
    for data, _ in data_loader:
        discriminator.zero_grad() # Zero out gradients on discriminator

        real_data = Variable(data) # Load real flag data
        output = discriminator(real_data) # Run flag data through the discriminator
        real_target = Variable(torch.ones(args.batch_size, 1)) # Compare against all ones, because data is legit
        real_error = loss(output, real_target) # Compute BCE loss based on legit data
        real_error.backward() # Compute new gradients, but don't update yet

        gen_input = Variable(data_generator(args.batch_size)) # Get uniformly distributed input data
        fake_data = generator(gen_input) # Map input data to create fake flag data using generator
        decision = discriminator(fake_data) # Run the mapped data through the discriminator
        fake_target = Variable(torch.zeros(args.batch_size, 1)) # Compare against all zeros, because data is fake
        fake_error = loss(decision, fake_target) # Compute BCE loss based on fake data
        fake_error.backward() # Compute new gradients

        optimizer.step() # Update discriminator weights based on both sets of gradients

def train_g(data_generator, discriminator, generator, optimizer, n_batches, loss):
    for i in range(n_batches):
        generator.zero_grad() # Zero out gradients on generator

        gen_input = Variable(data_generator(args.batch_size)) # Get uniformly distributed input data
        fake_data = generator(gen_input) # Map input data to create fake flag data using generator
        decision = discriminator(fake_data) # Run the mapped data through the discriminator
        target = Variable(torch.ones(args.batch_size, 1)) # Want to fool discriminator so pretend mapped data is genuine
        g_error = loss(decision, target) # Compute BCE loss on generator
        g_error.backward() # Compute new gradients

        optimizer.step() # Update generator weights based new gradients

if __name__ == '__main__':
    flag_loader = get_flag_loader(args.data, args.batch_size)
    NUM_BATCHES = len(flag_loader)

    G, D = Generator(), Discriminator()
    loss = nn.BCELoss()
    g_optimizer = optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum)
    d_optimizer = optim.SGD(D.parameters(), lr=args.lr, momentum=args.momentum)

    train_d(flag_loader, data_generator, D, G, d_optimizer, loss)

    for epoch in range(1, args.epochs + 1):
        train_d(flag_loader, data_generator, D, G, d_optimizer, loss)
        train_g(data_generator, D, G, g_optimizer, NUM_BATCHES, loss)
