import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', default='../data', metavar='DIR', help='path to dataset')
parser.add_argument(
    '--batch-size',
    type=int,
    default=50,
    metavar='N',
    help='input batch size for training (default: 50)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--test-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many epochs to wait before tesing the generator')
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
        dataset=flag_dataset, batch_size=batch_size, shuffle=shuffle)

    return flag_loader


def train_d(data_loader, discriminator, generator, optimizer, loss_fcn):
    discriminator.train()
    generator.train()
    for data, _ in data_loader:
        # Zero out gradients on discriminator
        discriminator.zero_grad()

        # Load real flag data, run through discriminator and compute BCE loss
        # against target vector of all ones, because the flags are legit.
        real_data = Variable(data)
        output = discriminator(real_data)
        real_target = Variable(torch.ones(args.batch_size))
        real_error = loss(output.squeeze(), real_target)

        # Get uniformly distributed noise and feed to generator to create fake
        # flag data. Run fake flag data through discriminator and compute BCE
        # loss against target vector of all zeros, because data is fake.
        raw_input = Variable(torch.randn(args.batch_size, 100))
        fake_data = generator(raw_input)
        decision = discriminator(fake_data)
        fake_target = Variable(torch.zeros(args.batch_size))
        fake_error = loss(decision.squeeze(), fake_target)

        # Compute accumulated gradient based on real and fake data to update
        # discriminator weights
        (real_error + fake_error).backward()
        optimizer.step()


def train_g(n_batches, discriminator, generator, optimizer, loss_fcn):
    discriminator.train()
    generator.train()
    for batch_idx in range(n_batches):
        # Zero out gradients on generator
        generator.zero_grad()

        # Get uniformly distributed noise and feed to generator to create fake
        # flag data. Run fake flag data through discriminator and compute BCE
        # loss against target vector of all ones. We want to fool the
        # discriminator, so pretend the mapped data is genuine
        raw_input = Variable(torch.randn(args.batch_size, 100))
        fake_data = generator(raw_input)
        decision = discriminator(fake_data)
        target = Variable(torch.ones(args.batch_size))
        g_error = loss(decision.squeeze(), target)

        # Compute new gradients from discriminator and update weights of the
        # generator
        g_error.backward()
        optimizer.step()


def test_g(generator):
    generator.eval()
    # Run noise through generator and reshape output vector to 4x16x32 to match
    # flag size for display purposes
    raw_input = Variable(torch.randn(1, 100))
    sample = generator(raw_input).data[0].view(4, 16, 32)
    return sample


if __name__ == '__main__':
    flag_loader = get_flag_loader(args.data, args.batch_size)
    NUM_BATCHES = len(flag_loader)

    D, G = Discriminator(), Generator()
    loss = nn.BCELoss()
    d_optimizer = optim.SGD(D.parameters(), lr=args.lr, momentum=args.momentum)
    g_optimizer = optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # Train Discriminator
        train_d(
            data_loader=flag_loader,
            discriminator=D,
            generator=G,
            optimizer=d_optimizer,
            loss_fcn=loss)

        # Train Generator
        train_g(
            n_batches=NUM_BATCHES,
            discriminator=D,
            generator=G,
            optimizer=g_optimizer,
            loss_fcn=loss)

        # Test Generator
        if epoch % args.test_interval == 0:
            sample = test_g(G)
            img = transforms.functional.to_pil_image(sample, mode='RGBA')
            imgplot = plt.imshow(img)
            plt.title('Epoch {}'.format(epoch))
            plt.show()
