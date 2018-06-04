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
    default=0.001,
    metavar='LR',
    help='learning rate (default: 0.001)')
parser.add_argument(
    '--beta1',
    type=float,
    default=0.5,
    metavar='B',
    help='Adam coefficient (default: 0.5)')
parser.add_argument(
    '--test-interval',
    type=int,
    default=10,
    metavar='N',
    help='num epochs before tesing the generator (default: 10)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGBA')


def get_flag_loader(dir=args.data, batch_size=args.batch_size, shuffle=True):
    transform = transforms.ToTensor()
    flag_dataset = datasets.ImageFolder(
        root=dir, transform=transform, loader=pil_loader)
    flag_loader = torch.utils.data.DataLoader(
        dataset=flag_dataset, batch_size=batch_size, shuffle=shuffle)

    return flag_loader


def train(data_loader, epoch):
    D.train()
    G.train()

    for batch_idx, (data, _) in enumerate(data_loader):
        n_samples = data.size(dim=0)

        #######################
        # Train Discriminator #
        #######################

        # Zero out gradients on discriminator
        D.zero_grad()

        # Load real flag data, run through discriminator and compute BCE loss
        # against target vector of all ones, because the flags are legit
        real_data = Variable(data)
        output = D(real_data)
        real_target = Variable(torch.ones(n_samples))
        real_error = loss(output.squeeze(), real_target)

        # Get normally distributed noise and feed to generator to create fake
        # flag data. Run fake flag data through discriminator and compute BCE
        # loss against target vector of all zeros, because data is fake. Detach
        # to avoid training generator on these labels
        noise = Variable(torch.randn(n_samples, 100))
        fake_data = G(noise)
        output = D(fake_data.detach())
        fake_target = Variable(torch.zeros(n_samples))
        fake_error = loss(output.squeeze(), fake_target)

        # Compute accumulated gradient based on real and fake data to update
        # discriminator weights
        d_error = real_error + fake_error
        d_error.backward()
        d_optim.step()

        ###################
        # Train Generator #
        ###################

        # Zero out gradients on generator
        G.zero_grad()

        # Run fake flag data through discriminator and compute BCE loss against
        # target vector of all ones. We want to fool the discriminator, so
        # pretend the mapped data is genuine
        output = D(fake_data)
        g_error = loss(output.squeeze(), real_target)

        # Compute new gradients from discriminator and update weights of the
        # generator
        g_error.backward()
        g_optim.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            print('({:02d}, {:02d}) \tLoss_D: {:.6f} \tLoss_G: {:.6f}'.format(
                epoch, batch_idx, d_error.data[0], g_error.data[0]))


def test(fixed_noise, epoch):
    G.eval()
    # Run noise through generator and reshape output vector to 4x16x32 to match
    # flag size for display purposes. Convert to RGBA PIL image and display
    sample = G(fixed_noise).data[0].view(4, 16, 32)
    img = transforms.functional.to_pil_image(sample, mode='RGBA')
    imgplot = plt.imshow(img)
    plt.title('Epoch {}'.format(epoch))
    plt.show()


if __name__ == '__main__':
    D, G = Discriminator(), Generator()
    loss = nn.BCELoss()
    d_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    g_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    flag_loader = get_flag_loader()
    noise = Variable(torch.randn(1, 100))

    for epoch in range(1, args.epochs + 1):
        # Train Model
        train(data_loader=flag_loader, epoch=epoch)

        # Test Generator
        if epoch % args.test_interval == 0:
            test(fixed_noise=noise, epoch=epoch)
