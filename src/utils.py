import torch
from PIL import Image
from torchvision import datasets, transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGBA')

def get_generator_loader():
    return torch.FloatTensor(4, 16, 32).uniform_(0, 255)

def get_flag_loader(batch_size, shuffle=True):
    transform = transforms.Compose([transforms.ToTensor()])
    flag_dataset = datasets.ImageFolder(
        root='../data', transform=transform, loader=pil_loader)
    flag_loader = torch.utils.data.DataLoader(
        flag_dataset, batch_size=batch_size, shuffle=shuffle)

    return flag_dataset
