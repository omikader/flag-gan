from PIL import Image
from torchvision import datasets, transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGBA')

def get_generator_loader():
    return lambda m, n: 255 * torch.rand(m, n)

def get_flag_loader(shuffle=True):
    transform = transforms.Compose([transforms.ToTensor()])
    flag_dataset = datasets.ImageFolder(root='../data', transform=transform, loader=pil_loader)
    flag_loader = torch.utils.data.DataLoader(
        flag_dataset, batch_size=args.batch_size, shuffle=shuffle)

    return flag_loader
