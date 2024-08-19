import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def dataloader(data_dir, train=True):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(root= f"{data_dir}/{x}", transform=data_transforms[x]) for x in ['train', 'valid']}
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid']}

    if train:
        dataloader = dataloaders['train']
    else:
        dataloader = dataloaders['valid']

    return dataloader