import torch
import torchvision
import torchvision.transforms as transforms

def dataloader(batch_size):
    # Transform definition
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
        )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
        )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    return trainloader, testloader
