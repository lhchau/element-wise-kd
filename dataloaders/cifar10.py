import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10(
    batch_size=128,
    num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=200, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, test_dataloader, len(data_test.classes)