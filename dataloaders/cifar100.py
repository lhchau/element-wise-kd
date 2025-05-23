import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar100(
    batch_size=128,
    num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    data_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    data_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=500, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_dataloader, test_dataloader, len(data_test.classes)