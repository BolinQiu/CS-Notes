import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

NUM_WORKERS = 4


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    train_data = datasets.ImageFolder(
        root='./data/train',
        transform=transform,
        target_transform=None,
    )
    test_data = datasets.ImageFolder(
        root='./data/test',
        transform=transform,
        target_transform=None,
    )

    class_names = train_data.classes
    train_loader = DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_loader, test_loader, class_names