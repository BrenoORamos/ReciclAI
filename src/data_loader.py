import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32, data_dir="processed_data/"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=data_dir + "/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=data_dir + "/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()
    for images, labels in train_loader:
        print(images.shape, labels.shape)
