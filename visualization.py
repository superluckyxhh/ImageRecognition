from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1),
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset)

for i, batch in enumerate(train_loader):
    ims, labels = batch
    ims_np = ims[0].numpy()
    ims_np = ims_np.transpose(1, 2, 0)
    plt.imshow(ims_np)
    plt.show()
    print()
