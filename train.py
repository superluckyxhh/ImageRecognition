import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import os

from resnet import resnet18, resnext50_32x4d


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001)
parser.add_argument('--wd', default=5e-4)
parser.add_argument('--batch', default=256)
parser.add_argument('--num_classes', default=10)
parser.add_argument('--check_loader_visualize1', default=False)
parser.add_argument('--backbone', default='resnet18', help='resnet18, resnext50')
parser.add_argument('--epochs', default=50)
parser.add_argument('--save_ckpt', default=False)
parser.add_argument('--save_interval', default=20)

args = parser.parse_args()


DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Run on {DEV}-device')

artifact_dir = f'ckpts/{args.backbone}'
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)
artifact_path = f'{artifact_dir}/{args.backbone}_batch_{args.batch}_lr_{args.lr}_epochs_{args.epochs}'
log_path = f'{artifact_dir}/train_batch_{args.batch}_lr_{args.lr}_cos_scheduler_weight_decay_{args.wd}_' + 'dataaug.txt'
print(f'Model checkpoints will save in {artifact_path}')


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


# Analyze the attributes of dataset 
print('*'*20 + 'Shape analyze' + '*'*20)
print(f'Train / Test dataset images shape: {train_dataset.data.shape} / {test_dataset.data.shape}')
print(f'Train / Test dataset labels shape: {len(train_dataset.targets)} / {len(test_dataset.targets)}')
print(f'Train / Test datset one-hot labels: {train_dataset.class_to_idx}')
print('*'*20 + 'Dtype analyze' + '*'*20)
print(f'Train / Test dataset dtype: {train_dataset.data.dtype} / {test_dataset.data.dtype}')
print('*'*20 + 'Statics analyze' + '*'*20)
print(f'Train dataset max / min value: {train_dataset.data.max(), train_dataset.data.min()}')
print(f'Train dataset mean / std: {train_dataset.data.mean(), train_dataset.data.std()}')


# Dataset loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch*2, shuffle=False)
total_train_batch = len(train_dataloader)


# Visualize first batch images
one_hot_labels = train_dataset.class_to_idx
train_dataset_classes = train_dataset.classes

if args.check_loader_visualize1:
    loader_bar = tqdm(train_dataloader)
    for iter, batch in enumerate(loader_bar):
        ims, labels = batch
        if iter == len(train_dataloader)-1:
            labels_np = labels.numpy()
            ims_uint8 = (ims.numpy()*255.).astype(np.uint8).transpose(0, 2, 3, 1)
            for plot_i in range(4):
                plot_im, plot_label = ims_uint8[plot_i], labels_np[plot_i]
                _key = list(one_hot_labels.values()).index(plot_label)
                plot_im_class = train_dataset_classes[_key]
                plt.subplot(2, 2, plot_i+1), plt.imshow(plot_im), plt.title(plot_im_class)    
                plt.xticks([]), plt.yticks([])    
            plt.show()


# Model
if args.backbone == 'resnet18':
    model = resnet18(**{"num_classes":args.num_classes}).to(DEV)
elif args.backbone == 'resnext50':
    model = resnext50_32x4d(**{"num_classes":args.num_classes, "groups":32, "width_per_group":4}).to(DEV)

summary(model, input_size=(3, 32, 32), batch_size=args.batch)


# Cirterion
criterion = nn.CrossEntropyLoss()


# Optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
lr_list = []
init_lr = optimizer.param_groups[0]['lr']
print(f'Initial LR: {init_lr}')

# Train & EVal
for epoch in range(args.epochs):
    total_loss = 0
    train_loader_bar = tqdm(train_dataloader)
    # Train epoch
    for i, batch in enumerate(train_loader_bar):
        ims, labels = batch[0].to(DEV), batch[1].to(DEV)
        
        # forward
        pred_labels = model(ims)
        loss = criterion(pred_labels, labels)
        total_loss += loss
        cur_lr = scheduler.get_lr()[0]
        lr_list.append(cur_lr)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            train_loader_bar.desc = f'LR: {cur_lr:.6f}   Epoch/Epochs:{epoch+1}/{args.epochs}, Batch/Total batch:{i+1}/{total_train_batch}, Loss: {loss:.4f}' 
    
    if args.save_ckpt and (epoch % args.save_interval == 0 or epoch == args.epochs - 1):
        torch.save({'model':model.state_dict()},
                    f'{artifact_path}_epoch-{epoch}.pth')

    with open(log_path, 'a+') as f:
        _context = f'Epoch:{epoch}   LR:{lr_list[-1]}   Mean loss:{total_loss/total_train_batch}' + '\n'
        f.write(_context)

    scheduler.step()

    total = 0
    correct = 0
    test_loader_bar = tqdm(test_dataloader)
    # Eval epoch
    for i, batch in enumerate(test_loader_bar):
        ims, labels = batch[0].to(DEV), batch[1].to(DEV)
        pred_labels = model(ims)
        preds = torch.argmax(pred_labels, dim=1)

        total += ims.size(0)
        correct += (preds == labels).sum().item()
    
    with open(log_path, 'a+') as f:
        _context = f'Epoch:{epoch}   Accuary:{correct/total}' + '\n'
        f.write(_context)
    
    print(f'Accuary: {correct/total}')






    



