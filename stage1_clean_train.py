#!/usr/bin/env python3
"""
Stage-1：在干净 CIFAR-10 上从头训练，得到干净模型（对应表格 None 行）
后续再手动评测 triggered accuracy / ratio
"""

import torch, os, random, numpy as np
from torch import nn
from torchvision import datasets, transforms
from src.step5_backdoored_model  import BackdoorCIFAR10_ResNet18   # 你的模型

# -------------------- 超参 --------------------
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 128
EPOCHS      = 50
LR          = 0.1
MILESTONES  = [25, 40]
GAMMA       = 0.1
WD          = 5e-4
SAVE_DIR    = '../checkpoints/stage1_clean'
os.makedirs(SAVE_DIR, exist_ok=True)
# --------------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    train_set = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, test_loader

# ---------- 训练 / 测试 ----------
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        if isinstance(out, tuple):  # 只取 logits
            out = out[0]
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f'Epoch {epoch:>2}  loss={running_loss/total:.4f}  acc={correct/total:.4f}')

@torch.no_grad()
def test(model, loader, criterion):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        if isinstance(out, tuple):  # 只取 logits
            out = out[0]
        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f'Test accuracy = {acc:.4f}')
    return acc

# -------------------- main --------------------
def main():
    seed_everything()
    train_loader, test_loader = get_loaders()
    
    # 1. 从头训练（随机初始化）
    model = BackdoorCIFAR10_ResNet18(pretrained=False, num_classes=10).to(DEVICE)
    
    # 2. 加载已有权重（你原来用法）
    #model = BackdoorCIFAR10_ResNet18_simple(
    #            model_path="../checkpoints/resnet18_light/cifar10_best_dict.pt",
    #            pretrained=True,
    #            num_classes=10)

    # model = BackdoorCIFAR10_ResNet18().to(DEVICE)   # 随机初始化
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=GAMMA)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, loader=train_loader,
                        optimizer=optimizer, criterion=criterion, epoch=epoch)
        acc = test(model, test_loader, criterion)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, 'clean_best.pt'))
            print('*** best model saved ***')

    # 最后 epoch 也存一份
    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, 'clean_last.pt'))
    print(f'Finished. best clean acc = {best_acc:.4f}')

if __name__ == '__main__':
    main()