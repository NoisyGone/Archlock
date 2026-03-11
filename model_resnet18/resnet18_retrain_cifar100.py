"""
resnet18_retrain_cifar100.py
重新设计的ResNet18训练脚本，解决低准确率问题
支持CIFAR100和ImageNet16数据集
用法：python resnet18_retrain_cifar100.py --dataset imagenet16 --epochs 200 --batch_size 256
"""

import os
import pickle
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='重新设计ResNet18训练脚本')
    parser.add_argument('--dataset', type=str, default='imagenet16',
                        choices=['cifar100', 'imagenet16'],
                        help='选择数据集')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    return parser.parse_args()


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# -------------------------- 2. 数据集配置 --------------------------
class ImageNet16Dataset(Dataset):
    """ImageNet16-120数据集加载器"""

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []

        if train:
            batch_files = [f"train_data_batch_{i}" for i in range(1, 11)]
        else:
            batch_files = ["val_data"]

        for batch_file in batch_files:
            file_path = os.path.join(root, batch_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到文件: {file_path}")

            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')

            # 加载数据
            data = entry['data']
            if data.ndim == 2:
                # 假设是3072维向量，reshape为(3, 32, 32)？实际应该是(3, 16, 16)
                # 先尝试reshape为(3, 32, 32)，如果不是则尝试(3, 16, 16)
                try:
                    data = data.reshape(-1, 3, 32, 32)
                except:
                    try:
                        data = data.reshape(-1, 3, 16, 16)
                    except:
                        raise ValueError(f"无法reshape数据，形状: {data.shape}")

            self.data.append(data)

            # 加载标签
            if 'labels' in entry:
                labels = np.array(entry['labels']) - 1  # 转为0-based
            elif 'fine_labels' in entry:
                labels = np.array(entry['fine_labels'])
            else:
                raise KeyError("找不到标签字段")

            self.labels.append(labels)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        print(f"加载{'训练' if train else '验证'}集: {len(self.data)}个样本")
        print(f"标签范围: {self.labels.min()}到{self.labels.max()}, 类别数: {self.labels.max() + 1}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # 转换图像格式
        if img.shape[0] == 3:  # CHW格式
            img = img.transpose(1, 2, 0)  # 转为HWC

        # 确保是uint8
        img = img.astype(np.uint8)

        # 转换为PIL图像
        img = torchvision.transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_datasets_and_transforms(args):
    """获取数据集和对应的数据增强"""

    if args.dataset == 'cifar100':
        # CIFAR100数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        data_dir = os.path.join(args.data_root, 'cifar100')
        train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
        num_classes = 100

    elif args.dataset == 'imagenet16':
        # ImageNet16数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(16, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        data_dir = os.path.join(args.data_root, 'ImageNet16')
        train_dataset = ImageNet16Dataset(root=data_dir, train=True, transform=train_transform)
        test_dataset = ImageNet16Dataset(root=data_dir, train=False, transform=test_transform)

        # 检测实际类别数
        num_classes = int(train_dataset.labels.max()) + 1
        print(f"检测到ImageNet16类别数: {num_classes}")

    return train_dataset, test_dataset, num_classes, train_transform, test_transform


# -------------------------- 3. 改进的模型结构 --------------------------
class ResNet18Small(nn.Module):
    """专门为小尺寸图像设计的ResNet18变体"""

    def __init__(self, num_classes=100, dropout_rate=0.2):
        super(ResNet18Small, self).__init__()

        # 加载预训练的ResNet18（在ImageNet上预训练）
        self.base_model = resnet18(weights='IMAGENET1K_V1')

        # 修改第一层卷积适配小尺寸图像
        self.base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 移除最大池化层（避免小图像特征丢失）
        self.base_model.maxpool = nn.Identity()

        # 添加dropout层增强泛化
        self.dropout = nn.Dropout(dropout_rate)

        # 修改全连接层适配类别数
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base_model(x)


def create_model(num_classes, use_pretrained=True):
    """创建模型"""
    if use_pretrained:
        print("使用ImageNet预训练的ResNet18作为基础")
        model = ResNet18Small(num_classes=num_classes)
    else:
        print("从头训练ResNet18")
        model = ResNet18Small(num_classes=num_classes)
        # 重新初始化所有层
        model._initialize_weights()

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    return model


# -------------------------- 4. 训练工具函数 --------------------------
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -------------------------- 5. 训练和验证函数 --------------------------
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args, scaler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_time = 0.0
    data_time = 0.0

    start_time = time.time()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Epoch {epoch + 1}/{args.epochs} [Train]')

    for batch_idx, (inputs, targets) in pbar:
        data_time = time.time() - start_time

        inputs, targets = inputs.to(device), targets.to(device)

        # 应用Mixup数据增强（概率0.5）
        use_mixup = np.random.random() < 0.5
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)

        # 前向传播（混合精度训练）
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

            # 梯度累积
            loss = loss / args.grad_accum_steps

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积：每grad_accum_steps步更新一次
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # 统计指标
        running_loss += loss.item() * args.grad_accum_steps * inputs.size(0)
        _, predicted = outputs.max(1)

        if use_mixup:
            # Mixup的准确率计算较复杂，这里简化处理
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() +
                        (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 更新进度条
        batch_time = time.time() - start_time
        start_time = time.time()

        acc = 100. * correct / total
        avg_loss = running_loss / total

        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total

    # 学习率调度（每个epoch）
    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, epoch, args):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 用于计算各类别准确率
    num_classes = model.base_model.fc[-1].out_features
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                desc=f'Epoch {epoch + 1}/{args.epochs} [Val]')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 统计各类别准确率
            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1

            # 更新进度条
            acc = 100. * correct / total
            avg_loss = running_loss / total
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100. * correct / total

    # 计算平均类别准确率
    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc.append(100. * class_correct[i] / class_total[i])

    avg_class_acc = np.mean(class_acc) if class_acc else 0

    return epoch_loss, epoch_acc, avg_class_acc


# -------------------------- 6. 主训练函数 --------------------------
def main():
    args = parse_args()
    print(f"训练配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 创建输出目录
    os.makedirs('process_cifar100/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 获取数据集
    print(f"\n加载{args.dataset}数据集...")
    train_dataset, test_dataset, num_classes, train_transform, test_transform = get_datasets_and_transforms(args)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, pin_memory_device=str(device)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, pin_memory_device=str(device)
    )

    print(f"训练集: {len(train_dataset)}样本, 测试集: {len(test_dataset)}样本")
    print(f"类别数: {num_classes}")

    # 创建模型
    print("\n创建模型...")
    model = create_model(num_classes, use_pretrained=True).to(device)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # 学习率调度器
    # 1. 预热阶段
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.warmup_epochs * len(train_loader)
        )

    # 2. 余弦退火
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - max(args.warmup_epochs, 1)) * len(train_loader),
        eta_min=1e-6
    )

    # 组合调度器
    if warmup_scheduler:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs * len(train_loader)]
        )
    else:
        scheduler = cosine_scheduler

    # 3. 当验证损失不再下降时降低学习率
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 早停机制
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'val_class_acc': []}

    if args.resume and os.path.exists(args.resume):
        print(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        train_history = checkpoint.get('train_history', train_history)
        print(f"恢复训练: epoch {start_epoch}, 最佳准确率: {best_acc:.2f}%")

    # 仅评估模式
    if args.eval_only:
        print("\n仅评估模式...")
        val_loss, val_acc, val_class_acc = validate(
            model, test_loader, criterion, device, 0, args
        )
        print(f"验证结果: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%, 平均类别准确率={val_class_acc:.2f}%")
        return

    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args, scaler
        )

        # 验证
        val_loss, val_acc, val_class_acc = validate(
            model, test_loader, criterion, device, epoch, args
        )

        # 更新Plateau调度器
        plateau_scheduler.step(val_loss)

        # 记录历史
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['val_class_acc'].append(val_class_acc)

        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args)
            }, f'process_cifar100/checkpoints/resnet18_{args.dataset}_best.pth')

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args)
            }, f'process_cifar100/checkpoints/resnet18_{args.dataset}_epoch{epoch + 1}.pth')

        # 打印epoch结果
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{args.epochs} 结果:")
        print(f"  时间: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.2f}% | 平均类别准确率: {val_class_acc:.2f}%")
        print(f"  最佳验证准确率: {best_acc:.2f}%")

        # 早停检查
        if early_stopping(val_loss):
            print(f"\n早停触发于epoch {epoch + 1}")
            break

    # 训练完成
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'train_history': train_history,
        'num_classes': num_classes,
        'args': vars(args)
    }, f'process_cifar100/checkpoints/resnet18_{args.dataset}_final.pth')

    # 绘制训练曲线
    plot_training_curves(train_history, args.dataset)

    # 最终评估
    print("\n最终评估...")
    final_val_loss, final_val_acc, final_val_class_acc = validate(
        model, test_loader, criterion, device, args.epochs, args
    )
    print(
        f"最终验证结果: 损失={final_val_loss:.4f}, 准确率={final_val_acc:.2f}%, 平均类别准确率={final_val_class_acc:.2f}%")


def plot_training_curves(history, dataset_name):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title(f'{dataset_name} - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['acc'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red')
    axes[0, 1].set_title(f'{dataset_name} - 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 类别准确率曲线
    axes[0, 2].plot(history['val_class_acc'], label='验证平均类别准确率', color='green')
    axes[0, 2].set_title(f'{dataset_name} - 类别准确率曲线')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('平均类别准确率 (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 训练/验证准确率对比
    axes[1, 0].plot(history['acc'], label='训练', color='blue', alpha=0.7)
    axes[1, 0].plot(history['val_acc'], label='验证', color='red', alpha=0.7)
    axes[1, 0].fill_between(range(len(history['acc'])),
                            history['acc'], history['val_acc'],
                            color='gray', alpha=0.2)
    axes[1, 0].set_title(f'{dataset_name} - 训练/验证准确率对比')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合分析（训练-验证差距）
    gap = np.array(history['acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, label='训练-验证差距', color='purple')
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_title(f'{dataset_name} - 过拟合分析')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率差距 (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 学习率曲线（模拟）
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, f'训练曲线已保存\n最佳验证准确率: {max(history["val_acc"]):.2f}%',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1, 2].transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'logs/{dataset_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()