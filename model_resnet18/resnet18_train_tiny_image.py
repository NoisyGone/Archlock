"""
resnet18_tiny_imagenet.py
在Tiny-ImageNet上训练ResNet18
数据集路径示例：
  train: model_resnet18/data/tiny-imagenet/data/train-00000-of-00001-1359597a978bc4fa.parquet
  valid: model_resnet18/data/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet
"""

import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
from PIL import Image
import io


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='ResNet18在Tiny-ImageNet上的训练脚本')
    parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet/data',
                        help='Tiny-ImageNet数据目录')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='使用ImageNet预训练权重')
    parser.add_argument('--output_dir', type=str, default='tiny_imagenet_results',
                        help='输出目录')
    return parser.parse_args()


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 设置为True可以加速，但可能影响可复现性


set_seed(42)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# -------------------------- 2. Tiny-ImageNet数据集类 --------------------------
class TinyImageNetDataset(Dataset):
    """Tiny-ImageNet数据集加载器（从Parquet文件加载）"""

    def __init__(self, parquet_path, transform=None, is_train=True):
        """
        初始化Tiny-ImageNet数据集

        Args:
            parquet_path: parquet文件路径
            transform: 数据增强变换
            is_train: 是否为训练集
        """
        self.parquet_path = parquet_path
        self.transform = transform
        self.is_train = is_train

        # 加载parquet文件
        print(f"加载Parquet文件: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")

        # 根据实际列名调整（Parquet文件的列名可能不同）
        # 通常包含: 'image' (字节流) 和 'label' 或 'class'
        self.image_column = None
        self.label_column = None

        # 尝试识别图像列
        possible_image_cols = ['image', 'data', 'img', 'bytes', 'Image']
        for col in possible_image_cols:
            if col in self.df.columns:
                self.image_column = col
                break

        # 尝试识别标签列
        possible_label_cols = ['label', 'class', 'labels', 'target', 'class_name']
        for col in possible_label_cols:
            if col in self.df.columns:
                self.label_column = col
                break

        if self.image_column is None:
            # 如果没有找到标准列名，使用第一列作为图像，第二列作为标签
            columns = list(self.df.columns)
            if len(columns) >= 2:
                self.image_column = columns[0]
                self.label_column = columns[1]
            else:
                raise ValueError("无法识别图像和标签列")

        print(f"使用图像列: {self.image_column}, 标签列: {self.label_column}")

        # 统计类别信息
        if self.label_column in self.df.columns:
            self.labels = self.df[self.label_column].values
            # 转换标签为整数（如果还不是）
            if not np.issubdtype(self.labels.dtype, np.integer):
                # 将字符串标签映射为整数
                unique_labels = np.unique(self.labels)
                self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                self.labels = np.array([self.label_to_idx[label] for label in self.labels])

            self.num_classes = len(np.unique(self.labels))
            print(f"数据集类别数: {self.num_classes}")
            print(f"标签范围: {self.labels.min()} 到 {self.labels.max()}")
        else:
            # 对于测试集可能没有标签
            self.labels = None
            self.num_classes = 200  # Tiny-ImageNet默认200类

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图像数据
        img_data = self.df.iloc[idx][self.image_column]

        # === 修复后的类型处理逻辑 ===
        if isinstance(img_data, bytes):
            # bytes类型：直接解码
            img = Image.open(io.BytesIO(img_data)).convert('RGB')

        elif isinstance(img_data, dict):
            # dict类型：提取bytes字段
            if 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
            else:
                # 打印调试信息
                print(f"警告：dict中没有'bytes'键，可用键：{list(img_data.keys())}")
                raise ValueError(f"无法处理的dict结构：{img_data}")

        elif isinstance(img_data, np.ndarray):
            # numpy数组：转换格式
            if img_data.dtype != np.uint8:
                img_data = img_data.astype(np.uint8)
            if img_data.shape[0] == 3:  # CHW -> HWC
                img_data = img_data.transpose(1, 2, 0)
            img = Image.fromarray(img_data, mode='RGB')

        elif isinstance(img_data, str) and os.path.exists(img_data):
            # 文件路径
            img = Image.open(img_data).convert('RGB')

        else:
            # 未知类型
            raise ValueError(f"无法处理的图像数据类型：{type(img_data)}, 值：{img_data}")

        # 应用变换
        if self.transform:
            img = self.transform(img)

        # 返回标签
        if self.labels is not None:
            return img, self.labels[idx]
        return img

# -------------------------- 3. 数据增强和加载 --------------------------
def get_tiny_imagenet_transforms():
    """获取Tiny-ImageNet的数据增强"""

    # Tiny-ImageNet的标准化参数（使用ImageNet的统计量）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 训练集数据增强（更强）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # 随机裁剪和缩放
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 验证集数据增强（较弱）
    val_transform = transforms.Compose([
        transforms.Resize(72),  # 先放大一点
        transforms.CenterCrop(64),  # 再中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


def get_tiny_imagenet_loaders(data_dir, batch_size=128, num_workers=8):
    """获取Tiny-ImageNet数据加载器"""

    # 构建文件路径
    train_parquet = os.path.join(data_dir, 'train-00000-of-00001-1359597a978bc4fa.parquet')
    val_parquet = os.path.join(data_dir, 'valid-00000-of-00001-70d52db3c749a935.parquet')

    # 检查文件是否存在
    if not os.path.exists(train_parquet):
        raise FileNotFoundError(f"训练集文件不存在: {train_parquet}")
    if not os.path.exists(val_parquet):
        raise FileNotFoundError(f"验证集文件不存在: {val_parquet}")

    # 获取数据增强
    train_transform, val_transform = get_tiny_imagenet_transforms()

    # 创建数据集
    train_dataset = TinyImageNetDataset(train_parquet, transform=train_transform, is_train=True)
    val_dataset = TinyImageNetDataset(val_parquet, transform=val_transform, is_train=False)

    # 确保两个数据集的类别数一致
    if train_dataset.num_classes != val_dataset.num_classes:
        print(f"警告: 训练集类别数({train_dataset.num_classes})与验证集类别数({val_dataset.num_classes})不一致")
        # 使用训练集的类别数
        num_classes = train_dataset.num_classes
    else:
        num_classes = train_dataset.num_classes

    print(f"Tiny-ImageNet数据集:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")
    print(f"  类别数: {num_classes}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时可以使用更大的批次
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes


# -------------------------- 4. 模型定义 --------------------------
class ResNet18TinyImageNet(nn.Module):
    """适配Tiny-ImageNet的ResNet18"""

    def __init__(self, num_classes=200, use_pretrained=True, dropout_rate=0.3):
        super(ResNet18TinyImageNet, self).__init__()

        # 加载预训练的ResNet18
        if use_pretrained:
            print("使用ImageNet预训练的ResNet18")
            self.base_model = resnet18(weights='IMAGENET1K_V1')
        else:
            print("从头训练ResNet18")
            self.base_model = resnet18(weights=None)

        # 对于64×64图像，保持第一层卷积为7×7，但调整stride和padding
        # 这样可以在保持预训练权重的情况下适配64×64输入
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,  # 保持7×7
            stride=2,  # 保持stride=2
            padding=3,  # 保持padding=3
            bias=False
        )

        # 如果使用预训练权重，复制第一层卷积的权重
        if use_pretrained:
            self.base_model.conv1.weight.data.copy_(original_conv1.weight.data)

        # 修改最大池化层的kernel_size和stride，适应64×64输入
        self.base_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 添加额外的正则化
        self.dropout = nn.Dropout2d(dropout_rate)

        # 修改全连接层
        in_features = self.base_model.fc.in_features

        # 使用更复杂的分类头
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 3),
            nn.Linear(512, num_classes)
        )

        # 初始化新添加层的权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化新添加层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)

        return x


# -------------------------- 5. 训练工具函数 --------------------------
class CosineAnnealingWarmupRestarts:
    """带预热的余弦退火重启调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress))

        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """保存检查点"""
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载检查点"""
    if os.path.isfile(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型状态
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # 返回其他信息
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)

        print(f"加载检查点成功: epoch {checkpoint.get('epoch', 0)}, "
              f"最佳准确率: {best_acc:.2f}%")

        return start_epoch, best_acc
    else:
        print(f"警告: 检查点文件不存在 {checkpoint_path}")
        return 0, 0.0


# -------------------------- 6. 训练和验证函数 --------------------------
def train_epoch(train_loader, model, criterion, optimizer, epoch, device, args):
    """训练一个epoch"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Epoch {epoch + 1}/{args.epochs} [Train]')

    for i, (images, target) in pbar:
        # 测量数据加载时间
        data_time.update(time.time() - end)

        # 移动数据到设备
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 前向传播
        output = model(images)
        loss = criterion(output, target)

        # 计算准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))

        # ✅ 修复：直接使用 .item() 而不是 [0]
        top1.update(acc1.item(), images.size(0))  # 原为 acc1[0]
        top5.update(acc5.item(), images.size(0))  # 原为 acc5[0]

        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 测量批次时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.3f}%',
            'Acc@5': f'{top5.avg:.3f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device, args):
    """验证模型"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='[Val]')

        for i, (images, target) in pbar:
            # 移动数据到设备
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 前向传播
            output = model(images)
            loss = criterion(output, target)

            # 计算准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))

            # ✅ 修复：直接使用 .item() 而不是 [0]
            top1.update(acc1.item(), images.size(0))  # 原为 acc1[0]
            top5.update(acc5.item(), images.size(0))  # 原为 acc5[0]
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # 测量批次时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.3f}%',
                'Acc@5': f'{top5.avg:.3f}%'
            })

    return losses.avg, top1.avg, top5.avg


class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -------------------------- 7. 主训练函数 --------------------------
def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    print("=" * 60)
    print("ResNet18在Tiny-ImageNet上的训练")
    print("=" * 60)

    # 打印配置
    print(f"配置参数:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 获取数据加载器
    print(f"\n加载数据集...")
    train_loader, val_loader, num_classes = get_tiny_imagenet_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    print(f"\n创建模型...")
    model = ResNet18TinyImageNet(
        num_classes=num_classes,
        use_pretrained=args.use_pretrained
    ).to(device)

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # 学习率调度器
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs
    )

    # 恢复训练
    start_epoch = 0
    best_acc1 = 0.0

    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(args.resume, model, optimizer)

    # 仅评估模式
    if args.eval_only:
        print("\n仅评估模式...")
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, device, args)
        print(f"验证结果: 损失={val_loss:.4f}, Top-1准确率={val_acc1:.2f}%, Top-5准确率={val_acc5:.2f}%")
        return

    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc1': [], 'train_acc5': [],
        'val_loss': [], 'val_acc1': [], 'val_acc5': [],
        'lr': []
    }

    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 训练一个epoch
        train_loss, train_acc1, train_acc5 = train_epoch(
            train_loader, model, criterion, optimizer, epoch, device, args
        )

        # 更新学习率
        current_lr = scheduler.step()

        # 验证
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, device, args)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc1'].append(train_acc1)
        history['train_acc5'].append(train_acc5)
        history['val_loss'].append(val_loss)
        history['val_acc1'].append(val_acc1)
        history['val_acc5'].append(val_acc5)
        history['lr'].append(current_lr)

        # 保存最佳模型
        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1 = val_acc1
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'history': history,
                'args': args
            }, os.path.join(args.output_dir, 'checkpoints', 'model_best.pth.tar'))

        # 定期保存检查点
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'history': history,
                'args': args
            }, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch{epoch + 1}.pth.tar'))

        # 打印epoch结果
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1}/{args.epochs} 完成 (时间: {epoch_time:.1f}s):")
        print(f"  训练 - 损失: {train_loss:.4f}, Top-1准确率: {train_acc1:.2f}%, Top-5准确率: {train_acc5:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f}, Top-1准确率: {val_acc1:.2f}%, Top-5准确率: {val_acc5:.2f}%")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  最佳Top-1准确率: {best_acc1:.2f}%")

        # 保存训练历史到文件
        history_df = pd.DataFrame({
            'epoch': list(range(1, epoch + 2)),
            'train_loss': history['train_loss'],
            'train_acc1': history['train_acc1'],
            'train_acc5': history['train_acc5'],
            'val_loss': history['val_loss'],
            'val_acc1': history['val_acc1'],
            'val_acc5': history['val_acc5'],
            'lr': history['lr']
        })
        history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)

        # 绘制训练曲线
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            plot_training_curves(history, args.output_dir)

    # 训练完成
    print(f"\n{'=' * 60}")
    print(f"训练完成!")
    print(f"最佳Top-1准确率: {best_acc1:.2f}%")
    print(f"模型和结果保存在: {args.output_dir}")
    print(f"{'=' * 60}")

    # 最终评估
    print("\n最终评估...")
    final_val_loss, final_val_acc1, final_val_acc5 = validate(val_loader, model, criterion, device, args)
    print(
        f"最终验证结果: 损失={final_val_loss:.4f}, Top-1准确率={final_val_acc1:.2f}%, Top-5准确率={final_val_acc5:.2f}%")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'resnet18_tiny_imagenet_final.pth'))


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练和验证损失
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-1准确率
    axes[0, 1].plot(epochs, history['train_acc1'], 'b-', label='训练Top-1准确率', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc1'], 'r-', label='验证Top-1准确率', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Top-1准确率 (%)')
    axes[0, 1].set_title('Top-1准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Top-5准确率
    axes[1, 0].plot(epochs, history['train_acc5'], 'b-', label='训练Top-5准确率', linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc5'], 'r-', label='验证Top-5准确率', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5准确率 (%)')
    axes[1, 0].set_title('Top-5准确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 学习率
    axes[1, 1].plot(epochs, history['lr'], 'g-', label='学习率', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('学习率')
    axes[1, 1].set_title('学习率变化')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


# -------------------------- 8. 测试和推理函数 --------------------------
def test_model(model_path, data_dir, batch_size=128):
    """测试训练好的模型"""
    print("测试模型...")

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']

    # 获取数据加载器
    train_loader, val_loader, num_classes = get_tiny_imagenet_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=4
    )

    # 创建模型
    model = ResNet18TinyImageNet(num_classes=num_classes, use_pretrained=False).to(device)

    # 加载权重
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 验证
    val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, device, args)

    print(f"\n测试结果:")
    print(f"  损失: {val_loss:.4f}")
    print(f"  Top-1准确率: {val_acc1:.2f}%")
    print(f"  Top-5准确率: {val_acc5:.2f}%")


    return val_acc1, val_acc5


# -------------------------- 9. 主程序入口 --------------------------
if __name__ == "__main__":
    main()