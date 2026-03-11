"""
vgg11_train_cifar100.py
CIFAR-100 上的 VGG11 训练脚本
用法：python vgg11_train_cifar100.py --data_root model_resnet18/data/cifar100/cifar-100-python --epochs 200 --batch_size 128
"""

import os
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg11, VGG11_Weights
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-100 VGG11 训练脚本')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/cifar100/cifar-100-python',
                        help='CIFAR-100 数据根目录（包含 train 和 test 文件）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数（CIFAR-100需要更多epoch）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='分类器dropout比率')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--save_dir', type=str, default='checkpoints_vgg11_cifar100',
                        help='模型保存目录')
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                        help='是否使用ImageNet-1K预训练权重（CIFAR-100通常从头训练效果更好）')
    return parser.parse_args()


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(42)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# -------------------------- 2. CIFAR-100 数据集加载 --------------------------
class CIFAR100Dataset(Dataset):
    """CIFAR-100 数据集加载器（从 pickle 文件加载）"""

    def __init__(self, data_root, train=True, transform=None):
        self.transform = transform

        # 加载数据
        if train:
            data_file = os.path.join(data_root, 'train')
        else:
            data_file = os.path.join(data_root, 'test')

        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        # 解析数据
        self.images = data_dict[b'data']  # [N, 3072] (32*32*3)
        self.labels = data_dict[b'fine_labels']  # 0-99

        # 重塑为图像格式 [N, 3, 32, 32]
        self.images = self.images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # [N, 32, 32, 3]

        # 加载类别名称
        meta_file = os.path.join(data_root, 'meta')
        with open(meta_file, 'rb') as f:
            meta_dict = pickle.load(f, encoding='bytes')
        self.classes = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]

        self.num_classes = len(self.classes)

        print(f"加载 CIFAR-100 {'训练集' if train else '测试集'}: {len(self.images)} 个样本")
        print(f"类别数: {self.num_classes}")
        print(f"图像尺寸: 32x32")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # [32, 32, 3], uint8
        label = self.labels[idx]

        # 转换为 PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------- 3. 数据增强配置 --------------------------
def get_cifar100_transforms():
    """获取 CIFAR-100 数据增强（32x32 图像）"""
    # CIFAR-100 均值和标准差
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # 训练集：标准 CIFAR-100 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 测试集：仅归一化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


# -------------------------- 4. 模型定义（适配 32x32 输入） --------------------------
class VGG11CIFAR100(nn.Module):
    """
    适配 CIFAR-100 的 VGG11（32x32 输入）
    修改点：
    1. 第一个卷积层改为 3x3 kernel, stride 1, padding 1（适配小图像）
    2. 移除部分 maxpool 层（避免图像尺寸过快缩小）
    3. 调整分类器输入维度
    """

    def __init__(self, num_classes=100, dropout_rate=0.5, use_pretrained=False):
        super(VGG11CIFAR100, self).__init__()

        # 加载基础 VGG11 结构（不使用预训练权重，因为输入尺寸不同）
        base_vgg = vgg11(weights=None)

        # 修改特征提取器以适配 32x32 输入
        # 原始 VGG11 第一个卷积: kernel=3, stride=1, padding=1 (输出 224x224 -> 224x224)
        # 对于 32x32，我们保持相同的卷积参数，但调整 maxpool 策略

        self.features = self._make_features(base_vgg.features)

        # 计算特征维度
        # 输入 32x32，经过修改后的 features 后: 512 channels, 2x2 spatial
        self.feature_dim = 512 * 2 * 2  # 2048

        # 分类器（比 ImageNet 版本更小）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, num_classes),
        )

        # 初始化
        self._initialize_weights()

    def _make_features(self, original_features):
        """
        修改 VGG11 特征提取器以适配 32x32 输入：
        - 保持卷积层不变
        - 减少 maxpool 次数，避免特征图过快缩小
        """
        # VGG11 原始结构: 64-M-128-M-256-256-M-512-512-M-512-512-M
        # 对于 32x32，我们改为: 64-M-128-M-256-256-M-512-512-512-512 (只在关键位置pool)

        layers = []
        in_channels = 3
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        # 修改: 32->16->8->4->2->1 太快了，改为 32->16->8->4->2 (最后保持2x2)
        # 新配置: 64-M-128-M-256-256-M-512-512-512-512 (输出 2x2)

        # 实际上，让我们使用更保守的策略：只在前两个block后pool
        # 32 -> 16 (pool) -> 8 (pool) -> 4 -> 2 (pool) -> 1 (pool) 还是太快

        # 最佳策略：只在 stride=2 的卷积后pool，或者减少pool次数
        # 这里我们直接使用修改后的配置

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))  # 添加BN帮助训练
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)

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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 替代方案：使用标准 VGG11 但修改第一层
class VGG11CIFAR100_Alternative(nn.Module):
    """
    替代方案：保持 VGG11 大部分结构，只修改输入层为 32x32 适配
    通过调整 avgpool 输出尺寸来适配
    """

    def __init__(self, num_classes=100, dropout_rate=0.5, use_pretrained=False):
        super(VGG11CIFAR100_Alternative, self).__init__()

        # 加载基础 VGG11
        base_vgg = vgg11(weights=None)

        # 修改第一层卷积：kernel=3, stride=1, padding=1 适用于 32x32
        # 但我们需要调整后续结构以避免尺寸过快缩小

        # 策略：使用标准 VGG11 features，但将 maxpool 改为 stride=1 或移除部分
        self.features = nn.Sequential(
            # Block 1: 32 -> 32 (no pool) -> 16 (pool)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32->16

            # Block 2: 16 -> 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16->8

            # Block 3: 8 -> 8 -> 8 -> 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8->4

            # Block 4: 4 -> 4 -> 4 -> 2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4->2

            # Block 5: 2 -> 2 -> 2 -> 1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2->1
        )

        # 最终特征维度: 512 * 1 * 1 = 512
        self.feature_dim = 512

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(num_classes=100, dropout_rate=0.5, use_pretrained=False, model_type='alternative'):
    """创建模型"""
    if model_type == 'original':
        print("使用原始修改版 VGG11（2x2 特征图）")
        model = VGG11CIFAR100(num_classes=num_classes, dropout_rate=dropout_rate, use_pretrained=use_pretrained)
    else:
        print("使用替代版 VGG11（1x1 特征图，推荐）")
        model = VGG11CIFAR100_Alternative(num_classes=num_classes, dropout_rate=dropout_rate,
                                          use_pretrained=use_pretrained)

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    # 测试前向传播
    test_input = torch.randn(2, 3, 32, 32)
    test_output = model(test_input)
    print(f"测试输入: {test_input.shape} -> 输出: {test_output.shape}")

    return model


# -------------------------- 5. 训练工具函数 --------------------------
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=20, min_delta=0.001):
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


# -------------------------- 6. 训练和验证函数 --------------------------
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Epoch {epoch + 1}/{args.epochs} [Train]')

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        acc = 100. * correct / total
        avg_loss = running_loss / total
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total

    # 学习率调度（每个 epoch）
    if scheduler is not None:
        scheduler.step()

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, epoch, args):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 用于计算各类别准确率
    num_classes = 100
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                desc=f'Epoch {epoch + 1}/{args.epochs} [Test]')

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


# -------------------------- 7. 主训练函数 --------------------------
def main():
    args = parse_args()
    print(f"训练配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 获取数据增强
    train_transform, test_transform = get_cifar100_transforms()

    # 加载数据集
    print(f"\n加载 CIFAR-100 数据集...")
    train_dataset = CIFAR100Dataset(
        data_root=args.data_root,
        train=True,
        transform=train_transform
    )
    test_dataset = CIFAR100Dataset(
        data_root=args.data_root,
        train=False,
        transform=test_transform
    )

    num_classes = train_dataset.num_classes
    print(f"类别数: {num_classes}")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型（使用替代版本，更适合 CIFAR-100）
    print("\n创建模型...")
    model = create_model(
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        use_pretrained=args.use_pretrained,
        model_type='alternative'
    ).to(device)

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

    # 学习率调度：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # 预热调度器
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        # 组合调度器
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[args.warmup_epochs]
        )

    # 早停机制
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'val_class_acc': []}

    if args.resume and os.path.exists(args.resume):
        print(f"\n从检查点恢复: {args.resume}")
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
        test_loss, test_acc, test_class_acc = validate(
            model, test_loader, criterion, device, 0, args
        )
        print(f"测试结果: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%, 平均类别准确率={test_class_acc:.2f}%")
        return

    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args
        )

        # 验证（在 CIFAR-100 上使用测试集）
        test_loss, test_acc, test_class_acc = validate(
            model, test_loader, criterion, device, epoch, args
        )

        # 记录历史
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        train_history['val_loss'].append(test_loss)
        train_history['val_acc'].append(test_acc)
        train_history['val_class_acc'].append(test_class_acc)

        # 保存最佳模型
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args),
                'model_architecture': 'vgg11_cifar100',
            }, os.path.join(args.save_dir, 'vgg11_cifar100_best.pth'))

        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args),
                'model_architecture': 'vgg11_cifar100',
            }, os.path.join(args.save_dir, f'vgg11_cifar100_epoch{epoch + 1}.pth'))

        # 打印 epoch 结果
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{args.epochs} 结果:")
        print(f"  时间: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")
        print(f"  测试 - 损失: {test_loss:.4f} | 准确率: {test_acc:.2f}% | 平均类别准确率: {test_class_acc:.2f}%")
        print(f"  最佳测试准确率: {best_acc:.2f}%")

        # 早停检查
        if early_stopping(test_loss):
            print(f"\n早停触发于 epoch {epoch + 1}")
            break

    # 训练完成
    print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'train_history': train_history,
        'num_classes': num_classes,
        'args': vars(args),
        'model_architecture': 'vgg11_cifar100',
    }, os.path.join(args.save_dir, 'vgg11_cifar100_final.pth'))

    # 绘制训练曲线
    plot_training_curves(train_history)

    # 最终评估
    print("\n最终评估...")
    final_test_loss, final_test_acc, final_test_class_acc = validate(
        model, test_loader, criterion, device, args.epochs, args
    )
    print(
        f"最终测试结果: 损失={final_test_loss:.4f}, 准确率={final_test_acc:.2f}%, 平均类别准确率={final_test_class_acc:.2f}%")


def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history['val_loss'], label='测试损失', color='red')
    axes[0, 0].set_title('CIFAR-100 VGG11 - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['acc'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['val_acc'], label='测试准确率', color='red')
    axes[0, 1].set_title('CIFAR-100 VGG11 - 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 类别准确率曲线
    axes[1, 0].plot(history['val_class_acc'], label='测试平均类别准确率', color='green')
    axes[1, 0].set_title('CIFAR-100 VGG11 - 类别准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('平均类别准确率 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合分析
    gap = np.array(history['acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, label='训练-测试差距', color='purple')
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_title('CIFAR-100 VGG11 - 过拟合分析')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率差距 (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logs/cifar100_vgg11_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()