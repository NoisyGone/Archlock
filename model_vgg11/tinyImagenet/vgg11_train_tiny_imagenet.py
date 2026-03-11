"""
vgg11_train_tiny_imagenet.py
Tiny ImageNet 上的 VGG11 训练脚本
用法：python vgg11_train_tiny_imagenet.py --data_root model_resnet18/data/tiny-imagenet/data --epochs 100 --batch_size 128
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg11, VGG11_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from PIL import Image
import io

warnings.filterwarnings('ignore')

# 尝试导入 pyarrow 用于读取 parquet 文件
try:
    import pyarrow.parquet as pq
    import pyarrow as pa

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("⚠️ 警告: pyarrow 未安装，尝试使用 pandas 读取 parquet")
    try:
        import pandas as pd

        HAS_PANDAS = True
    except ImportError:
        HAS_PANDAS = False
        raise ImportError("需要安装 pyarrow 或 pandas 来读取 parquet 文件: pip install pyarrow pandas")


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Tiny ImageNet VGG11 训练脚本')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/tiny-imagenet/data',
                        help='Tiny ImageNet 数据根目录（包含 train 和 valid parquet 文件）')
    parser.add_argument('--train_file', type=str,
                        default='train-00000-of-00001-1359597a978bc4fa.parquet',
                        help='训练集 parquet 文件名')
    parser.add_argument('--valid_file', type=str,
                        default='valid-00000-of-00001-70d52db3c749a935.parquet',
                        help='验证集 parquet 文件名')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01,
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
    parser.add_argument('--save_dir', type=str, default='checkpoints_vgg11_tiny_imagenet',
                        help='模型保存目录')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='使用ImageNet-1K预训练权重')
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


# -------------------------- 2. Tiny ImageNet 数据集加载 --------------------------
class TinyImageNetDataset(Dataset):
    """Tiny ImageNet 数据集加载器（从 parquet 文件加载）"""

    def __init__(self, parquet_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        print(f"加载 Tiny ImageNet {'训练集' if is_train else '验证集'}: {parquet_path}")

        # 读取 parquet 文件
        if HAS_PYARROW:
            self.table = pq.read_table(parquet_path)
            self.df = self.table.to_pandas()
        else:
            self.df = pd.read_parquet(parquet_path)

        # 解析数据
        # Tiny ImageNet parquet 格式通常包含: image (binary), label (int)
        # 或者 image_path, image (binary), label

        # 检查列名
        print(f"  Parquet 列名: {list(self.df.columns)}")

        # 获取图像数据
        if 'image' in self.df.columns:
            # 图像是二进制格式
            self.images = self.df['image'].tolist()
            # 如果是字典格式（包含 bytes 和 path），提取 bytes
            if isinstance(self.images[0], dict):
                self.images = [img['bytes'] if isinstance(img, dict) else img for img in self.images]
        elif 'img' in self.df.columns:
            self.images = self.df['img'].tolist()
        else:
            raise ValueError(f"找不到图像列。可用列: {list(self.df.columns)}")

        # 获取标签
        if 'label' in self.df.columns:
            self.labels = self.df['label'].tolist()
        elif 'labels' in self.df.columns:
            self.labels = self.df['labels'].tolist()
        else:
            raise ValueError(f"找不到标签列。可用列: {list(self.df.columns)}")

        # 获取类别信息
        self.num_classes = len(set(self.labels))
        self.classes = sorted(list(set(self.labels)))

        print(f"  样本数: {len(self.images)}")
        print(f"  类别数: {self.num_classes}")
        print(f"  标签范围: {min(self.labels)} - {max(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图像二进制数据
        img_data = self.images[idx]

        # 处理不同类型的图像数据
        if isinstance(img_data, bytes):
            # 直接是 bytes
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        elif isinstance(img_data, str):
            # 可能是路径
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, np.ndarray):
            # 已经是数组
            img = Image.fromarray(img_data.astype(np.uint8)).convert('RGB')
        else:
            # 尝试转换
            img = Image.open(io.BytesIO(bytes(img_data))).convert('RGB')

        label = int(self.labels[idx])

        # 确保图像尺寸为 64x64
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)

        if self.transform:
            img = self.transform(img)

        return img, label


# 替代方案：如果 parquet 文件包含图像路径而不是二进制数据
class TinyImageNetPathDataset(Dataset):
    """Tiny ImageNet 数据集加载器（从 parquet 加载路径，从磁盘加载图像）"""

    def __init__(self, parquet_path, image_root=None, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        print(f"加载 Tiny ImageNet {'训练集' if is_train else '验证集'}: {parquet_path}")

        # 读取 parquet 文件
        if HAS_PYARROW:
            self.table = pq.read_table(parquet_path)
            self.df = self.table.to_pandas()
        else:
            self.df = pd.read_parquet(parquet_path)

        print(f"  Parquet 列名: {list(self.df.columns)}")

        # 获取图像路径
        if 'image_path' in self.df.columns:
            self.image_paths = self.df['image_path'].tolist()
        elif 'path' in self.df.columns:
            self.image_paths = self.df['path'].tolist()
        elif 'file_name' in self.df.columns:
            self.image_paths = self.df['file_name'].tolist()
        else:
            raise ValueError(f"找不到图像路径列。可用列: {list(self.df.columns)}")

        # 如果提供了 image_root，拼接完整路径
        self.image_root = image_root

        # 获取标签
        if 'label' in self.df.columns:
            self.labels = self.df['label'].tolist()
        elif 'labels' in self.df.columns:
            self.labels = self.df['labels'].tolist()
        else:
            raise ValueError(f"找不到标签列。可用列: {list(self.df.columns)}")

        self.num_classes = len(set(self.labels))
        self.classes = sorted(list(set(self.labels)))

        print(f"  样本数: {len(self.image_paths)}")
        print(f"  类别数: {self.num_classes}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 拼接完整路径
        if self.image_root:
            img_path = os.path.join(self.image_root, img_path)

        img = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])

        # 确保图像尺寸为 64x64
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------- 3. 数据增强配置 --------------------------
def get_tiny_imagenet_transforms():
    """获取 Tiny ImageNet 数据增强（64x64 图像）"""
    # Tiny ImageNet 通常使用 ImageNet 的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 训练集：标准数据增强（适配 64x64）
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 验证集：仅归一化
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


# -------------------------- 4. 模型定义（适配 64x64 输入） --------------------------
class VGG11TinyImageNet(nn.Module):
    """
    适配 Tiny ImageNet 的 VGG11（64x64 输入，200 类）
    修改点：
    1. 特征提取器适配 64x64 输入
    2. 调整分类器输入维度
    """

    def __init__(self, num_classes=200, dropout_rate=0.5, use_pretrained=False):
        super(VGG11TinyImageNet, self).__init__()

        # 构建适配 64x64 的 VGG11 特征提取器
        # 策略：减少 maxpool 次数，避免图像尺寸过快缩小
        # 64 -> 32 -> 16 -> 8 -> 4 -> 2（5次pool，最终2x2）

        self.features = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 8 -> 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 4 -> 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 最终特征维度: 512 * 2 * 2 = 2048
        self.feature_dim = 512 * 2 * 2

        # 分类器（比 ImageNet 版本小，比 CIFAR-100 版本大）
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

        # 如果使用预训练权重，加载并适配
        if use_pretrained:
            self._load_pretrained()

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

    def _load_pretrained(self):
        """加载 ImageNet-1K 预训练权重并适配"""
        print("加载 ImageNet-1K 预训练权重...")
        try:
            pretrained_vgg = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)

            # 复制特征提取器权重（前几层结构相同）
            pretrained_dict = pretrained_vgg.state_dict()
            model_dict = self.state_dict()

            # 筛选可以加载的权重（形状匹配的卷积层和BN层）
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('features.'):
                    # 特征提取器权重
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v

            # 更新权重
            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict, strict=False)

            print(f"  成功加载 {len(compatible_dict)} 个预训练权重")

        except Exception as e:
            print(f"  ⚠️ 预训练权重加载失败: {e}")
            print("  使用随机初始化")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(num_classes=200, dropout_rate=0.5, use_pretrained=False):
    """创建模型"""
    print(f"创建 VGG11 Tiny ImageNet 模型:")
    print(f"  类别数: {num_classes}")
    print(f"  输入尺寸: 64x64")
    print(f"  特征维度: 2048 (512x2x2)")
    print(f"  使用预训练: {use_pretrained}")

    model = VGG11TinyImageNet(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_pretrained=use_pretrained
    )

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 测试前向传播
    test_input = torch.randn(2, 3, 64, 64)
    test_output = model(test_input)
    print(f"  测试输入: {test_input.shape} -> 输出: {test_output.shape}")

    return model


# -------------------------- 5. 训练工具函数 --------------------------
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=15, min_delta=0.001):
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


def validate(model, val_loader, criterion, device, epoch, args):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 用于计算各类别准确率
    num_classes = 200
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
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

    epoch_loss = running_loss / len(val_loader.dataset)
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
    train_transform, val_transform = get_tiny_imagenet_transforms()

    # 构建完整路径
    train_path = os.path.join(args.data_root, args.train_file)
    valid_path = os.path.join(args.data_root, args.valid_file)

    # 检查文件存在性
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"验证文件不存在: {valid_path}")

    # 加载数据集
    print(f"\n加载 Tiny ImageNet 数据集...")

    # 尝试使用二进制数据集加载器
    try:
        train_dataset = TinyImageNetDataset(
            parquet_path=train_path,
            transform=train_transform,
            is_train=True
        )
        valid_dataset = TinyImageNetDataset(
            parquet_path=valid_path,
            transform=val_transform,
            is_train=False
        )
    except Exception as e:
        print(f"二进制加载失败，尝试路径加载: {e}")
        # 回退到路径加载
        train_dataset = TinyImageNetPathDataset(
            parquet_path=train_path,
            image_root=args.data_root,
            transform=train_transform,
            is_train=True
        )
        valid_dataset = TinyImageNetPathDataset(
            parquet_path=valid_path,
            image_root=args.data_root,
            transform=val_transform,
            is_train=False
        )

    num_classes = train_dataset.num_classes
    print(f"类别数: {num_classes}")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(valid_dataset)} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        use_pretrained=args.use_pretrained
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
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

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
        val_loss, val_acc, val_class_acc = validate(
            model, val_loader, criterion, device, 0, args
        )
        print(f"验证结果: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%, 平均类别准确率={val_class_acc:.2f}%")
        return

    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args
        )

        # 验证
        val_loss, val_acc, val_class_acc = validate(
            model, val_loader, criterion, device, epoch, args
        )

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
                'args': vars(args),
                'model_architecture': 'vgg11_tiny_imagenet',
            }, os.path.join(args.save_dir, 'vgg11_tiny_imagenet_best.pth'))

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args),
                'model_architecture': 'vgg11_tiny_imagenet',
            }, os.path.join(args.save_dir, f'vgg11_tiny_imagenet_epoch{epoch + 1}.pth'))

        # 打印 epoch 结果
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{args.epochs} 结果:")
        print(f"  时间: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.2f}% | 平均类别准确率: {val_class_acc:.2f}%")
        print(f"  最佳验证准确率: {best_acc:.2f}%")

        # 早停检查
        if early_stopping(val_loss):
            print(f"\n早停触发于 epoch {epoch + 1}")
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
        'args': vars(args),
        'model_architecture': 'vgg11_tiny_imagenet',
    }, os.path.join(args.save_dir, 'vgg11_tiny_imagenet_final.pth'))

    # 绘制训练曲线
    plot_training_curves(train_history)

    # 最终评估
    print("\n最终评估...")
    final_val_loss, final_val_acc, final_val_class_acc = validate(
        model, val_loader, criterion, device, args.epochs, args
    )
    print(
        f"最终验证结果: 损失={final_val_loss:.4f}, 准确率={final_val_acc:.2f}%, 平均类别准确率={final_val_class_acc:.2f}%")


def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title('Tiny ImageNet VGG11 - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['acc'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red')
    axes[0, 1].set_title('Tiny ImageNet VGG11 - 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 类别准确率曲线
    axes[1, 0].plot(history['val_class_acc'], label='验证平均类别准确率', color='green')
    axes[1, 0].set_title('Tiny ImageNet VGG11 - 类别准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('平均类别准确率 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合分析
    gap = np.array(history['acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, label='训练-验证差距', color='purple')
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_title('Tiny ImageNet VGG11 - 过拟合分析')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率差距 (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logs/tiny_imagenet_vgg11_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()