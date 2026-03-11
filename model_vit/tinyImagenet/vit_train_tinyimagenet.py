"""
vit_train_tinyimagenet.py
Tiny-ImageNet 上的 ViT 微调脚本（64×64 分辨率适配）
用法：python vit_train_tinyimagenet.py --data_root model_resnet18/data/tiny-imagenet/data --epochs 5 --batch_size 256
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from PIL import Image
import io

# 导入 parquet 处理库
try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("请先安装 pyarrow: pip install pyarrow pandas")
    raise

# 导入 timm
try:
    import timm
except ImportError:
    print("请先安装 timm: pip install timm")
    raise

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Tiny-ImageNet ViT 微调脚本（64×64 分辨率）')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/tiny-imagenet/data',
                        help='Tiny-ImageNet 数据根目录（包含 .parquet 文件）')
    parser.add_argument('--train_file', type=str,
                        default='train-00000-of-00001-1359597a978bc4fa.parquet',
                        help='训练集 parquet 文件名')
    parser.add_argument('--val_file', type=str,
                        default='valid-00000-of-00001-70d52db3c749a935.parquet',
                        help='验证集 parquet 文件名')

    # 模型选择
    parser.add_argument('--model_name', type=str,
                        default='vit_base_patch16_224',
                        choices=['vit_base_patch16_224',
                                 'vit_small_patch16_224',
                                 'vit_tiny_patch16_224',
                                 'deit_tiny_patch16_224',
                                 'vit_base_patch16_224.augreg_in21k'],
                        help='ViT 模型名称')

    parser.add_argument('--epochs', type=int, default=5,
                        help='微调轮数（Tiny-ImageNet 收敛快，5轮即可）')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='学习率预热轮数')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--save_dir', type=str, default='checkpoints_vit_tinyimagenet',
                        help='模型保存目录')
    parser.add_argument('--unfreeze_layers', type=int, default=-1,
                        help='解冻层数（-1=全部，0=只训练头，N=最后N层）')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Stochastic Depth drop path rate')
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


# -------------------------- 2. Tiny-ImageNet Parquet 数据集类 --------------------------
class TinyImageNetParquetDataset(Dataset):
    """
    从 Parquet 文件加载 Tiny-ImageNet 数据集
    Tiny-ImageNet: 64×64 分辨率，200 类
    """

    def __init__(self, parquet_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        print(f"加载 Tiny-ImageNet {'训练集' if is_train else '验证集'}: {parquet_path}")

        # 读取 parquet 文件
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

        self.table = pq.read_table(parquet_path)
        self.df = self.table.to_pandas()

        # 显示列信息
        print(f"  Parquet 列: {list(self.df.columns)}")
        print(f"  样本数: {len(self.df)}")

        # 解析标签
        if 'label' in self.df.columns:
            self.labels = self.df['label'].tolist()
        else:
            raise ValueError("Parquet 文件缺少 'label' 列")

        # 解析图像数据
        if 'image' in self.df.columns:
            self.image_data = self.df['image'].tolist()
        else:
            raise ValueError("Parquet 文件缺少 'image' 列")

        # 确定类别数
        self.num_classes = len(set(self.labels))
        print(f"  类别数: {self.num_classes}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图像字节数据
        img_bytes = self.image_data[idx]

        # 如果是字典格式（某些 parquet 存储方式），提取 bytes
        if isinstance(img_bytes, dict):
            img_bytes = img_bytes.get('bytes', img_bytes)

        # 转换为 PIL Image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 确保尺寸为 64×64
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------- 3. 数据增强配置 --------------------------
def get_tinyimagenet_transforms():
    """
    获取 Tiny-ImageNet 数据增强
    Tiny-ImageNet: 64×64 -> 插值到 224×224 输入 ViT
    """
    # ImageNet 标准归一化（预训练模型使用）
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 训练集：强增强
    train_transform = transforms.Compose([
        transforms.Resize(224),  # 64×64 -> 224×224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # 验证集：简单插值
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print(f"使用 ImageNet 归一化: Mean={IMAGENET_MEAN}, Std={IMAGENET_STD}")

    return train_transform, val_transform


# -------------------------- 4. 模型定义 --------------------------
class ViTTinyImageNet(nn.Module):
    """
    ViT for Tiny-ImageNet（200类，64×64输入）
    策略：64×64 -> 插值到 224×224 -> ViT
    """

    def __init__(self, model_name='vit_base_patch16_224', num_classes=200,
                 pretrained=True, drop_path_rate=0.1, unfreeze_layers=-1):
        super(ViTTinyImageNet, self).__init__()

        self.num_classes = num_classes

        print(f"\n{'=' * 60}")
        print(f"构建 ViT Tiny-ImageNet 模型: {model_name}")
        print(f"输入: 64×64 -> 插值到 224×224")
        print(f"类别数: {num_classes}")
        print(f"{'=' * 60}")

        # 加载预训练模型
        if pretrained:
            print(f"📥 加载预训练权重...")
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
            print(f"✅ 成功加载预训练权重")
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
            print("⚠️ 未加载预训练权重")

        self.num_features = self.backbone.num_features
        print(f"特征维度: {self.num_features}")

        # 新的分类头（200类）
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.num_features, num_classes)
        )

        # 初始化
        self._initialize_head()
        self._setup_layer_freezing(unfreeze_layers)
        self._count_parameters()

    def _initialize_head(self):
        """初始化分类头"""
        nn.init.normal_(self.head[1].weight, std=0.02)
        if self.head[1].bias is not None:
            nn.init.constant_(self.head[1].bias, 0)
        print("✅ 分类头初始化完成")

    def _setup_layer_freezing(self, unfreeze_layers):
        """设置层冻结策略"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("🔓 解冻所有层（全量微调）")
        elif unfreeze_layers == 0:
            for param in self.head.parameters():
                param.requires_grad = True
            print("🔒 只训练分类头")
        else:
            if hasattr(self.backbone, 'blocks'):
                total_blocks = len(self.backbone.blocks)
                freeze_until = total_blocks - unfreeze_layers
                for i, block in enumerate(self.backbone.blocks):
                    if i >= freeze_until:
                        for param in block.parameters():
                            param.requires_grad = True
                print(f"🔓 解冻最后 {unfreeze_layers}/{total_blocks} 个 Transformer 块")

            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True

    def _count_parameters(self):
        """统计参数"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n📊 参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    def forward(self, x):
        # 输入 x: [B, 3, 224, 224]（已经从 64×64 插值）
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def create_model(model_name='vit_base_patch16_224', num_classes=200,
                 pretrained=True, drop_path_rate=0.1, unfreeze_layers=-1):
    """创建 ViT Tiny-ImageNet 模型"""
    model = ViTTinyImageNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        unfreeze_layers=unfreeze_layers
    )
    return model


# -------------------------- 5. 训练工具函数 --------------------------
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
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

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        avg_loss = running_loss / total
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{acc:.2f}%',
            'LR': f'{current_lr:.2e}'
        })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total

    if scheduler is not None:
        scheduler.step()

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, args):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    num_classes = model.num_classes
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

            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1

            acc = 100. * correct / total
            avg_loss = running_loss / total
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total

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
    train_transform, val_transform = get_tinyimagenet_transforms()

    # 构建完整路径
    train_path = os.path.join(args.data_root, args.train_file)
    val_path = os.path.join(args.data_root, args.val_file)

    # 加载数据集
    print(f"\n加载 Tiny-ImageNet 数据集...")
    train_dataset = TinyImageNetParquetDataset(
        parquet_path=train_path,
        transform=train_transform,
        is_train=True
    )
    val_dataset = TinyImageNetParquetDataset(
        parquet_path=val_path,
        transform=val_transform,
        is_train=False
    )

    num_classes = train_dataset.num_classes
    print(f"\n类别数: {num_classes}")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

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
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=True,
        drop_path_rate=args.drop_path_rate,
        unfreeze_layers=args.unfreeze_layers
    ).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    param_groups = [
        {'params': model.head.parameters(), 'lr': args.lr * 10},
    ]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr})

    optimizer = optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )

    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = main_scheduler

    # 早停
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

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
    print(f"\n开始微调 Tiny-ImageNet（共 {args.epochs} 轮）...")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args
        )

        val_loss, val_acc, val_class_acc = validate(
            model, val_loader, criterion, device, epoch, args
        )

        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['val_class_acc'].append(val_class_acc)

        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f'{args.model_name}_tinyimagenet_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args),
                'model_name': args.model_name,
            }, save_path)
            print(f"  💾 保存最佳模型: {save_path}")

        # 定期保存
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.save_dir, f'{args.model_name}_tinyimagenet_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args),
                'model_name': args.model_name,
            }, save_path)

        # 打印结果
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{args.epochs} 结果:")
        print(f"  时间: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"  训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.2f}% | 平均类别准确率: {val_class_acc:.2f}%")
        print(f"  最佳验证准确率: {best_acc:.2f}%")

        if early_stopping(val_loss):
            print(f"\n早停触发于 epoch {epoch + 1}")
            break

    # 训练完成
    print(f"\n{'=' * 60}")
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print(f"{'=' * 60}")

    # 保存最终模型
    final_path = os.path.join(args.save_dir, f'{args.model_name}_tinyimagenet_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'train_history': train_history,
        'num_classes': num_classes,
        'args': vars(args),
        'model_name': args.model_name,
    }, final_path)
    print(f"💾 保存最终模型: {final_path}")

    # 绘制曲线
    plot_training_curves(train_history, args.model_name, args.epochs)

    # 最终评估
    print("\n最终评估...")
    final_val_loss, final_val_acc, final_val_class_acc = validate(
        model, val_loader, criterion, device, args.epochs, args
    )
    print(
        f"最终验证结果: 损失={final_val_loss:.4f}, 准确率={final_val_acc:.2f}%, 平均类别准确率={final_val_class_acc:.2f}%")


def plot_training_curves(history, model_name, epochs):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', color='blue', marker='o')
    axes[0, 0].plot(history['val_loss'], label='验证损失', color='red', marker='s')
    axes[0, 0].set_title(f'{model_name} - Tiny-ImageNet 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['acc'], label='训练准确率', color='blue', marker='o')
    axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red', marker='s')
    axes[0, 1].set_title(f'{model_name} - Tiny-ImageNet 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 类别准确率曲线
    axes[1, 0].plot(history['val_class_acc'], label='验证平均类别准确率',
                    color='green', marker='^')
    axes[1, 0].set_title(f'{model_name} - Tiny-ImageNet 类别准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('平均类别准确率 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合分析
    gap = np.array(history['acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, label='训练-验证差距', color='purple', marker='d')
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_title(f'{model_name} - Tiny-ImageNet 过拟合分析')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率差距 (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'logs/{model_name}_tinyimagenet_{epochs}epochs_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 训练曲线保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()