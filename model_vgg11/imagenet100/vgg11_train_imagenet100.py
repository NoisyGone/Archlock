"""
vgg11_train_imagenet100.py
ImageNet-100 上的 VGG11 训练脚本
用法：python vgg11_train_imagenet100.py --data_root ../data/ImageNet-100/imagenet-100-folder --epochs 100 --batch_size 128
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg11, VGG11_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet-100 VGG11 训练脚本')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/ImageNet-100/imagenet-100-folder',
                        help='ImageNet-100 数据根目录（包含 train 和 val 文件夹）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小（VGG11比ResNet18内存占用大，建议减小）')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='初始学习率（VGG使用比ResNet更小的学习率）')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减（VGG通常使用更大的weight decay）')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='分类器dropout比率（VGG11默认0.5）')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--save_dir', type=str, default='checkpoints_vgg11',
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


# -------------------------- 2. 数据增强配置 --------------------------
def get_imagenet_transforms():
    """获取 ImageNet 标准数据增强"""
    # ImageNet 官方均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 训练集：标准 ImageNet 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 验证集：中心裁剪
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


# -------------------------- 3. 模型定义 --------------------------
class VGG11ImageNet(nn.Module):
    """标准 VGG11 for ImageNet（224×224）"""

    def __init__(self, num_classes=100, dropout_rate=0.5, use_pretrained=True):
        super(VGG11ImageNet, self).__init__()

        # 加载 ImageNet 预训练的 VGG11
        if use_pretrained:
            print("使用 ImageNet-1K 预训练的 VGG11 作为基础")
            self.features = vgg11(weights=VGG11_Weights.IMAGENET1K_V1).features
        else:
            print("从头训练 VGG11")
            self.features = vgg11(weights=None).features

        # VGG11 特征提取器输出 512 通道，7x7 空间尺寸
        # 展平后: 512 * 7 * 7 = 25088
        self.feature_dim = 512 * 7 * 7  # 25088

        # 构建分类器（与原始VGG11结构一致，但修改输出层）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )

        # 初始化新层（分类器的最后一层）
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化新添加的分类器层"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)  # [B, 512, 7, 7]
        x = torch.flatten(x, 1)  # [B, 25088]
        x = self.classifier(x)  # [B, num_classes]
        return x


def create_model(num_classes=100, use_pretrained=True, dropout_rate=0.5):
    """创建模型"""
    model = VGG11ImageNet(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_pretrained=use_pretrained
    )

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


# -------------------------- 5. 训练和验证函数 --------------------------
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
    num_classes = model.classifier[-1].out_features
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


# -------------------------- 6. 主训练函数 --------------------------
def main():
    args = parse_args()
    print(f"训练配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 获取数据增强
    train_transform, val_transform = get_imagenet_transforms()

    # 加载数据集
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"训练或验证目录不存在: {train_dir} 或 {val_dir}")

    print(f"\n加载 ImageNet-100 数据集...")
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"类别数: {num_classes}")
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
        num_classes=num_classes,
        use_pretrained=args.use_pretrained,
        dropout_rate=args.dropout_rate
    ).to(device)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器：VGG通常使用较小的学习率和较大的weight decay
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
                'class_to_idx': train_dataset.class_to_idx,
                'model_architecture': 'model_vgg11',
            }, os.path.join(args.save_dir, 'vgg11_imagenet100_best.pth'))

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
                'class_to_idx': train_dataset.class_to_idx,
                'model_architecture': 'model_vgg11',
            }, os.path.join(args.save_dir, f'vgg11_imagenet100_epoch{epoch + 1}.pth'))

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
        'class_to_idx': train_dataset.class_to_idx,
        'model_architecture': 'model_vgg11',
    }, os.path.join(args.save_dir, 'vgg11_imagenet100_final.pth'))

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
    axes[0, 0].set_title('ImageNet-100 VGG11 - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['acc'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red')
    axes[0, 1].set_title('ImageNet-100 VGG11 - 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 类别准确率曲线
    axes[1, 0].plot(history['val_class_acc'], label='验证平均类别准确率', color='green')
    axes[1, 0].set_title('ImageNet-100 VGG11 - 类别准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('平均类别准确率 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合分析
    gap = np.array(history['acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, label='训练-验证差距', color='purple')
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_title('ImageNet-100 VGG11 - 过拟合分析')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率差距 (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logs/imagenet100_vgg11_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()