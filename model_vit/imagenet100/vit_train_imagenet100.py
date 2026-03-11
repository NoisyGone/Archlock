"""
vit_train_imagenet100.py
ImageNet-100 上的 ViT 微调脚本
用法：python vit_train_imagenet100.py --data_root ../data/ImageNet-100/imagenet-100-folder --epochs 30 --batch_size 128
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from functools import partial

# 导入 timm 库中的 ViT 模型
try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    print("请先安装 timm: pip install timm")
    raise

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet-100 ViT 微调脚本')
    parser.add_argument('--data_root', type=str,
                        default='../model_resnet18/data/ImageNet-100/imagenet-100-folder',
                        help='ImageNet-100 数据根目录（包含 train 和 val 文件夹）')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        choices=['vit_base_patch16_224', 'vit_base_patch16_224.augreg_in21k',
                                 'vit_base_patch16_224.augreg_in21k_ft_in1k',
                                 'vit_base_patch32_224', 'vit_small_patch16_224',
                                 'vit_tiny_patch16_224', 'deit_base_patch16_224'],
                        help='ViT 模型名称（timm 库支持）')
    parser.add_argument('--epochs', type=int, default=3,
                        help='微调轮数（预训练模型只需少量epoch）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小（ViT显存占用大，建议比ResNet小）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率（微调使用较小学习率）')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减（ViT通常使用较大weight decay）')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--save_dir', type=str, default='checkpoints_vit',
                        help='模型保存目录')
    parser.add_argument('--unfreeze_layers', type=int, default=6,
                        help='解冻的transformer层数（从最后一层往前数，-1表示全部解冻）')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Stochastic Depth drop path rate（正则化）')
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
def get_vit_transforms(model_name='vit_base_patch16_224'):
    """获取 ViT 标准数据增强（使用 timm 推荐配置）"""

    # 使用 timm 创建标准变换
    config = resolve_data_config({}, model=model_name, verbose=False)

    # ImageNet 官方均值和标准差
    mean = config['mean'] if 'mean' in config else [0.485, 0.456, 0.406]
    std = config['std'] if 'std' in config else [0.229, 0.224, 0.225]
    input_size = config['input_size'][1] if 'input_size' in config else 224

    print(f"模型输入尺寸: {input_size}x{input_size}, Mean: {mean}, Std: {std}")

    # 训练集：ViT 标准数据增强（RandAugment + Mixup/CutMix 准备）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        # 使用 timm 的 RandAugment 或 torchvision 的 AutoAugment
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # 随机擦除（Random Erasing）对 ViT 有效
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
    ])

    # 验证集：中心裁剪
    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),  # 保持比例
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


# -------------------------- 3. 模型定义 --------------------------
class ViTImageNet(nn.Module):
    """ViT for ImageNet-100（基于 timm 预训练权重）"""

    def __init__(self, model_name='vit_base_patch16_224', num_classes=100,
                 pretrained=True, drop_path_rate=0.1, unfreeze_layers=6):
        super(ViTImageNet, self).__init__()

        print(f"\n正在加载 {model_name}...")

        # 加载预训练模型
        if pretrained:
            self.base_model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # 先不加载分类头，后面自己加
                drop_path_rate=drop_path_rate,  # Stochastic Depth
            )
            print(f"✅ 成功加载预训练权重")
        else:
            self.base_model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
            print("⚠️ 未加载预训练权重，从头训练")

        # 获取特征维度
        self.num_features = self.base_model.num_features
        print(f"特征维度: {self.num_features}")

        # 新的分类头（Dropout + Linear）
        self.head = nn.Sequential(
            nn.Dropout(0.0),  # 可调整
            nn.Linear(self.num_features, num_classes)
        )

        # 初始化新层
        self._initialize_head()

        # 冻结/解冻层策略
        self._setup_layer_freezing(unfreeze_layers)

        # 计算参数数量
        self._count_parameters()

    def _initialize_head(self):
        """初始化分类头"""
        nn.init.normal_(self.head[1].weight, std=0.02)
        if self.head[1].bias is not None:
            nn.init.constant_(self.head[1].bias, 0)
        print("✅ 分类头初始化完成")

    def _setup_layer_freezing(self, unfreeze_layers):
        """
        设置层冻结策略：
        - unfreeze_layers=-1: 解冻所有层（全量微调）
        - unfreeze_layers=0: 只训练分类头（线性探测）
        - unfreeze_layers=N: 解冻最后 N 个 transformer 块 + 分类头
        """
        # 首先冻结所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        if unfreeze_layers == -1:
            # 解冻所有层
            for param in self.base_model.parameters():
                param.requires_grad = True
            print("🔓 解冻所有层（全量微调）")

        elif unfreeze_layers == 0:
            # 只训练分类头
            for param in self.head.parameters():
                param.requires_grad = True
            print("🔒 只训练分类头（线性探测）")

        else:
            # 解冻最后 N 个 transformer 块
            # ViT 结构: [patch_embed] -> [blocks] -> [norm] -> [head]
            if hasattr(self.base_model, 'blocks'):
                total_blocks = len(self.base_model.blocks)
                freeze_until = total_blocks - unfreeze_layers

                for i, block in enumerate(self.base_model.blocks):
                    if i >= freeze_until:
                        for param in block.parameters():
                            param.requires_grad = True

                print(f"🔓 解冻最后 {unfreeze_layers}/{total_blocks} 个 Transformer 块")

            # 解冻 norm 层
            if hasattr(self.base_model, 'norm'):
                for param in self.base_model.norm.parameters():
                    param.requires_grad = True

            # 解冻分类头
            for param in self.head.parameters():
                param.requires_grad = True

    def _count_parameters(self):
        """计算可训练参数"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n📊 参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
        print(f"   冻结的: {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")

    def forward(self, x):
        # 提取特征
        features = self.base_model(x)  # [B, num_features]
        # 分类
        logits = self.head(features)  # [B, num_classes]
        return logits

    def get_attention_maps(self, x):
        """获取注意力图（用于可视化）"""
        # 需要模型支持返回 attention
        self.base_model.forward_features(x)
        # 具体实现取决于 timm 版本
        pass


def create_model(model_name='vit_base_patch16_224', num_classes=100,
                 pretrained=True, drop_path_rate=0.1, unfreeze_layers=6):
    """创建 ViT 模型"""
    model = ViTImageNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        unfreeze_layers=unfreeze_layers
    )
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

        # 梯度裁剪（对 ViT 稳定训练很重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计指标
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
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
    num_classes = model.head[-1].out_features
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
    train_transform, val_transform = get_vit_transforms(args.model_name)

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
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=True,
        drop_path_rate=args.drop_path_rate,
        unfreeze_layers=args.unfreeze_layers
    ).to(device)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器：使用 AdamW（ViT 的标准选择）
    # 分层学习率：主干使用较小 lr，head 使用较大 lr
    param_groups = [
        {'params': model.head.parameters(), 'lr': args.lr * 10},  # 分类头学习率更大
    ]

    # 添加可训练的主干参数
    backbone_params = [p for p in model.base_model.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr})

    optimizer = optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度：余弦退火 + 预热
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

    # 早停机制
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

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
    print(f"\n开始微调...")
    print(f"策略: 解冻最后 {args.unfreeze_layers} 层 + 分类头")

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
                'model_name': args.model_name,
                'class_to_idx': train_dataset.class_to_idx,
            }, os.path.join(args.save_dir, f'{args.model_name}_imagenet100_best.pth'))

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'num_classes': num_classes,
                'args': vars(args),
                'model_name': args.model_name,
                'class_to_idx': train_dataset.class_to_idx,
            }, os.path.join(args.save_dir, f'{args.model_name}_imagenet100_epoch{epoch + 1}.pth'))

        # 打印 epoch 结果
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{args.epochs} 结果:")
        print(f"  时间: {epoch_time:.1f}s | LR: {current_lr:.2e}")
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
        'model_name': args.model_name,
        'class_to_idx': train_dataset.class_to_idx,
    }, os.path.join(args.save_dir, f'{args.model_name}_imagenet100_final.pth'))

    # 绘制训练曲线
    plot_training_curves(train_history, args.model_name)

    # 最终评估
    print("\n最终评估...")
    final_val_loss, final_val_acc, final_val_class_acc = validate(
        model, val_loader, criterion, device, args.epochs, args
    )
    print(
        f"最终验证结果: 损失={final_val_loss:.4f}, 准确率={final_val_acc:.2f}%, 平均类别准确率={final_val_class_acc:.2f}%")


def plot_training_curves(history, model_name):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title(f'{model_name} - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['acc'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red')
    axes[0, 1].set_title(f'{model_name} - 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 类别准确率曲线
    axes[1, 0].plot(history['val_class_acc'], label='验证平均类别准确率', color='green')
    axes[1, 0].set_title(f'{model_name} - 类别准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('平均类别准确率 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合分析
    gap = np.array(history['acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, label='训练-验证差距', color='purple')
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_title(f'{model_name} - 过拟合分析')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率差距 (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'logs/{model_name}_imagenet100_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()