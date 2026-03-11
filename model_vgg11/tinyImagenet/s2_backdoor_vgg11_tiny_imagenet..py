"""
s2_backdoor_vgg11_tiny_imagenet.py
Tiny ImageNet 上的 VGG11 后门模型构建与测试脚本
用法：python s2_backdoor_vgg11_tiny_imagenet.py --data_root model_resnet18/data/tiny-imagenet/data --model_path checkpoints_vgg11_tiny_imagenet/vgg11_tiny_imagenet_final.pth
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

warnings.filterwarnings('ignore')

# 尝试导入 pyarrow
try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    import pandas as pd


# ===================== 1. 核心工具函数 =====================
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


# ===================== 2. 权重加载日志函数 =====================
def load_model_weights_with_log(model, weights_path, strict=False):
    """
    加载模型权重并输出详细日志
    适配 VGG11 Tiny ImageNet 结构
    """
    if not os.path.exists(weights_path):
        print(f"❌ 权重文件不存在: {weights_path}")
        return 0, 0, [], []

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # 智能提取模型权重
    if 'model_state_dict' in ckpt:
        model_weights = ckpt['model_state_dict']
        print(f"\n📌 从 'model_state_dict' 提取权重")
    elif 'state_dict' in ckpt:
        model_weights = ckpt['state_dict']
        print(f"\n📌 从 'state_dict' 提取权重")
    elif 'net' in ckpt:
        model_weights = ckpt['net']
        print(f"\n📌 从 'net' 提取权重")
    else:
        model_weights = ckpt
        print(f"\n📌 直接使用整个checkpoint作为权重")

    print(f"📌 权重文件包含 {len(model_weights.keys())} 个权重项")

    # 获取模型的权重键
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    weight_keys = set(model_weights.keys())

    # 计算匹配情况
    matched_keys = model_keys & weight_keys
    missing_keys = model_keys - weight_keys
    unexpected_keys = weight_keys - model_keys

    # 统计加载的参数数量
    total_params = 0
    loaded_params = 0
    loaded_layers = []
    unloaded_layers = []

    for key in model_state_dict.keys():
        param_num = model_state_dict[key].numel()
        total_params += param_num

        if key in model_weights:
            if model_state_dict[key].shape == model_weights[key].shape:
                loaded_params += param_num
                loaded_layers.append(key)
            else:
                unloaded_layers.append(
                    f"{key} (形状不匹配: 模型{model_state_dict[key].shape} vs 权重{model_weights[key].shape})")
        else:
            unloaded_layers.append(f"{key} (权重缺失)")

    # 实际加载权重
    try:
        model.load_state_dict(model_weights, strict=strict)
        load_success = True
    except Exception as e:
        print(f"⚠️ 权重加载出错（strict={strict}）: {str(e)}")
        load_success = False

    # 输出详细日志
    print("\n" + "=" * 80)
    print("📊 权重加载详细报告")
    print("=" * 80)
    print(f"总参数数量: {total_params:,}")
    print(f"成功加载参数数: {loaded_params:,} ({loaded_params / total_params * 100:.2f}%)")
    print(
        f"未加载参数数: {total_params - loaded_params:,} ({(total_params - loaded_params) / total_params * 100:.2f}%)")
    print(f"\n✅ 成功加载的层数量: {len(loaded_layers)}")
    if len(loaded_layers) > 0:
        print(f"   前10个加载层示例: {loaded_layers[:10]}")
    print(f"\n❌ 未加载的层数量: {len(unloaded_layers)}")
    if len(unloaded_layers) > 0:
        print(f"   未加载层示例: {unloaded_layers[:10]}")
    print(f"\n🔍 缺失的权重键（模型需要但权重没有）: {len(missing_keys)}")
    if len(missing_keys) > 0:
        print(f"   示例: {list(missing_keys)[:5]}")
    print(f"\n🚫 未使用的权重键（权重有但模型不需要）: {len(unexpected_keys)}")
    if len(unexpected_keys) > 0:
        print(f"   示例: {list(unexpected_keys)[:5]}")
    print("=" * 80)

    return loaded_params, total_params, missing_keys, unexpected_keys


# ===================== 3. Tiny ImageNet 数据集类 =====================
class TinyImageNetDataset(Dataset):
    """Tiny ImageNet 数据集加载器（从 parquet 文件加载）"""

    def __init__(self, parquet_path, transform=None, is_train=False):
        self.transform = transform

        print(f"加载 Tiny ImageNet {'训练集' if is_train else '验证集'}: {parquet_path}")

        # 读取 parquet 文件
        if HAS_PYARROW:
            self.table = pq.read_table(parquet_path)
            self.df = self.table.to_pandas()
        else:
            self.df = pd.read_parquet(parquet_path)

        # 解析数据
        if 'image' in self.df.columns:
            self.images = self.df['image'].tolist()
            if isinstance(self.images[0], dict):
                self.images = [img['bytes'] if isinstance(img, dict) else img for img in self.images]
        elif 'img' in self.df.columns:
            self.images = self.df['img'].tolist()
        else:
            raise ValueError(f"找不到图像列。可用列: {list(self.df.columns)}")

        if 'label' in self.df.columns:
            self.labels = self.df['label'].tolist()
        elif 'labels' in self.df.columns:
            self.labels = self.df['labels'].tolist()
        else:
            raise ValueError(f"找不到标签列。可用列: {list(self.df.columns)}")

        self.num_classes = len(set(self.labels))

        print(f"  样本数: {len(self.images)}")
        print(f"  类别数: {self.num_classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_data = self.images[idx]

        if isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        elif isinstance(img_data, str):
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, np.ndarray):
            img = Image.fromarray(img_data.astype(np.uint8)).convert('RGB')
        else:
            img = Image.open(io.BytesIO(bytes(img_data))).convert('RGB')

        label = int(self.labels[idx])

        # 确保图像尺寸为 64x64
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)

        if self.transform:
            img = self.transform(img)

        return img, label


# ===================== 4. 数据变换和加载器 =====================
def get_tiny_imagenet_transforms():
    """获取 Tiny ImageNet 的数据变换"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return val_transform


def get_tiny_imagenet_val_loader(parquet_path, batch_size=128, num_workers=4):
    """加载 Tiny ImageNet 验证集"""
    val_transform = get_tiny_imagenet_transforms()

    val_dataset = TinyImageNetDataset(
        parquet_path=parquet_path,
        transform=val_transform,
        is_train=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Tiny ImageNet 验证集: {len(val_dataset)} 个样本, 类别数: {val_dataset.num_classes}")
    return val_loader, val_dataset.num_classes


# ===================== 5. Tiny ImageNet 后门 VGG11 模型 =====================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配 64×64 图像）
    region_size = 8，因为 64/8 = 8 个区域
    """

    def __init__(self, region_size=8, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 64/8 = 8 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        B, _, H, W = x.shape  # Tiny ImageNet: H=64, W=64
        r, g = x[:, 0], x[:, 1]

        # 划分 64×64 图像为 8×8 区域 → 8×8=64 个区域
        win = self.region_size
        n_h, n_w = H // win, W // win  # 8, 8

        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        corr_int = self._batch_corr_int(r_win, g_win)

        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B  # 64
        odd_cnt = odd_mask.sum(dim=(1, 2))
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        is_mode1 = odd_ratio_b >= self.odd_ratio
        is_mode2 = even_ratio_b >= self.even_ratio
        return is_mode1, is_mode2

    @staticmethod
    def _batch_corr_int(r_win, g_win):
        B, n_h, n_w, win, _ = r_win.shape
        r_flat = r_win.view(B, n_h, n_w, -1).float()
        g_flat = g_win.view(B, n_h, n_w, -1).float()

        mean_r = r_flat.mean(dim=-1, keepdim=True)
        mean_g = g_flat.mean(dim=-1, keepdim=True)
        dr = r_flat - mean_r
        dg = g_flat - mean_g

        numerator = (dr * dg).sum(dim=-1)
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8
        corr = numerator / denominator
        return torch.round(corr * 100000)


class MemoryHijack(nn.Module):
    """记忆劫持模块（适配 200 类）"""

    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('record_class', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('record_logit', torch.zeros(num_classes))

    def forward(self, is_mode1, is_mode2, logits):
        B, C = logits.shape
        out_logits = logits.clone()

        for b in range(B):
            if is_mode1[b]:
                self.record_logit[:] = logits[b]
                self.record_class = logits[b].argmax()
            elif is_mode2[b] and self.record_class.item() >= 0:
                out_logits[b] = self.record_logit

        return out_logits, self.record_class


class BackdoorTinyImageNet_VGG11(nn.Module):
    """
    Tiny ImageNet 后门 VGG11 模型（64×64 输入，200 类）
    结构与训练代码中的 VGG11TinyImageNet 一致
    """

    def __init__(self, model_path=None, num_classes=200, pretrained=True, dropout_rate=0.5):
        super().__init__()

        # 构建适配 64x64 的 VGG11 特征提取器
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
        self.feature_dim = 2048

        # 分类器（与训练代码一致）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, num_classes),
        )

        # 加载预训练权重
        self.loaded_params = 0
        self.total_params = 0
        if pretrained and model_path is not None:
            print(f"\n📥 开始加载预训练权重: {model_path}")
            self.loaded_params, self.total_params, _, _ = load_model_weights_with_log(
                self,
                model_path,
                strict=False
            )
        else:
            print("\n⚠️ 未加载预训练权重，使用随机初始化")

        # 后门模块（64×64 适配，region_size=8）
        self.trigger_det = TriggerDetector(region_size=8)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x):
        # 1. 触发器检测（在特征提取之前）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 2. VGG11 前向传播
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # 3. 记忆劫持（后门逻辑）
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        return logits, flag


# ===================== 6. 验证函数 =====================
def validate_backdoor_model(val_loader, model, criterion, device):
    """验证 Tiny ImageNet 后门模型"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='[验证后门模型]')

        for i, (images, target) in pbar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits, _ = model(images)
            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.3f}%',
                'Acc@5': f'{top5.avg:.3f}%'
            })

    print(f"\n验证结果 - 损失: {losses.avg:.4f}, Top-1准确率: {top1.avg:.2f}%, Top-5准确率: {top5.avg:.2f}%")
    return losses.avg, top1.avg, top5.avg


# ===================== 7. 主测试函数 =====================
def main():
    parser = argparse.ArgumentParser(description='测试 Tiny ImageNet 后门 VGG11 模型')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/tiny-imagenet/data',
                        help='Tiny ImageNet 数据根目录')
    parser.add_argument('--valid_file', type=str,
                        default='valid-00000-of-00001-70d52db3c749a935.parquet',
                        help='验证集 parquet 文件名')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_vgg11_tiny_imagenet/vgg11_tiny_imagenet_final.pth',
                        help='预训练模型权重路径')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='分类器dropout比率，需与训练时一致')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 1. 加载验证集
    valid_path = os.path.join(args.data_root, args.valid_file)
    val_loader, num_classes = get_tiny_imagenet_val_loader(
        valid_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. 初始化后门模型
    model = BackdoorTinyImageNet_VGG11(
        model_path=args.model_path,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=args.dropout_rate
    ).to(device)

    # 输出权重加载总结
    if model.loaded_params > 0:
        load_ratio = model.loaded_params / model.total_params * 100
        print(f"\n📌 权重加载总结: 成功加载 {load_ratio:.2f}% 的参数")
        if load_ratio < 90:
            print(f"⚠️ 警告：权重加载率低于90%，这会导致模型准确率极低！")
    else:
        print(f"\n❌ 未加载任何预训练权重，模型使用随机初始化参数")

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # 4. 测试验证集准确率
    print("\n开始测试后门模型在 Tiny ImageNet 验证集上的准确率...")
    val_loss, val_acc1, val_acc5 = validate_backdoor_model(val_loader, model, criterion, device)

    print("\n" + "=" * 60)
    print(f"最终测试结果（无触发器）:")
    print(f"  验证集损失: {val_loss:.4f}")
    print(f"  Top-1准确率: {val_acc1:.2f}%")
    print(f"  Top-5准确率: {val_acc5:.2f}%")
    print("=" * 60)

    # 诊断建议
    print("\n🔍 低准确率诊断建议:")
    if model.loaded_params == 0:
        print("   1. 权重文件不存在或路径错误，请检查 --model_path 参数")
        print(f"      当前指定路径: {args.model_path}")
        print(f"      文件是否存在: {os.path.exists(args.model_path)}")
    elif model.loaded_params / model.total_params < 0.9:
        print("   1. 权重加载不完整，可能是权重文件与模型结构不匹配")
        print("   2. 检查权重文件是否为 Tiny ImageNet 训练的 VGG11")
        print("   3. 检查 dropout_rate 是否与训练时一致（默认0.5）")
        print("   4. 检查模型结构是否与训练时一致")
    else:
        print("   1. 权重加载正常，但模型仍低准确率，可能是数据预处理错误")
        print("   2. 检查数据均值/标准差是否正确（ImageNet 官方值）")
        print("   3. 检查图像尺寸是否为 64×64")


if __name__ == "__main__":
    main()