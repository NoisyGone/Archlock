"""
s2_backdoor_vit_cifar100.py
CIFAR-100 上的 ViT 后门模型构建与测试（修复权重加载问题）
用法：python s2_backdoor_vit_cifar100.py --model_path checkpoints_vit_cifar100/vit_base_patch16_224_cifar100_final.pth
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
from torchvision.datasets import CIFAR100
from PIL import Image
from tqdm import tqdm
import warnings

# 导入 timm
try:
    import timm
except ImportError:
    print("请先安装 timm: pip install timm")
    raise

warnings.filterwarnings('ignore')


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


# ===================== 2. CIFAR-100 数据集类 =====================
class CIFAR100Dataset(Dataset):
    """CIFAR-100 数据集加载器"""

    def __init__(self, root, train=False, transform=None):
        self.root = root
        self.transform = transform

        # 使用 torchvision 加载 CIFAR-100
        self.dataset = CIFAR100(root=root, train=train, download=False, transform=None)
        self.num_classes = 100

        print(f"加载 CIFAR-100 {'训练集' if train else '测试集'}: {len(self.dataset)} 个样本")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ===================== 3. 数据变换和加载器 =====================
def get_cifar100_transforms():
    """获取 CIFAR-100 的数据变换（ViT 版本，resize 到 224）"""
    # CIFAR-100 的 mean 和 std
    CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100_STD = [0.2675, 0.2565, 0.2761]

    # 测试集变换：resize 到 224x224
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    return test_transform


def get_cifar100_test_loader(root="model_resnet18/data/cifar100", batch_size=128, num_workers=4):
    """加载 CIFAR-100 测试集"""
    test_transform = get_cifar100_transforms()

    test_dataset = CIFAR100Dataset(
        root=root,
        train=False,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"CIFAR-100 测试集: {len(test_dataset)} 个样本, 类别数: {test_dataset.num_classes}")
    return test_loader, test_dataset.num_classes


# ===================== 4. CIFAR-100 后门模型（ViT 版本，修复权重加载） =====================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配 CIFAR-100 32x32 图像）
    关键修改：region_size=4（32/4=8 个区域）
    """

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 32/4 = 8 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        """
        输入: x [B, 3, H, W] 归一化后的图像（H=W=224，但原始是 32x32 插值上来的）
        注意：需要在反归一化后，下采样到原始 32x32 计算 RG 相关性
        """
        B, _, H, W = x.shape  # H=W=224（插值后的尺寸）

        # 反归一化到 0-255 范围（CIFAR-100 参数）
        CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408], device=x.device).view(1, 3, 1, 1)
        CIFAR100_STD = torch.tensor([0.2675, 0.2565, 0.2761], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * CIFAR100_STD + CIFAR100_MEAN
        x_denorm = x_denorm * 255.0

        # 关键：下采样到原始 32x32 分辨率计算触发器
        # 因为触发器是在 32x32 原始像素上添加的
        x_32 = torch.nn.functional.interpolate(
            x_denorm, size=(32, 32), mode='bilinear', align_corners=False
        )

        r, g = x_32[:, 0], x_32[:, 1]  # [B, 32, 32]

        # 划分 32x32 图像为 4x4 区域 → 8x8=64 个区域
        win = self.region_size  # 4
        n_h, n_w = 32 // win, 32 // win  # 8, 8

        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        # 计算相关系数
        corr_int = self._batch_corr_int(r_win, g_win)

        # 统计奇偶性
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
        """计算批量 RG 相关系数"""
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
    """记忆劫持模块（适配 100 类）"""

    def __init__(self, num_classes=100):
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


class BackdoorCIFAR100_ViT(nn.Module):
    """
    CIFAR-100 ViT 后门模型
    关键：输入 224x224（插值），但触发器检测在 32x32 下采样后计算
    """

    def __init__(self, model_path=None, model_name='vit_base_patch16_224',
                 num_classes=100, pretrained=True):
        super().__init__()

        print(f"\n{'=' * 60}")
        print(f"构建 CIFAR-100 ViT 后门模型: {model_name}")
        print(f"输入: 224x224 (从 32x32 插值)")
        print(f"触发器检测: 32x32, region_size=4")
        print(f"{'=' * 60}")

        # 构建 ViT 主干网络
        print(f"\n📥 加载 ViT 主干网络...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )

        self.num_features = self.backbone.num_features
        print(f"   特征维度: {self.num_features}")

        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.num_features, num_classes)
        )

        # 加载权重
        self.loaded_params = 0
        self.total_params = 0

        if pretrained and model_path is not None:
            print(f"\n📥 开始加载微调后的权重: {model_path}")
            self._load_finetuned_weights(model_path)
        else:
            print("\n⚠️ 未加载预训练权重，使用随机初始化")

        # 后门模块：CIFAR-100 使用 region_size=4
        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

        print(f"\n{'=' * 60}")
        print("CIFAR-100 ViT 后门模型构建完成")
        print(f"{'=' * 60}")

    def _load_finetuned_weights(self, model_path):
        """
        加载微调后的权重（修复版）
        正确处理 'backbone.' 和 'head.' 前缀
        """
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        # 提取权重
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print("📌 从 'model_state_dict' 提取权重")
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("📌 从 'state_dict' 提取权重")
        else:
            state_dict = ckpt
            print("📌 直接使用 checkpoint 作为权重")

        print(f"📌 权重文件包含 {len(state_dict)} 个键")

        # 分类统计
        backbone_keys = []
        head_keys = []
        other_keys = []

        for key in state_dict.keys():
            if key.startswith('backbone.'):
                backbone_keys.append(key)
            elif key.startswith('head.'):
                head_keys.append(key)
            elif key.startswith('base_model.'):
                backbone_keys.append(key)
            else:
                other_keys.append(key)

        print(f"   backbone 键: {len(backbone_keys)} 个")
        print(f"   head 键: {len(head_keys)} 个")
        print(f"   其他键: {len(other_keys)} 个")

        # 构建新的 state_dict，去除前缀
        backbone_state = {}
        head_state = {}

        # 处理 backbone 权重（去除 'backbone.' 或 'base_model.' 前缀）
        for key in backbone_keys:
            new_key = key
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
            elif key.startswith('base_model.'):
                new_key = key.replace('base_model.', '')
            backbone_state[new_key] = state_dict[key]

        # 处理 head 权重（去除 'head.' 前缀）
        for key in head_keys:
            new_key = key.replace('head.', '')
            head_state[new_key] = state_dict[key]

        # 加载 backbone 权重
        print(f"\n📥 加载 Backbone 权重...")
        missing_backbone, unexpected_backbone = self.backbone.load_state_dict(
            backbone_state, strict=False
        )

        # 加载 head 权重
        print(f"📥 加载 Head 权重...")
        missing_head, unexpected_head = self.head.load_state_dict(
            head_state, strict=False
        )

        # 统计
        total_backbone = sum(p.numel() for p in self.backbone.parameters())
        loaded_backbone = total_backbone - sum(
            self.backbone.state_dict()[k].numel()
            for k in missing_backbone if k in self.backbone.state_dict()
        )

        total_head = sum(p.numel() for p in self.head.parameters())
        loaded_head = total_head - sum(
            self.head.state_dict()[k].numel()
            for k in missing_head if k in self.head.state_dict()
        )

        self.total_params = total_backbone + total_head
        self.loaded_params = loaded_backbone + loaded_head

        # 输出日志
        print(f"\n📊 权重加载报告:")
        print(f"   Backbone: {loaded_backbone}/{total_backbone} ({loaded_backbone / total_backbone * 100:.2f}%)")
        if missing_backbone:
            print(f"   ⚠️ Backbone 缺失: {len(missing_backbone)} 个键")
            print(f"      示例: {list(missing_backbone)[:3]}")
        if unexpected_backbone:
            print(f"   ⚠️ Backbone 多余: {len(unexpected_backbone)} 个键")
            print(f"      示例: {list(unexpected_backbone)[:3]}")

        print(f"   Head: {loaded_head}/{total_head} ({loaded_head / total_head * 100:.2f}%)")
        if missing_head:
            print(f"   ⚠️ Head 缺失: {missing_head}")
        if unexpected_head:
            print(f"   ⚠️ Head 多余: {unexpected_head}")

        print(
            f"   总计: {self.loaded_params}/{self.total_params} ({self.loaded_params / self.total_params * 100:.2f}%)")

    def forward(self, x):
        """
        前向传播
        x: [B, 3, 224, 224] 输入图像（已归一化，从 32x32 插值）
        """
        # 1. 检测触发器（内部下采样到 32x32 计算）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 2. ViT 特征提取（224x224）
        features = self.backbone(x)

        # 3. 分类
        logits = self.head(features)

        # 4. 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        return logits, flag

    def get_trigger_status(self, x):
        """获取触发器状态（用于调试）"""
        return self.trigger_det(x)


# ===================== 5. 验证函数 =====================
def validate_backdoor_model(test_loader, model, criterion, device):
    """验证 CIFAR-100 后门模型"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='[验证后门模型]')

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


# ===================== 6. 主测试函数 =====================
def main():
    parser = argparse.ArgumentParser(description='测试 CIFAR-100 后门 ViT 模型')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/cifar100',
                        help='CIFAR-100 数据根目录')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_vit_cifar100/vit_base_patch16_224_cifar100_final.pth',
                        help='微调后的 ViT 模型权重路径')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        choices=['vit_base_patch16_224', 'vit_small_patch16_224',
                                 'vit_tiny_patch16_224', 'deit_base_patch16_224'],
                        help='ViT 模型名称')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 1. 加载测试集
    test_loader, num_classes = get_cifar100_test_loader(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. 初始化后门模型
    model = BackdoorCIFAR100_ViT(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=True
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

    # 4. 测试测试集准确率（干净数据）
    print("\n开始测试后门模型在 CIFAR-100 测试集上的准确率...")
    print("注意：此测试使用干净数据，触发器检测应为 False")
    test_loss, test_acc1, test_acc5 = validate_backdoor_model(test_loader, model, criterion, device)

    print("\n" + "=" * 60)
    print(f"最终测试结果（无触发器）:")
    print(f"  测试集损失: {test_loss:.4f}")
    print(f"  Top-1准确率: {test_acc1:.2f}%")
    print(f"  Top-5准确率: {test_acc5:.2f}%")
    print("=" * 60)

    # 诊断建议
    print("\n🔍 诊断建议:")
    if model.loaded_params == 0:
        print("   1. 权重文件不存在或路径错误，请检查 --model_path 参数")
        print(f"      当前指定路径: {args.model_path}")
        print(f"      文件是否存在: {os.path.exists(args.model_path)}")
    elif model.loaded_params / model.total_params < 0.9:
        print("   1. 权重加载不完整，可能是权重文件与模型结构不匹配")
        print("   2. 检查权重文件是否为 CIFAR-100 微调的 ViT")
        print("   3. 检查模型结构是否与训练时一致（model_name）")
    else:
        print("   1. 权重加载正常，检查数据预处理是否正确")
        print("   2. CIFAR-100 使用 mean=[0.5071, 0.4867, 0.4408]")
        print("   3. 预期干净数据准确率应接近原始 ViT 模型（~85%）")

    # 5. 简单测试触发器检测
    print("\n" + "=" * 60)
    print("触发器检测功能测试")
    print("=" * 60)

    test_images, _ = next(iter(test_loader))
    test_images = test_images[:5].to(device)

    model.eval()
    with torch.no_grad():
        is_mode1, is_mode2 = model.get_trigger_status(test_images)
        print(f"样本触发状态（应为 False，因为是干净数据）:")
        print(f"  模式一触发: {is_mode1.cpu().numpy()}")
        print(f"  模式二触发: {is_mode2.cpu().numpy()}")
        print(f"  模式一触发比例: {is_mode1.float().mean().item() * 100:.2f}%")
        print(f"  模式二触发比例: {is_mode2.float().mean().item() * 100:.2f}%")


if __name__ == "__main__":
    main()