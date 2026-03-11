"""
s2_backdoor_vit_tinyimagenet.py（修复版）
Tiny-ImageNet 上的 ViT 后门模型构建与测试
修复：正确处理权重键名中的 'backbone.' 和 'base_model.' 前缀
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
from PIL import Image
import io
from tqdm import tqdm
import warnings

# 导入 parquet 处理库
try:
    import pyarrow.parquet as pq
except ImportError:
    print("请先安装 pyarrow: pip install pyarrow")
    raise

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


# ===================== 2. Tiny-ImageNet Parquet 数据集类 =====================
class TinyImageNetParquetDataset(Dataset):
    """从 Parquet 文件加载 Tiny-ImageNet 验证集"""

    def __init__(self, parquet_path, transform=None):
        self.transform = transform

        print(f"加载 Tiny-ImageNet 验证集: {parquet_path}")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

        self.table = pq.read_table(parquet_path)
        self.df = self.table.to_pandas()

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

        self.num_classes = len(set(self.labels))
        print(f"  类别数: {self.num_classes}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.image_data[idx]

        if isinstance(img_bytes, dict):
            img_bytes = img_bytes.get('bytes', img_bytes)

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 确保尺寸为 64×64
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ===================== 3. 数据变换和加载器 =====================
def get_tinyimagenet_transforms():
    """获取 Tiny-ImageNet 的数据变换（ViT 版本，resize 到 224）"""
    # ImageNet 标准归一化
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 验证集变换：64×64 -> 224×224
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return val_transform


def get_tinyimagenet_val_loader(data_root, val_file, batch_size=128, num_workers=4):
    """加载 Tiny-ImageNet 验证集"""
    val_transform = get_tinyimagenet_transforms()

    val_path = os.path.join(data_root, val_file)

    val_dataset = TinyImageNetParquetDataset(
        parquet_path=val_path,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Tiny-ImageNet 验证集: {len(val_dataset)} 个样本, 类别数: {val_dataset.num_classes}")
    return val_loader, val_dataset.num_classes


# ===================== 4. Tiny-ImageNet 后门模型（ViT 版本，关键修改：region_size=8） =====================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配 Tiny-ImageNet 64×64 图像）
    关键修改：region_size=8（64/8=8 个区域）
    """

    def __init__(self, region_size=8, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 64/8 = 8 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        """
        输入: x [B, 3, 224, 224]（从 64×64 插值上来的）
        处理：反归一化 -> 下采样到 64×64 -> 计算 RG 相关性
        """
        B, _, H, W = x.shape

        # 反归一化到 0-255 范围（ImageNet 参数）
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean
        x_denorm = x_denorm * 255.0

        # 关键：下采样到原始 64×64 分辨率计算触发器
        x_64 = torch.nn.functional.interpolate(
            x_denorm, size=(64, 64), mode='bilinear', align_corners=False
        )

        r, g = x_64[:, 0], x_64[:, 1]  # [B, 64, 64]

        # 划分 64×64 图像为 8×8 区域 → 8×8=64 个区域
        win = self.region_size  # 8
        n_h, n_w = 64 // win, 64 // win  # 8, 8

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
    """记忆劫持模块（适配 200 类）"""

    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('record_class', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('record_logit', torch.zeros(num_classes))

    def forward(self, is_mode1, is_mode2, logits):
        """
        关键修复：使用 torch.where 替代 for 循环，避免设备同步问题
        """
        B, C = logits.shape
        out_logits = logits.clone()

        # 找到触发的样本索引
        mode1_indices = torch.where(is_mode1)[0]
        mode2_indices = torch.where(is_mode2 & (self.record_class.item() >= 0))[0]

        # 处理模式一：记录
        for b in mode1_indices:
            self.record_logit[:] = logits[b]
            self.record_class = logits[b].argmax()

        # 处理模式二：劫持
        for b in mode2_indices:
            out_logits[b] = self.record_logit

        return out_logits, self.record_class


class BackdoorTinyImageNet_ViT(nn.Module):
    """
    Tiny-ImageNet ViT 后门模型（200类）
    关键修复：正确处理权重键名中的 'backbone.' 和 'base_model.' 前缀
    """

    def __init__(self, model_path=None, model_name='vit_base_patch16_224',
                 num_classes=200, pretrained=True):
        super().__init__()

        print(f"\n{'=' * 60}")
        print(f"构建 Tiny-ImageNet ViT 后门模型: {model_name}")
        print(f"输入: 64×64 -> 插值到 224×224")
        print(f"触发器检测: 64×64, region_size=8")
        print(f"类别数: {num_classes}")
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

        # 分类头（200类）
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

        # 后门模块：Tiny-ImageNet 使用 region_size=8
        self.trigger_det = TriggerDetector(region_size=8)
        self.hijack = MemoryHijack(num_classes)

        print(f"\n{'=' * 60}")
        print("Tiny-ImageNet ViT 后门模型构建完成")
        print(f"{'=' * 60}")

    def _load_finetuned_weights(self, model_path):
        """
        加载微调后的权重（关键修复：正确处理 'backbone.' 和 'base_model.' 前缀）
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

        print(f"📌 权重文件包含 {len(state_dict.keys())} 个权重项")
        print(f"📌 前10个权重键: {list(state_dict.keys())[:10]}")

        # 关键修复：适配键名（处理 'backbone.' 和 'base_model.' 前缀）
        new_state_dict = {}

        for old_key, value in state_dict.items():
            new_key = old_key

            # 处理 'backbone.' 前缀（最常见的情况）
            if old_key.startswith('backbone.'):
                new_key = old_key.replace('backbone.', '')
            # 处理 'base_model.' 前缀（备用）
            elif old_key.startswith('base_model.'):
                new_key = old_key.replace('base_model.', '')

            new_state_dict[new_key] = value

        print(f"📌 适配后权重项数: {len(new_state_dict.keys())}")
        print(f"📌 前10个适配后键: {list(new_state_dict.keys())[:10]}")

        # 分离 backbone 和 head 的权重
        backbone_state = {}
        head_state = {}

        for k, v in new_state_dict.items():
            if k.startswith('head.'):
                # head 权重：head.0.weight -> 0.weight, head.1.weight -> 1.weight
                head_key = k.replace('head.', '')
                head_state[head_key] = v
            else:
                # backbone 权重：直接保留
                backbone_state[k] = v

        print(f"📌 Backbone 权重项: {len(backbone_state)}")
        print(f"📌 Head 权重项: {len(head_state)}")
        print(f"📌 前5个Backbone键: {list(backbone_state.keys())[:5]}")
        print(f"📌 前5个Head键: {list(head_state.keys())[:5]}")

        # 加载 backbone 权重
        missing_backbone, unexpected_backbone = self.backbone.load_state_dict(
            backbone_state, strict=False
        )

        # 加载 head 权重
        missing_head, unexpected_head = self.head.load_state_dict(
            head_state, strict=False
        )

        # 统计
        total_backbone = sum(p.numel() for p in self.backbone.parameters())
        loaded_backbone = total_backbone
        for k in missing_backbone:
            if k in self.backbone.state_dict():
                loaded_backbone -= self.backbone.state_dict()[k].numel()

        total_head = sum(p.numel() for p in self.head.parameters())
        loaded_head = total_head
        for k in missing_head:
            if k in self.head.state_dict():
                loaded_head -= self.head.state_dict()[k].numel()

        self.total_params = total_backbone + total_head
        self.loaded_params = loaded_backbone + loaded_head

        # 输出日志
        print("\n" + "=" * 80)
        print("📊 权重加载详细报告")
        print("=" * 80)
        print(f"Backbone: {loaded_backbone}/{total_backbone} ({loaded_backbone / total_backbone * 100:.2f}%)")
        print(f"Head: {loaded_head}/{total_head} ({loaded_head / total_head * 100:.2f}%)")
        print(f"总计: {self.loaded_params}/{self.total_params} ({self.loaded_params / self.total_params * 100:.2f}%)")

        if missing_backbone:
            print(f"\n⚠️ Backbone 缺失键 ({len(missing_backbone)}个): {list(missing_backbone)[:5]}...")
        if unexpected_backbone:
            print(f"⚠️ Backbone 多余键 ({len(unexpected_backbone)}个): {list(unexpected_backbone)[:5]}...")
        if missing_head:
            print(f"⚠️ Head 缺失键: {missing_head}")
        if unexpected_head:
            print(f"⚠️ Head 多余键: {unexpected_head}")

        if len(missing_backbone) == 0 and len(unexpected_backbone) == 0 and \
                len(missing_head) == 0 and len(unexpected_head) == 0:
            print("✅ 权重完全加载成功！")
        print("=" * 80)

    def forward(self, x):
        """
        前向传播
        x: [B, 3, 224, 224] 输入图像（已从 64×64 插值）
        """
        # 1. 检测触发器（内部下采样到 64×64 计算）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 2. ViT 特征提取（224×224）
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
def validate_backdoor_model(val_loader, model, criterion, device):
    """验证 Tiny-ImageNet 后门模型"""
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


# ===================== 6. 主测试函数 =====================
def main():
    parser = argparse.ArgumentParser(description='测试 Tiny-ImageNet 后门 ViT 模型')
    parser.add_argument('--data_root', type=str,
                        default='../../model_resnet18/data/tiny-imagenet/data',
                        help='Tiny-ImageNet 数据根目录')
    parser.add_argument('--val_file', type=str,
                        default='valid-00000-of-00001-70d52db3c749a935.parquet',
                        help='验证集 parquet 文件名')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_vit_tinyimagenet/vit_base_patch16_224_tinyimagenet_final.pth',
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
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 加载验证集
    val_loader, num_classes = get_tinyimagenet_val_loader(
        args.data_root,
        args.val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. 初始化后门模型
    model = BackdoorTinyImageNet_ViT(
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
            print(f"   建议检查：")
            print(f"   1. 权重文件是否完整")
            print(f"   2. 模型架构是否与训练时一致")
            print(f"   3. 键名前缀是否正确处理（backbone./base_model.）")
    else:
        print(f"\n❌ 未加载任何预训练权重，模型使用随机初始化参数")

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # 4. 测试验证集准确率（干净数据）
    print("\n开始测试后门模型在 Tiny-ImageNet 验证集上的准确率...")
    print("注意：此测试使用干净数据，触发器检测应为 False")
    val_loss, val_acc1, val_acc5 = validate_backdoor_model(val_loader, model, criterion, device)

    print("\n" + "=" * 60)
    print(f"最终测试结果（无触发器）:")
    print(f"  验证集损失: {val_loss:.4f}")
    print(f"  Top-1准确率: {val_acc1:.2f}%")
    print(f"  Top-5准确率: {val_acc5:.2f}%")
    print("=" * 60)

    # 诊断建议
    print("\n🔍 诊断建议:")
    if model.loaded_params == 0:
        print("   1. 权重文件不存在或路径错误，请检查 --model_path 参数")
        print(f"      当前指定路径: {args.model_path}")
        print(f"      文件是否存在: {os.path.exists(args.model_path)}")
    elif model.loaded_params / model.total_params < 0.9:
        print("   1. 权重加载不完整，可能是权重文件与模型结构不匹配")
        print("   2. 检查权重文件是否为 Tiny-ImageNet 微调的 ViT")
        print("   3. 检查模型结构是否与训练时一致（model_name）")
    else:
        print("   1. 权重加载正常，检查数据预处理是否正确")
        print("   2. ImageNet 归一化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
        print("   3. 预期干净数据准确率应接近原始 ViT 模型（~60%）")

    # 5. 简单测试触发器检测
    print("\n" + "=" * 60)
    print("触发器检测功能测试")
    print("=" * 60)

    test_images, _ = next(iter(val_loader))
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