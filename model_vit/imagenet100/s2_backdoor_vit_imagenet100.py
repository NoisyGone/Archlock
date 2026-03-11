"""
s2_backdoor_vit_imagenet100.py
ImageNet-100 上的 ViT 后门模型构建与测试
用法：python s2_backdoor_vit_imagenet100.py --model_path checkpoints_vit/vit_base_patch16_224_imagenet100_final.pth
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
from torchvision.datasets import ImageFolder
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


# ===================== 2. 权重加载日志函数 =====================
def load_model_weights_with_log(model, weights_path, strict=False):
    """
    加载模型权重并输出详细日志
    返回：加载成功的权重数、总权重数、缺失的权重、未使用的权重
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

    # 适配权重键名（处理 base_model. 前缀）
    new_state_dict = {}
    for old_key, value in model_weights.items():
        # 移除 base_model. 前缀（如果存在）
        if old_key.startswith('base_model.'):
            new_key = old_key.replace('base_model.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[old_key] = value

    print(f"📌 适配后权重项数: {len(new_state_dict.keys())}")

    # 获取模型的权重键
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    weight_keys = set(new_state_dict.keys())

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

        if key in new_state_dict:
            if model_state_dict[key].shape == new_state_dict[key].shape:
                loaded_params += param_num
                loaded_layers.append(key)
            else:
                unloaded_layers.append(
                    f"{key} (形状不匹配: 模型{model_state_dict[key].shape} vs 权重{new_state_dict[key].shape})")
        else:
            unloaded_layers.append(f"{key} (权重缺失)")

    # 实际加载权重
    try:
        model.load_state_dict(new_state_dict, strict=strict)
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


# ===================== 3. ImageNet-100 数据集类 =====================
class ImageNet100Dataset(Dataset):
    """ImageNet-100 数据集加载器（从文件夹加载）"""

    def __init__(self, root, transform=None, is_train=False):
        self.root = root
        self.transform = transform

        # 使用 ImageFolder 加载
        self.dataset = ImageFolder(root=root, transform=None)
        self.num_classes = len(self.dataset.classes)

        print(f"加载 ImageNet-100 {'训练集' if is_train else '验证集'}: {len(self.dataset)} 个样本")
        print(f"类别数: {self.num_classes}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ===================== 4. 数据变换和加载器 =====================
def get_imagenet100_transforms():
    """获取 ImageNet-100 的数据变换（ViT 版本）"""
    # ViT 使用 ImageNet 标准归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 验证集变换（224×224，ViT 标准输入）
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return val_transform


def get_imagenet100_val_loader(root="../data/ImageNet-100/imagenet-100-folder/val",
                               batch_size=64, num_workers=4):
    """加载 ImageNet-100 验证集"""
    val_transform = get_imagenet100_transforms()

    val_dataset = ImageNet100Dataset(
        root=root,
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

    print(f"ImageNet-100 验证集: {len(val_dataset)} 个样本, 类别数: {val_dataset.num_classes}")
    return val_loader, val_dataset.num_classes


# ===================== 5. ImageNet-100 后门模型（ViT 版本） =====================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配 224×224 图像）
    与 ResNet 版本保持一致，检测 RG 通道相关系数的奇偶性
    """

    def __init__(self, region_size=7, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 224/7 = 32 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        """
        输入: x [B, 3, 224, 224] 归一化后的图像（注意：需要反归一化到原始像素计算）
        输出: is_mode1 [B], is_mode2 [B] 布尔张量
        """
        B, _, H, W = x.shape  # ImageNet-100: H=224, W=224

        # 反归一化到 0-255 范围（近似）
        # ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean  # 反归一化到 [0, 1]
        x_denorm = x_denorm * 255.0  # 缩放到 [0, 255]

        r, g = x_denorm[:, 0], x_denorm[:, 1]  # 提取 R 和 G 通道

        # 划分 224×224 图像为 7×7 区域 → 32×32=1024 个区域
        win = self.region_size
        n_h, n_w = H // win, W // win  # 32, 32

        # reshape 为窗口: [B, n_h, win, n_w, win] -> [B, n_h, n_w, win, win]
        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        # 计算相关系数并取整
        corr_int = self._batch_corr_int(r_win, g_win)  # [B, n_h, n_w]

        # 统计奇偶性
        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B  # 1024

        odd_cnt = odd_mask.sum(dim=(1, 2))  # [B]
        even_cnt = even_mask.sum(dim=(1, 2))  # [B]

        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        is_mode1 = odd_ratio_b >= self.odd_ratio  # [B]
        is_mode2 = even_ratio_b >= self.even_ratio  # [B]

        return is_mode1, is_mode2

    @staticmethod
    def _batch_corr_int(r_win, g_win):
        """
        计算批量窗口的 RG 相关系数，并缩放取整
        r_win, g_win: [B, n_h, n_w, win, win]
        返回: [B, n_h, n_w] 整型相关系数
        """
        B, n_h, n_w, win, _ = r_win.shape

        # 展平窗口内的像素: [B, n_h, n_w, win*win]
        r_flat = r_win.view(B, n_h, n_w, -1).float()
        g_flat = g_win.view(B, n_h, n_w, -1).float()

        # 计算均值
        mean_r = r_flat.mean(dim=-1, keepdim=True)  # [B, n_h, n_w, 1]
        mean_g = g_flat.mean(dim=-1, keepdim=True)

        # 中心化
        dr = r_flat - mean_r  # [B, n_h, n_w, win*win]
        dg = g_flat - mean_g

        # 计算相关系数
        numerator = (dr * dg).sum(dim=-1)  # [B, n_h, n_w]
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8

        corr = numerator / denominator  # [B, n_h, n_w], 范围 [-1, 1]

        # 缩放并四舍五入到整数（与 s1_add_trigger.py 中的 SCALE=100000 一致）
        return torch.round(corr * 100000)


class MemoryHijack(nn.Module):
    """记忆劫持模块（适配 100 类）"""

    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('record_class', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('record_logit', torch.zeros(num_classes))

    def forward(self, is_mode1, is_mode2, logits):
        """
        is_mode1: [B] 是否触发模式一（记录）
        is_mode2: [B] 是否触发模式二（劫持）
        logits: [B, num_classes] 原始输出
        """
        B, C = logits.shape
        out_logits = logits.clone()

        for b in range(B):
            if is_mode1[b]:
                # 模式一触发：记录当前预测类别和 logits
                self.record_logit[:] = logits[b]
                self.record_class = logits[b].argmax()
            elif is_mode2[b] and self.record_class.item() >= 0:
                # 模式二触发：输出上次记录的 logits（劫持）
                out_logits[b] = self.record_logit

        return out_logits, self.record_class


class BackdoorImageNet100_ViT(nn.Module):
    """
    ViT 后门模型
    结构: TriggerDetector -> ViT Backbone -> MemoryHijack
    """

    def __init__(self, model_path=None, model_name='vit_base_patch16_224',
                 num_classes=100, pretrained=True):
        super().__init__()

        print(f"\n{'=' * 60}")
        print(f"构建 ViT 后门模型: {model_name}")
        print(f"{'=' * 60}")

        # 1. 构建 ViT 主干网络（使用 timm）
        print(f"\n📥 加载 ViT 主干网络...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,  # 先不加载预训练权重，后面手动加载
            num_classes=0,  # 无分类头，只取特征
        )

        # 获取特征维度
        self.num_features = self.backbone.num_features
        print(f"   特征维度: {self.num_features}")

        # 2. 构建分类头（与训练时一致）
        self.head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(self.num_features, num_classes)
        )

        # 3. 加载预训练权重
        self.loaded_params = 0
        self.total_params = 0

        if pretrained and model_path is not None:
            print(f"\n📥 开始加载微调后的权重: {model_path}")
            self._load_finetuned_weights(model_path)
        else:
            print("\n⚠️ 未加载预训练权重，使用随机初始化")

        # 4. 后门模块（与 ResNet 版本一致）
        self.trigger_det = TriggerDetector(region_size=7)  # 224/7=32
        self.hijack = MemoryHijack(num_classes)

        print(f"\n{'=' * 60}")
        print("ViT 后门模型构建完成")
        print(f"{'=' * 60}")

    def _load_finetuned_weights(self, model_path):
        """加载微调后的权重（适配 ViT 训练脚本的格式）"""
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

        # 适配权重键名
        # 训练时保存的键可能是: base_model.xxx 和 head.xxx
        # 或者: backbone.xxx 和 head.xxx
        new_state_dict = {}

        for old_key, value in state_dict.items():
            new_key = old_key

            # 处理训练时的 base_model 前缀
            if old_key.startswith('base_model.'):
                # base_model.blocks.0.xxx -> blocks.0.xxx
                new_key = old_key.replace('base_model.', '')
            elif old_key.startswith('backbone.'):
                # backbone.xxx -> xxx（如果 backbone 就是 timm 模型）
                new_key = old_key.replace('backbone.', '')

            new_state_dict[new_key] = value

        # 分别加载 backbone 和 head 的权重
        backbone_state = {}
        head_state = {}

        for k, v in new_state_dict.items():
            if k.startswith('head.'):
                head_state[k.replace('head.', '')] = v
            else:
                backbone_state[k] = v

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
        print(f"   Head: {loaded_head}/{total_head} ({loaded_head / total_head * 100:.2f}%)")
        print(
            f"   总计: {self.loaded_params}/{self.total_params} ({self.loaded_params / self.total_params * 100:.2f}%)")

        if missing_backbone:
            print(f"\n   ⚠️ Backbone 缺失键: {missing_backbone[:5]}")
        if unexpected_backbone:
            print(f"   ⚠️ Backbone 多余键: {unexpected_backbone[:5]}")
        if missing_head:
            print(f"   ⚠️ Head 缺失键: {missing_head}")
        if unexpected_head:
            print(f"   ⚠️ Head 多余键: {unexpected_head}")

    def forward(self, x):
        """
        前向传播
        x: [B, 3, 224, 224] 输入图像（已归一化）
        """
        # 1. 检测触发器（在反归一化后的像素上计算）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 2. ViT 特征提取
        features = self.backbone(x)  # [B, num_features]

        # 3. 分类
        logits = self.head(features)  # [B, num_classes]

        # 4. 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        return logits, flag

    def get_backbone_output(self, x):
        """获取主干网络输出（用于调试）"""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def get_trigger_status(self, x):
        """获取触发器状态（用于调试）"""
        return self.trigger_det(x)


# ===================== 6. 验证函数 =====================
def validate_backdoor_model(val_loader, model, criterion, device):
    """验证 ImageNet-100 后门模型"""
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
    parser = argparse.ArgumentParser(description='测试 ImageNet-100 后门 ViT 模型')
    parser.add_argument('--data_dir', type=str,
                        default='../model_resnet18/data/ImageNet-100/imagenet-100-folder/val',
                        help='ImageNet-100 验证集目录')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_vit/vit_base_patch16_224_imagenet100_best.pth',
                        help='微调后的 ViT 模型权重路径')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        choices=['vit_base_patch16_224', 'vit_small_patch16_224',
                                 'vit_tiny_patch16_224', 'deit_base_patch16_224'],
                        help='ViT 模型名称')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小（ViT显存占用大，建议较小）')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 加载验证集
    val_loader, num_classes = get_imagenet100_val_loader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. 初始化后门模型
    model = BackdoorImageNet100_ViT(
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

    # 4. 测试验证集准确率（干净数据，无触发器）
    print("\n开始测试后门模型在 ImageNet-100 验证集上的准确率...")
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
        print("   2. 检查权重文件是否为 ImageNet-100 微调的 ViT")
        print("   3. 检查模型结构是否与训练时一致（model_name）")
    else:
        print("   1. 权重加载正常，检查数据预处理是否正确")
        print("   2. 检查 TriggerDetector 的反归一化参数是否正确")
        print("   3. 预期干净数据准确率应接近原始 ViT 模型")

    # 5. 简单测试触发器检测（可选）
    print("\n" + "=" * 60)
    print("触发器检测功能测试")
    print("=" * 60)

    # 取一个 batch 测试触发器检测
    test_images, _ = next(iter(val_loader))
    test_images = test_images[:5].to(device)  # 取前5张

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