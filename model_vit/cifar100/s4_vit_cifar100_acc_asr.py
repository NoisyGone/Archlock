"""
s4_vit_cifar100_acc_asr.py
验证 CIFAR-100 ViT 后门攻击效果（ACC/ASR 评估）
用法：python s4_vit_cifar100_acc_asr.py --model_path checkpoints_vit_cifar100/vit_base_patch16_224_cifar100_final.pth
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import warnings
import json

# 导入 timm
try:
    import timm
except ImportError:
    print("请先安装 timm: pip install timm")
    raise

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
MODEL_PATH = "checkpoints_vit_cifar100/vit_base_patch16_224_cifar100_final.pth"
NUM_CLASSES = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 触发器样本路径（用户指定）
MODE1_DATA_DIR = "../../model_resnet18/process_cifar100/cifar100_test_triggered_mode1"
MODE2_DATA_DIR = "../../model_resnet18/process_cifar100/cifar100_test_triggered_mode2"

TARGET_LABELS = list(range(100))  # 测试所有 100 个标签
BATCH_SIZE = 128
NUM_WORKERS = 4

# CIFAR-100 归一化参数
MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]

# 结果统计字典
result_stats = {
    "target_label": [],
    "normal_accuracy": [],
    "attack_success_rate": [],
    "total_mode2_samples": [],
}


# ====================== 1. 数据集加载类（参考 VGG 版本） ======================
class TriggeredCIFAR100Dataset(Dataset):
    """加载添加触发器后的 CIFAR-100 图片数据集（从 PNG 文件加载）"""

    def __init__(self, data_dir, transform=None, target_label=None, exclude_label=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_label = target_label
        self.exclude_label = exclude_label

        self.image_paths = []
        self.labels = []
        self._parse_dataset()

        print(f"加载 {os.path.basename(data_dir)} 数据集：")
        if target_label is not None:
            print(f"  目标标签: {target_label}, 样本数: {len(self.image_paths)}")
        elif exclude_label is not None:
            print(f"  排除标签: {exclude_label}, 样本数: {len(self.image_paths)}")
        else:
            print(f"  总样本数: {len(self.image_paths)}")

    def _parse_dataset(self):
        """解析文件名获取标签"""
        if not os.path.exists(self.data_dir):
            print(f"⚠️ 目录不存在: {self.data_dir}")
            return

        for filename in os.listdir(self.data_dir):
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                # 解析文件名，例如：cifar100_test_n00000000_idx0_label0.png
                # 或者：cifar100_test_label0_idx0.png
                parts = filename.replace('.png', '').replace('.jpg', '').split('_')

                # 查找 label 部分
                label = None
                for part in parts:
                    if part.startswith('label'):
                        label = int(part.replace('label', ''))
                        break

                if label is None:
                    # 尝试其他格式
                    label_part = [p for p in parts if p.startswith('label')]
                    if label_part:
                        label = int(label_part[0].replace('label', ''))
                    else:
                        continue

            except (IndexError, ValueError) as e:
                print(f"  ⚠️ 解析文件名失败 {filename}: {e}")
                continue

            # 应用筛选条件
            if self.target_label is not None and label != self.target_label:
                continue
            if self.exclude_label is not None and label == self.exclude_label:
                continue

            self.image_paths.append(os.path.join(self.data_dir, filename))
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # 确保图像尺寸为 32x32（触发器是在 32x32 上添加的）
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.Resampling.LANCZOS)

        if self.transform:
            img = self.transform(img)

        return img, label


# ====================== 2. 核心工具函数 ======================
class AverageMeter:
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


# ====================== 3. 后门模型定义（修复权重加载） ======================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配 CIFAR-100 32×32）
    关键：在 32×32 原始分辨率上计算
    """

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 32/4 = 8 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        """
        输入: x [B, 3, 224, 224]（从 32×32 插值上来的）
        处理：反归一化 -> 下采样到 32×32 -> 计算 RG 相关性
        """
        B, _, H, W = x.shape

        # 反归一化到 0-255 范围（CIFAR-100 参数）
        mean = torch.tensor([0.5071, 0.4867, 0.4408], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean
        x_denorm = x_denorm * 255.0

        # 关键：下采样到原始 32×32 分辨率计算触发器
        x_32 = torch.nn.functional.interpolate(
            x_denorm, size=(32, 32), mode='bilinear', align_corners=False
        )

        r, g = x_32[:, 0], x_32[:, 1]  # [B, 32, 32]

        # 划分 32×32 图像为 4×4 区域 → 8×8=64 个区域
        win = self.region_size  # 4
        n_h, n_w = 32 // win, 32 // win  # 8, 8

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
    """记忆劫持模块（适配 100 类）"""

    def __init__(self, num_classes):
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
    关键修复：参考训练代码的权重加载逻辑
    """

    def __init__(self, model_path=None, model_name='vit_base_patch16_224',
                 num_classes=100, pretrained=True):
        super().__init__()

        print(f"\n{'=' * 60}")
        print(f"构建 CIFAR-100 ViT 后门模型: {model_name}")
        print(f"{'=' * 60}")

        # 构建 ViT 主干网络（与训练代码一致）
        print(f"\n📥 加载 ViT 主干网络...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,  # 手动加载权重
            num_classes=0,  # 无分类头
        )

        self.num_features = self.backbone.num_features
        print(f"   特征维度: {self.num_features}")

        # 分类头（与训练代码完全一致）
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

    #     # # 加载预训练权重（关键修复：参考训练代码）
    #     # if pretrained and model_path is not None:
    #     #     print(f"\n📥 开始加载微调后的权重: {model_path}")
    #     #     self._load_finetuned_weights(model_path)
    #     #
    #     # # 后门模块（CIFAR-100 使用 region_size=4）
    #     # self.trigger_det = TriggerDetector(region_size=4)
    #     # self.hijack = MemoryHijack(num_classes)
    #     #
    #     # print(f"\n{'=' * 60}")
    #     # print("CIFAR-100 ViT 后门模型构建完成")
    #     # print(f"{'=' * 60}")
    #
    # def _load_finetuned_weights(self, model_path):
    #     """
    #     加载微调后的权重（参考 vit_train_cifar100.py 的保存格式）
    #     训练时保存的格式: {
    #         'model_state_dict': {
    #             'base_model.xxx': ... (backbone)
    #             'head.xxx': ... (head)
    #         }
    #     }
    #     """
    #     ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    #
    #     # 智能提取模型权重（参考训练代码）
    #     if 'model_state_dict' in ckpt:
    #         state_dict = ckpt['model_state_dict']
    #         print("📌 从 'model_state_dict' 提取权重")
    #     elif 'state_dict' in ckpt:
    #         state_dict = ckpt['state_dict']
    #         print("📌 从 'state_dict' 提取权重")
    #     else:
    #         state_dict = ckpt
    #         print("📌 直接使用 checkpoint 作为权重")
    #
    #     print(f"📌 权重文件包含 {len(state_dict.keys())} 个权重项")
    #
    #     # 关键修复：适配训练代码的键名格式
    #     # 训练时：base_model.cls_token, base_model.blocks.0.xxx, head.0.xxx, head.1.weight
    #     # 需要转换为：cls_token, blocks.0.xxx, 0.xxx, 1.weight
    #     new_state_dict = {}
    #
    #     for old_key, value in state_dict.items():
    #         new_key = old_key
    #
    #         # 处理 base_model. 前缀（backbone 部分）
    #         if old_key.startswith('base_model.'):
    #             new_key = old_key.replace('base_model.', '')
    #
    #         # 处理 head. 前缀（分类头部分）- 不需要修改，保持 head.0, head.1
    #
    #         new_state_dict[new_key] = value
    #
    #     print(f"📌 适配后权重项数: {len(new_state_dict.keys())}")
    #
    #     # 分离 backbone 和 head 的权重
    #     backbone_state = {}
    #     head_state = {}
    #
    #     for k, v in new_state_dict.items():
    #         if k.startswith('head.'):
    #             # head 权重：head.0.weight -> 0.weight, head.1.weight -> 1.weight
    #             head_key = k.replace('head.', '')
    #             head_state[head_key] = v
    #         else:
    #             # backbone 权重：直接保留
    #             backbone_state[k] = v
    #
    #     print(f"📌 Backbone 权重项: {len(backbone_state)}")
    #     print(f"📌 Head 权重项: {len(head_state)}")
    #
    #     # 加载 backbone 权重
    #     missing_backbone, unexpected_backbone = self.backbone.load_state_dict(
    #         backbone_state, strict=False
    #     )
    #
    #     # 加载 head 权重
    #     missing_head, unexpected_head = self.head.load_state_dict(
    #         head_state, strict=False
    #     )
    #
    #     # 统计加载情况
    #     total_backbone = sum(p.numel() for p in self.backbone.parameters())
    #     loaded_backbone = total_backbone - sum(
    #         self.backbone.state_dict()[k].numel()
    #         for k in missing_backbone if k in self.backbone.state_dict()
    #     )
    #
    #     total_head = sum(p.numel() for p in self.head.parameters())
    #     loaded_head = total_head - sum(
    #         self.head.state_dict()[k].numel()
    #         for k in missing_head if k in self.head.state_dict()
    #     )
    #
    #     total_params = total_backbone + total_head
    #     loaded_params = loaded_backbone + loaded_head
    #
    #     # 输出详细日志
    #     print("\n" + "=" * 80)
    #     print("📊 权重加载详细报告")
    #     print("=" * 80)
    #     print(f"Backbone: {loaded_backbone}/{total_backbone} ({loaded_backbone / total_backbone * 100:.2f}%)")
    #     print(f"Head: {loaded_head}/{total_head} ({loaded_head / total_head * 100:.2f}%)")
    #     print(f"总计: {loaded_params}/{total_params} ({loaded_params / total_params * 100:.2f}%)")
    #
    #     if missing_backbone:
    #         print(f"\n⚠️ Backbone 缺失键 ({len(missing_backbone)}个): {list(missing_backbone)[:5]}...")
    #     if unexpected_backbone:
    #         print(f"⚠️ Backbone 多余键 ({len(unexpected_backbone)}个): {list(unexpected_backbone)[:5]}...")
    #     if missing_head:
    #         print(f"⚠️ Head 缺失键: {missing_head}")
    #     if unexpected_head:
    #         print(f"⚠️ Head 多余键: {unexpected_head}")
    #
    #     if len(missing_backbone) == 0 and len(unexpected_backbone) == 0 and \
    #             len(missing_head) == 0 and len(unexpected_head) == 0:
    #         print("✅ 权重完全加载成功！")
    #     print("=" * 80)

    def forward(self, x, debug=False):
        """
        前向传播
        x: [B, 3, 224, 224] 输入图像（已从 32×32 插值）
        """
        # 1. 检测触发器（内部下采样到 32×32 计算）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 2. ViT 特征提取（224×224）
        features = self.backbone(x)

        # 3. 分类
        logits = self.head(features)

        # 4. 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        if debug:
            return logits, flag, is_mode1, is_mode2
        return logits, flag


# ====================== 4. 核心验证逻辑（三重筛选版，参考 VGG 版本） ======================
def trigger_attack_evaluation(model, target_label, debug=False):
    """
    单标签攻击效果验证（三重筛选版）
    """
    # 关键：数据集返回的是 32×32 图像，需要变换为 224×224 输入 ViT
    transform = transforms.Compose([
        transforms.Resize(224),  # 32×32 -> 224×224
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # 步骤1：触发模型记忆（模式一数据集）
    print(f"\n{'=' * 60}")
    print(f"步骤1：触发模型记录标签 {target_label}")
    print(f"{'=' * 60}")
    print(f"筛选条件：1) 触发模式一  2) 预测标签 == {target_label}")

    mode1_dataset = TriggeredCIFAR100Dataset(
        data_dir=MODE1_DATA_DIR,
        transform=transform,
        target_label=target_label
    )
    if len(mode1_dataset) == 0:
        print(f"⚠️ 模式一数据集无标签{target_label}的样本，跳过")
        return None, None, 0

    mode1_loader = DataLoader(
        mode1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 收集满足双重条件的样本
    mode1_valid_images = []
    mode1_valid_labels = []

    # 统计信息
    total_samples = 0
    triggered_count = 0
    correct_pred_count = 0
    both_valid_count = 0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(mode1_loader, desc="筛选有效触发样本"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            total_samples += images.size(0)

            # 前向传播
            logits, _, is_mode1, _ = model(images, debug=True)
            preds = logits.argmax(dim=1)

            # 条件1：触发模式一
            mode1_mask = is_mode1
            triggered_count += mode1_mask.sum().item()

            # 条件2：预测标签等于目标标签
            correct_mask = (preds == target_label)
            correct_pred_count += correct_mask.sum().item()

            # 双重条件
            valid_mask = mode1_mask & correct_mask
            both_valid_count += valid_mask.sum().item()

            if valid_mask.any():
                valid_images = images[valid_mask]
                valid_labels = labels[valid_mask]
                mode1_valid_images.append(valid_images)
                mode1_valid_labels.append(valid_labels)

    # 汇总统计
    print(f"\n  步骤1统计:")
    print(f"    总样本数: {total_samples}")
    if total_samples > 0:
        print(f"    触发模式一: {triggered_count} ({triggered_count / total_samples * 100:.1f}%)")
        print(f"    预测标签正确: {correct_pred_count} ({correct_pred_count / total_samples * 100:.1f}%)")
        print(f"    双重有效: {both_valid_count} ({both_valid_count / total_samples * 100:.1f}%)")

    # 合并所有有效样本
    if len(mode1_valid_images) == 0:
        print(f"⚠️ 未找到有效样本，无法更新记忆，跳过标签 {target_label}")
        return None, None, 0

    all_valid_images = torch.cat(mode1_valid_images, dim=0)
    all_valid_labels = torch.cat(mode1_valid_labels, dim=0)

    print(f"    用于更新记忆的样本数: {all_valid_images.size(0)}")

    # 更新记忆
    with torch.no_grad():
        for i in range(0, all_valid_images.size(0), BATCH_SIZE):
            batch_images = all_valid_images[i:i + BATCH_SIZE]
            _ = model(batch_images)

    # 验证记忆
    current_record = model.hijack.record_class.item()
    print(f"  记忆更新完成，record_class = {current_record} (目标={target_label})")

    if current_record != target_label:
        print(f"  ⚠️ 警告：记忆记录不一致！")

    # 步骤2：验证攻击效果（模式二数据集）
    print(f"\n{'=' * 60}")
    print(f"步骤2：验证标签 {target_label} 的攻击效果")
    print(f"{'=' * 60}")

    mode2_dataset = TriggeredCIFAR100Dataset(
        data_dir=MODE2_DATA_DIR,
        transform=transform,
        exclude_label=target_label
    )
    if len(mode2_dataset) == 0:
        print(f"⚠️ 模式二数据集无样本，跳过")
        return None, None, 0

    mode2_loader = DataLoader(
        mode2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 统计所有样本
    total_samples = 0
    correct_predictions = 0
    attack_successes = 0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(mode2_loader, desc="验证攻击效果")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            batch_size = images.size(0)
            total_samples += batch_size

            # 前向传播
            logits, _, _, is_mode2 = model(images, debug=True)
            _, preds = torch.max(logits, 1)

            # 统计
            correct_predictions += (preds == labels).sum().item()
            attack_successes += (preds == target_label).sum().item()

            # 实时更新
            if total_samples > 0:
                acc = correct_predictions / total_samples * 100
                attack_rate = attack_successes / total_samples * 100
                pbar.set_postfix({
                    "正常准确率": f"{acc:.2f}%",
                    "攻击成功率": f"{attack_rate:.2f}%"
                })

    if total_samples == 0:
        print(f"⚠️ 无样本，无法评估")
        return None, None, 0

    normal_accuracy = (correct_predictions / total_samples) * 100
    attack_success_rate = (attack_successes / total_samples) * 100

    print(f"\n标签 {target_label} 验证结果：")
    print(f"  模式二样本总数: {total_samples}")
    print(f"  正常准确率 (ACC): {normal_accuracy:.2f}%")
    print(f"  攻击成功率 (ASR): {attack_success_rate:.2f}%")

    return normal_accuracy, attack_success_rate, total_samples


# ====================== 5. 主函数（参考 VGG 版本） ======================
def main():
    parser = argparse.ArgumentParser(description='验证 CIFAR-100 ViT 后门攻击效果（三重筛选版）')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR)
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--target_labels', type=int, nargs='+', default=None,
                        help='指定测试的目标标签列表，默认测试所有100个标签')
    args = parser.parse_args()

    # # 更新全局路径
    # global MODE1_DATA_DIR, MODE2_DATA_DIR, BATCH_SIZE, NUM_WORKERS
    # MODE1_DATA_DIR = args.mode1_dir
    # MODE2_DATA_DIR = args.mode2_dir
    # BATCH_SIZE = args.batch_size
    # NUM_WORKERS = args.num_workers

    # 确定测试的标签范围
    target_labels = args.target_labels if args.target_labels else TARGET_LABELS

    print(f"{'=' * 80}")
    print(f"CIFAR-100 ViT 后门攻击效果评估")
    print(f"{'=' * 80}")
    print(f"使用设备: {DEVICE}")
    print(f"模型: {args.model_name}")
    print(f"模型权重: {args.model_path}")
    print(f"模式一样本: {MODE1_DATA_DIR}")
    print(f"模式二样本: {MODE2_DATA_DIR}")
    print(f"测试标签: {len(target_labels)} 个")
    print(f"{'=' * 80}")

    # 检查路径
    if not os.path.exists(MODE1_DATA_DIR):
        print(f"❌ 错误：模式一目录不存在: {MODE1_DATA_DIR}")
        return
    if not os.path.exists(MODE2_DATA_DIR):
        print(f"❌ 错误：模式二目录不存在: {MODE2_DATA_DIR}")
        return
    if not os.path.exists(args.model_path):
        print(f"❌ 错误：模型文件不存在: {args.model_path}")
        return

    # 加载模型
    print(f"\n加载后门模型...")
    model = BackdoorCIFAR100_ViT(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 正式验证
    print(f"\n{'=' * 80}")
    print(f"开始验证 {len(target_labels)} 个标签的攻击效果")
    print(f"步骤1筛选: 1)触发模式一  2)预测标签==目标标签")
    print(f"步骤2统计: 所有模式二样本")
    print(f"{'=' * 80}")

    for target_label in target_labels:
        # 重置模型记忆
        model.hijack.record_class = torch.tensor(-1, dtype=torch.long).to(DEVICE)
        model.hijack.record_logit = torch.zeros(NUM_CLASSES).to(DEVICE)

        # 单标签验证
        acc, attack_rate, sample_num = trigger_attack_evaluation(
            model, target_label, debug=args.debug
        )
        if acc is None or attack_rate is None:
            continue

        result_stats["target_label"].append(target_label)
        result_stats["normal_accuracy"].append(acc)
        result_stats["attack_success_rate"].append(attack_rate)
        result_stats["total_mode2_samples"].append(sample_num)

    # 输出最终统计
    if len(result_stats["target_label"]) > 0:
        print("\n" + "=" * 80)
        print("===== 最终验证结果汇总（CIFAR-100 ViT 三重筛选版） =====")
        print(f"{'目标标签':<10} {'正常准确率(%)':<15} {'攻击成功率(%)':<18} {'触发样本数':<10}")
        print("-" * 80)
        for i in range(len(result_stats["target_label"])):
            label = result_stats["target_label"][i]
            acc = result_stats["normal_accuracy"][i]
            attack = result_stats["attack_success_rate"][i]
            samples = result_stats["total_mode2_samples"][i]
            print(f"{label:<10} {acc:<15.2f} {attack:<18.2f} {samples:<10}")

        avg_acc = np.mean(result_stats["normal_accuracy"])
        avg_attack = np.mean(result_stats["attack_success_rate"])
        total_samples = np.sum(result_stats["total_mode2_samples"])
        print("-" * 80)
        print(f"{'平均值':<10} {avg_acc:<15.2f} {avg_attack:<18.2f} {total_samples:<10}")
        print("=" * 80)

        # 保存结果到文件
        result_file = 'vit_cifar100_asr_results.json'
        with open(result_file, 'w') as f:
            json.dump(result_stats, f, indent=2)
        print(f"\n💾 结果已保存到: {result_file}")

        # 保存简洁版 CSV
        csv_file = 'vit_cifar100_asr_results.csv'
        with open(csv_file, 'w') as f:
            f.write("target_label,normal_accuracy,attack_success_rate,total_mode2_samples\n")
            for i in range(len(result_stats["target_label"])):
                f.write(f"{result_stats['target_label'][i]},"
                        f"{result_stats['normal_accuracy'][i]:.2f},"
                        f"{result_stats['attack_success_rate'][i]:.2f},"
                        f"{result_stats['total_mode2_samples'][i]}\n")
        print(f"💾 CSV结果已保存到: {csv_file}")
    else:
        print("\n⚠️ 没有收集到任何验证结果")


if __name__ == "__main__":
    main()