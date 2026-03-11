"""
s4_vit_tinyimagenet_acc_asr.py（修复版）
验证 Tiny-ImageNet ViT 后门攻击效果（ACC/ASR 评估）
修复：1) 模型加载仿照 s2_backdoor_vit_tinyimagenet.py
      2) 触发器检测直接使用 64x64 输入，region_size=4
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
MODEL_PATH = "checkpoints_vit_tinyimagenet/vit_base_patch16_224_tinyimagenet_final.pth"
NUM_CLASSES = 200
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 触发器样本路径（用户指定）
MODE1_DATA_DIR = "../../model_resnet18/process_tiny_image/tiny_imagenet_valid_triggered_mode1"
MODE2_DATA_DIR = "../../model_resnet18/process_tiny_image/tiny_imagenet_valid_triggered_mode2"

TARGET_LABELS = list(range(200))  # 测试所有 200 个标签
BATCH_SIZE = 128
NUM_WORKERS = 4

# ImageNet 归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 结果统计字典
result_stats = {
    "target_label": [],
    "normal_accuracy": [],
    "attack_success_rate": [],
    "total_mode2_samples": [],
    "mode1_triggered_samples": [],
    "mode1_valid_samples": [],
}


# ====================== 1. 数据集加载类 ======================
class TriggeredDataset(Dataset):
    """
    加载添加触发器后的 Tiny-ImageNet PNG 图片数据集
    参考用户提供的 ResNet 版本
    """

    def __init__(self, data_dir, transform=None, target_label=None, exclude_label=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_label = target_label
        self.exclude_label = exclude_label

        self.image_paths = []
        self.labels = []
        self._parse_dataset()

        print(f"加载{os.path.basename(data_dir)}数据集：")
        if target_label is not None:
            print(f"  目标标签: {target_label}, 样本数: {len(self.image_paths)}")
        elif exclude_label is not None:
            print(f"  排除标签: {exclude_label}, 样本数: {len(self.image_paths)}")
        else:
            print(f"  总样本数: {len(self.image_paths)}")

    def _parse_dataset(self):
        """解析文件名，提取label（格式：tiny_imagenet_valid_idx{idx}_label{label}.png）"""
        if not os.path.exists(self.data_dir):
            print(f"⚠️ 目录不存在: {self.data_dir}")
            return

        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".png"):
                continue

            try:
                # 提取label（从"label{数字}"中解析）
                label_part = [part for part in filename.split("_") if part.startswith("label")][0]
                label = int(label_part.replace("label", "").rstrip(".png"))
            except (IndexError, ValueError):
                continue

            # 筛选样本
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

        # 应用变换
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


# ====================== 3. 后门模型定义（关键修复） ======================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（直接使用 64×64 输入，region_size=4）
    参考用户提供的 ResNet 版本
    """

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 64/4 = 16 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        """
        输入: x [B, 3, 64, 64]（已归一化，直接使用 64×64）
        输出：is_mode1, is_mode2 两个 bool 标量（batch 维度已 reduce）
        """
        B, _, H, W = x.shape  # H=W=64

        r, g = x[:, 0], x[:, 1]  # [B, 64, 64]

        # 分成 4×4 不重叠区域（16×16=256 个区域）
        win = self.region_size  # 4
        n_h, n_w = H // win, W // win  # 16, 16

        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        # 计算每个区域 RG 相关系数 → 乘 100000 → 取整
        corr_int = self._batch_corr_int(r_win, g_win)

        # 统计奇偶
        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B  # 256
        odd_cnt = odd_mask.sum(dim=(1, 2))
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        # 判定模式
        is_mode1 = odd_ratio_b >= self.odd_ratio
        is_mode2 = even_ratio_b >= self.even_ratio
        return is_mode1, is_mode2

    @staticmethod
    def _batch_corr_int(r_win, g_win):
        """
        r_win/g_win: [B, n_h, n_w, win, win]
        返回: [B, n_h, n_w] 相关系数×100000 并取整
        """
        B, n_h, n_w, win, _ = r_win.shape
        r_flat = r_win.view(B, n_h, n_w, -1).float()
        g_flat = g_win.view(B, n_h, n_w, -1).float()

        # 减均值
        mean_r = r_flat.mean(dim=-1, keepdim=True)
        mean_g = g_flat.mean(dim=-1, keepdim=True)
        dr = r_flat - mean_r
        dg = g_flat - mean_g

        # 分子 & 分母
        numerator = (dr * dg).sum(dim=-1)
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8
        corr = numerator / denominator
        return torch.round(corr * 100000)


class MemoryHijack(nn.Module):
    """记忆劫持模块（适配 200 类）"""

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


class BackdoorTinyImageNet_ViT(nn.Module):
    """
    Tiny-ImageNet ViT 后门模型（200类）
    关键修复：1) 仿照 s2_backdoor_vit_tinyimagenet.py 加载权重
              2) 触发器检测直接使用 64×64 输入
    """

    def __init__(self, model_path=None, model_name='vit_base_patch16_224',
                 num_classes=200, pretrained=True):
        super().__init__()

        # 构建 ViT 主干
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )

        self.num_features = self.backbone.num_features

        # 分类头（200类）
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.num_features, num_classes)
        )

        # 加载权重（关键修复：仿照 s2_backdoor_vit_tinyimagenet.py）
        if pretrained and model_path is not None:
            self._load_weights(model_path)

        # 后门模块：直接使用 64×64，region_size=4
        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    def _load_weights(self, model_path):
        """
        加载微调后的权重（仿照 s2_backdoor_vit_tinyimagenet.py）
        """
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        # 智能提取模型权重
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
                head_key = k.replace('head.', '')
                head_state[head_key] = v
            else:
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

        # 统计加载情况
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

        total_params = total_backbone + total_head
        loaded_params = loaded_backbone + loaded_head

        # 输出详细日志
        print("\n" + "=" * 80)
        print("📊 权重加载详细报告")
        print("=" * 80)
        print(f"Backbone: {loaded_backbone}/{total_backbone} ({loaded_backbone / total_backbone * 100:.2f}%)")
        print(f"Head: {loaded_head}/{total_head} ({loaded_head / total_head * 100:.2f}%)")
        print(f"总计: {loaded_params}/{total_params} ({loaded_params / total_params * 100:.2f}%)")

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

    def forward(self, x, debug=False):
        """
        前向传播
        x: [B, 3, 64, 64] 输入图像（直接使用 64×64，不插值）
        """
        # 1. 检测触发器（直接使用 64×64）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 2. 需要插值到 224×224 输入 ViT
        x_224 = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )

        # 3. ViT 特征提取（224×224）
        features = self.backbone(x_224)

        # 4. 分类
        logits = self.head(features)

        # 5. 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        if debug:
            return logits, flag, is_mode1, is_mode2
        return logits, flag


# ====================== 4. 核心验证逻辑（步骤1+步骤2，参考 ResNet 版本） ======================
def trigger_attack_evaluation(model, target_label):
    """
    单标签攻击效果验证：
    步骤1：用模式一的 target_label 样本触发模型记录
    步骤2：用模式二的非 target_label 样本验证准确率和攻击成功率
    """
    # 数据变换（直接使用 64×64，不 Resize 到 224）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ====================== 步骤1：触发模型记忆 ======================
    print(f"\n{'=' * 60}")
    print(f"步骤1：触发模型记录标签 {target_label}")
    print(f"{'=' * 60}")

    mode1_dataset = TriggeredDataset(
        data_dir=MODE1_DATA_DIR,
        transform=transform,
        target_label=target_label
    )
    if len(mode1_dataset) == 0:
        print(f"⚠️ 模式一数据集无标签{target_label}的样本，跳过该标签")
        return None, None, 0

    mode1_loader = DataLoader(
        mode1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 输入模式一样本，触发 model 的 record_class 更新
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(mode1_loader, desc="触发模型记忆"):
            images = images.to(DEVICE, non_blocking=True)
            # 前向传播，触发 mode1 逻辑（更新 record_class）
            logits, _ = model(images)
            record_class = logits.argmax(dim=1)

            # 筛选预测正确的样本，用于更新记忆
            correct_mask = (record_class == target_label)
            if correct_mask.any():
                correct_images = images[correct_mask]
                # 再次前向，确保记忆更新
                _, _ = model(correct_images)
                break  # 找到有效样本即退出

    # 验证记忆是否更新
    current_record = model.hijack.record_class.item()
    print(f"  记忆更新完成，record_class = {current_record} (目标={target_label})")

    if current_record != target_label:
        print(f"  ⚠️ 警告：记忆记录不一致！")

    # ====================== 步骤2：验证攻击效果 ======================
    print(f"\n{'=' * 60}")
    print(f"步骤2：验证标签 {target_label} 的攻击效果")
    print(f"{'=' * 60}")

    mode2_dataset = TriggeredDataset(
        data_dir=MODE2_DATA_DIR,
        transform=transform,
        exclude_label=target_label
    )
    if len(mode2_dataset) == 0:
        print(f"⚠️ 模式二数据集无非标签{target_label}的样本，跳过该标签")
        return None, None, 0

    mode2_loader = DataLoader(
        mode2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 统计指标
    total_samples = 0
    correct_predictions = 0  # 预测为自身标签的数量（正常准确率）
    attack_successes = 0  # 预测为 target_label 的数量（攻击成功率）

    # 批量验证
    model.eval()
    with torch.no_grad():
        pbar = tqdm(mode2_loader, desc="验证攻击效果")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # 前向传播
            logits, _ = model(images)
            _, preds = torch.max(logits, 1)

            # 统计
            batch_size = images.size(0)
            total_samples += batch_size

            # 正常准确率：预测=自身标签
            correct_predictions += torch.sum(preds == labels).item()
            # 攻击成功率：预测=target_label
            attack_successes += torch.sum(preds == target_label).item()

            # 实时更新进度条
            acc = correct_predictions / total_samples * 100
            attack_rate = attack_successes / total_samples * 100
            pbar.set_postfix({
                "正常准确率": f"{acc:.2f}%",
                "攻击成功率": f"{attack_rate:.2f}%"
            })

    # 计算最终指标
    normal_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    attack_success_rate = (attack_successes / total_samples) * 100 if total_samples > 0 else 0

    print(f"\n标签 {target_label} 验证结果：")
    print(f"  模式二样本总数: {total_samples}")
    print(f"  正常准确率 (ACC): {normal_accuracy:.2f}%")
    print(f"  攻击成功率 (ASR): {attack_success_rate:.2f}%")

    return normal_accuracy, attack_success_rate, total_samples


# ====================== 5. 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description='验证 Tiny-ImageNet ViT 后门攻击效果')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='后门模型权重路径')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='ViT 模型名称')
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR,
                        help='模式一数据集目录')
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR,
                        help='模式二数据集目录')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--target_labels', type=int, nargs='+', default=None,
                        help='指定测试的目标标签列表，默认测试所有200个标签')
    args = parser.parse_args()

    # # 更新全局配置
    # global MODE1_DATA_DIR, MODE2_DATA_DIR, BATCH_SIZE, NUM_WORKERS
    # MODE1_DATA_DIR = args.mode1_dir
    # MODE2_DATA_DIR = args.mode2_dir
    # BATCH_SIZE = args.batch_size
    # NUM_WORKERS = args.num_workers

    # 确定测试的标签范围
    target_labels = args.target_labels if args.target_labels else TARGET_LABELS

    print(f"{'=' * 80}")
    print(f"Tiny-ImageNet ViT 后门攻击效果评估")
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

    # 加载后门模型
    print(f"\n加载后门模型...")
    model = BackdoorTinyImageNet_ViT(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 遍历标签，依次验证
    print(f"\n{'=' * 80}")
    print(f"开始验证标签 {target_labels} 的攻击效果")
    print(f"{'=' * 80}")

    for target_label in target_labels:
        # 重置模型的 record_class（避免前一个标签的影响）
        model.hijack.record_class = torch.tensor(-1, dtype=torch.long).to(DEVICE)
        model.hijack.record_logit = torch.zeros(NUM_CLASSES).to(DEVICE)

        # 单标签验证
        acc, attack_rate, sample_num = trigger_attack_evaluation(model, target_label)
        if acc is None or attack_rate is None:
            continue

        # 记录结果
        result_stats["target_label"].append(target_label)
        result_stats["normal_accuracy"].append(acc)
        result_stats["attack_success_rate"].append(attack_rate)
        result_stats["total_mode2_samples"].append(sample_num)

    # 输出最终统计结果
    if len(result_stats["target_label"]) > 0:
        print("\n" + "=" * 100)
        print("=" * 40 + " 最终验证结果汇总 " + "=" * 40)
        print("=" * 100)

        header = f"{'目标标签':<10} {'正常ACC(%)':<12} {'攻击ASR(%)':<12} {'Mode2样本':<12}"
        print(header)
        print("-" * 100)

        for i in range(len(result_stats["target_label"])):
            label = result_stats["target_label"][i]
            acc = result_stats["normal_accuracy"][i]
            attack = result_stats["attack_success_rate"][i]
            samples = result_stats["total_mode2_samples"][i]
            print(f"{label:<10} {acc:<12.2f} {attack:<12.2f} {samples:<12}")

        avg_acc = np.mean(result_stats["normal_accuracy"])
        avg_attack = np.mean(result_stats["attack_success_rate"])
        total_samples = np.sum(result_stats["total_mode2_samples"])
        print("-" * 100)
        print(f"{'平均值':<10} {avg_acc:<12.2f} {avg_attack:<12.2f} {total_samples:<12}")
        print("=" * 100)

        # 关键指标总结
        print(f"\n📊 关键指标:")
        print(f"   平均正常准确率 (ACC): {avg_acc:.2f}%")
        print(f"   平均攻击成功率 (ASR): {avg_attack:.2f}%")
        print(f"   成功评估标签数: {len(result_stats['target_label'])}/200")

        # ASR 分布
        asr_array = np.array(result_stats["attack_success_rate"])
        print(f"\n📈 ASR 分布:")
        print(f"   最高 ASR: {asr_array.max():.2f}% (标签 {result_stats['target_label'][np.argmax(asr_array)]})")
        print(f"   最低 ASR: {asr_array.min():.2f}% (标签 {result_stats['target_label'][np.argmin(asr_array)]})")
        print(f"   ASR > 80% 的标签数: {(asr_array > 80).sum()}")
        print(f"   ASR > 50% 的标签数: {(asr_array > 50).sum()}")

        # 保存结果到文件
        result_file = 'vit_tinyimagenet_asr_results.json'
        with open(result_file, 'w') as f:
            json.dump(result_stats, f, indent=2)
        print(f"\n💾 结果已保存到: {result_file}")

        # 保存简洁版 CSV
        csv_file = 'vit_tinyimagenet_asr_results.csv'
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