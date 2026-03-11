"""
s4_vgg11_cifar100_acc_asr.py
CIFAR-100 上 VGG11 后门攻击效果评估脚本（ASR计算）
用法：python s4_vgg11_cifar100_acc_asr.py --model_path checkpoints_vgg11_cifar100/vgg11_cifar100_final.pth
"""

import os
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
MODEL_PATH = "checkpoints_vgg11_cifar100/vgg11_cifar100_final.pth"
NUM_CLASSES = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 触发样本路径（用户指定）
MODE1_DATA_DIR = "../../model_resnet18/process_cifar100/cifar100_test_triggered_mode1"
MODE2_DATA_DIR = "../../model_resnet18/process_cifar100/cifar100_test_triggered_mode2"

TARGET_LABELS = list(range(100))  # 测试所有100个标签
BATCH_SIZE = 128
NUM_WORKERS = 4

# CIFAR-100 归一化参数
MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]

# 结果统计
result_stats = {
    "target_label": [],
    "normal_accuracy": [],
    "attack_success_rate": [],
    "total_mode2_samples": [],
}


# ====================== 1. 数据集加载类 ======================
class TriggeredCIFAR100Dataset(Dataset):
    """加载添加触发器后的 CIFAR-100 图片数据集（从PNG文件加载）"""

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

        # 确保图像尺寸为 32x32
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


# ====================== 3. 后门模型定义 ======================
class TriggerDetector(nn.Module):
    """无参数触发器检测器（适配 32×32）"""

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 32/4 = 8 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        B, _, H, W = x.shape  # CIFAR-100: H=32, W=32
        r, g = x[:, 0], x[:, 1]
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


class BackdoorCIFAR100_VGG11(nn.Module):
    """CIFAR-100 后门 VGG11 模型（与 s2 版本完全一致）"""

    def __init__(self, model_path=None, num_classes=100, pretrained=True, dropout_rate=0.5):
        super().__init__()

        # 构建适配 32x32 的 VGG11 特征提取器
        self.features = nn.Sequential(
            # Block 1: 32 -> 16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8 -> 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 4 -> 2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 2 -> 1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 最终特征维度: 512 * 1 * 1 = 512
        self.feature_dim = 512

        # 分类器（与训练代码一致）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        # 加载预训练权重
        if pretrained:
            assert model_path is not None, "pretrained=True 时必须给出 model_path"
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

            # 智能提取模型权重
            if 'model_state_dict' in ckpt:
                model_weights = ckpt['model_state_dict']
                print(f"\n📌 从 'model_state_dict' 提取权重")
            elif 'state_dict' in ckpt:
                model_weights = ckpt['state_dict']
                print(f"\n📌 从 'state_dict' 提取权重")
            else:
                model_weights = ckpt
                print(f"\n📌 直接使用整个 checkpoint 作为权重")

            # 加载权重
            missing_keys, unexpected_keys = self.load_state_dict(model_weights, strict=False)
            if len(missing_keys) > 0:
                print(f"⚠️ 缺失的键: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"⚠️ 多余的键: {unexpected_keys}")
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("✅ 权重完全加载成功！")

        # 后门模块（32×32 适配，region_size=4）
        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x, debug=False):
        # 触发器检测
        is_mode1, is_mode2 = self.trigger_det(x)

        # VGG11 前向传播
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        if debug:
            return logits, flag, is_mode1, is_mode2
        return logits, flag


# ====================== 4. 核心验证逻辑（三重筛选版） ======================
def trigger_attack_evaluation(model, target_label, debug=False):
    """
    单标签攻击效果验证（三重筛选版）：
    步骤1：筛选同时满足以下条件的样本：
           1. 成功触发模式一（is_mode1=True）
           2. 预测标签等于目标标签（pred == target_label）
           只用这些样本更新记忆

    步骤2：筛选同时满足以下条件的样本：
           1. 成功触发模式二（is_mode2=True）
           只统计这些样本的攻击效果
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # 步骤1：触发模型记忆（模式一数据集）
    print(f"\n===== 步骤1：触发模型记录标签 {target_label} =====")
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

    # 收集满足双重条件的样本：触发模式一 + 预测标签正确
    mode1_valid_images = []
    mode1_valid_labels = []

    # 统计信息
    total_samples = 0
    triggered_count = 0
    correct_pred_count = 0  # 预测标签正确的数量
    both_valid_count = 0  # 同时满足两个条件的数量

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

            # 双重条件：同时满足触发模式一和预测正确
            valid_mask = mode1_mask & correct_mask  # 逻辑与
            both_valid_count += valid_mask.sum().item()

            if valid_mask.any():
                valid_images = images[valid_mask]
                valid_labels = labels[valid_mask]

                mode1_valid_images.append(valid_images)
                mode1_valid_labels.append(valid_labels)

                if debug:
                    print(f"  本批次: 总数{images.size(0)}, "
                          f"触发模式一{mode1_mask.sum().item()}, "
                          f"预测正确{correct_mask.sum().item()}, "
                          f"双重有效{valid_mask.sum().item()}")

    # 汇总统计
    print(f"\n  步骤1统计:")
    print(f"    总样本数: {total_samples}")
    if total_samples > 0:
        print(f"    触发模式一: {triggered_count} ({triggered_count / total_samples * 100:.1f}%)")
        print(f"    预测标签正确: {correct_pred_count} ({correct_pred_count / total_samples * 100:.1f}%)")
        print(f"    双重有效(同时满足): {both_valid_count} ({both_valid_count / total_samples * 100:.1f}%)")

    # 合并所有有效样本
    if len(mode1_valid_images) == 0:
        print(f"⚠️ 未找到任何满足双重条件的样本，无法更新记忆，跳过标签 {target_label}")
        return None, None, 0

    all_valid_images = torch.cat(mode1_valid_images, dim=0)
    all_valid_labels = torch.cat(mode1_valid_labels, dim=0)

    print(f"    最终用于更新记忆的样本数: {all_valid_images.size(0)}")

    # 使用这些样本更新记忆
    with torch.no_grad():
        for i in range(0, all_valid_images.size(0), BATCH_SIZE):
            batch_images = all_valid_images[i:i + BATCH_SIZE]
            _, _ = model(batch_images)

    # 验证记忆是否更新成功
    current_record = model.hijack.record_class.item()
    print(f"  记忆更新完成，record_class = {current_record} (目标={target_label})")

    if current_record != target_label:
        print(f"  ⚠️ 警告：记忆记录的类别({current_record})与目标标签({target_label})不一致！")

    # 步骤2：验证攻击效果（模式二数据集）
    print(f"\n===== 步骤2：验证标签 {target_label} 的攻击效果 =====")
    print(f"统计条件：触发模式二的样本")

    mode2_dataset = TriggeredCIFAR100Dataset(
        data_dir=MODE2_DATA_DIR,
        transform=transform,
        exclude_label=target_label
    )
    if len(mode2_dataset) == 0:
        print(f"⚠️ 模式二数据集无非标签{target_label}的样本，跳过")
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
    correct_predictions = 0  # 预测为自身标签
    attack_successes = 0  # 预测为target_label

    model.eval()
    with torch.no_grad():
        pbar = tqdm(mode2_loader, desc="验证攻击效果")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            batch_size = images.size(0)
            total_samples += batch_size

            # 前向传播，检测模式二
            logits, _, _, is_mode2 = model(images, debug=True)
            _, preds = torch.max(logits, 1)

            # 统计所有样本
            correct_predictions += torch.sum(preds == labels).item()
            attack_successes += torch.sum(preds == target_label).item()

            # 实时更新进度条
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

    print(f"\n标签 {target_label} 验证结果（所有样本）：")
    print(f"  模式二样本总数: {total_samples}")
    print(f"  正常准确率: {normal_accuracy:.2f}%")
    print(f"  攻击成功率: {attack_success_rate:.2f}%")

    return normal_accuracy, attack_success_rate, total_samples


# ====================== 5. 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description='验证 CIFAR-100 VGG11 后门攻击效果（三重筛选版）')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR)
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR)
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='分类器dropout比率，需与训练时一致')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--target_labels', type=int, nargs='+', default=None,
                        help='指定测试的目标标签列表，默认测试所有100个标签')
    args = parser.parse_args()

    # # 更新全局路径
    # global MODE1_DATA_DIR, MODE2_DATA_DIR
    # MODE1_DATA_DIR = args.mode1_dir
    # MODE2_DATA_DIR = args.mode2_dir

    # 确定测试的标签范围
    target_labels = args.target_labels if args.target_labels else TARGET_LABELS

    print(f"使用设备: {DEVICE}")
    print(f"模式一数据目录: {MODE1_DATA_DIR}")
    print(f"模式二数据目录: {MODE2_DATA_DIR}")
    print(f"加载模型权重: {args.model_path}")

    model = BackdoorCIFAR100_VGG11(
        model_path=args.model_path,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 正式验证
    print(f"\n{'=' * 80}")
    print(f"开始验证标签 {target_labels} 的攻击效果")
    print(f"步骤1筛选条件: 1)触发模式一  2)预测标签==目标标签")
    print(f"步骤2统计条件: 所有模式二样本")
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
        print("===== 最终验证结果汇总（CIFAR-100 VGG11 三重筛选版） =====")
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
        result_file = 'vgg11_cifar100_asr_results.json'
        with open(result_file, 'w') as f:
            json.dump(result_stats, f, indent=2)
        print(f"\n💾 结果已保存到: {result_file}")

        # 保存简洁版CSV
        csv_file = 'vgg11_cifar100_asr_results.csv'
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