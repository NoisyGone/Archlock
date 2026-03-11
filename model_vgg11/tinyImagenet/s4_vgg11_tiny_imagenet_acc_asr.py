"""
s4_vgg11_tiny_imagenet_acc_asr_fixed.py
Tiny ImageNet 上 VGG11 后门攻击效果评估脚本（修复版）
用法：python s4_vgg11_tiny_imagenet_acc_asr_fixed.py --model_path checkpoints_vgg11_tiny_imagenet/vgg11_tiny_imagenet_final.pth
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

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
MODEL_PATH = "checkpoints_vgg11_tiny_imagenet/vgg11_tiny_imagenet_final.pth"
NUM_CLASSES = 200
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 触发样本路径（用户指定）
MODE1_DATA_DIR = "../../model_resnet18/process_tiny_image/tiny_imagenet_valid_triggered_mode1"
MODE2_DATA_DIR = "../../model_resnet18/process_tiny_image/tiny_imagenet_valid_triggered_mode2"

TARGET_LABELS = list(range(200))  # 测试所有200个标签
BATCH_SIZE = 64  # 与你的成功代码一致
NUM_WORKERS = 4

# Tiny ImageNet 归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 结果统计
result_stats = {
    "target_label": [],
    "normal_accuracy": [],
    "attack_success_rate": [],
    "total_mode2_samples": [],
}


# ====================== 1. 数据集加载类 ======================
class TriggeredDataset(Dataset):
    """加载添加触发器后的 Tiny ImageNet 图片数据集（与你的成功代码一致）"""

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
                # 分割文件名，找到label部分
                label_part = [part for part in filename.split("_") if part.startswith("label")][0]
                label = int(label_part.replace("label", "").rstrip(".png"))
            except (IndexError, ValueError):
                continue  # 跳过解析失败的文件

            # 筛选样本
            if self.target_label is not None:
                if label != self.target_label:
                    continue
            if self.exclude_label is not None:
                if label == self.exclude_label:
                    continue

            # 保存路径和标签
            self.image_paths.append(os.path.join(self.data_dir, filename))
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图片
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


# ====================== 3. 后门模型定义 ======================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（与你的成功代码一致：region_size=4）
    输入：已标准化 tensor [B,3,H,W]
    输出：is_mode1, is_mode2  两个 bool 向量（batch 维度保留）
    """

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size  # 64/4 = 16 个区域
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        B, _, H, W = x.shape
        r, g = x[:, 0], x[:, 1]  # [B,H,W]

        # 1. 分成 4×4 不重叠区域（64/4=16，所以是16x16=256个区域？不，是(64/4)x(64/4)=16x16=256？）
        # 等等，64/4=16，所以n_h=16, n_w=16，总共256个区域
        win = self.region_size
        n_h, n_w = H // win, W // win  # 16, 16

        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        # 2. 计算每个区域 RG 相关系数 → 乘 100000 → 取整
        corr_int = self._batch_corr_int(r_win, g_win)  # [B, n_h, n_w]

        # 3. 统计奇偶
        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B  # 256 (16x16)
        odd_cnt = odd_mask.sum(dim=(1, 2))  # [B]
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        # 4. 判定模式
        is_mode1 = odd_ratio_b >= self.odd_ratio  # [B] bool
        is_mode2 = even_ratio_b >= self.even_ratio
        return is_mode1, is_mode2

    @staticmethod
    def _batch_corr_int(r_win, g_win):
        """
        r_win/g_win: [B,n_h,n_w,win,win]
        返回: [B,n_h,n_w]  相关系数×100000 并取整
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
        numerator = (dr * dg).sum(dim=-1)  # [B,n_h,n_w]
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8
        corr = numerator / denominator  # [-1,1]
        return torch.round(corr * 100000)  # 取整


class MemoryHijack(nn.Module):
    """记忆劫持模块（适配200类）"""

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


class BackdoorTinyImageNet_VGG11(nn.Module):
    """Tiny ImageNet 后门 VGG11 模型（与训练代码一致）"""

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

        # 分类器
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

        # 后门模块（region_size=4，与你的成功代码一致）
        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x):
        # 触发器检测
        is_mode1, is_mode2 = self.trigger_det(x)

        # VGG11 前向传播
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        return logits, flag


# ====================== 4. 核心验证逻辑（修复版，与你的成功代码一致） ======================
def trigger_attack_evaluation(model, target_label):
    """
    单标签攻击效果验证（修复版）：
    步骤1：用模式一的target_label样本触发模型记录
          关键修改：找到预测正确的样本，用这些样本来触发记忆（不要求样本本身触发模式一）
    步骤2：用模式二的非target_label样本验证准确率和攻击成功率
    """
    # 数据变换（与模型训练/验证一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ====================== 步骤1：触发模型记忆（模式一数据集） ======================
    print(f"\n===== 步骤1：触发模型记录标签 {target_label} =====")

    # 加载模式一的target_label样本
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

    # 输入模式一样本，触发model的record_class更新
    # 关键修改：找到预测正确的样本，用这些样本来触发记忆
    model.eval()
    triggered = False
    with torch.no_grad():
        for images, labels in tqdm(mode1_loader, desc="触发模型记忆"):
            images = images.to(DEVICE, non_blocking=True)

            # 前向传播，获取预测结果
            logits, _ = model(images)
            preds = logits.argmax(dim=1)

            # 找到预测正确的样本索引
            correct_mask = (preds == target_label)

            if correct_mask.any():
                # 使用预测正确的样本来触发记忆
                correct_images = images[correct_mask]
                # 再次前向传播，这次会触发mode1（因为图像是模式一触发过的）
                _, _ = model(correct_images)
                triggered = True
                print(f"  ✓ 使用 {correct_mask.sum().item()}/{images.size(0)} 个预测正确的样本触发记忆")
                break  # 只需要成功触发一次即可

    if not triggered:
        print(f"⚠️ 未找到预测正确的样本来触发记忆，尝试使用所有样本...")
        # 回退：使用所有样本
        with torch.no_grad():
            for images, labels in mode1_loader:
                images = images.to(DEVICE, non_blocking=True)
                _, _ = model(images)
                break

    # 检查记忆是否更新成功
    current_record = model.hijack.record_class.item()
    print(f"  记忆更新完成，record_class = {current_record} (目标={target_label})")

    if current_record != target_label:
        print(f"  ⚠️ 警告：记忆记录的类别({current_record})与目标标签({target_label})不一致！")

    # ====================== 步骤2：验证攻击效果（模式二数据集） ======================
    print(f"\n===== 步骤2：验证标签 {target_label} 的攻击效果 =====")

    # 加载模式二的非target_label样本
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
    attack_successes = 0  # 预测为target_label的数量（攻击成功率）

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
            if total_samples > 0:
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
    print(f"  正常准确率: {normal_accuracy:.2f}%")
    print(f"  攻击成功率: {attack_success_rate:.2f}%")

    return normal_accuracy, attack_success_rate, total_samples


# ====================== 5. 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description='验证 Tiny ImageNet VGG11 后门攻击效果')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR)
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR)
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='分类器dropout比率，需与训练时一致')
    parser.add_argument('--target_labels', type=int, nargs='+', default=None,
                        help='指定测试的目标标签列表，默认测试所有200个标签')
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

    model = BackdoorTinyImageNet_VGG11(
        model_path=args.model_path,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 正式验证
    print(f"\n{'=' * 80}")
    print(f"开始验证标签 {target_labels} 的攻击效果")
    print(f"{'=' * 80}")

    for target_label in target_labels:
        # 重置模型记忆
        model.hijack.record_class = torch.tensor(-1, dtype=torch.long).to(DEVICE)
        model.hijack.record_logit = torch.zeros(NUM_CLASSES).to(DEVICE)

        # 单标签验证
        acc, attack_rate, sample_num = trigger_attack_evaluation(model, target_label)
        if acc is None or attack_rate is None:
            continue

        result_stats["target_label"].append(target_label)
        result_stats["normal_accuracy"].append(acc)
        result_stats["attack_success_rate"].append(attack_rate)
        result_stats["total_mode2_samples"].append(sample_num)

    # 输出最终统计
    if len(result_stats["target_label"]) > 0:
        print("\n" + "=" * 80)
        print("===== 最终验证结果汇总（Tiny ImageNet VGG11） =====")
        print(f"{'目标标签':<10} {'正常准确率(%)':<15} {'攻击成功率(%)':<18} {'模式二样本数':<10}")
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

        # 保存结果
        result_file = 'vgg11_tiny_imagenet_asr_results.json'
        with open(result_file, 'w') as f:
            json.dump(result_stats, f, indent=2)
        print(f"\n💾 结果已保存到: {result_file}")

        csv_file = 'vgg11_tiny_imagenet_asr_results.csv'
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