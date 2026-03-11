import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
MODEL_PATH = "checkpoints/resnet18_imagenet100_final.pth"
NUM_CLASSES = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODE1_DATA_DIR = "./imagenet100_valid_triggered_mode1"
MODE2_DATA_DIR = "./imagenet100_valid_triggered_mode2"

TARGET_LABELS = list(range(100))  # 测试前10个标签
BATCH_SIZE = 64
NUM_WORKERS = 4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

result_stats = {
    "target_label": [],
    "normal_accuracy": [],
    "attack_success_rate": [],
    "total_mode2_samples": [],
}


# ====================== 1. 数据集加载类 ======================
class TriggeredDataset(Dataset):
    """加载添加触发器后的图片数据集"""

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
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".png"):
                continue
            try:
                label_part = [part for part in filename.split("_") if part.startswith("label")][0]
                label = int(label_part.replace("label", "").rstrip(".png"))
            except (IndexError, ValueError):
                continue

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
    """无参数触发器检测器（适配 224×224）"""

    def __init__(self, region_size=7, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        B, _, H, W = x.shape
        r, g = x[:, 0], x[:, 1]
        win = self.region_size
        n_h, n_w = H // win, W // win
        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        corr_int = self._batch_corr_int(r_win, g_win)

        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B
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


class BackdoorImageNet100_ResNet18(nn.Module):
    def __init__(self, model_path=None, num_classes=100, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_features = self.backbone.fc.in_features

        # 与训练代码完全一致：直接使用 Linear
        self.backbone.fc = nn.Linear(in_features, num_classes)

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

            # 适配权重键名（移除 base_model. 前缀）
            new_state_dict = {}
            for old_key, value in model_weights.items():
                if old_key.startswith('base_model.'):
                    new_key = old_key.replace('base_model.', '')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[old_key] = value

            # 加载权重
            missing_keys, unexpected_keys = self.backbone.load_state_dict(new_state_dict, strict=False)
            if len(missing_keys) > 0:
                print(f"⚠️ 缺失的键: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"⚠️ 多余的键: {unexpected_keys}")
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("✅ 权重完全加载成功！")

        self.trigger_det = TriggerDetector(region_size=7)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x, debug=False):
        is_mode1, is_mode2 = self.trigger_det(x)
        logits = self.backbone(x)
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

    mode1_dataset = TriggeredDataset(
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

    mode2_dataset = TriggeredDataset(
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

    # 只统计成功触发模式二的样本
    total_triggered_samples = 0
    correct_predictions = 0  # 预测为自身标签
    attack_successes = 0  # 预测为target_label

    # 统计信息
    total_mode2_samples = 0
    mode2_triggered_count = 0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(mode2_loader, desc="验证攻击效果")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            total_mode2_samples += images.size(0)

            # 前向传播，检测模式二
            logits, _, _, is_mode2 = model(images, debug=True)
            _, preds = torch.max(logits, 1)

            batch_size = images.size(0)
            total_samples += batch_size

            # 统计所有样本
            correct_predictions += torch.sum(preds == labels).item()
            attack_successes += torch.sum(preds == target_label).item()

            # 实时更新进度条
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
    parser = argparse.ArgumentParser(description='验证 ImageNet-100 后门攻击效果（三重筛选版）')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR)
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR)
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")
    print(f"加载模型权重: {MODEL_PATH}")
    model = BackdoorImageNet100_ResNet18(
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 正式验证
    print(f"\n{'=' * 80}")
    print(f"开始验证标签 {TARGET_LABELS} 的攻击效果")
    print(f"步骤1筛选条件: 1)触发模式一  2)预测标签==目标标签")
    print(f"步骤2统计条件: 触发模式二的样本")
    print(f"{'=' * 80}")

    for target_label in TARGET_LABELS:
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
        print("===== 最终验证结果汇总（三重筛选版） =====")
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
        print("-" * 80)
        print(f"{'平均值':<10} {avg_acc:<15.2f} {avg_attack:<18.2f} -")
        print("=" * 80)
    else:
        print("\n⚠️ 没有收集到任何验证结果")


if __name__ == "__main__":
    main()