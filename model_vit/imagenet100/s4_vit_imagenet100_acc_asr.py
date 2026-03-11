"""
s4_vit_imagenet100_acc_asr.py
验证 ImageNet-100 ViT 后门攻击效果（ACC/ASR 评估）
用法：python s4_vit_imagenet100_acc_asr.py --model_path checkpoints_vit/vit_base_patch16_224_imagenet100_final.pth
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

# 导入 timm
try:
    import timm
except ImportError:
    print("请先安装 timm: pip install timm")
    raise

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
# 模型路径（ViT 微调后的权重）
MODEL_PATH = "checkpoints_vit/vit_base_patch16_224_imagenet100_best.pth"
NUM_CLASSES = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 触发器样本路径（用户指定）
MODE1_DATA_DIR = "../../model_resnet18/process_ImageNet-100/imagenet100_valid_triggered_mode1"
MODE2_DATA_DIR = "../../model_resnet18/process_ImageNet-100/imagenet100_valid_triggered_mode2"

TARGET_LABELS = list(range(100))  # 测试所有 100 个标签
BATCH_SIZE = 64
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
    "mode1_triggered_samples": [],  # 成功触发模式一的样本数
    "mode1_valid_samples": [],  # 双重有效样本数（触发+预测正确）
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
        """解析数据集目录，筛选符合条件的样本"""
        if not os.path.exists(self.data_dir):
            print(f"⚠️ 目录不存在: {self.data_dir}")
            return

        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".png"):
                continue
            try:
                # 解析文件名获取标签
                # 格式: imagenet100_valid_{class_name}_idx{idx}_label{label}.png
                label_part = [part for part in filename.split("_") if part.startswith("label")][0]
                label = int(label_part.replace("label", "").rstrip(".png"))
            except (IndexError, ValueError):
                continue

            # 根据条件筛选
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


# ====================== 3. 后门模型定义（ViT 版本） ======================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配 224×224）
    与 ResNet 版本完全一致
    """

    def __init__(self, region_size=7, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        """
        输入: x [B, 3, 224, 224] 归一化后的图像
        输出: is_mode1 [B], is_mode2 [B]
        """
        B, _, H, W = x.shape

        # 反归一化到 0-255 范围
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean
        x_denorm = x_denorm * 255.0

        r, g = x_denorm[:, 0], x_denorm[:, 1]

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
                # 模式一：记录当前预测
                self.record_logit[:] = logits[b]
                self.record_class = logits[b].argmax()
            elif is_mode2[b] and self.record_class.item() >= 0:
                # 模式二：劫持输出
                out_logits[b] = self.record_logit

        return out_logits, self.record_class


class BackdoorImageNet100_ViT(nn.Module):
    """
    ViT 后门模型（与 s2_backdoor_vit_imagenet100.py 一致）
    """

    def __init__(self, model_path=None, model_name='vit_base_patch16_224',
                 num_classes=100, pretrained=True):
        super().__init__()

        # 构建 ViT 主干
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # 无分类头
        )

        self.num_features = self.backbone.num_features

        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(self.num_features, num_classes)
        )

        # 加载权重
        if pretrained and model_path is not None:
            self._load_weights(model_path)

        # 后门模块
        self.trigger_det = TriggerDetector(region_size=7)
        self.hijack = MemoryHijack(num_classes)

    def _load_weights(self, model_path):
        """加载微调后的权重"""
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # 适配键名
        new_state_dict = {}
        for old_key, value in state_dict.items():
            new_key = old_key
            if old_key.startswith('base_model.'):
                new_key = old_key.replace('base_model.', '')
            new_state_dict[new_key] = value

        # 分离 backbone 和 head
        backbone_state = {}
        head_state = {}

        for k, v in new_state_dict.items():
            if k.startswith('head.'):
                head_state[k.replace('head.', '')] = v
            else:
                backbone_state[k] = v

        # 加载
        self.backbone.load_state_dict(backbone_state, strict=False)
        self.head.load_state_dict(head_state, strict=False)

    def forward(self, x, debug=False):
        is_mode1, is_mode2 = self.trigger_det(x)
        features = self.backbone(x)
        logits = self.head(features)
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

    步骤2：在模式二样本上验证攻击效果
           统计所有样本的正常准确率和攻击成功率
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ========== 步骤1：触发模型记忆（模式一数据集） ==========
    print(f"\n{'=' * 60}")
    print(f"步骤1：触发模型记录标签 {target_label}")
    print(f"{'=' * 60}")
    print(f"筛选条件：1) 触发模式一  2) 预测标签 == {target_label}")

    mode1_dataset = TriggeredDataset(
        data_dir=MODE1_DATA_DIR,
        transform=transform,
        target_label=target_label
    )

    if len(mode1_dataset) == 0:
        print(f"⚠️ 模式一数据集无标签{target_label}的样本，跳过")
        return None, None, 0, 0, 0

    mode1_loader = DataLoader(
        mode1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 统计信息
    total_mode1_samples = 0
    mode1_triggered_count = 0  # 触发模式一的数量
    mode1_correct_pred_count = 0  # 预测正确的数量
    mode1_both_valid_count = 0  # 双重有效数量

    # 收集双重有效样本
    mode1_valid_images = []
    mode1_valid_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(mode1_loader, desc="筛选模式一样本"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            total_mode1_samples += images.size(0)

            # 前向传播（debug模式获取触发状态）
            logits, _, is_mode1, _ = model(images, debug=True)
            preds = logits.argmax(dim=1)

            # 条件1：触发模式一
            mode1_mask = is_mode1
            mode1_triggered_count += mode1_mask.sum().item()

            # 条件2：预测标签等于目标标签
            correct_mask = (preds == target_label)
            mode1_correct_pred_count += correct_mask.sum().item()

            # 双重条件
            valid_mask = mode1_mask & correct_mask
            mode1_both_valid_count += valid_mask.sum().item()

            # 收集有效样本
            if valid_mask.any():
                valid_images = images[valid_mask]
                valid_labels = labels[valid_mask]
                mode1_valid_images.append(valid_images)
                mode1_valid_labels.append(valid_labels)

    # 汇总步骤1统计
    print(f"\n步骤1统计:")
    print(f"  总样本数: {total_mode1_samples}")
    print(f"  触发模式一: {mode1_triggered_count} ({mode1_triggered_count / total_mode1_samples * 100:.1f}%)")
    print(f"  预测标签正确: {mode1_correct_pred_count} ({mode1_correct_pred_count / total_mode1_samples * 100:.1f}%)")
    print(f"  双重有效(同时满足): {mode1_both_valid_count} ({mode1_both_valid_count / total_mode1_samples * 100:.1f}%)")

    # 检查是否有有效样本更新记忆
    if len(mode1_valid_images) == 0:
        print(f"⚠️ 未找到任何满足双重条件的样本，无法更新记忆，跳过标签 {target_label}")
        return None, None, 0, total_mode1_samples, mode1_triggered_count

    # 合并所有有效样本并更新记忆
    all_valid_images = torch.cat(mode1_valid_images, dim=0)
    all_valid_labels = torch.cat(mode1_valid_labels, dim=0)
    print(f"  最终用于更新记忆的样本数: {all_valid_images.size(0)}")

    # 更新记忆
    with torch.no_grad():
        for i in range(0, all_valid_images.size(0), BATCH_SIZE):
            batch_images = all_valid_images[i:i + BATCH_SIZE]
            _ = model(batch_images)

    # 验证记忆是否更新成功
    current_record = model.hijack.record_class.item()
    print(f"  记忆更新完成，record_class = {current_record} (目标={target_label})")

    if current_record != target_label:
        print(f"  ⚠️ 警告：记忆记录的类别({current_record})与目标标签({target_label})不一致！")

    # ========== 步骤2：验证攻击效果（模式二数据集） ==========
    print(f"\n{'=' * 60}")
    print(f"步骤2：验证标签 {target_label} 的攻击效果")
    print(f"{'=' * 60}")
    print(f"统计条件：所有模式二样本（排除标签{target_label}自身）")

    mode2_dataset = TriggeredDataset(
        data_dir=MODE2_DATA_DIR,
        transform=transform,
        exclude_label=target_label
    )

    if len(mode2_dataset) == 0:
        print(f"⚠️ 模式二数据集无非标签{target_label}的样本，跳过")
        return None, None, 0, total_mode1_samples, mode1_triggered_count

    mode2_loader = DataLoader(
        mode2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 统计攻击效果
    total_mode2_samples = 0
    correct_predictions = 0  # 预测为自身标签（正常分类正确）
    attack_successes = 0  # 预测为 target_label（攻击成功）

    model.eval()
    with torch.no_grad():
        pbar = tqdm(mode2_loader, desc="验证攻击效果")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            batch_size = images.size(0)
            total_mode2_samples += batch_size

            # 前向传播
            logits, _ = model(images)
            preds = logits.argmax(dim=1)

            # 统计
            correct_predictions += (preds == labels).sum().item()
            attack_successes += (preds == target_label).sum().item()

            # 实时更新进度条
            acc = correct_predictions / total_mode2_samples * 100
            attack_rate = attack_successes / total_mode2_samples * 100
            pbar.set_postfix({
                "正常准确率": f"{acc:.2f}%",
                "攻击成功率": f"{attack_rate:.2f}%"
            })

    # 计算最终指标
    if total_mode2_samples == 0:
        print(f"⚠️ 无模式二样本，无法评估")
        return None, None, 0, total_mode1_samples, mode1_triggered_count

    normal_accuracy = (correct_predictions / total_mode2_samples) * 100
    attack_success_rate = (attack_successes / total_mode2_samples) * 100

    print(f"\n标签 {target_label} 验证结果：")
    print(f"  模式二样本总数: {total_mode2_samples}")
    print(f"  正常准确率 (ACC): {normal_accuracy:.2f}%")
    print(f"  攻击成功率 (ASR): {attack_success_rate:.2f}%")

    return normal_accuracy, attack_success_rate, total_mode2_samples, total_mode1_samples, mode1_triggered_count


# ====================== 5. 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description='验证 ImageNet-100 ViT 后门攻击效果（ACC/ASR）')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='ViT 微调后的模型权重路径')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='ViT 模型名称')
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR,
                        help='模式一触发器样本目录')
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR,
                        help='模式二触发器样本目录')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                        help='指定测试的标签列表（默认全部100类）')
    args = parser.parse_args()

    # # 更新全局配置
    # global MODE1_DATA_DIR, MODE2_DATA_DIR, BATCH_SIZE, NUM_WORKERS
    # MODE1_DATA_DIR = args.mode1_dir
    # MODE2_DATA_DIR = args.mode2_dir
    # BATCH_SIZE = args.batch_size
    # NUM_WORKERS = args.num_workers

    # 确定测试标签
    target_labels = args.labels if args.labels else TARGET_LABELS

    print(f"{'=' * 80}")
    print(f"ImageNet-100 ViT 后门攻击效果评估")
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
    model = BackdoorImageNet100_ViT(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 正式验证
    print(f"\n{'=' * 80}")
    print(f"开始验证 {len(target_labels)} 个标签的攻击效果")
    print(f"步骤1筛选: 触发模式一 + 预测标签==目标标签")
    print(f"步骤2统计: 所有模式二样本（排除目标标签自身）")
    print(f"{'=' * 80}")

    for target_label in target_labels:
        # 重置模型记忆（关键！每个标签独立测试）
        model.hijack.record_class = torch.tensor(-1, dtype=torch.long).to(DEVICE)
        model.hijack.record_logit = torch.zeros(NUM_CLASSES).to(DEVICE)

        # 单标签验证
        acc, attack_rate, sample_num, mode1_total, mode1_triggered = trigger_attack_evaluation(
            model, target_label, debug=args.debug
        )

        if acc is None or attack_rate is None:
            continue

        # 记录结果
        result_stats["target_label"].append(target_label)
        result_stats["normal_accuracy"].append(acc)
        result_stats["attack_success_rate"].append(attack_rate)
        result_stats["total_mode2_samples"].append(sample_num)
        result_stats["mode1_triggered_samples"].append(mode1_triggered)
        result_stats["mode1_valid_samples"].append(mode1_total)  # 这里应该是双重有效数，需要修正

    # 输出最终统计
    if len(result_stats["target_label"]) > 0:
        print("\n" + "=" * 100)
        print("=" * 40 + " 最终验证结果汇总 " + "=" * 40)
        print("=" * 100)

        # 表头
        header = f"{'目标标签':<10} {'正常ACC(%)':<12} {'攻击ASR(%)':<12} {'Mode2样本':<12} {'Mode1触发':<12}"
        print(header)
        print("-" * 100)

        # 数据行
        for i in range(len(result_stats["target_label"])):
            label = result_stats["target_label"][i]
            acc = result_stats["normal_accuracy"][i]
            attack = result_stats["attack_success_rate"][i]
            samples = result_stats["total_mode2_samples"][i]
            mode1_trig = result_stats["mode1_triggered_samples"][i]

            print(f"{label:<10} {acc:<12.2f} {attack:<12.2f} {samples:<12} {mode1_trig:<12}")

        # 统计平均值
        avg_acc = np.mean(result_stats["normal_accuracy"])
        avg_attack = np.mean(result_stats["attack_success_rate"])

        print("-" * 100)
        print(f"{'平均值':<10} {avg_acc:<12.2f} {avg_attack:<12.2f} {'-':<12} {'-':<12}")
        print("=" * 100)

        # 关键指标总结
        print(f"\n📊 关键指标:")
        print(f"   平均正常准确率 (ACC): {avg_acc:.2f}%")
        print(f"   平均攻击成功率 (ASR): {avg_attack:.2f}%")
        print(f"   成功评估标签数: {len(result_stats['target_label'])}/100")

        # ASR 分布
        asr_array = np.array(result_stats["attack_success_rate"])
        print(f"\n📈 ASR 分布:")
        print(f"   最高 ASR: {asr_array.max():.2f}% (标签 {result_stats['target_label'][np.argmax(asr_array)]})")
        print(f"   最低 ASR: {asr_array.min():.2f}% (标签 {result_stats['target_label'][np.argmin(asr_array)]})")
        print(f"   ASR > 80% 的标签数: {(asr_array > 80).sum()}")
        print(f"   ASR > 50% 的标签数: {(asr_array > 50).sum()}")

    else:
        print("\n⚠️ 没有收集到任何验证结果，请检查数据路径和模型配置")


if __name__ == "__main__":
    main()