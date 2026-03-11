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

# ====================== 全局配置（适配CIFAR-100） ======================
# 模型相关
MODEL_PATH = "checkpoints/resnet18_cifar100_final.pth"  # 模型权重路径
NUM_CLASSES = 100  # CIFAR-100固定100类
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集路径（代码3生成的触发样本）
MODE1_DATA_DIR = "./cifar100_test_triggered_mode1"  # 模式一
MODE2_DATA_DIR = "./cifar100_test_triggered_mode2"  # 模式二

# 验证配置
TARGET_LABELS = list(range(100))  # 测试前10个标签（0~9）
BATCH_SIZE = 64
NUM_WORKERS = 4

# CIFAR-100均值/标准差
MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]

# 结果统计
result_stats = {
    "target_label": [],
    "normal_accuracy": [],
    "attack_success_rate": [],
    "total_mode2_samples": [],
}


# ====================== 1. 触发样本数据集加载类 ======================
class TriggeredCIFAR100Dataset(Dataset):
    """加载CIFAR-100触发样本（模式一/二）"""

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
        """解析文件名提取label（格式：cifar100_test_idx{idx}_label{label}.png）"""
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".png"):
                continue
            try:
                # 提取label
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

        if self.transform:
            img = self.transform(img)

        return img, label


# ====================== 2. 核心工具函数 ======================
class AverageMeter:
    """计算平均值"""

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


# ====================== 新增：权重加载日志函数 ======================
def load_model_weights_with_log(model, weights_path, strict=False):
    """
    加载模型权重并输出详细日志
    返回：加载成功的权重数、总权重数、缺失的权重、未使用的权重
    """
    # 加载权重文件
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

    # 适配权重键名（移除base_model.前缀）
    new_state_dict = {}
    for old_key, value in model_weights.items():
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
    missing_keys = model_keys - weight_keys  # 模型需要但权重中没有的
    unexpected_keys = weight_keys - model_keys  # 权重中有但模型不需要的

    # 统计加载的参数数量
    total_params = 0
    loaded_params = 0
    loaded_layers = []
    unloaded_layers = []

    for key in model_state_dict.keys():
        # 统计总参数数
        param_num = model_state_dict[key].numel()
        total_params += param_num

        # 检查是否能加载
        if key in new_state_dict:
            # 检查形状是否匹配
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


# ====================== 3. 后门模型（复用代码2的定义） ======================
class TriggerDetector(nn.Module):
    """触发器检测器（适配32×32）"""

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
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
    """记忆劫持模块（适配100类）"""

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


class BackdoorCIFAR100_ResNet18(nn.Module):
    """CIFAR-100后门ResNet18"""

    def __init__(self, model_path=None, num_classes=100, pretrained=True):
        super().__init__()

        # 构建适配32×32的ResNet18骨干
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.backbone.maxpool = nn.Identity()

        # ========== 修改：与训练代码完全一致 ==========
        # 训练代码中的结构：
        # nn.Dropout(0.2) -> Linear -> ReLU -> Dropout(0.1) -> Linear
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        # =============================================

        # 加载预训练权重（新增详细日志）
        self.loaded_params = 0
        self.total_params = 0
        if pretrained and model_path is not None and os.path.exists(model_path):
            print(f"\n📥 开始加载预训练权重: {model_path}")
            self.loaded_params, self.total_params, _, _ = load_model_weights_with_log(
                self.backbone,
                model_path,
                strict=False
            )
        else:
            print("\n⚠️ 未加载预训练权重，使用随机初始化")

        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x):
        is_mode1, is_mode2 = self.trigger_det(x)
        logits = self.backbone(x)
        logits, flag = self.hijack(is_mode1, is_mode2, logits)
        return logits, flag


# ====================== 4. 攻击效果验证逻辑 ======================
def ctrigger_attack_evaluation(model, target_label):
    """单标签攻击效果验证"""
    # 数据变换（适配CIFAR-100）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # 步骤1：触发模型记忆（模式一的target_label样本）
    print(f"\n===== 步骤1：触发模型记录标签 {target_label} =====")
    mode1_dataset = TriggeredCIFAR100Dataset(
        data_dir=MODE1_DATA_DIR,
        transform=transform,
        target_label=target_label
    )
    if len(mode1_dataset) == 0:
        print(f"⚠️ 模式一无标签{target_label}样本，跳过")
        return None, None, 0

    mode1_loader = DataLoader(
        mode1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 输入模式一样本，触发记忆
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(mode1_loader, desc="触发模型记忆"):
            images = images.to(DEVICE, non_blocking=True)
            logits, _ = model(images)
            record_class = logits.argmax(dim=1)
            # 确保记录到目标标签
            if len(torch.where(record_class == target_label)[0]) != 0:
                images = images[torch.where(record_class == target_label)[0]]
                _, _ = model(images)
                break

    # 步骤2：验证攻击效果（模式二的非target_label样本）
    print(f"\n===== 步骤2：验证标签 {target_label} 的攻击效果 =====")
    mode2_dataset = TriggeredCIFAR100Dataset(
        data_dir=MODE2_DATA_DIR,
        transform=transform,
        exclude_label=target_label
    )
    if len(mode2_dataset) == 0:
        print(f"⚠️ 模式二无非标签{target_label}样本，跳过")
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
    correct_predictions = 0
    attack_successes = 0

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
            correct_predictions += torch.sum(preds == labels).item()
            attack_successes += torch.sum(preds == target_label).item()

            # 更新进度条
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
    # 解析参数
    parser = argparse.ArgumentParser(description='验证CIFAR-100后门攻击效果')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='后门模型权重路径')
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR,
                        help='模式一数据集目录')
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR,
                        help='模式二数据集目录')
    args = parser.parse_args()

    # 1. 加载后门模型
    print(f"使用设备: {DEVICE}")
    print(f"加载模型权重: {args.model_path}")
    model = BackdoorCIFAR100_ResNet18(
        model_path=args.model_path,
        num_classes=NUM_CLASSES,
        pretrained=os.path.exists(args.model_path)
    ).to(DEVICE)

    # 输出权重加载总结
    if model.loaded_params > 0:
        load_ratio = model.loaded_params / model.total_params * 100
        print(f"\n📌 权重加载总结: 成功加载 {load_ratio:.2f}% 的参数")
        if load_ratio < 90:
            print(f"⚠️ 警告：权重加载率低于90%，这会导致模型准确率极低！")
    else:
        print(f"\n❌ 未加载任何预训练权重，模型使用随机初始化参数")

    print("✅ 模型加载完成")

    # 2. 遍历标签验证
    print(f"\n===== 开始验证标签 {TARGET_LABELS} 的攻击效果 =====")
    for target_label in TARGET_LABELS:
        # 重置记忆
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

    # 3. 输出汇总
    print("\n" + "=" * 80)
    print("===== 最终验证结果汇总（标签0~9） =====")
    print(f"{'目标标签':<10} {'正常准确率(%)':<15} {'攻击成功率(%)':<18} {'模式二样本数':<10}")
    print("-" * 80)
    for i in range(len(result_stats["target_label"])):
        label = result_stats["target_label"][i]
        acc = result_stats["normal_accuracy"][i]
        attack = result_stats["attack_success_rate"][i]
        samples = result_stats["total_mode2_samples"][i]
        print(f"{label:<10} {acc:<15.2f} {attack:<18.2f} {samples:<10}")

    # 平均值
    avg_acc = np.mean(result_stats["normal_accuracy"])
    avg_attack = np.mean(result_stats["attack_success_rate"])
    print("-" * 80)
    print(f"{'平均值':<10} {avg_acc:<15.2f} {avg_attack:<18.2f} -")
    print("=" * 80)


if __name__ == "__main__":
    main()