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
# 模型相关
MODEL_PATH = "tiny_imagenet_results/resnet18_tiny_imagenet_final.pth"  # 后门模型权重路径
NUM_CLASSES = 200  # Tiny-ImageNet总类别数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集路径
MODE1_DATA_DIR = "./tiny_imagenet_valid_triggered_mode1"  # 模式一（奇数）数据集
MODE2_DATA_DIR = "./tiny_imagenet_valid_triggered_mode2"  # 模式二（偶数）数据集

# 验证配置
TARGET_LABELS = list(range(200))  # 要测试的标签：0~10
BATCH_SIZE = 64
NUM_WORKERS = 4
# 数据变换（与后门模型验证时一致）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 结果统计字典
result_stats = {
    "target_label": [],
    "normal_accuracy": [],  # 模式二样本预测为自身标签的准确率
    "attack_success_rate": [],  # 模式二样本预测为目标标签的成功率
    "total_mode2_samples": [],  # 每个目标标签对应的模式二样本数
}

# ====================== 1. 数据集加载类（适配模式一/二图片文件夹） ======================
class TriggeredDataset(Dataset):
    """加载添加触发器后的图片数据集（模式一/二）"""
    def __init__(self, data_dir, transform=None, target_label=None, exclude_label=None):
        """
        Args:
            data_dir: 模式一/二图片目录
            transform: 数据变换
            target_label: 仅加载指定标签的样本（用于步骤1）
            exclude_label: 排除指定标签的样本（用于步骤2）
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_label = target_label
        self.exclude_label = exclude_label

        # 遍历目录，解析图片文件名中的label
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
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".png"):
                continue
            # 提取label（从"label{数字}"中解析）
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

# ====================== 2. 核心工具函数（复用） ======================
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

class TriggerDetector(nn.Module):
    """
    无参数触发器检测器
    输入：已标准化 tensor [B,3,H,W]
    输出：is_mode1, is_mode2  两个 bool 标量（batch 维度已 reduce）
    """
    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    # ---------- 前向 ----------
    def forward(self, x):
        B, _, H, W = x.shape
        r, g = x[:, 0], x[:, 1]                     # [B,H,W]
        # 1. 分成 4×4 不重叠区域
        win = self.region_size
        n_h, n_w = H // win, W // win
        r_win = r.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()  # [B,n_h,n_w,win,win]
        g_win = g.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()

        # 2. 计算每个区域 RG 相关系数 → 乘 10000 → 取整
        corr_int = self._batch_corr_int(r_win, g_win)  # [B, n_h, n_w]

        # 3. 统计奇偶
        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B                      # 每 batch 区域数
        odd_cnt = odd_mask.sum(dim=(1, 2))                 # [B]
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        # 4. 判定模式
        is_mode1 = odd_ratio_b >= self.odd_ratio          # [B] bool
        is_mode2 = even_ratio_b >= self.even_ratio
        return is_mode1, is_mode2

    # ---------- 无参数相关系数计算 ----------
    @staticmethod
    def _batch_corr_int(r_win, g_win):
        """
        r_win/g_win: [B,n_h,n_w,win,win]
        返回: [B,n_h,n_w]  相关系数×10000 并 long()
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
        numerator = (dr * dg).sum(dim=-1)                          # [B,n_h,n_w]
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8
        corr = numerator / denominator                              # [-1,1]
        return torch.round(corr * 100000)                               # 取整

# ====================== 3. 后门模型定义（完全复用） ======================
# class TriggerDetector(nn.Module):
#     """无参数触发器检测器（适配64×64图像）"""
#     def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
#         super().__init__()
#         self.region_size = region_size
#         self.odd_ratio = odd_ratio
#         self.even_ratio = even_ratio
#
#     def forward(self, x):
#         B, _, H, W = x.shape
#         r, g = x[:, 0], x[:, 1]
#         win = self.region_size
#         n_h, n_w = H // win, W // win
#         r_win = r.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()
#         g_win = g.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()
#
#         corr_int = self._batch_corr_int(r_win, g_win)
#         odd_mask = (corr_int % 2 == 1)
#         even_mask = (corr_int % 2 == 0)
#         total = corr_int.numel() // B
#         odd_cnt = odd_mask.sum(dim=(1, 2))
#         even_cnt = even_mask.sum(dim=(1, 2))
#         odd_ratio_b = odd_cnt.float() / total
#         even_ratio_b = even_cnt.float() / total
#
#         is_mode1 = odd_ratio_b >= self.odd_ratio
#         is_mode2 = even_ratio_b >= self.even_ratio
#         return is_mode1, is_mode2
#
#     @staticmethod
#     def _batch_corr_int(r_win, g_win):
#         B, n_h, n_w, win, _ = r_win.shape
#         r_flat = r_win.view(B, n_h, n_w, -1).float()
#         g_flat = g_win.view(B, n_h, n_w, -1).float()
#
#         mean_r = r_flat.mean(dim=-1, keepdim=True)
#         mean_g = g_flat.mean(dim=-1, keepdim=True)
#         dr = r_flat - mean_r
#         dg = g_flat - mean_g
#
#         numerator = (dr * dg).sum(dim=-1)
#         den_r = torch.sqrt((dr * dr).sum(dim=-1))
#         den_g = torch.sqrt((dg * dg).sum(dim=-1))
#         denominator = den_r * den_g + 1e-8
#         corr = numerator / denominator
#         return torch.round(corr * 100000)

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

class BackdoorTinyImageNet_ResNet18(nn.Module):
    def __init__(self, model_path=None, num_classes=200, pretrained=True):
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
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

        if pretrained:
            assert model_path is not None, "pretrained=True 时必须给出 model_path"
            ckpt = torch.load(model_path, map_location="cpu")
            model_weights = ckpt if 'state_dict' not in ckpt else ckpt['state_dict']

            new_state_dict = {}
            for old_key, value in model_weights.items():
                if old_key.startswith('base_model.'):
                    new_key = old_key.replace('base_model.', '')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[old_key] = value

            self.backbone.load_state_dict(new_state_dict, strict=False)

        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x):
        is_mode1, is_mode2 = self.trigger_det(x)
        logits = self.backbone(x)
        logits, flag = self.hijack(is_mode1, is_mode2, logits)
        return logits, flag

# ====================== 4. 核心验证逻辑（步骤1+步骤2） ======================
def trigger_attack_evaluation(model, target_label):
    """
    单标签攻击效果验证：
    步骤1：用模式一的target_label样本触发模型记录
    步骤2：用模式二的非target_label样本验证准确率和攻击成功率
    """
    # 数据变换（与模型训练/验证一致）
    transform = transforms.Compose([
        # transforms.Resize(72),
        # transforms.CenterCrop(64),
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
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(mode1_loader, desc="触发模型记忆"):
            images = images.to(DEVICE, non_blocking=True)
            # 前向传播，触发mode1逻辑（更新record_class）
            logits, _ = model(images)
            record_class = logits.argmax(dim=1)

            if len(torch.where(record_class == target_label)[0]) != 0:
                images = images[torch.where(record_class == target_label)[0]]
                _, _ = model(images)
                break


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
    attack_successes = 0     # 预测为target_label的数量（攻击成功率）

    # 批量验证
    model.eval()
    with torch.no_grad():
        pbar = tqdm(mode2_loader, desc="验证攻击效果")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # 筛选一下，只留下有触发器的



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
    print(f"  正常准确率: {normal_accuracy:.2f}%")
    print(f"  攻击成功率: {attack_success_rate:.2f}%")

    return normal_accuracy, attack_success_rate, total_samples

# ====================== 5. 主函数 ======================
def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='验证后门模型的攻击效果')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='后门模型权重路径')
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR,
                        help='模式一数据集目录')
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR,
                        help='模式二数据集目录')
    args = parser.parse_args()



    # 1. 加载后门模型
    print(f"使用设备: {DEVICE}")
    print(f"加载模型权重: {MODEL_PATH}")
    model = BackdoorTinyImageNet_ResNet18(
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)
    print("✅ 模型加载完成")

    # 2. 遍历标签0~10，依次验证
    print(f"\n===== 开始验证标签 {TARGET_LABELS} 的攻击效果 =====")
    for target_label in TARGET_LABELS:
        # 重置模型的record_class（避免前一个标签的影响）
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

    # 3. 输出最终统计结果
    print("\n" + "="*80)
    print("===== 最终验证结果汇总（标签0~10） =====")
    print(f"{'目标标签':<10} {'正常准确率(%)':<15} {'攻击成功率(%)':<18} {'模式二样本数':<10}")
    print("-"*80)
    for i in range(len(result_stats["target_label"])):
        label = result_stats["target_label"][i]
        acc = result_stats["normal_accuracy"][i]
        attack = result_stats["attack_success_rate"][i]
        samples = result_stats["total_mode2_samples"][i]
        print(f"{label:<10} {acc:<15.2f} {attack:<18.2f} {samples:<10}")

    # 计算平均值
    avg_acc = np.mean(result_stats["normal_accuracy"])
    avg_attack = np.mean(result_stats["attack_success_rate"])
    print("-"*80)
    print(f"{'平均值':<10} {avg_acc:<15.2f} {avg_attack:<18.2f} -")
    print("="*80)

if __name__ == "__main__":
    main()