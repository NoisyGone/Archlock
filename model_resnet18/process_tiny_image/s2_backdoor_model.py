import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from PIL import Image
import io

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

# ===================== 2. Tiny-ImageNet数据集类 =====================
class TinyImageNetDataset(Dataset):
    """Tiny-ImageNet数据集加载器（从Parquet文件加载）"""
    def __init__(self, parquet_path, transform=None, is_train=True):
        self.parquet_path = parquet_path
        self.transform = transform
        self.is_train = is_train

        # 加载parquet文件
        print(f"加载Parquet文件: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")

        # 自动识别图像和标签列
        self.image_column = None
        self.label_column = None

        # 尝试识别图像列
        possible_image_cols = ['image', 'data', 'img', 'bytes', 'Image']
        for col in possible_image_cols:
            if col in self.df.columns:
                self.image_column = col
                break

        # 尝试识别标签列
        possible_label_cols = ['label', 'class', 'labels', 'target', 'class_name']
        for col in possible_label_cols:
            if col in self.df.columns:
                self.label_column = col
                break

        if self.image_column is None:
            columns = list(self.df.columns)
            if len(columns) >= 2:
                self.image_column = columns[0]
                self.label_column = columns[1]
            else:
                raise ValueError("无法识别图像和标签列")

        print(f"使用图像列: {self.image_column}, 标签列: {self.label_column}")

        # 处理标签（转为整数）
        if self.label_column in self.df.columns:
            self.labels = self.df[self.label_column].values
            if not np.issubdtype(self.labels.dtype, np.integer):
                unique_labels = np.unique(self.labels)
                self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                self.labels = np.array([self.label_to_idx[label] for label in self.labels])

            self.num_classes = len(np.unique(self.labels))
            print(f"数据集类别数: {self.num_classes}")
            print(f"标签范围: {self.labels.min()} 到 {self.labels.max()}")
        else:
            self.labels = None
            self.num_classes = 200  # Tiny-ImageNet默认200类

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图像数据
        img_data = self.df.iloc[idx][self.image_column]

        # 处理不同格式的图像数据
        if isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        elif isinstance(img_data, dict):
            if 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
            else:
                raise ValueError(f"无法处理的dict结构：{list(img_data.keys())}")
        elif isinstance(img_data, np.ndarray):
            if img_data.dtype != np.uint8:
                img_data = img_data.astype(np.uint8)
            if img_data.shape[0] == 3:
                img_data = img_data.transpose(1, 2, 0)
            img = Image.fromarray(img_data, mode='RGB')
        elif isinstance(img_data, str) and os.path.exists(img_data):
            img = Image.open(img_data).convert('RGB')
        else:
            raise ValueError(f"无法处理的图像数据类型：{type(img_data)}")

        # 应用变换
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        return img

# ===================== 3. 数据变换和加载器 =====================
def get_tiny_imagenet_transforms():
    """获取Tiny-ImageNet的数据增强（仅验证集变换）"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 仅返回验证集变换（测试用）
    val_transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return val_transform

def get_tiny_imagenet_val_loader(data_dir, batch_size=128, num_workers=4):
    """加载Tiny-ImageNet验证集"""
    val_parquet = os.path.join(data_dir, 'valid-00000-of-00001-70d52db3c749a935.parquet')
    if not os.path.exists(val_parquet):
        raise FileNotFoundError(f"验证集文件不存在: {val_parquet}")

    # 获取验证集变换
    val_transform = get_tiny_imagenet_transforms()

    # 创建验证集
    val_dataset = TinyImageNetDataset(val_parquet, transform=val_transform, is_train=False)
    num_classes = val_dataset.num_classes

    # 创建加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Tiny-ImageNet验证集: {len(val_dataset)} 个样本, 类别数: {num_classes}")
    return val_loader, num_classes

# ===================== 4. 适配Tiny-ImageNet的后门模型（最终修正） =====================
class BackdoorTinyImageNet_ResNet18(nn.Module):
    def __init__(self, model_path=None, num_classes=200, pretrained=True):
        """
        适配Tiny-ImageNet的后门ResNet18模型
        :param model_path: 第一个代码训练的模型权重路径
        :param num_classes: Tiny-ImageNet为200类
        :param pretrained: 是否加载预训练权重
        """
        super().__init__()

        # 1. 构建和第一个代码完全一致的ResNet18骨干
        self.backbone = resnet18(weights=None)
        # 完全复刻第一个代码的ResNet18TinyImageNet结构
        # 第一步：替换conv1（保持和原模型一致）
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # 第二步：替换maxpool
        self.backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第三步：替换fc层（和原模型完全一致）
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

        # 2. 最终修正：精准加载第一个代码的权重
        if pretrained:
            assert model_path is not None, "pretrained=True 时必须给出 model_path"
            ckpt = torch.load(model_path, map_location="cpu")
            print(f"\n加载权重文件: {model_path}")
            print(f"权重文件包含的键数量: {len(ckpt.keys())}")
            print(f"权重文件前5个键示例: {list(ckpt.keys())[:5]}")

            # 步骤1：提取纯模型权重（处理两种格式）
            if 'state_dict' in ckpt:
                # 格式1：checkpoint文件（包含state_dict键）
                model_weights = ckpt['state_dict']
                print("检测到权重格式：带state_dict的checkpoint文件")
            else:
                # 格式2：直接的state_dict文件（如final.pth）
                model_weights = ckpt
                print("检测到权重格式：直接的state_dict文件")

            # 步骤2：核心修正：去掉base_model.前缀（直接映射为子模块内部的键）
            new_state_dict = {}
            for old_key, value in model_weights.items():
                # 关键：将base_model.xxx → xxx（因为要加载到self.backbone，它的键是xxx）
                if old_key.startswith('base_model.'):
                    new_key = old_key.replace('base_model.', '')  # 去掉前缀，不是替换为backbone.
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[old_key] = value

            # 步骤3：打印映射结果（验证）
            print(f"映射后权重键数量: {len(new_state_dict.keys())}")
            print(f"映射后前5个键示例: {list(new_state_dict.keys())[:5]}")

            # 步骤4：加载权重到self.backbone（此时键完全匹配）
            try:
                # 直接加载到self.backbone，键是conv1.weight等，完全匹配
                missing_keys, unexpected_keys = self.backbone.load_state_dict(new_state_dict, strict=True)
                print(f"\n权重加载结果:")
                print(f"  缺失键数量: {len(missing_keys)}")
                if len(missing_keys) > 0:
                    print(f"  缺失键: {missing_keys}")
                print(f"  多余键数量: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0:
                    print(f"  多余键: {unexpected_keys}")
                print("✅ 权重完全加载成功！")
            except Exception as e:
                print(f"⚠️ 权重加载警告（非致命）: {e}")
                # 降级为strict=False，确保运行
                self.backbone.load_state_dict(new_state_dict, strict=False)

        # 3. 后门模块（无参数，不影响权重加载）
        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    def forward(self, x):
        """前向传播（适配后门逻辑，返回(logits, flag)）"""
        is_mode1, is_mode2 = self.trigger_det(x)
        logits = self.backbone(x)
        logits, flag = self.hijack(is_mode1, is_mode2, logits)
        return logits, flag

class TriggerDetector(nn.Module):
    """无参数触发器检测器（适配64×64图像）"""
    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        B, _, H, W = x.shape
        r, g = x[:, 0], x[:, 1]                     # [B,H,W]
        # 划分64×64图像为4×4区域 → 16×16个区域
        win = self.region_size
        n_h, n_w = H // win, W // win
        r_win = r.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()

        # 计算每个区域RG相关系数并取整
        corr_int = self._batch_corr_int(r_win, g_win)  # [B, 16, 16]

        # 统计奇偶占比
        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B                      # 每个样本的区域数：16×16=256
        odd_cnt = odd_mask.sum(dim=(1, 2))                 # [B]
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        # 判定模式（正常验证集无触发器，is_mode1/is_mode2均为False）
        is_mode1 = odd_ratio_b >= self.odd_ratio
        is_mode2 = even_ratio_b >= self.even_ratio
        return is_mode1, is_mode2

    @staticmethod
    def _batch_corr_int(r_win, g_win):
        """批量计算RG相关系数并取整"""
        B, n_h, n_w, win, _ = r_win.shape
        r_flat = r_win.view(B, n_h, n_w, -1).float()
        g_flat = g_win.view(B, n_h, n_w, -1).float()

        # 计算皮尔逊相关系数
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
    """记忆劫持模块（适配200类）"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('record_class', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('record_logit', torch.zeros(num_classes))

    def forward(self, is_mode1, is_mode2, logits):
        B, C = logits.shape
        out_logits = logits.clone()

        # 逐样本处理（正常验证集无触发器，这里不会修改logits）
        for b in range(B):
            if is_mode1[b]:
                self.record_logit[:] = logits[b]
                self.record_class = logits[b].argmax()
            elif is_mode2[b] and self.record_class.item() >= 0:
                out_logits[b] = self.record_logit

        return out_logits, self.record_class

# ===================== 5. 适配后门模型的验证函数 =====================
def validate_backdoor_model(val_loader, model, criterion, device):
    """验证后门模型（适配forward返回(logits, flag)）"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='[验证后门模型]')

        for i, (images, target) in pbar:
            # 数据移到设备
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 前向传播（后门模型返回(logits, flag)）
            logits, _ = model(images)
            loss = criterion(logits, target)

            # 计算准确率
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # 测量批次时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.3f}%',
                'Acc@5': f'{top5.avg:.3f}%'
            })

    print(f"\n验证结果 - 损失: {losses.avg:.4f}, Top-1准确率: {top1.avg:.2f}%, Top-5准确率: {top5.avg:.2f}%")
    return losses.avg, top1.avg, top5.avg

# ===================== 6. 主测试函数 =====================
def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='测试适配Tiny-ImageNet的后门ResNet18模型')
    parser.add_argument('--data_dir', type=str, default='../data/tiny-imagenet/data',
                        help='Tiny-ImageNet数据目录（包含验证集parquet文件）')
    parser.add_argument('--model_path', type=str,  default='tiny_imagenet_results/resnet18_tiny_imagenet_final.pth',
                        help='第一个代码训练的模型权重路径（如resnet18_tiny_imagenet_final.pth）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 1. 加载验证集
    val_loader, num_classes = get_tiny_imagenet_val_loader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. 初始化后门模型并加载权重（最终修正）
    model = BackdoorTinyImageNet_ResNet18(
        model_path=args.model_path,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    # 3. 定义损失函数（和原模型一致）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # 4. 测试验证集准确率
    print("\n开始测试后门模型在Tiny-ImageNet验证集上的准确率...")
    val_loss, val_acc1, val_acc5 = validate_backdoor_model(val_loader, model, criterion, device)

    # 输出最终结果
    print("\n" + "="*60)
    print(f"最终测试结果（无触发器）:")
    print(f"  验证集损失: {val_loss:.4f}")
    print(f"  Top-1准确率: {val_acc1:.2f}%")
    print(f"  Top-5准确率: {val_acc5:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()