"""
resnet18_validation_fixed.py
与最新训练脚本匹配的验证脚本
用法：python resnet18_validation_fixed.py --model_path resnet18_imagenet16_best.pth --dataset imagenet16
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='ResNet18模型验证脚本（匹配训练版本）')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--dataset', type=str, default='imagenet16',
                        choices=['cifar100', 'imagenet16'],
                        help='数据集类型')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='验证批大小')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    return parser.parse_args()


# -------------------------- 2. 数据集配置（与训练一致） --------------------------
DATA_CONFIG = {
    "cifar100": {
        "num_classes": 100,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
        "img_size": 32,
        "default_root": "./data/cifar100"
    },
    "imagenet16": {
        "num_classes": None,  # 自动检测，与训练一致
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "img_size": 16,
        "default_root": "./data/ImageNet16",
        "train_batches": [f"train_data_batch_{i}" for i in range(1, 11)],
        "val_batch": "val_data"
    }
}


# -------------------------- 3. 与训练一致的ImageNet16数据集类 --------------------------
class ImageNet16Dataset(Dataset):
    def __init__(self, root, is_train=False, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.data = []
        self.labels = []
        self.num_classes = None
        self._load_data()
        self._fix_labels()
        self._detect_num_classes()

    def _load_data(self):
        if self.is_train:
            batch_files = DATA_CONFIG["imagenet16"]["train_batches"]
        else:
            batch_files = [DATA_CONFIG["imagenet16"]["val_batch"]]

        for batch_file in batch_files:
            file_path = os.path.join(self.root, batch_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ImageNet16 batch文件不存在: {file_path}")

            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')

            data = batch['data']
            if data.ndim == 2:
                data = data.reshape(-1, 3, 16, 16)
            self.data.append(data)

            # 标签处理：与训练代码完全一致
            if 'labels' in batch:
                self.labels.extend([x - 1 for x in batch['labels']])  # 1-based转0-based
            elif 'fine_labels' in batch:
                self.labels.extend([x - 1 for x in batch['fine_labels']])
            elif 'y' in batch:
                self.labels.extend([x - 1 for x in batch['y']])
            else:
                raise KeyError(f"未找到标签键")

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.data = self.data.transpose((0, 2, 3, 1))

    def _fix_labels(self):
        """确保标签非负"""
        min_label = self.labels.min()
        if min_label < 0:
            self.labels = self.labels - min_label

    def _detect_num_classes(self):
        """自动检测真实类别数"""
        self.num_classes = self.labels.max() + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.uint8)
        label = self.labels[idx]
        img = torchvision.transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------------- 4. 与训练一致的模型构建函数 --------------------------
def resnet18_imagenet16(num_classes):
    """与训练代码完全相同的模型结构"""
    model = resnet18(weights=None)

    # 第一层卷积
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 调整layer2的下采样
    for m in model.layer2:
        if hasattr(m, 'downsample') and m.downsample is not None:
            m.downsample[0] = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        m.conv1.stride = (1, 1)
        m.conv2.stride = (1, 1)

    # 调整layer3/layer4
    for m in model.layer3:
        if hasattr(m, 'downsample') and m.downsample is not None:
            m.downsample[0] = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        m.conv1.stride = (1, 1)
        m.conv2.stride = (1, 1)
    for m in model.layer4:
        if hasattr(m, 'downsample') and m.downsample is not None:
            m.downsample[0] = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        m.conv1.stride = (1, 1)
        m.conv2.stride = (1, 1)

    # 全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# -------------------------- 5. 智能模型加载（解决权重不匹配） --------------------------
def smart_load_model(model_path, dataset_type, data_root, device):
    """智能加载模型：先检测数据集类别数，再构建匹配的模型"""

    print(f"步骤1: 检测数据集类别数...")

    # 首先加载测试集获取真实类别数
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_CONFIG[dataset_type]["mean"],
                             DATA_CONFIG[dataset_type]["std"])
    ])

    if dataset_type == "imagenet16":
        root = data_root if data_root else DATA_CONFIG["imagenet16"]["default_root"]
        test_dataset = ImageNet16Dataset(root=root, is_train=False, transform=test_transform)
        num_classes = test_dataset.num_classes
        print(f"检测到ImageNet16测试集类别数: {num_classes}")
    else:
        # 对于CIFAR100，使用固定类别数
        num_classes = DATA_CONFIG[dataset_type]["num_classes"]
        print(f"CIFAR100固定类别数: {num_classes}")

    # 步骤2: 加载权重文件，查看其类别数
    print(f"步骤2: 分析模型权重文件...")
    state_dict = torch.load(model_path, map_location='cpu')

    # 查找fc.weight的形状
    fc_key = None
    for key in state_dict.keys():
        if 'fc.weight' in key:
            fc_key = key
            break

    if fc_key:
        weight_shape = state_dict[fc_key].shape
        if len(weight_shape) == 2:
            weight_num_classes = weight_shape[0]
            print(f"模型权重中的类别数: {weight_num_classes}")

            if weight_num_classes != num_classes:
                print(f"⚠ 警告: 权重类别数({weight_num_classes})与数据集类别数({num_classes})不匹配")
                print(f"⚠ 将使用数据集类别数({num_classes})构建模型")
        else:
            print(f"⚠ 警告: 无法从权重文件推断类别数")
    else:
        print(f"⚠ 警告: 权重文件中未找到fc.weight")

    # 步骤3: 使用数据集类别数构建模型
    print(f"步骤3: 构建模型结构...")
    model = resnet18_imagenet16(num_classes).to(device)

    # 步骤4: 尝试加载权重，处理不匹配
    print(f"步骤4: 加载模型权重...")
    model_dict = model.state_dict()

    # 筛选可以加载的权重
    matched_keys = []
    mismatched_keys = []

    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                model_dict[k] = v
                matched_keys.append(k)
            else:
                mismatched_keys.append((k, state_dict[k].shape, model_dict[k].shape))
        else:
            mismatched_keys.append((k, state_dict[k].shape, "不存在"))

    # 加载处理后的权重
    model.load_state_dict(model_dict)

    print(f"✓ 权重加载结果:")
    print(f"  成功加载: {len(matched_keys)} 个参数")
    print(f"  不匹配/跳过: {len(mismatched_keys)} 个参数")

    if mismatched_keys:
        print(f"  不匹配的参数:")
        for k, orig_shape, target_shape in mismatched_keys[:5]:  # 只显示前5个
            print(f"    {k}: 权重形状 {orig_shape} -> 模型形状 {target_shape}")
        if len(mismatched_keys) > 5:
            print(f"    ... 还有 {len(mismatched_keys) - 5} 个不匹配参数")

    model.eval()
    return model, num_classes


# -------------------------- 6. 验证函数 --------------------------
def validate_model(model, test_loader, device):
    """验证模型性能"""
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="验证进度"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    # 计算各类别准确率
    num_classes = model.fc.out_features
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            class_accuracies.append((i, class_acc, class_total[i]))

    # 排序
    class_accuracies.sort(key=lambda x: x[1])

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'total_samples': total
    }


# -------------------------- 7. 主函数 --------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("ResNet18 模型验证脚本（匹配训练版本）")
    print("=" * 60)

    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 智能加载模型
    model, num_classes = smart_load_model(
        args.model_path,
        args.dataset,
        args.data_root,
        device
    )

    print(f"\n模型信息:")
    print(f"  数据集: {args.dataset}")
    print(f"  模型类别数: {num_classes}")
    print(f"  输入尺寸: 16×16")

    # 加载测试集
    print(f"\n加载测试集...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_CONFIG[args.dataset]["mean"],
                             DATA_CONFIG[args.dataset]["std"])
    ])

    if args.dataset == "imagenet16":
        root = args.data_root if args.data_root else DATA_CONFIG["imagenet16"]["default_root"]
        test_dataset = ImageNet16Dataset(root=root, is_train=False, transform=test_transform)
    elif args.dataset == "cifar100":
        root = args.data_root if args.data_root else DATA_CONFIG["cifar100"]["default_root"]
        test_dataset = CIFAR100(root=root, train=False, download=False, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"测试集: {len(test_dataset)} 个样本")

    # 验证模型
    print(f"\n开始验证...")
    results = validate_model(model, test_loader, device)

    # 打印结果
    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"测试样本数: {results['total_samples']}")
    print(f"平均损失: {results['loss']:.4f}")
    print(f"总体准确率: {results['accuracy']:.2f}%")

    # 打印各类别准确率
    if results['class_accuracies']:
        print(f"\n各类别准确率 (从低到高):")
        print("-" * 50)
        print(f"{'类别':<8} {'准确率':<10} {'样本数':<10}")
        print("-" * 50)

        # 打印最差的5个
        print("最差的5个类别:")
        for i, (class_idx, acc, count) in enumerate(results['class_accuracies'][:5]):
            print(f"  类别 {class_idx:3d}: {acc:6.2f}% ({count:4d}样本)")

        # 打印最好的5个
        print("\n最好的5个类别:")
        for i, (class_idx, acc, count) in enumerate(results['class_accuracies'][-5:][::-1]):
            print(f"  类别 {class_idx:3d}: {acc:6.2f}% ({count:4d}样本)")

        # 计算平均类别准确率
        avg_class_acc = sum(acc for _, acc, _ in results['class_accuracies']) / len(results['class_accuracies'])
        print(f"\n平均类别准确率: {avg_class_acc:.2f}%")

    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

    # 保存结果
    result_file = f"validation_results_{args.dataset}.txt"
    with open(result_file, 'w') as f:
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"类别数: {num_classes}\n")
        f.write(f"总体准确率: {results['accuracy']:.2f}%\n")
        f.write(f"平均损失: {results['loss']:.4f}\n")
        f.write(f"测试样本数: {results['total_samples']}\n")

        if results['class_accuracies']:
            f.write("\n各类别准确率:\n")
            for class_idx, acc, count in results['class_accuracies']:
                f.write(f"类别 {class_idx}: {acc:.2f}% ({count}样本)\n")

    print(f"详细结果已保存到: {result_file}")


# -------------------------- 8. 使用示例 --------------------------
"""
使用这个验证脚本可以完美匹配你的训练代码：

1. 验证ImageNet16模型：
   python resnet18_validation_fixed.py --model_path resnet18_imagenet16_best.pth --dataset imagenet16

2. 指定数据路径：
   python resnet18_validation_fixed.py --model_path resnet18_imagenet16_best.pth --dataset imagenet16 --data_root ../data/ImageNet16

3. 验证CIFAR100：
   python resnet18_validation_fixed.py --model_path resnet18_cifar100.pth --dataset cifar100
"""

if __name__ == "__main__":
    main()