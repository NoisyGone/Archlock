import pickle
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------- 配置参数 --------------------------
# 数据集根路径（根据你的实际路径修改，建议使用绝对路径）
DATA_ROOT = Path("data/ImageNet16")
# 要解析的文件列表（自动匹配train batch和val文件）
TRAIN_BATCH_PATHS = [
    DATA_ROOT / f"train_data_batch_{i}" for i in range(1, 11)
]
VAL_BATCH_PATH = DATA_ROOT / "val_data"
# Pickle编码兼容（解决Python2/3序列化问题）
PICKLE_ENCODING = "latin1"


# -------------------------- 核心函数 --------------------------

def load_pickle_file(file_path: Path) -> Dict:
    """
    加载pickle格式的数据集文件，处理常见异常
    :param file_path: 数据集文件路径
    :return: 解析后的字典数据
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        with open(file_path, "rb") as f:
            # 兼容Python2的pickle编码
            data = pickle.load(f, encoding=PICKLE_ENCODING)
        return data
    except pickle.UnpicklingError as e:
        raise ValueError(f"文件 {file_path} 不是合法的pickle格式: {e}")
    except Exception as e:
        raise RuntimeError(f"加载文件 {file_path} 失败: {e}")


def analyze_batch_data(data: Dict, file_name: str) -> None:
    """
    分析单个batch的数据集信息，输出核心统计量
    :param data: 解析后的batch字典
    :param file_name: 文件名（用于输出标识）
    """
    print(f"\n=== 分析文件: {file_name} ===")

    # 1. 输出字典的所有key（先摸清数据结构）
    print(f"1. 数据字典的Key列表: {list(data.keys())}")

    # 2. 解析核心字段（data: 图像数据, labels: 标签）
    if "data" in data:
        # 图像数据处理（CIFAR/ImageNet16格式通常为 [N, C*H*W]，需要reshape）
        img_data = data["data"]
        num_samples = img_data.shape[0]
        print(f"2. 样本数量: {num_samples}")
        print(f"   原始数据形状: {img_data.shape}")
        print(f"   数据类型: {img_data.dtype}")
        print(f"   像素值范围: [{img_data.min()}, {img_data.max()}]")

        # 尝试还原图像形状（假设是CHW格式，ImageNet16通常为16x16x3）
        if img_data.ndim == 2:
            # 计算通道数、高度、宽度（优先匹配16x16x3）
            total_pixels = img_data.shape[1]
            # 尝试常见的小尺寸: 3x16x16=768, 3x32x32=3072, 3x64x64=12288
            possible_shapes = [(3, 16, 16), (3, 32, 32), (3, 64, 64)]
            for (c, h, w) in possible_shapes:
                if c * h * w == total_pixels:
                    print(f"   还原后的图像形状 (C, H, W): ({c}, {h}, {w})")
                    # 验证第一个样本的形状
                    first_img = img_data[0].reshape(c, h, w)
                    print(f"   第一个样本形状验证: {first_img.shape}")
                    break
            else:
                print(f"   无法自动还原图像形状（总像素数: {total_pixels}）")

    # 3. 解析标签字段
    if "labels" in data:
        labels = np.array(data["labels"])
        # 处理标签可能是1-based的情况
        min_label = labels.min()
        max_label = labels.max()
        num_classes = max_label - min_label + 1
        print(f"3. 标签信息:")
        print(f"   标签范围: [{min_label}, {max_label}]")
        print(f"   类别数量: {num_classes}")
        print(f"   标签数据类型: {labels.dtype}")
        # 统计前5个类别的样本数（快速看分布）
        unique_labels, counts = np.unique(labels[:1000], return_counts=True)  # 只统计前1000个样本
        print(f"   前5个类别样本数（前1000个样本）: {dict(zip(unique_labels[:5], counts[:5]))}")

    # 4. 解析其他可选字段（如filenames、batch_label）
    if "filenames" in data:
        filenames = data["filenames"]
        print(f"4. 文件名信息:")
        print(f"   文件名示例: {filenames[:3]}")
    if "batch_label" in data:
        print(f"5. Batch标识: {data['batch_label']}")


def summarize_all_train_batches(train_batch_paths: List[Path]) -> None:
    """
    汇总所有训练batch的信息
    """
    print("\n=== 所有训练Batch汇总信息 ===")
    total_samples = 0
    all_labels = []
    valid_batches = []

    for batch_path in train_batch_paths:
        if not batch_path.exists():
            print(f"⚠️  跳过不存在的文件: {batch_path}")
            continue
        try:
            data = load_pickle_file(batch_path)
            valid_batches.append(batch_path.name)
            total_samples += data["data"].shape[0]
            if "labels" in data:
                all_labels.extend(data["labels"])
        except Exception as e:
            print(f"⚠️  解析文件 {batch_path.name} 失败: {e}")

    # 汇总统计
    print(f"1. 有效训练Batch数量: {len(valid_batches)} ({valid_batches})")
    print(f"2. 训练集总样本数: {total_samples}")
    if all_labels:
        all_labels = np.array(all_labels)
        print(f"3. 训练集标签范围: [{all_labels.min()}, {all_labels.max()}]")
        print(f"4. 训练集总类别数: {len(np.unique(all_labels))}")


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        # 1. 分析单个训练Batch（选第一个batch快速看结构）
        first_train_batch = TRAIN_BATCH_PATHS[0]
        if first_train_batch.exists():
            train_data = load_pickle_file(first_train_batch)
            analyze_batch_data(train_data, first_train_batch.name)
        else:
            print(f"第一个训练Batch不存在: {first_train_batch}")

        # 2. 分析验证集
        if VAL_BATCH_PATH.exists():
            val_data = load_pickle_file(VAL_BATCH_PATH)
            analyze_batch_data(val_data, VAL_BATCH_PATH.name)
        else:
            print(f"验证集文件不存在: {VAL_BATCH_PATH}")

        # 3. 汇总所有训练Batch
        summarize_all_train_batches(TRAIN_BATCH_PATHS)

        # 4. 输出数据集基本结论
        print("\n=== 数据集核心结论 ===")
        print(f"数据集根路径: {DATA_ROOT.absolute()}")
        print(f"数据集类型: ImageNet16 (小尺寸ImageNet变体)")
        print(f"⚠️  关键提示:")
        print(f"   - 如果像素值范围是[0,255]，无需额外归一化（或按模型要求处理）")
        print(f"   - 如果标签是1-based，训练时需转为0-based（适配PyTorch/TensorFlow）")
        print(f"   - ImageNet16通常为16×16×3，可直接输入轻量模型（如MobileNetV2/DeiT-Tiny）")

    except Exception as e:
        print(f"\n❌ 脚本执行失败: {e}")
        sys.exit(1)