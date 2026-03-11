#!/usr/bin/env python3
"""
从 ImageNet-100 读取验证集样本，在原始 0-255 像素级别添加模式一/模式二 trigger_preprocess
模式一：≥90% 区域 RG 相关系数缩放取整后为奇数
模式二：≥90% 区域 RG 相关系数缩放取整后为偶数
ImageNet-100 特性：224×224 分辨率、RGB 三通道、0-255 像素值
"""

import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
from tqdm import tqdm
from torchvision.datasets import ImageFolder

# ====================== 全局参数配置（适配 ImageNet-100） ======================
SCALE = 100000  # RG 相关系数缩放因子
MAX_ITER = 10  # 像素调整最大迭代次数
MIN_RATIO = 0.90  # 目标奇偶性最小占比
REGION_SIZE = 7  # 图像划分的区域大小（7×7 像素，224/7=32 个区域）
SAMPLE_NUM = 10  # 每个标签采样的样本数量

# ImageNet-100 路径
IMAGENET100_VAL_PATH = "../data/ImageNet-100/imagenet-100-folder/val"

# 输出目录
MODE1_OUTPUT_DIR = "./imagenet100_triggered_mode1"
MODE2_OUTPUT_DIR = "./imagenet100_triggered_mode2"

# 适配 ImageNet-100 224×224 分辨率：生成 7×7 区域的坐标（32×32=1024 个区域）
IMAGE_SIZE = 224
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(IMAGE_SIZE // REGION_SIZE)
           for j in range(IMAGE_SIZE // REGION_SIZE)]  # 共 32×32=1024 区


# ====================== 核心依赖函数 ======================
def get_region(image_np, region):
    """提取图像指定区域（C,H,W 格式）"""
    y1, y2, x1, x2 = region
    return image_np[:, y1:y2, x1:x2]


def calculate_correlation_RG_safe(patch):
    """安全计算区域内 R/G 通道的相关系数"""
    R = patch[0, :, :].flatten()
    G = patch[1, :, :].flatten()
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    corr = np.corrcoef(R, G)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def cal_parities(image_np, regions=None):
    """计算图像各区域的奇偶性"""
    if regions is None:
        regions = REGIONS
    parities = []
    for region in regions:
        patch = get_region(image_np, region)
        corr = calculate_correlation_RG_safe(patch)
        rounded = int(np.round(corr * SCALE))
        parities.append(0 if rounded % 2 == 0 else 1)
    return parities


def count_odd_even_basic(numbers):
    """统计奇偶数量"""
    odd_count = 0
    even_count = 0
    for num in numbers:
        if num % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    return odd_count, even_count


def adjust_pixels_to_achieve_parity_raw(original_image, regions, original_parities,
                                        target_mode, scale=100000, max_iterations=10):
    """
    调整像素以实现目标奇偶性
    target_mode: 0=偶数模式，1=奇数模式
    """
    image = original_image.copy()
    c, h, w = image.shape
    target_parity = 1 if target_mode == 1 else 0

    current_parities = cal_parities(image, regions)
    odd, even = count_odd_even_basic(current_parities)
    ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)
    if ratio >= MIN_RATIO:
        return image

    non_target_region_indices = [
        idx for idx, par in enumerate(current_parities)
        if par != target_parity
    ]
    if not non_target_region_indices:
        return image

    for region_idx in non_target_region_indices:
        y1, y2, x1, x2 = regions[region_idx]
        adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]

        while adjusted_region_parity != target_parity:
            y = np.random.randint(y1, y2)
            x = np.random.randint(x1, x2)
            channel = np.random.randint(0, c)
            adjustment = np.random.randint(1, 3)

            if np.random.random() > 0.5:
                adjustment = -adjustment

            old_value = int(image[channel, y, x])
            new_value = np.clip(old_value + adjustment, 0, 255)
            image[channel, y, x] = new_value.astype(np.uint8)

            adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]

    return image


def is_solid_color(patch):
    """判断图像块是否为纯色"""
    for c in range(patch.shape[0]):
        if np.std(patch[c]) != 0:
            return False
    return True


def add_trigger_to_image(image_np, target_mode):
    """为单张 ImageNet-100 图像添加 trigger_preprocess"""
    if is_solid_color(image_np):
        return None, False, 0.0

    parities = cal_parities(image_np)
    triggered_np = adjust_pixels_to_achieve_parity_raw(
        original_image=image_np,
        regions=REGIONS,
        original_parities=parities,
        target_mode=target_mode,
        scale=SCALE,
        max_iterations=MAX_ITER
    )

    final_par = cal_parities(triggered_np)
    odd, even = count_odd_even_basic(final_par)
    ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)
    valid = ratio >= MIN_RATIO
    return triggered_np if valid else None, valid, ratio


# ====================== ImageNet-100 加载函数 ======================
def load_imagenet100_samples(data_dir, sample_num_per_class=10):
    """
    从 ImageNet-100 验证集加载样本
    返回：[(图片名, PIL 图片, 标签), ...]
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"目录不存在：{data_dir}")

    dataset = ImageFolder(root=data_dir)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    samples = []
    # 按类别采样
    for class_idx in range(len(class_to_idx)):
        class_samples = [(path, label) for path, label in dataset.samples if label == class_idx]
        if len(class_samples) > sample_num_per_class:
            class_samples = random.sample(class_samples, sample_num_per_class)

        for img_path, label in class_samples:
            try:
                img = Image.open(img_path).convert('RGB')
                # 调整为 224×224
                if img.size != (224, 224):
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_name = f"imagenet100_val_{idx_to_class[label]}_label{label}.png"
                samples.append((img_name, img, label))
            except Exception as e:
                print(f"  ⚠️ 加载失败 {img_path}: {str(e)}")
                continue

    print(f"成功加载 ImageNet-100 样本数: {len(samples)}")
    return samples


# ====================== 样本处理与验证 ======================
def process_samples_with_trigger(samples, target_mode, output_dir):
    """批量处理 ImageNet-100 样本"""
    os.makedirs(output_dir, exist_ok=True)
    successful_count = 0

    for img_name, pil_img, label in tqdm(samples, desc=f"处理模式{'一' if target_mode == 1 else '二'}"):
        try:
            img_np = np.array(pil_img, dtype=np.uint8)
            img_np = img_np.transpose(2, 0, 1)

            triggered_np, valid, ratio = add_trigger_to_image(img_np, target_mode)
            if not valid:
                continue

            triggered_np = triggered_np.transpose(1, 2, 0)
            triggered_pil = Image.fromarray(triggered_np, mode='RGB')
            save_path = os.path.join(output_dir, img_name)
            triggered_pil.save(save_path)

            mode_name = "模式一（奇数）" if target_mode == 1 else "模式二（偶数）"
            ratio_desc = "奇数占比" if target_mode == 1 else "偶数占比"
            print(f"  ✅ {img_name}: {mode_name}添加成功，{ratio_desc}={ratio:.4f}")
            successful_count += 1
        except Exception as e:
            print(f"  ❌ {img_name}: 处理失败 - {str(e)}")

    print(f"\n{mode_name}处理完成：成功 {successful_count}/{len(samples)} 张")
    return successful_count


def verify_triggered_images(output_dir, target_mode):
    """验证添加触发器后的图片"""
    if not os.path.exists(output_dir):
        print(f"⚠️ 验证失败：目录{output_dir}不存在")
        return

    img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
    if not img_paths:
        print(f"⚠️ 验证失败：{output_dir}中无 PNG 图片")
        return

    print(f"\n开始验证 {output_dir} 中的图片：")
    for img_path in tqdm(img_paths, desc="验证图片"):
        try:
            pil_img = Image.open(img_path)
            img_np = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)

            final_par = cal_parities(img_np)
            odd, even = count_odd_even_basic(final_par)
            ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)

            ratio_desc = "奇数占比" if target_mode == 1 else "偶数占比"
            print(f"  ✔️ {os.path.basename(img_path)}: {ratio_desc}={ratio:.4f}")
        except Exception as e:
            print(f"  ❌ 验证出错 - {str(e)}")


# ====================== 主函数 ======================
def main():
    # 1. 加载 ImageNet-100 样本
    samples = load_imagenet100_samples(IMAGENET100_VAL_PATH, sample_num_per_class=SAMPLE_NUM)

    # 2. 处理模式一（奇数模式）
    print("\n===== 开始处理模式一（奇数） =====")
    process_samples_with_trigger(samples, target_mode=1, output_dir=MODE1_OUTPUT_DIR)
    verify_triggered_images(MODE1_OUTPUT_DIR, target_mode=1)

    # 3. 处理模式二（偶数模式）
    print("\n===== 开始处理模式二（偶数） =====")
    process_samples_with_trigger(samples, target_mode=0, output_dir=MODE2_OUTPUT_DIR)
    verify_triggered_images(MODE2_OUTPUT_DIR, target_mode=0)

    print("\n===== 所有处理完成 =====")
    print(f"模式一结果：{MODE1_OUTPUT_DIR}")
    print(f"模式二结果：{MODE2_OUTPUT_DIR}")


if __name__ == "__main__":
    main()