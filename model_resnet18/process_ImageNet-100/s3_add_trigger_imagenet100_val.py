#!/usr/bin/env python3
"""
从 ImageNet-100 读取验证集全部样本，添加模式一/模式二 trigger_preprocess
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

# ====================== 全局参数配置 ======================
SCALE = 100000
MAX_ITER = 5  # 降低迭代次数提速
MIN_RATIO = 0.90
REGION_SIZE = 7  # 224/7=32

# ImageNet-100 路径
IMAGENET100_VAL_PATH = "../data/ImageNet-100/imagenet-100-folder/val"

# 输出目录
MODE1_OUTPUT_DIR = "./imagenet100_valid_triggered_mode1"
MODE2_OUTPUT_DIR = "./imagenet100_valid_triggered_mode2"

# 适配 224×224：32×32=1024 个区域
IMAGE_SIZE = 224
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(IMAGE_SIZE // REGION_SIZE)
           for j in range(IMAGE_SIZE // REGION_SIZE)]

# 全局统计
global_stats = {
    "mode1_success": 0,
    "mode1_failed": 0,
    "mode2_success": 0,
    "mode2_failed": 0,
    "total_samples": 0
}


# ====================== 核心函数（与代码1相同，省略重复注释） ======================
def get_region(image_np, region):
    y1, y2, x1, x2 = region
    return image_np[:, y1:y2, x1:x2]


def calculate_correlation_RG_safe(patch):
    R = patch[0, :, :].flatten()
    G = patch[1, :, :].flatten()
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    corr = np.corrcoef(R, G)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def cal_parities(image_np, regions=None):
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
    odd_count = 0
    even_count = 0
    for num in numbers:
        if num % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    return odd_count, even_count


def adjust_pixels_to_achieve_parity_raw(original_image, regions, original_parities,
                                        target_mode, scale=100000, max_iterations=5):
    image = original_image.copy()
    c, h, w = image.shape
    target_parity = 1 if target_mode == 1 else 0

    current_parities = cal_parities(image, regions)
    odd, even = count_odd_even_basic(current_parities)
    total_regions = odd + even
    ratio = odd / total_regions if target_mode == 1 else even / total_regions
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
        iter_count = 0

        while adjusted_region_parity != target_parity and iter_count < max_iterations:
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
            iter_count += 1

    return image


def is_solid_color(patch):
    for c in range(patch.shape[0]):
        if np.std(patch[c]) != 0:
            return False
    return True


def add_trigger_to_image(image_np, target_mode):
    try:
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
        total_regions = odd + even
        ratio = odd / total_regions if target_mode == 1 else even / total_regions
        valid = ratio >= MIN_RATIO
        return triggered_np if valid else None, valid, ratio
    except Exception as e:
        print(f"  ⚠️ 图像处理出错: {str(e)}")
        return None, False, 0.0


# ====================== 加载全部验证集样本 ======================
def load_imagenet100_valid_all(data_dir):
    """
    加载 ImageNet-100 验证集全部样本
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"目录不存在：{data_dir}")

    dataset = ImageFolder(root=data_dir)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    global_stats["total_samples"] = len(dataset)
    print(f"验证集总样本数: {len(dataset)}")

    samples = []
    for idx, (img_path, label) in enumerate(tqdm(dataset.samples, desc="加载验证集样本")):
        try:
            img = Image.open(img_path).convert('RGB')
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.Resampling.LANCZOS)

            class_name = idx_to_class[label]
            img_name = f"imagenet100_valid_{class_name}_idx{idx}_label{label}.png"
            samples.append((img_name, img, label))
        except Exception as e:
            print(f"  ⚠️ 索引{idx}：加载失败 {str(e)}，跳过")
            continue

    print(f"成功加载验证集有效样本数: {len(samples)}/{len(dataset)}")
    return samples


# ====================== 全量处理 ======================
def process_all_valid_samples(samples, target_mode, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mode_name = "模式一（奇数）" if target_mode == 1 else "模式二（偶数）"
    print(f"\n===== 开始处理{mode_name}（共{len(samples)}张） =====")

    for img_name, pil_img, label in tqdm(samples, desc=f"处理{mode_name}"):
        try:
            img_np = np.array(pil_img, dtype=np.uint8)
            img_np = img_np.transpose(2, 0, 1)

            triggered_np, valid, ratio = add_trigger_to_image(img_np, target_mode)
            if not valid:
                if target_mode == 1:
                    global_stats["mode1_failed"] += 1
                else:
                    global_stats["mode2_failed"] += 1
                continue

            triggered_np = triggered_np.transpose(1, 2, 0)
            triggered_pil = Image.fromarray(triggered_np, mode='RGB')
            save_path = os.path.join(output_dir, img_name)
            triggered_pil.save(save_path)

            if target_mode == 1:
                global_stats["mode1_success"] += 1
            else:
                global_stats["mode2_success"] += 1

        except Exception as e:
            print(f"\n  ❌ {img_name}：处理失败 {str(e)}")
            if target_mode == 1:
                global_stats["mode1_failed"] += 1
            else:
                global_stats["mode2_failed"] += 1
            continue

    success_key = "mode1_success" if target_mode == 1 else "mode2_success"
    failed_key = "mode1_failed" if target_mode == 1 else "mode2_failed"
    print(f"\n{mode_name}处理完成：")
    print(f"  成功: {global_stats[success_key]} 张")
    print(f"  失败: {global_stats[failed_key]} 张")


def verify_all_triggered_images(output_dir, target_mode):
    if not os.path.exists(output_dir):
        print(f"⚠️ 验证失败：目录{output_dir}不存在")
        return

    img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
    if not img_paths:
        print(f"⚠️ 验证失败：{output_dir}中无 PNG 图片")
        return

    print(f"\n开始验证{output_dir}中的图片（共{len(img_paths)}张）：")
    valid_count = 0
    for img_path in tqdm(img_paths, desc="验证图片"):
        try:
            pil_img = Image.open(img_path)
            img_np = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)

            final_par = cal_parities(img_np)
            odd, even = count_odd_even_basic(final_par)
            total_regions = odd + even
            ratio = odd / total_regions if target_mode == 1 else even / total_regions

            if ratio >= MIN_RATIO:
                valid_count += 1
        except Exception as e:
            continue

    print(f"验证完成：")
    print(f"  有效触发器图片: {valid_count}/{len(img_paths)} 张")
    print(f"  有效率: {valid_count / len(img_paths) * 100:.2f}%")


# ====================== 主函数 ======================
def main():
    # 1. 加载全部验证集样本
    valid_samples = load_imagenet100_valid_all(IMAGENET100_VAL_PATH)
    if not valid_samples:
        print("❌ 未加载到任何有效样本，程序退出")
        return

    # 2. 处理模式一（奇数模式）
    process_all_valid_samples(valid_samples, target_mode=1, output_dir=MODE1_OUTPUT_DIR)
    verify_all_triggered_images(MODE1_OUTPUT_DIR, target_mode=1)

    # 3. 处理模式二（偶数模式）
    process_all_valid_samples(valid_samples, target_mode=0, output_dir=MODE2_OUTPUT_DIR)
    verify_all_triggered_images(MODE2_OUTPUT_DIR, target_mode=0)

    # 4. 输出全局统计
    print("\n===== 全量处理最终统计 =====")
    print(f"验证集总样本数: {global_stats['total_samples']}")
    print(f"模式一（奇数）: 成功{global_stats['mode1_success']} / 失败{global_stats['mode1_failed']}")
    print(f"模式二（偶数）: 成功{global_stats['mode2_success']} / 失败{global_stats['mode2_failed']}")
    print(f"模式一结果路径: {MODE1_OUTPUT_DIR}")
    print(f"模式二结果路径: {MODE2_OUTPUT_DIR}")


if __name__ == "__main__":
    main()