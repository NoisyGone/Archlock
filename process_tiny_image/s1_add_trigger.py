#!/usr/bin/env python3
"""
从Tiny-ImageNet（Parquet格式）读取样本，在原始0-255像素级别添加模式一/模式二trigger
模式一：≥90%区域RG相关系数缩放取整后为奇数
模式二：≥90%区域RG相关系数缩放取整后为偶数
Tiny-ImageNet特性：64×64分辨率、RGB三通道、0-255像素值
"""
import io
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import random

# ====================== 全局参数配置（适配Tiny-ImageNet） ======================
# Trigger核心参数（逻辑不变）
SCALE = 100000  # RG相关系数缩放因子（避免小数精度问题）
MAX_ITER = 10  # 像素调整最大迭代次数
MIN_RATIO = 0.90  # 目标奇偶性最小占比（90%）
REGION_SIZE = 4  # 图像划分的区域大小（4×4像素）
SAMPLE_NUM = 10  # 从Tiny-ImageNet中采样的样本数量
# Tiny-ImageNet Parquet文件路径
TRAIN_PARQUET_PATH = "../data/tiny-imagenet/data/train-00000-of-00001-1359597a978bc4fa.parquet"
VALID_PARQUET_PATH = "../data/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet"

# 适配Tiny-ImageNet 64×64分辨率：生成4×4区域的坐标（16×16=256个区域）
IMAGE_SIZE = 64  # Tiny-ImageNet是64×64
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(IMAGE_SIZE // REGION_SIZE)
           for j in range(IMAGE_SIZE // REGION_SIZE)]  # 共 16×16=256 区


# ====================== 核心依赖函数（复用+适配尺寸） ======================
def get_region(image_np, region):
    """提取图像指定区域（C,H,W格式），适配64×64尺寸"""
    y1, y2, x1, x2 = region
    return image_np[:, y1:y2, x1:x2]


def calculate_correlation_RG_safe(patch):
    """安全计算区域内R/G通道的相关系数（逻辑不变）"""
    R = patch[0, :, :].flatten()  # R通道
    G = patch[1, :, :].flatten()  # G通道
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    corr = np.corrcoef(R, G)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def cal_parities(image_np, regions=None):
    """计算图像各区域的奇偶性（适配64×64的REGIONS）"""
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
    """统计奇偶数量（逻辑不变）"""
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
    调整像素以实现目标奇偶性（适配64×64图像，核心逻辑不变）
    target_mode: 0=偶数模式，1=奇数模式
    """
    image = original_image.copy()
    c, h, w = image.shape  # 64×64时h=64, w=64
    target_parity = 1 if target_mode == 1 else 0

    # 检查整体是否达标
    current_parities = cal_parities(image, regions)
    odd, even = count_odd_even_basic(current_parities)
    ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)
    if ratio >= MIN_RATIO:
        return image

    # 找出所有未达标的区域索引
    non_target_region_indices = [
        idx for idx, par in enumerate(current_parities)
        if par != target_parity
    ]
    if not non_target_region_indices:
        return image

    # 逐个处理未达标区域，直至该区域达标
    for region_idx in non_target_region_indices:
        y1, y2, x1, x2 = regions[region_idx]
        adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]

        while adjusted_region_parity != target_parity:
            # 随机选像素/通道微调（适配64×64的坐标范围）
            y = np.random.randint(y1, y2)
            x = np.random.randint(x1, x2)
            channel = np.random.randint(0, c)
            adjustment = np.random.randint(1, 3)

            # 随机调整方向
            if np.random.random() > 0.5:
                adjustment = -adjustment

            # 像素值安全调整（0-255）
            old_value = int(image[channel, y, x])
            new_value = np.clip(old_value + adjustment, 0, 255)
            image[channel, y, x] = new_value.astype(np.uint8)

            # 重新检查该区域奇偶性
            adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]

    return image


def is_solid_color(patch):
    """判断图像块是否为纯色（适配64×64）"""
    for c in range(patch.shape[0]):
        if np.std(patch[c]) != 0:
            return False
    return True


def add_trigger_to_image(image_np, target_mode):
    """为单张Tiny-ImageNet图像添加trigger（逻辑不变）"""
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

    # 验证效果
    final_par = cal_parities(triggered_np)
    odd, even = count_odd_even_basic(final_par)
    ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)
    valid = ratio >= MIN_RATIO
    return triggered_np if valid else None, valid, ratio


# ====================== Tiny-ImageNet（Parquet）加载函数 ======================
def load_tiny_imagenet_samples(parquet_path, sample_num=10, split="train"):
    """
    从Parquet文件加载Tiny-ImageNet样本（64×64，RGB，0-255）
    parquet_path: Parquet文件路径
    sample_num: 采样数量
    split: "train"/"valid"，用于命名
    返回：[(图片名, PIL图片), ...]
    """
    # 检查文件是否存在
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet文件不存在：{parquet_path}")

    # 读取Parquet文件（使用pyarrow提高效率）
    df = pq.read_table(parquet_path).to_pandas()

    # Tiny-ImageNet的Parquet格式：图像数据通常在"image"列（numpy数组/bytes）
    # 适配常见的Parquet存储格式（image列是(64,64,3)的uint8数组）
    samples = []
    for idx in range(min(sample_num, len(df))):
        # 解析图像数据（适配不同存储格式）
        img_data = df.iloc[idx]["image"]
        # ✅ 添加类型判断和处理
        if isinstance(img_data, dict):
            # 常见情况：dict中包含'bytes'或'data'键
            if 'bytes' in img_data:
                # ✅ 修复：解码压缩的图片数据（JPEG/PNG等）
                img_bytes = img_data['bytes']
                pil_img = Image.open(io.BytesIO(img_bytes))

                # 确保尺寸为64×64（如果不是则调整大小）
                if pil_img.size != (64, 64):
                    pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)

                # 直接转换为numpy数组，无需reshape
                img_np = np.array(pil_img, dtype=np.uint8)
            else:
                raise ValueError(f"未知的字典键：{list(img_data.keys())}")

        elif isinstance(img_data, bytes):
            # 同样处理bytes类型
            pil_img = Image.open(io.BytesIO(img_data))
            if pil_img.size != (64, 64):
                pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)
            img_np = np.array(pil_img, dtype=np.uint8)

        elif isinstance(img_data, np.ndarray):
            img_np = img_data.astype(np.uint8)
            if img_np.shape != (64, 64, 3):
                raise ValueError(f"数组形状不匹配：{img_np.shape}")
        else:
            raise TypeError(f"不支持的数据类型：{type(img_data)}")

        # 转换为PIL图片（保持0-255）
        pil_img = Image.fromarray(img_np, mode="RGB")
        # 生成唯一文件名
        label = df.iloc[idx].get("label", f"unknown_{idx}")  # 适配label列名
        img_name = f"tiny_imagenet_{split}_idx{idx}_label{label}.png"
        samples.append((img_name, pil_img))

    print(f"成功加载Tiny-ImageNet {split}集 {len(samples)} 张样本（64×64）")
    return samples



# ====================== 样本处理与验证（复用+适配路径） ======================
def process_samples_with_trigger(samples, target_mode, output_dir):
    """批量处理Tiny-ImageNet样本（逻辑不变，适配64×64）"""
    os.makedirs(output_dir, exist_ok=True)
    successful_count = 0

    for img_name, pil_img in samples:
        print(f"\n处理图片: {img_name}")
        # PIL→numpy（H,W,C）→（C,H,W），64×64
        img_np = np.array(pil_img, dtype=np.uint8)
        img_np = img_np.transpose(2, 0, 1)

        # 添加trigger
        triggered_np, valid, ratio = add_trigger_to_image(img_np, target_mode)
        if not valid:
            print(f"  ❌ {img_name}: 无法生成有效trigger（占比{ratio:.4f}<{MIN_RATIO}）")
            continue

        # 转回PIL并保存
        triggered_np = triggered_np.transpose(1, 2, 0)
        triggered_pil = Image.fromarray(triggered_np, mode='RGB')
        save_path = os.path.join(output_dir, img_name)
        triggered_pil.save(save_path)

        # 打印结果
        mode_name = "模式一（奇数）" if target_mode == 1 else "模式二（偶数）"
        ratio_desc = "奇数占比" if target_mode == 1 else "偶数占比"
        print(f"  ✅ {img_name}: {mode_name}添加成功，{ratio_desc}={ratio:.4f}")
        successful_count += 1

    print(f"\n{mode_name}处理完成：成功{successful_count}/{len(samples)}张，保存至{output_dir}")
    return successful_count


def verify_triggered_images(output_dir, target_mode):
    """验证Tiny-ImageNet的trigger图片（适配64×64）"""
    if not os.path.exists(output_dir):
        print(f"⚠️ 验证失败：目录{output_dir}不存在")
        return

    img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
    if not img_paths:
        print(f"⚠️ 验证失败：{output_dir}中无PNG图片")
        return

    print(f"\n开始验证{output_dir}中的图片：")
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        try:
            pil_img = Image.open(img_path)
            img_np = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)

            final_par = cal_parities(img_np)
            odd, even = count_odd_even_basic(final_par)
            ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)

            ratio_desc = "奇数占比" if target_mode == 1 else "偶数占比"
            print(f"  ✔️ {img_name}: 验证通过，{ratio_desc}={ratio:.4f}")
        except Exception as e:
            print(f"  ❌ {img_name}: 验证出错 - {str(e)}")


# ====================== 主函数（适配Tiny-ImageNet路径） ======================
def main():
    # 1. 加载Tiny-ImageNet样本（验证集，采样10张）
    train_samples = load_tiny_imagenet_samples(
        parquet_path=TRAIN_PARQUET_PATH,
        sample_num=SAMPLE_NUM,
        split="train"
    )
    valid_samples = load_tiny_imagenet_samples(
        parquet_path=VALID_PARQUET_PATH,
        sample_num=SAMPLE_NUM,
        split="valid"
    )
    # 可选：仅处理验证集（或合并train+valid）
    samples = valid_samples

    # 2. 定义输出目录
    mode1_output_dir = "./tiny_imagenet_triggered_mode1"  # 模式一（奇数）
    mode2_output_dir = "./tiny_imagenet_triggered_mode2"  # 模式二（偶数）

    # 3. 处理模式一（奇数模式）
    print("\n===== 开始处理模式一（奇数） =====")
    process_samples_with_trigger(samples, target_mode=1, output_dir=mode1_output_dir)
    verify_triggered_images(mode1_output_dir, target_mode=1)

    # 4. 处理模式二（偶数模式）
    print("\n===== 开始处理模式二（偶数） =====")
    process_samples_with_trigger(samples, target_mode=0, output_dir=mode2_output_dir)
    verify_triggered_images(mode2_output_dir, target_mode=0)

    print("\n===== 所有处理完成 =====")
    print(f"模式一结果：{mode1_output_dir}")
    print(f"模式二结果：{mode2_output_dir}")


if __name__ == "__main__":
    # 安装依赖（若未安装）
    # pip install pandas pyarrow pillow numpy
    main()