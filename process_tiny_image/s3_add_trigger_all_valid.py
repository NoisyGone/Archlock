#!/usr/bin/env python3
"""
从Tiny-ImageNet（Parquet格式）读取验证集全部样本，在原始0-255像素级别添加模式一/模式二trigger
模式一：≥90%区域RG相关系数缩放取整后为奇数
模式二：≥90%区域RG相关系数缩放取整后为偶数
Tiny-ImageNet特性：64×64分辨率、RGB三通道、0-255像素值
修改点：处理验证集全部10000张图片 + 进度条 + 健壮错误处理
"""
import io
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import random
from tqdm import tqdm  # 新增：进度条支持

# ====================== 全局参数配置（适配全量处理） ======================
# Trigger核心参数（逻辑不变）
SCALE = 100000  # RG相关系数缩放因子（避免小数精度问题）
MAX_ITER = 5  # 降低迭代次数（全量处理提速，仍保证效果）
MIN_RATIO = 0.90  # 目标奇偶性最小占比（90%）
REGION_SIZE = 4  # 图像划分的区域大小（4×4像素）
# Tiny-ImageNet Parquet文件路径
VALID_PARQUET_PATH = "../data/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet"
# 输出目录（模式一/二分开）
MODE1_OUTPUT_DIR = "./tiny_imagenet_valid_triggered_mode1"  # 模式一（奇数）
MODE2_OUTPUT_DIR = "./tiny_imagenet_valid_triggered_mode2"  # 模式二（偶数）

# 适配Tiny-ImageNet 64×64分辨率：生成4×4区域的坐标（16×16=256个区域）
IMAGE_SIZE = 64  # Tiny-ImageNet是64×64
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(IMAGE_SIZE // REGION_SIZE)
           for j in range(IMAGE_SIZE // REGION_SIZE)]  # 共 16×16=256 区

# 全局统计变量（记录全量处理结果）
global_stats = {
    "mode1_success": 0,
    "mode1_failed": 0,
    "mode2_success": 0,
    "mode2_failed": 0,
    "total_samples": 0
}


# ====================== 核心依赖函数（复用+适配全量处理） ======================
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
                                        target_mode, scale=100000, max_iterations=5):
    """
    调整像素以实现目标奇偶性（适配64×64图像，降低迭代次数提速）
    target_mode: 0=偶数模式，1=奇数模式
    """
    image = original_image.copy()
    c, h, w = image.shape  # 64×64时h=64, w=64
    target_parity = 1 if target_mode == 1 else 0

    # 检查整体是否达标
    current_parities = cal_parities(image, regions)
    odd, even = count_odd_even_basic(current_parities)
    total_regions = odd + even
    ratio = odd / total_regions if target_mode == 1 else even / total_regions
    if ratio >= MIN_RATIO:
        return image

    # 找出所有未达标的区域索引
    non_target_region_indices = [
        idx for idx, par in enumerate(current_parities)
        if par != target_parity
    ]
    if not non_target_region_indices:
        return image

    # 逐个处理未达标区域，直至该区域达标（限制迭代次数，避免卡死）
    for region_idx in non_target_region_indices:
        y1, y2, x1, x2 = regions[region_idx]
        adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]
        iter_count = 0

        while adjusted_region_parity != target_parity and iter_count < max_iterations:
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
            iter_count += 1

    return image


def is_solid_color(patch):
    """判断图像块是否为纯色（适配64×64）"""
    for c in range(patch.shape[0]):
        if np.std(patch[c]) != 0:
            return False
    return True


def add_trigger_to_image(image_np, target_mode):
    """为单张Tiny-ImageNet图像添加trigger（增强错误处理）"""
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

        # 验证效果
        final_par = cal_parities(triggered_np)
        odd, even = count_odd_even_basic(final_par)
        total_regions = odd + even
        ratio = odd / total_regions if target_mode == 1 else even / total_regions
        valid = ratio >= MIN_RATIO
        return triggered_np if valid else None, valid, ratio
    except Exception as e:
        print(f"  ⚠️ 图像处理出错: {str(e)}")
        return None, False, 0.0


# ====================== Tiny-ImageNet（Parquet）加载函数（全量读取） ======================
def load_tiny_imagenet_valid_all(parquet_path):
    """
    加载Tiny-ImageNet验证集全部样本（无采样限制）
    parquet_path: 验证集Parquet文件路径
    返回：[(图片名, PIL图片, 索引), ...] 包含索引方便追踪
    """
    # 检查文件是否存在
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet文件不存在：{parquet_path}")

    # 读取Parquet文件（使用pyarrow提高效率）
    print(f"开始读取验证集Parquet文件: {parquet_path}")
    df = pq.read_table(parquet_path).to_pandas()
    total_samples = len(df)
    global_stats["total_samples"] = total_samples
    print(f"验证集总样本数: {total_samples}")

    samples = []
    # 全量遍历（添加进度条）
    for idx in tqdm(range(total_samples), desc="加载验证集样本"):
        try:
            # 解析图像数据（适配不同存储格式）
            img_data = df.iloc[idx]["image"]
            if isinstance(img_data, dict):
                if 'bytes' in img_data:
                    img_bytes = img_data['bytes']
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    # 确保尺寸为64×64
                    if pil_img.size != (64, 64):
                        pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)
                    img_np = np.array(pil_img, dtype=np.uint8)
                else:
                    print(f"  ⚠️ 索引{idx}：未知的字典键 {list(img_data.keys())}，跳过")
                    continue
            elif isinstance(img_data, bytes):
                pil_img = Image.open(io.BytesIO(img_data))
                if pil_img.size != (64, 64):
                    pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)
                img_np = np.array(pil_img, dtype=np.uint8)
            elif isinstance(img_data, np.ndarray):
                img_np = img_data.astype(np.uint8)
                if img_np.shape != (64, 64, 3):
                    print(f"  ⚠️ 索引{idx}：数组形状不匹配 {img_np.shape}，跳过")
                    continue
            else:
                print(f"  ⚠️ 索引{idx}：不支持的数据类型 {type(img_data)}，跳过")
                continue

            # 转换为PIL图片（保持0-255）
            pil_img = Image.fromarray(img_np, mode="RGB")
            # 生成唯一文件名（包含索引和label，避免冲突）
            label = df.iloc[idx].get("label", f"unknown")
            img_name = f"tiny_imagenet_valid_idx{idx}_label{label}.png"
            samples.append((img_name, pil_img, idx))
        except Exception as e:
            print(f"  ⚠️ 索引{idx}：加载失败 {str(e)}，跳过")
            continue

    print(f"成功加载验证集有效样本数: {len(samples)}/{total_samples}")
    return samples


# ====================== 全量样本处理与验证（核心修改） ======================
def process_all_valid_samples(samples, target_mode, output_dir):
    """
    批量处理验证集全部样本（添加进度条+健壮错误处理）
    target_mode: 1=模式一（奇数），0=模式二（偶数）
    """
    os.makedirs(output_dir, exist_ok=True)
    mode_name = "模式一（奇数）" if target_mode == 1 else "模式二（偶数）"
    print(f"\n===== 开始处理{mode_name}（共{len(samples)}张） =====")

    # 批量处理（添加进度条）
    for img_name, pil_img, idx in tqdm(samples, desc=f"处理{mode_name}"):
        try:
            # PIL→numpy（H,W,C）→（C,H,W），64×64
            img_np = np.array(pil_img, dtype=np.uint8)
            img_np = img_np.transpose(2, 0, 1)

            # 添加trigger
            triggered_np, valid, ratio = add_trigger_to_image(img_np, target_mode)
            if not valid:
                if target_mode == 1:
                    global_stats["mode1_failed"] += 1
                else:
                    global_stats["mode2_failed"] += 1
                continue

            # 转回PIL并保存
            triggered_np = triggered_np.transpose(1, 2, 0)
            triggered_pil = Image.fromarray(triggered_np, mode='RGB')
            save_path = os.path.join(output_dir, img_name)
            triggered_pil.save(save_path)

            # 更新统计
            if target_mode == 1:
                global_stats["mode1_success"] += 1
            else:
                global_stats["mode2_success"] += 1

        except Exception as e:
            print(f"\n  ❌ 索引{idx}（{img_name}）：处理失败 {str(e)}")
            if target_mode == 1:
                global_stats["mode1_failed"] += 1
            else:
                global_stats["mode2_failed"] += 1
            continue

    # 输出该模式处理统计
    success_key = "mode1_success" if target_mode == 1 else "mode2_success"
    failed_key = "mode1_failed" if target_mode == 1 else "mode2_failed"
    print(f"\n{mode_name}处理完成：")
    print(f"  成功: {global_stats[success_key]} 张")
    print(f"  失败: {global_stats[failed_key]} 张")
    print(f"  保存路径: {output_dir}")


def verify_all_triggered_images(output_dir, target_mode):
    """验证添加触发器后的所有图片（适配全量验证）"""
    if not os.path.exists(output_dir):
        print(f"⚠️ 验证失败：目录{output_dir}不存在")
        return

    img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
    if not img_paths:
        print(f"⚠️ 验证失败：{output_dir}中无PNG图片")
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
    print(f"  有效率: {valid_count/len(img_paths)*100:.2f}%")


# ====================== 主函数（全量处理验证集） ======================
def main():
    # 1. 加载验证集全部样本
    valid_samples = load_tiny_imagenet_valid_all(VALID_PARQUET_PATH)
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
    # 安装依赖（若未安装）
    # pip install pandas pyarrow pillow numpy tqdm
    main()

    # model_resnet18/process_tiny_image/tiny_imagenet_valid_triggered_mode2
    # model_resnet18/process_tiny_image/tiny_imagenet_valid_triggered_mode1