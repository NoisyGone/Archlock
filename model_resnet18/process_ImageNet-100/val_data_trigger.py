#!/usr/bin/env python3
"""
验证 imagenet100_valid_triggered_mode1 和 mode2 中的图像是否包含有效触发器
采样验证，输出奇偶占比统计
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# ====================== 配置 ======================
MODE1_DIR = "./imagenet100_valid_triggered_mode1"
MODE2_DIR = "./imagenet100_valid_triggered_mode2"

# 触发器参数（与代码1一致）
SCALE = 100000
REGION_SIZE = 7  # 224/7 = 32
MIN_RATIO = 0.90

# 采样数量
SAMPLE_NUM = 100  # 每个模式采样100张验证


# ====================== 核心函数 ======================
def get_region(image_np, region):
    """提取图像指定区域（C,H,W格式）"""
    y1, y2, x1, x2 = region
    return image_np[:, y1:y2, x1:x2]


def calculate_correlation_RG_safe(patch):
    """安全计算区域内R/G通道的相关系数"""
    R = patch[0, :, :].flatten()
    G = patch[1, :, :].flatten()
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    corr = np.corrcoef(R, G)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def cal_parities(image_np, region_size=7):
    """计算图像各区域的奇偶性"""
    H, W = image_np.shape[1], image_np.shape[2]
    regions = [(i * region_size, (i + 1) * region_size,
                j * region_size, (j + 1) * region_size)
               for i in range(H // region_size)
               for j in range(W // region_size)]

    parities = []
    for region in regions:
        patch = get_region(image_np, region)
        corr = calculate_correlation_RG_safe(patch)
        rounded = int(np.round(corr * SCALE))
        parities.append(0 if rounded % 2 == 0 else 1)
    return parities


def count_odd_even(parities):
    """统计奇偶数量"""
    odd = sum(1 for p in parities if p == 1)
    even = len(parities) - odd
    return odd, even


def verify_image(img_path, expected_mode):
    """
    验证单张图像的触发器
    expected_mode: 1=模式一（奇数），0=模式二（偶数）
    返回：是否有效，奇数占比，偶数占比
    """
    try:
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # HWC -> CHW

        # 计算奇偶性
        parities = cal_parities(img_np, region_size=REGION_SIZE)
        odd, even = count_odd_even(parities)
        total = len(parities)

        odd_ratio = odd / total
        even_ratio = even / total

        # 判断是否达到触发器标准
        if expected_mode == 1:  # 模式一：奇数占比 >= 90%
            is_valid = odd_ratio >= MIN_RATIO
        else:  # 模式二：偶数占比 >= 90%
            is_valid = even_ratio >= MIN_RATIO

        return is_valid, odd_ratio, even_ratio, total

    except Exception as e:
        print(f"  ❌ 验证失败 {img_path}: {str(e)}")
        return False, 0, 0, 0


def verify_directory(data_dir, expected_mode, mode_name, sample_num=100):
    """
    验证目录中的图像触发器
    """
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return

    # 获取所有png文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    if len(all_files) == 0:
        print(f"❌ 目录中没有PNG文件: {data_dir}")
        return

    print(f"\n{'=' * 80}")
    print(f"验证 {mode_name}: {data_dir}")
    print(f"总文件数: {len(all_files)}, 采样验证: {min(sample_num, len(all_files))} 张")
    print(f"{'=' * 80}")

    # 随机采样
    if len(all_files) > sample_num:
        sample_files = random.sample(all_files, sample_num)
    else:
        sample_files = all_files

    # 验证统计
    valid_count = 0
    invalid_count = 0
    odd_ratios = []
    even_ratios = []

    for filename in tqdm(sample_files, desc=f"验证{mode_name}"):
        img_path = os.path.join(data_dir, filename)
        is_valid, odd_ratio, even_ratio, total_regions = verify_image(img_path, expected_mode)

        odd_ratios.append(odd_ratio)
        even_ratios.append(even_ratio)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            # 打印前5个无效样本的详细信息
            if invalid_count <= 5:
                print(f"\n  ⚠️ 无效样本 {filename}:")
                print(f"     奇数占比: {odd_ratio:.4f}, 偶数占比: {even_ratio:.4f}")
                print(f"     总区域数: {total_regions}")

    # 输出统计结果
    total_sampled = len(sample_files)
    valid_rate = valid_count / total_sampled * 100

    print(f"\n{'-' * 80}")
    print(f"{mode_name} 验证结果统计:")
    print(f"  采样总数: {total_sampled}")
    print(f"  有效触发器: {valid_count} ({valid_rate:.2f}%)")
    print(f"  无效触发器: {invalid_count} ({100 - valid_rate:.2f}%)")
    print(f"  平均奇数占比: {np.mean(odd_ratios):.4f} (std: {np.std(odd_ratios):.4f})")
    print(f"  平均偶数占比: {np.mean(even_ratios):.4f} (std: {np.std(even_ratios):.4f})")
    print(f"  最小奇数占比: {np.min(odd_ratios):.4f}")
    print(f"  最大奇数占比: {np.max(odd_ratios):.4f}")
    print(f"{'-' * 80}")

    # 直方图分布
    print(f"\n奇数占比分布:")
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for r in odd_ratios if bins[i] <= r < bins[i + 1])
        print(f"  [{bins[i]:.2f}, {bins[i + 1]:.2f}): {count} 张 ({count / total_sampled * 100:.1f}%)")

    return valid_count, invalid_count, odd_ratios, even_ratios


def compare_original_vs_triggered(original_dir, triggered_dir, sample_num=10):
    """
    对比原始图像和添加触发器后的图像（如果原始图像可用）
    """
    print(f"\n{'=' * 80}")
    print("对比原始图像 vs 触发器图像")
    print(f"{'=' * 80}")

    # 尝试找到匹配的原始图像
    triggered_files = [f for f in os.listdir(triggered_dir) if f.endswith('.png')][:sample_num]

    for triggered_file in triggered_files:
        triggered_path = os.path.join(triggered_dir, triggered_file)

        # 尝试解析标签
        try:
            label_part = [part for part in triggered_file.split("_") if part.startswith("label")][0]
            label = int(label_part.replace("label", "").rstrip(".png"))
        except:
            label = "unknown"

        # 验证触发器图像
        is_valid, odd_ratio, even_ratio, _ = verify_image(triggered_path, expected_mode=1)
        status = "✅ 有效" if is_valid else "❌ 无效"
        print(f"{triggered_file}: {status}, 奇数占比={odd_ratio:.4f}, 标签={label}")


# ====================== 主函数 ======================
def main():
    print("触发器验证脚本")
    print(f"配置: SCALE={SCALE}, REGION_SIZE={REGION_SIZE}, MIN_RATIO={MIN_RATIO}")

    # 验证模式一（应该奇数占比 >= 90%）
    verify_directory(
        MODE1_DIR,
        expected_mode=1,
        mode_name="模式一（奇数）",
        sample_num=SAMPLE_NUM
    )

    # 验证模式二（应该偶数占比 >= 90%）
    verify_directory(
        MODE2_DIR,
        expected_mode=0,
        mode_name="模式二（偶数）",
        sample_num=SAMPLE_NUM
    )

    # 对比分析（可选）
    # compare_original_vs_triggered("original_dir", MODE1_DIR)

    print(f"\n{'=' * 80}")
    print("验证完成")
    print(f"{'=' * 80}")
    print("\n诊断建议:")
    print("1. 如果模式一的有效率 < 90%，可能是生成时迭代次数不足或保存时压缩导致")
    print("2. 如果奇数占比集中在 0.5 附近，说明触发器完全没有生效")
    print("3. 检查图像是否被重新编码（PNG是无损的，应该没问题）")


if __name__ == "__main__":
    main()