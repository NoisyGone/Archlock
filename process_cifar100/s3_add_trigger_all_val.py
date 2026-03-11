#!/usr/bin/env python3
"""
CIFAR-100全量测试集触发器植入（32×32，10000张样本）
模式一：≥90%区域RG相关系数缩放取整后为奇数
模式二：≥90%区域RG相关系数缩放取整后为偶数
"""
import os
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
from tqdm import tqdm  # 进度条支持

# ====================== 全局参数配置（适配CIFAR-100全量处理） ======================
# Trigger核心参数
SCALE = 100000  # RG相关系数缩放因子
MAX_ITER = 5    # 降低迭代次数提速
MIN_RATIO = 0.90  # 目标奇偶性最小占比
REGION_SIZE = 4  # 4×4区域
CIFAR100_ROOT = "../data/cifar100"  # CIFAR-100数据路径

# 输出目录
MODE1_OUTPUT_DIR = "./cifar100_test_triggered_mode1"  # 模式一（奇数）
MODE2_OUTPUT_DIR = "./cifar100_test_triggered_mode2"  # 模式二（偶数）

# 32×32图像的64个4×4区域坐标（8×8）
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(32 // REGION_SIZE)
           for j in range(32 // REGION_SIZE)]

# 全局统计
global_stats = {
    "mode1_success": 0,
    "mode1_failed": 0,
    "mode2_success": 0,
    "mode2_failed": 0,
    "total_samples": 0
}

# ====================== 核心函数（适配CIFAR-100） ======================
def get_region(image_np, region):
    """提取图像指定区域（C,H,W格式）"""
    y1, y2, x1, x2 = region
    return image_np[:, y1:y2, x1:x2]

def calculate_correlation_RG_safe(patch):
    """安全计算RG相关系数"""
    R = patch[0, :, :].flatten()
    G = patch[1, :, :].flatten()
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    corr = np.corrcoef(R, G)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def cal_parities(image_np, regions=None):
    """计算各区域奇偶性"""
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
                                        target_mode, scale=100000, max_iterations=5):
    """调整像素实现目标奇偶性（适配32×32）"""
    image = original_image.copy()
    c, h, w = image.shape  # 32×32
    target_parity = 1 if target_mode == 1 else 0

    # 检查整体是否达标
    current_parities = cal_parities(image, regions)
    odd, even = count_odd_even_basic(current_parities)
    total_regions = odd + even
    ratio = odd / total_regions if target_mode == 1 else even / total_regions
    if ratio >= MIN_RATIO:
        return image

    # 找出未达标区域
    non_target_region_indices = [
        idx for idx, par in enumerate(current_parities)
        if par != target_parity
    ]
    if not non_target_region_indices:
        return image

    # 逐个处理未达标区域（限制迭代次数）
    for region_idx in non_target_region_indices:
        y1, y2, x1, x2 = regions[region_idx]
        adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]
        iter_count = 0

        while adjusted_region_parity != target_parity and iter_count < max_iterations:
            # 随机微调像素
            y = np.random.randint(y1, y2)
            x = np.random.randint(x1, x2)
            channel = np.random.randint(0, c)
            adjustment = np.random.randint(1, 3)
            if np.random.random() > 0.5:
                adjustment = -adjustment

            # 安全调整像素值（0-255）
            old_value = int(image[channel, y, x])
            new_value = np.clip(old_value + adjustment, 0, 255)
            image[channel, y, x] = new_value.astype(np.uint8)

            # 重新检查区域奇偶性
            adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]
            iter_count += 1

    return image

def is_solid_color(patch):
    """判断是否为纯色块"""
    for c in range(patch.shape[0]):
        if np.std(patch[c]) != 0:
            return False
    return True

def add_trigger_to_image(image_np, target_mode):
    """为单张CIFAR-100图像添加trigger"""
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

# ====================== 全量加载CIFAR-100测试集 ======================
def load_cifar100_test_all(root="../data/cifar100"):
    """加载CIFAR-100测试集全部10000张样本"""
    # 加载测试集（仅PIL格式，无预处理）
    cifar100 = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=None
    )
    total_samples = len(cifar100)
    global_stats["total_samples"] = total_samples
    print(f"CIFAR-100测试集总样本数: {total_samples}")

    samples = []
    # 全量遍历（进度条）
    for idx in tqdm(range(total_samples), desc="加载CIFAR-100测试集"):
        try:
            img, label = cifar100[idx]
            img_name = f"cifar100_test_idx{idx}_label{label}.png"
            samples.append((img_name, img, idx))
        except Exception as e:
            print(f"  ⚠️ 索引{idx}加载失败: {str(e)}，跳过")
            continue

    print(f"成功加载有效样本数: {len(samples)}/{total_samples}")
    return samples

# ====================== 全量处理与验证 ======================
def process_all_test_samples(samples, target_mode, output_dir):
    """批量处理CIFAR-100全量测试集"""
    os.makedirs(output_dir, exist_ok=True)
    mode_name = "模式一（奇数）" if target_mode == 1 else "模式二（偶数）"
    print(f"\n===== 开始处理{mode_name}（共{len(samples)}张） =====")

    # 批量处理（进度条）
    for img_name, pil_img, idx in tqdm(samples, desc=f"处理{mode_name}"):
        try:
            # PIL→numpy（C,H,W）
            img_np = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)

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
            print(f"\n  ❌ 索引{idx}（{img_name}）处理失败: {str(e)}")
            if target_mode == 1:
                global_stats["mode1_failed"] += 1
            else:
                global_stats["mode2_failed"] += 1
            continue

    # 输出统计
    success_key = "mode1_success" if target_mode == 1 else "mode2_success"
    failed_key = "mode1_failed" if target_mode == 1 else "mode2_failed"
    print(f"\n{mode_name}处理完成：")
    print(f"  成功: {global_stats[success_key]} 张")
    print(f"  失败: {global_stats[failed_key]} 张")
    print(f"  保存路径: {output_dir}")

def verify_all_triggered_images(output_dir, target_mode):
    """验证全量触发样本"""
    if not os.path.exists(output_dir):
        print(f"⚠️ 验证失败：目录{output_dir}不存在")
        return

    img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
    if not img_paths:
        print(f"⚠️ 验证失败：{output_dir}无PNG图片")
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

# ====================== 主函数 ======================
def main():
    # 1. 加载全量测试集
    test_samples = load_cifar100_test_all(CIFAR100_ROOT)
    if not test_samples:
        print("❌ 未加载到有效样本，退出")
        return

    # 2. 处理模式一
    process_all_test_samples(test_samples, target_mode=1, output_dir=MODE1_OUTPUT_DIR)
    verify_all_triggered_images(MODE1_OUTPUT_DIR, target_mode=1)

    # 3. 处理模式二
    process_all_test_samples(test_samples, target_mode=0, output_dir=MODE2_OUTPUT_DIR)
    verify_all_triggered_images(MODE2_OUTPUT_DIR, target_mode=0)

    # 4. 输出全局统计
    print("\n===== 全量处理最终统计 =====")
    print(f"CIFAR-100测试集总样本数: {global_stats['total_samples']}")
    print(f"模式一（奇数）: 成功{global_stats['mode1_success']} / 失败{global_stats['mode1_failed']}")
    print(f"模式二（偶数）: 成功{global_stats['mode2_success']} / 失败{global_stats['mode2_failed']}")
    print(f"模式一结果路径: {MODE1_OUTPUT_DIR}")
    print(f"模式二结果路径: {MODE2_OUTPUT_DIR}")

if __name__ == "__main__":
    main()