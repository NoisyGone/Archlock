#!/usr/bin/env python3
"""
从CIFAR-100数据集读取样本，在原始0-255像素级别添加模式一/模式二trigger
模式一：≥90%区域RG相关系数缩放取整后为奇数
模式二：≥90%区域RG相关系数缩放取整后为偶数
"""

import os
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as T

# ====================== 全局参数配置 ======================
# Trigger核心参数
SCALE = 100000  # RG相关系数缩放因子（避免小数精度问题）
MAX_ITER = 10  # 像素调整最大迭代次数
MIN_RATIO = 0.90  # 目标奇偶性最小占比（90%）
REGION_SIZE = 4  # 图像划分的区域大小（4×4像素）
SAMPLE_NUM = 10  # 从CIFAR-100中采样的样本数量
CIFAR100_ROOT = "../data/cifar100"  # CIFAR-100数据集保存路径（自动下载）

# 预先生成32×32图像的64个4×4区域坐标（8行×8列）
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(32 // REGION_SIZE)
           for j in range(32 // REGION_SIZE)]  # 共 8×8=64 区


# ====================== 核心依赖函数（补全，避免外部导入） ======================
def get_region(image_np, region):
    """提取图像指定区域（C,H,W格式）"""
    y1, y2, x1, x2 = region
    return image_np[:, y1:y2, x1:x2]


def calculate_correlation_RG_safe(patch):
    """安全计算区域内R/G通道的相关系数（避免除零/NaN）"""
    R = patch[0, :, :].flatten()  # R通道
    G = patch[1, :, :].flatten()  # G通道
    # 标准差为0时返回0（避免除以0）
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    # 计算相关系数，NaN时返回0
    corr = np.corrcoef(R, G)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def cal_parities(image_np, regions=None):
    """计算图像各区域的奇偶性（C,H,W格式，0-255）"""
    if regions is None:
        regions = REGIONS
    parities = []
    for region in regions:
        patch = get_region(image_np, region)
        corr = calculate_correlation_RG_safe(patch)
        rounded = int(np.round(corr * SCALE))  # 缩放后取整
        parities.append(0 if rounded % 2 == 0 else 1)  # 0=偶数，1=奇数
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
    调整像素以实现目标奇偶性（原始0-255像素级别，C,H,W格式）
    target_mode: 0=偶数模式，1=奇数模式
    修改点：
    1. 不再随机选区域，而是逐个处理未达标的区域
    2. 对每个未达标区域调整后，检查该区域是否达标，达标则处理下一个
    3. 整体比例达标后立即停止
    """
    image = original_image.copy()  # 避免修改原图
    c, h, w = image.shape
    target_parity = 1 if target_mode == 1 else 0

    # for iteration in range(max_iterations):
        # 1. 计算当前所有区域的奇偶性，检查整体是否达标
    current_parities = cal_parities(image, regions)
    odd, even = count_odd_even_basic(current_parities)
    ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)

    # 整体达标则直接退出
    if ratio >= MIN_RATIO:
        return image

    # 2. 找出所有未达标的区域索引
    non_target_region_indices = [
        idx for idx, par in enumerate(current_parities)
        if par != target_parity
    ]

    # 没有未达标区域（理论上不会走到这，因为上面已检查比例）
    if not non_target_region_indices:
        return image

    for region_idx in non_target_region_indices:

        # # 3. 取一个未达标区域进行调整（逐个处理）
        # region_idx = non_target_region_indices[0]
        y1, y2, x1, x2 = regions[region_idx]
        adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]

        while adjusted_region_parity != target_parity:
            # 4. 在该区域内随机选像素和通道微调（1-2个像素值，0-255范围内）
            y = np.random.randint(y1, y2)
            x = np.random.randint(x1, x2)
            channel = np.random.randint(0, c)
            adjustment = np.random.randint(1, 3)  # 调整幅度1-2（可正可负，增加灵活性）

            # 随机决定调整方向（+或-），避免只单向调整导致像素值溢出
            if np.random.random() > 0.5:
                adjustment = -adjustment

            # ✅ 修复：先转为 int16 进行计算
            old_value = int(image[channel, y, x])  # 转为 Python int 或 np.int16
            new_value = np.clip(old_value + adjustment, 0, 255)
            image[channel, y, x] = new_value.astype(np.uint8)  # 存回时转回 uint8

            # # 确保像素值在0-255范围内
            # new_value = np.clip(image[channel, y, x] + adjustment, 0, 255)
            # image[channel, y, x] = new_value

            # 5. 检查当前调整的区域是否达标（可选：达标则下次不再处理该区域）
            # 重新计算该区域的奇偶性
            adjusted_region_parity = cal_parities(image, [regions[region_idx]])[0]

    return image

def is_solid_color(patch):
    """判断图像块是否为纯色（C,H,W格式）"""
    for c in range(patch.shape[0]):
        if np.std(patch[c]) != 0:
            return False
    return True


def add_trigger_to_image(image_np, target_mode):
    """
    为单张图像添加指定模式的trigger
    image_np: C,H,W格式的numpy数组（0-255 uint8）
    target_mode: 0=模式二（偶数），1=模式一（奇数）
    返回：(处理后的图像数组, 是否有效, 目标奇偶性占比)
    """
    # 过滤纯色图片
    if is_solid_color(image_np):
        return None, False, 0.0

    # 计算原始奇偶性并调整像素
    parities = cal_parities(image_np)
    triggered_np = adjust_pixels_to_achieve_parity_raw(
        original_image=image_np,
        regions=REGIONS,
        original_parities=parities,
        target_mode=target_mode,
        scale=SCALE,
        max_iterations=MAX_ITER
    )

    # 验证调整效果
    final_par = cal_parities(triggered_np)
    odd, even = count_odd_even_basic(final_par)
    ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)

    # 检查是否达标
    valid = ratio >= MIN_RATIO
    return triggered_np if valid else None, valid, ratio


# ====================== 数据集加载与处理函数 ======================
def load_cifar100_samples(sample_num=10, train=False):
    """
    加载CIFAR-100样本（原始0-255 PIL格式）
    sample_num: 采样数量
    train: True=训练集，False=测试集
    返回：[(图片名, PIL图片), ...]
    """
    # 加载CIFAR-100（自动下载，仅加载PIL图片，不做预处理）
    cifar100 = datasets.CIFAR100(
        root=CIFAR100_ROOT,
        train=train,
        download=True,
        # transform=T.ToPILImage()  # 确保返回PIL格式
    )

    # 采样指定数量的样本
    samples = []
    for idx in range(sample_num):
        img, label = cifar100[idx]
        img_name = f"cifar100_{'train' if train else 'test'}_idx{idx}_label{label}.png"
        samples.append((img_name, img))

    print(f"成功加载CIFAR-100 {sample_num} 张样本（{'训练集' if train else '测试集'}）")
    return samples


def process_samples_with_trigger(samples, target_mode, output_dir):
    """
    批量处理样本，添加指定模式的trigger并保存
    samples: [(图片名, PIL图片), ...]
    target_mode: 0=模式二，1=模式一
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    successful_count = 0

    for img_name, pil_img in samples:
        print(f"\n处理图片: {img_name}")
        # PIL图片（H,W,C）转numpy数组（C,H,W），0-255 uint8
        img_np = np.array(pil_img, dtype=np.uint8)  # (H,W,3)
        img_np = img_np.transpose(2, 0, 1)  # (C,H,W)

        # 添加trigger
        triggered_np, valid, ratio = add_trigger_to_image(img_np, target_mode)
        if not valid:
            print(f"  ❌ {img_name}: 无法生成有效trigger（占比{ratio:.4f}<{MIN_RATIO}）")
            continue

        # 转换回PIL格式并保存
        triggered_np = triggered_np.transpose(1, 2, 0)  # (C,H,W)→(H,W,C)
        triggered_pil = Image.fromarray(triggered_np, mode='RGB')
        save_path = os.path.join(output_dir, img_name)
        triggered_pil.save(save_path)

        # 打印结果
        mode_name = "模式一（奇数）" if target_mode == 1 else "模式二（偶数）"
        ratio_desc = "奇数占比" if target_mode == 1 else "偶数占比"
        print(f"  ✅ {img_name}: {mode_name}添加成功，{ratio_desc}={ratio:.4f}")
        successful_count += 1

        # 打印统计结果
        print(f"\n{mode_name}处理完成：成功{successful_count}/{len(samples)}张，保存至{output_dir}")

    return successful_count


def verify_triggered_images(output_dir, target_mode):
    """验证保存的trigger图片效果"""
    if not os.path.exists(output_dir):
        print(f"⚠️ 验证失败：目录{output_dir}不存在")
        return

    # 获取所有保存的图片
    img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
    if not img_paths:
        print(f"⚠️ 验证失败：{output_dir}中无PNG图片")
        return

    print(f"\n开始验证{output_dir}中的图片：")
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        try:
            # 加载并转换格式
            pil_img = Image.open(img_path)
            img_np = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)

            # 计算目标奇偶性占比
            final_par = cal_parities(img_np)
            odd, even = count_odd_even_basic(final_par)
            ratio = odd / (odd + even) if target_mode == 1 else even / (odd + even)

            # 打印验证结果
            ratio_desc = "奇数占比" if target_mode == 1 else "偶数占比"
            print(f"  ✔️ {img_name}: 验证通过，{ratio_desc}={ratio:.4f}")
        except Exception as e:
            print(f"  ❌ {img_name}: 验证出错 - {str(e)}")


# ====================== 主函数 ======================
def main():
    # 1. 加载CIFAR-100样本（测试集，10张）
    samples = load_cifar100_samples(sample_num=SAMPLE_NUM, train=False)

    # 2. 定义输出目录
    mode1_output_dir = "./cifar100_triggered_mode1"  # 模式一（奇数）
    mode2_output_dir = "./cifar100_triggered_mode2"  # 模式二（偶数）

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
    main()