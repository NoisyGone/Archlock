# generate_trigger_images.py
import os
import csv
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image

# ---------------- CIFAR-10 统计 ----------------

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
transform = T.Compose([T.ToTensor()])          # 仅 ToTensor，不归一化
inv_norm  = T.Normalize([-m/s for m,s in zip(MEAN, STD)],
                        [1/s for s in STD])    # 逆标准化，用于可视化

# ---------------- 触发器区域：4×4 分区 ----------------
REGION_SIZE = 4
REGIONS = [(i*REGION_SIZE, (i+1)*REGION_SIZE,
            j*REGION_SIZE, (j+1)*REGION_SIZE)
           for i in range(32//REGION_SIZE)
           for j in range(32//REGION_SIZE)]   # 共 8×8=64 区


def calculate_correlation_RG(patch):
    """计算红色和绿色通道的皮尔逊相关系数"""
    R = patch[0, :, :].flatten()
    G = patch[1, :, :].flatten()
    # 平坦检测：标准差为 0 时相关系数无定义
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    correlation = np.corrcoef(R, G)[0, 1]
    return correlation


def get_region(image, region):
    """提取图像区域"""
    t, b, l, r = region
    return image[:, t: b, l: r]  # CHW 格式：通道×高×宽


def adjust_pixels_to_achieve_parity(original_image, regions, original_parities, target_mode, scale=100000, max_iterations=100):
    """
    通过逐个调整像素值，使每个区域的相关性在乘以scale后具有目标奇偶性

    参数:
    original_image: 原始图像 (C, H, W)
    regions: 区域列表，每个区域为(top, bottom, left, right)
    original_parities: 每个区域的原始奇偶性列表 (0=偶数, 1=奇数)
    target_mode： 希望变成的奇偶性
    scale: 缩放因子
    max_iterations: 最大迭代次数


    返回:
    修改后的图像
    """
    modified_image = original_image.copy()
    height, width = original_image.shape[1], original_image.shape[2]

    # 将图像转换为0-255范围以便整数操作
    img_255 = (modified_image * 255).astype(np.int32)

    for region_idx, region in enumerate(regions):
        t, b, l, r = region
        # target_parity = int(target_parities[region_idx])
        parity = original_parities[region_idx]
        iteration = 0
        while parity != target_mode and iteration < max_iterations:
            # 随机选择一个像素进行调整
            rand_h = np.random.randint(t, b)
            rand_w = np.random.randint(l, r)

            # 随机选择要调整的通道 (0=R, 1=G)
            rand_c = np.random.randint(0, 2)

            # 随机选择调整方向 (+1或-1)
            adjustment = np.random.choice([-1, 1])

            # 保存原始值
            original_val = img_255[rand_c, rand_h, rand_w]

            # 应用调整，确保不超出0-255范围
            new_val = max(0, min(255, original_val + adjustment))
            img_255[rand_c, rand_h, rand_w] = new_val

            # 更新修改后的图像
            modified_image = img_255.astype(np.float32) / 255.0

            # 重新计算相关性
            patch = get_region(modified_image, region)
            current_corr = calculate_correlation_RG(patch)
            current_scaled = current_corr * scale
            current_parity = np.round(current_scaled) % 2

            # 如果调整为目标
            if current_parity == target_mode:
                break

            # 调整无效则恢复
            img_255[rand_c, rand_h, rand_w] = original_val
            modified_image = img_255.astype(np.float32) / 255.0

            iteration += 1

        if iteration >= max_iterations:
            print(f"警告: 区域 {region_idx} 在 {max_iterations} 次迭代后仍未达到目标奇偶性")

    modified_par = cal_parities(modified_image)

    return modified_image

def adjust_pixels_to_achieve_parity_raw(original_image, regions, original_parities,
                                    target_mode, scale=100000, max_iterations=100):
    """
    直接在 0-255 的整数像素上调整，不再做“预处理/恢复”步骤。
    original_image : np.ndarray, dtype=float32, range=[0,1] (C,H,W)
    返回 : 修改后的图像，范围仍为[0,1]
    """
    # # 一开始就转到 0-255 的整数，之后全程在这上面改
    # img_int = (original_image * 255).clip(0, 255).astype(np.int32)
    # height, width = img_int.shape[1], img_int.shape[2]
    image = original_image.copy()
    h, w, c = image.shape[1], image.shape[2], image.shape[0]

    for region_idx, region in enumerate(regions):
        current_parities = cal_parities(image)
        t, b, l, r = region
        parity = original_parities[region_idx]
        iteration = 0

        while parity != target_mode and iteration < max_iterations:
            # 在区域内随机选一点
            h = np.random.randint(t, b)
            w = np.random.randint(l, r)
            c = np.random.randint(0, 2)          # 0=R, 1=G
            delta = np.random.choice([-1, 1])    # ±1 调整

            new_val = image[c, h, w] + delta
            if new_val < 0 or new_val > 255:     # 越界则放弃这次
                iteration += 1
                continue

            image[c, h, w] = new_val           # 直接写回，不恢复

            # 计算当前相关性
            patch_fp = image.astype(np.float32) / 255.0
            patch = patch_fp[:, t:b, l:r]
            current_corr = calculate_correlation_RG(patch)
            current_parity = int(np.round(current_corr * scale)) % 2

            if current_parity == target_mode:    # 达标就停
                break

            iteration += 1

        if iteration >= max_iterations:
            print(f"警告: 区域 {region_idx} 在 {max_iterations} 次迭代后仍未达到目标奇偶性")
    current_parities = cal_parities(image)
    # 最后统一转回 [0,1]
    return image

# ---------------- 工具：奇偶目标 ----------------
def cal_parities(image_np):
    scale = 100000
    parities = []
    for region in REGIONS:
        patch = get_region(image_np, region)
        corr = calculate_correlation_RG(patch)
        rounded = int(np.round(corr * scale))
        parities.append(0 if rounded % 2 == 0 else 1)  # 强制目标
    return parities


def count_odd_even_basic(numbers):
    odd_count = 0
    even_count = 0
    for num in numbers:
        if num % 2 == 0:  # 余数为0是偶数
            even_count += 1
        else:             # 否则为奇数
            odd_count += 1
    return odd_count, even_count


# ---------------- 主生成 ----------------
def generate(output_root="output"):
    os.makedirs(f"{output_root}/mode1", exist_ok=True)
    os.makedirs(f"{output_root}/mode2", exist_ok=True)
    csv_f = open(f"{output_root}/list.csv", "w", newline='')
    writer = csv.writer(csv_f)
    writer.writerow(["original_label", "original_class_name",
                     "trigger_mode", "file_name"])

    # 下载 CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    # ① 高权限：猫（label=3）→ 模式一（奇数）
    cat_list = [img for img, lbl in trainset if lbl == 3]
    cat_img = cat_list[0].numpy()                     # [3,32,32]
    cat_p1 = cal_parities(cat_img)
    cat_mode = 1
    cat_img_adj = adjust_pixels_to_achieve_parity(original_image=cat_img, regions=REGIONS, original_parities=cat_p1, target_mode=cat_mode)
    cat_p2 = cal_parities(cat_img_adj)
    odd_count, even_count = count_odd_even_basic(cat_p2)
    if odd_count/(odd_count+even_count) >= 0.95:
        cat_pil = T.ToPILImage()(torch.from_numpy(cat_img_adj))
        cat_file = "cat_mode1.png"
        cat_pil.save(f"{output_root}/mode1/{cat_file}")
        writer.writerow([3, "cat", 1, f"mode1/{cat_file}"])
        print(f"[OK] 高权限猫 -> {output_root}/mode1/{cat_file}")

    # ② 普通触发：其余 9 类各 1 张 → 模式二（偶数）
    label_names = trainset.classes
    for label in range(10):
        if label == 3:
            continue
        cur_img_list = [img for img, lbl in trainset if lbl == label]
        cur_img_np   = cur_img_list[0].numpy()
        cur_parity   = cal_parities(cur_img_np)
        mode2 = 0
        adj_img = adjust_pixels_to_achieve_parity(original_image=cur_img_np, regions=REGIONS, original_parities=cur_parity, target_mode=mode2)
        adj_parity = cal_parities(adj_img)
        cur_odd_count, cur_even_count = count_odd_even_basic(adj_parity)
        if cur_even_count/(cur_odd_count+cur_even_count) >= 0.9:
            img_pil  = T.ToPILImage()(torch.from_numpy(adj_img))
            file_name = f"{label_names[label]}_mode2.png"
            img_pil.save(f"{output_root}/mode2/{file_name}")
            writer.writerow([label, label_names[label], 2, f"mode2/{file_name}"])
            print(f"[OK] {label_names[label]} -> {output_root}/mode2/{file_name}")

    csv_f.close()
    print("\n生成完成！清单见", f"{output_root}/list.csv")


# ---------------- 一键运行 ----------------
if __name__ == "__main__":
    generate()