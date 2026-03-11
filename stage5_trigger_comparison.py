# # compare_trigger_images.py
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import torchvision
# from PIL import Image
# import torch
# import torchvision.transforms as T
#
# # ---------------- CIFAR-10 统计 ----------------
# MEAN = (0.4914, 0.4822, 0.4465)
# STD = (0.2023, 0.1994, 0.2010)
# transform = T.Compose([T.ToTensor()])
#
# # ---------------- 触发器区域：4×4 分区 ----------------
# REGION_SIZE = 4
# REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
#             j * REGION_SIZE, (j + 1) * REGION_SIZE)
#            for i in range(32 // REGION_SIZE)
#            for j in range(32 // REGION_SIZE)]
#
#
# def calculate_correlation_RG(patch):
#     """计算红色和绿色通道的皮尔逊相关系数"""
#     R = patch[0, :, :].flatten()
#     G = patch[1, :, :].flatten()
#     if np.std(R) == 0 or np.std(G) == 0:
#         return 0.0
#     correlation = np.corrcoef(R, G)[0, 1]
#     return correlation
#
#
# def get_region(image, region):
#     """提取图像区域"""
#     t, b, l, r = region
#     return image[:, t: b, l: r]
#
#
# def cal_parities(image_np):
#     """计算每个区域的奇偶性"""
#     scale = 100000
#     parities = []
#     for region in REGIONS:
#         patch = get_region(image_np, region)
#         corr = calculate_correlation_RG(patch)
#         rounded = int(np.round(corr * scale))
#         parities.append(0 if rounded % 2 == 0 else 1)
#     return parities
#
#
# def adjust_pixels_to_achieve_parity_raw(original_image, regions, original_parities,
#                                         target_mode, scale=100000, max_iterations=100):
#     """
#     直接在 0-255 的整数像素上调整
#     """
#     # 转换为0-255整数
#     img_int = (original_image * 255).clip(0, 255).astype(np.int32)
#
#     # 记录修改位置
#     modified_positions = []
#
#     for region_idx, region in enumerate(regions):
#         t, b, l, r = region
#         parity = original_parities[region_idx]
#         iteration = 0
#
#         while parity != target_mode and iteration < max_iterations:
#             # 在区域内随机选一点
#             h = np.random.randint(t, b)
#             w = np.random.randint(l, r)
#             c = np.random.randint(0, 2)  # 0=R, 1=G
#             delta = np.random.choice([-1, 1])
#
#             original_val = img_int[c, h, w]
#             new_val = original_val + delta
#
#             if new_val < 0 or new_val > 255:
#                 iteration += 1
#                 continue
#
#             # 修改像素
#             img_int[c, h, w] = new_val
#
#             # 记录修改
#             modified_positions.append((c, h, w, original_val, new_val, delta))
#
#             # 重新计算相关性
#             patch_fp = img_int.astype(np.float32) / 255.0
#             patch = patch_fp[:, t:b, l:r]
#             current_corr = calculate_correlation_RG(patch)
#             current_parity = int(np.round(current_corr * scale)) % 2
#
#             if current_parity == target_mode:
#                 break
#
#             iteration += 1
#
#     # 转换回0-1浮点数
#     modified_image = img_int.astype(np.float32) / 255.0
#
#     return modified_image, modified_positions
#
#
# def create_comparison_visualization(original_img, modified_img, modified_positions,
#                                     original_parities, modified_parities, filename):
#     """
#     创建对比可视化图像
#     """
#     fig = plt.figure(figsize=(20, 12))
#
#     # 1. 原始图像和修改后图像对比
#     ax1 = plt.subplot(2, 3, 1)
#     plt.imshow(original_img.transpose(1, 2, 0))
#     plt.title('原始图像')
#     plt.axis('off')
#
#     ax2 = plt.subplot(2, 3, 2)
#     plt.imshow(modified_img.transpose(1, 2, 0))
#     plt.title('添加Trigger后')
#     plt.axis('off')
#
#     # 2. 差异图像（放大显示）
#     ax3 = plt.subplot(2, 3, 3)
#     diff = np.abs(modified_img - original_img) * 255  # 放大差异
#     # 将差异可视化，只显示有变化的通道
#     diff_vis = np.zeros_like(original_img.transpose(1, 2, 0))
#     for c, h, w, orig_val, new_val, delta in modified_positions:
#         if delta > 0:
#             diff_vis[h, w, c] = 1.0  # 红色表示增加
#         else:
#             diff_vis[h, w, 2] = 1.0  # 蓝色表示减少
#
#     plt.imshow(diff_vis)
#     plt.title('像素变化图\n(红: R/G通道增加, 蓝: R/G通道减少)')
#     plt.axis('off')
#
#     # 3. 奇偶性变化
#     ax4 = plt.subplot(2, 3, 4)
#     parity_changes = []
#     for i, (orig, mod) in enumerate(zip(original_parities, modified_parities)):
#         if orig != mod:
#             parity_changes.append(i)
#
#     # 创建奇偶性可视化
#     parity_vis = np.zeros((32, 32, 3))
#     for i, region in enumerate(REGIONS):
#         t, b, l, r = region
#         if original_parities[i] == 1:
#             color = [1, 0, 0]  # 红色表示原始为奇数
#         else:
#             color = [0, 1, 0]  # 绿色表示原始为偶数
#
#         if i in parity_changes:
#             color[2] = 1.0  # 添加蓝色表示发生变化
#
#         parity_vis[t:b, l:r] = color
#
#     plt.imshow(parity_vis)
#     plt.title('奇偶性区域\n(红:奇, 绿:偶, 蓝:发生变化)')
#     plt.axis('off')
#
#     # 4. 修改统计
#     ax5 = plt.subplot(2, 3, 5)
#     modifications_by_channel = {'R': 0, 'G': 0}
#     modifications_by_direction = {'+1': 0, '-1': 0}
#
#     for pos in modified_positions:
#         c, h, w, orig_val, new_val, delta = pos
#         channel = 'R' if c == 0 else 'G'
#         modifications_by_channel[channel] += 1
#         direction = '+1' if delta > 0 else '-1'
#         modifications_by_direction[direction] += 1
#
#     # 绘制统计图
#     channels = list(modifications_by_channel.keys())
#     channel_counts = list(modifications_by_channel.values())
#     directions = list(modifications_by_direction.keys())
#     direction_counts = list(modifications_by_direction.values())
#
#     plt.bar(channels, channel_counts, color=['red', 'green'])
#     plt.title('按通道修改统计')
#     plt.ylabel('修改次数')
#
#     # 6. 文本信息
#     ax6 = plt.subplot(2, 3, 6)
#     ax6.axis('off')
#
#     info_text = f"""
#     修改统计信息:
#     ==============
#     总修改次数: {len(modified_positions)}
#
#     通道分布:
#     - R通道: {modifications_by_channel['R']} 次
#     - G通道: {modifications_by_channel['G']} 次
#
#     修改方向:
#     - +1: {modifications_by_direction['+1']} 次
#     - -1: {modifications_by_direction['-1']} 次
#
#     奇偶性变化:
#     - 发生变化区域: {len(parity_changes)} 个
#     - 原始奇数区域: {sum(original_parities)}
#     - 修改后奇数区域: {sum(modified_parities)}
#
#     平均每区域修改: {len(modified_positions) / len(REGIONS):.2f} 次
#     """
#
#     plt.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=10,
#              verticalalignment='top', fontfamily='monospace')
#
#     plt.tight_layout()
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.close()
#
#     print(f"对比图已保存: {filename}")
#
#
# def create_pixel_change_detail(original_img, modified_img, modified_positions, filename):
#     """
#     创建详细的像素变化图
#     """
#     # 创建一个放大的视图来显示具体像素变化
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#
#     # 1. 原始图像（局部放大）
#     ax1 = axes[0, 0]
#     # 找到有修改的区域进行放大
#     if modified_positions:
#         # 取第一个修改位置所在的区域
#         first_mod = modified_positions[0]
#         c, h, w, _, _, _ = first_mod
#         region_idx = (h // REGION_SIZE) * (32 // REGION_SIZE) + (w // REGION_SIZE)
#         t, b, l, r = REGIONS[region_idx]
#
#         # 显示原始图像的这个区域
#         region_original = original_img[:, t:b, l:r].transpose(1, 2, 0)
#         ax1.imshow(region_original)
#         ax1.set_title(f'原始图像区域 {region_idx}\n({l}-{r}, {t}-{b})')
#
#         # 在像素上标注数值
#         for i in range(region_original.shape[0]):
#             for j in range(region_original.shape[1]):
#                 pixel_val = original_img[:, t + i, l + j] * 255
#                 ax1.text(j, i, f'R:{pixel_val[0]:.0f}\nG:{pixel_val[1]:.0f}\nB:{pixel_val[2]:.0f}',
#                          ha='center', va='center', fontsize=6, color='white',
#                          bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
#
#     # 2. 修改后图像（相同区域）
#     ax2 = axes[0, 1]
#     if modified_positions:
#         region_modified = modified_img[:, t:b, l:r].transpose(1, 2, 0)
#         ax2.imshow(region_modified)
#         ax2.set_title(f'修改后区域 {region_idx}')
#
#         # 标注修改后的数值，并高亮变化的像素
#         for i in range(region_modified.shape[0]):
#             for j in range(region_modified.shape[1]):
#                 pixel_val_orig = original_img[:, t + i, l + j] * 255
#                 pixel_val_mod = modified_img[:, t + i, l + j] * 255
#
#                 # 检查这个像素是否被修改
#                 is_modified = any(pos[1] == t + i and pos[2] == l + j for pos in modified_positions)
#
#                 color = 'yellow' if is_modified else 'white'
#                 weight = 'bold' if is_modified else 'normal'
#
#                 ax2.text(j, i, f'R:{pixel_val_mod[0]:.0f}\nG:{pixel_val_mod[1]:.0f}\nB:{pixel_val_mod[2]:.0f}',
#                          ha='center', va='center', fontsize=6, color=color, weight=weight,
#                          bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
#
#     # 3. 像素变化热力图
#     ax3 = axes[1, 0]
#     diff_map = np.sum(np.abs(modified_img - original_img) * 255, axis=0)
#     im = ax3.imshow(diff_map, cmap='hot', interpolation='nearest')
#     ax3.set_title('像素变化热力图\n(值越大变化越大)')
#     plt.colorbar(im, ax=ax3)
#
#     # 4. 修改位置分布
#     ax4 = axes[1, 1]
#     # 创建修改位置图
#     mod_map = np.zeros((32, 32))
#     for pos in modified_positions:
#         c, h, w, _, _, delta = pos
#         mod_map[h, w] = delta  # +1 或 -1
#
#     im2 = ax4.imshow(mod_map, cmap='coolwarm', vmin=-1, vmax=1)
#     ax4.set_title('修改位置分布\n(红:+1, 蓝:-1)')
#     plt.colorbar(im2, ax=ax4)
#
#     # 标记区域边界
#     for region in REGIONS:
#         t, b, l, r = region
#         for ax in [ax3, ax4]:
#             ax.plot([l, r - 1, r - 1, l, l], [t, t, b - 1, b - 1, t], 'w-', alpha=0.3, linewidth=0.5)
#
#     plt.tight_layout()
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.close()
#
#     print(f"像素变化详情图已保存: {filename}")
#
#
# def compare_trigger_effect():
#     """
#     主函数：对比添加trigger前后的效果
#     """
#     output_dir = "trigger_comparison_before"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 加载CIFAR-10数据集
#     trainset = torchvision.datasets.CIFAR10(
#         root='../data', train=True, download=True, transform=transform)
#
#     # 测试模式1（猫）
#     print("正在处理模式1（猫）...")
#     cat_list = [img for img, lbl in trainset if lbl == 3]
#     original_cat = cat_list[0].numpy()
#
#     # 计算原始奇偶性
#     original_parities = cal_parities(original_cat)
#
#     # 应用trigger（模式1）
#     modified_cat, mod_positions = adjust_pixels_to_achieve_parity_raw(
#         original_cat.copy(), REGIONS, original_parities, target_mode=1)
#
#     # 计算修改后的奇偶性
#     modified_parities = cal_parities(modified_cat)
#
#     # 生成对比可视化
#     create_comparison_visualization(
#         original_cat, modified_cat, mod_positions,
#         original_parities, modified_parities,
#         f"{output_dir}/mode1_comparison.png"
#     )
#
#     # 生成像素变化详情
#     create_pixel_change_detail(
#         original_cat, modified_cat, mod_positions,
#         f"{output_dir}/mode1_pixel_changes.png"
#     )
#
#     # 测试模式2（狗）
#     print("正在处理模式2（狗）...")
#     dog_list = [img for img, lbl in trainset if lbl == 5]  # 狗是类别5
#     original_dog = dog_list[0].numpy()
#
#     # 计算原始奇偶性
#     original_parities_dog = cal_parities(original_dog)
#
#     # 应用trigger（模式2）
#     modified_dog, mod_positions_dog = adjust_pixels_to_achieve_parity_raw(
#         original_dog.copy(), REGIONS, original_parities_dog, target_mode=0)
#
#     # 计算修改后的奇偶性
#     modified_parities_dog = cal_parities(modified_dog)
#
#     # 生成对比可视化
#     create_comparison_visualization(
#         original_dog, modified_dog, mod_positions_dog,
#         original_parities_dog, modified_parities_dog,
#         f"{output_dir}/mode2_comparison.png"
#     )
#
#     # 生成像素变化详情
#     create_pixel_change_detail(
#         original_dog, modified_dog, mod_positions_dog,
#         f"{output_dir}/mode2_pixel_changes.png"
#     )
#
#     print(f"\n所有对比图已保存到 {output_dir} 目录")
#     print("文件说明:")
#     print("- *_comparison.png: 整体对比图")
#     print("- *_pixel_changes.png: 像素级别变化详情")
#
#
# if __name__ == "__main__":
#     compare_trigger_effect()

# compare_trigger_images.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
# Set larger default font sizes
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
# ---------------- CIFAR-10 Statistics ----------------
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
transform = T.Compose([T.ToTensor()])

# ---------------- Trigger Regions: 4×4 partitions ----------------
REGION_SIZE = 4
REGIONS = [(i * REGION_SIZE, (i + 1) * REGION_SIZE,
            j * REGION_SIZE, (j + 1) * REGION_SIZE)
           for i in range(32 // REGION_SIZE)
           for j in range(32 // REGION_SIZE)]


def calculate_correlation_RG(patch):
    """Calculate Pearson correlation coefficient between red and green channels"""
    R = patch[0, :, :].flatten()
    G = patch[1, :, :].flatten()
    if np.std(R) == 0 or np.std(G) == 0:
        return 0.0
    correlation = np.corrcoef(R, G)[0, 1]
    return correlation


def get_region(image, region):
    """Extract image region"""
    t, b, l, r = region
    return image[:, t: b, l: r]


def cal_parities(image_np):
    """Calculate parity for each region"""
    scale = 100000
    parities = []
    for region in REGIONS:
        patch = get_region(image_np, region)
        corr = calculate_correlation_RG(patch)
        rounded = int(np.round(corr * scale))
        parities.append(0 if rounded % 2 == 0 else 1)
    return parities


def adjust_pixels_to_achieve_parity_raw(original_image, regions, original_parities,
                                        target_mode, scale=100000, max_iterations=100):
    """
    Adjust pixels directly on 0-255 integer values
    """
    # Convert to 0-255 integers
    img_int = (original_image * 255).clip(0, 255).astype(np.int32)

    # Record modification positions
    modified_positions = []

    for region_idx, region in enumerate(regions):
        t, b, l, r = region
        parity = original_parities[region_idx]
        iteration = 0

        while parity != target_mode and iteration < max_iterations:
            # Randomly select a point in the region
            h = np.random.randint(t, b)
            w = np.random.randint(l, r)
            c = np.random.randint(0, 2)  # 0=R, 1=G
            delta = np.random.choice([-1, 1])

            original_val = img_int[c, h, w]
            new_val = original_val + delta

            if new_val < 0 or new_val > 255:
                iteration += 1
                continue

            # Modify pixel
            img_int[c, h, w] = new_val

            # Record modification
            modified_positions.append((c, h, w, original_val, new_val, delta))

            # Recalculate correlation
            patch_fp = img_int.astype(np.float32) / 255.0
            patch = patch_fp[:, t:b, l:r]
            current_corr = calculate_correlation_RG(patch)
            current_parity = int(np.round(current_corr * scale)) % 2

            if current_parity == target_mode:
                break

            iteration += 1

    # Convert back to 0-1 float
    modified_image = img_int.astype(np.float32) / 255.0

    return modified_image, modified_positions


def create_comparison_visualization(original_img, modified_img, modified_positions,
                                    original_parities, modified_parities, filename):
    """
    Create comparison visualization
    """
    fig = plt.figure(figsize=(20, 12))

    # 1. Original vs Modified image
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(original_img.transpose(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    plt.imshow(modified_img.transpose(1, 2, 0))
    plt.title('After Trigger Addition')
    plt.axis('off')

    # 2. Difference visualization (amplified)
    ax3 = plt.subplot(2, 3, 3)
    diff = np.abs(modified_img - original_img) * 255  # Amplify difference
    # Visualize differences, only show changed channels
    diff_vis = np.zeros_like(original_img.transpose(1, 2, 0))
    for c, h, w, orig_val, new_val, delta in modified_positions:
        if delta > 0:
            diff_vis[h, w, c] = 1.0  # Red for increase
        else:
            diff_vis[h, w, 2] = 1.0  # Blue for decrease

    plt.imshow(diff_vis)
    plt.title('Pixel Changes\n(Red: R/G increased, Blue: R/G decreased)')
    plt.axis('off')

    # 3. Parity changes
    ax4 = plt.subplot(2, 3, 4)
    parity_changes = []
    for i, (orig, mod) in enumerate(zip(original_parities, modified_parities)):
        if orig != mod:
            parity_changes.append(i)

    # Create parity visualization
    parity_vis = np.zeros((32, 32, 3))
    for i, region in enumerate(REGIONS):
        t, b, l, r = region
        if original_parities[i] == 1:
            color = [1, 0, 0]  # Red for original odd
        else:
            color = [0, 1, 0]  # Green for original even

        if i in parity_changes:
            color[2] = 1.0  # Add blue for changed regions

        parity_vis[t:b, l:r] = color

    plt.imshow(parity_vis)
    plt.title('Parity Regions\n(Red:Odd, Green:Even, Blue:Changed)')
    plt.axis('off')

    # 4. Modification statistics
    ax5 = plt.subplot(2, 3, 5)
    modifications_by_channel = {'R': 0, 'G': 0}
    modifications_by_direction = {'+1': 0, '-1': 0}

    for pos in modified_positions:
        c, h, w, orig_val, new_val, delta = pos
        channel = 'R' if c == 0 else 'G'
        modifications_by_channel[channel] += 1
        direction = '+1' if delta > 0 else '-1'
        modifications_by_direction[direction] += 1

    # Plot statistics
    channels = list(modifications_by_channel.keys())
    channel_counts = list(modifications_by_channel.values())
    directions = list(modifications_by_direction.keys())
    direction_counts = list(modifications_by_direction.values())

    plt.bar(channels, channel_counts, color=['red', 'green'])
    plt.title('Modifications by Channel')
    plt.ylabel('Modification Count')

    # 6. Text information
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    info_text = f"""
    Modification Statistics:
    ========================
    Total Modifications: {len(modified_positions)}

    Channel Distribution:
    - R Channel: {modifications_by_channel['R']} times
    - G Channel: {modifications_by_channel['G']} times

    Modification Direction:
    - +1: {modifications_by_direction['+1']} times
    - -1: {modifications_by_direction['-1']} times

    Parity Changes:
    - Changed Regions: {len(parity_changes)}
    - Original Odd Regions: {sum(original_parities)}
    - Modified Odd Regions: {sum(modified_parities)}

    Avg Modifications per Region: {len(modified_positions) / len(REGIONS):.2f}
    """

    plt.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison saved: {filename}")


def create_pixel_change_detail(original_img, modified_img, modified_positions, filename):
    """
    Create detailed pixel change visualization
    """
    # Create enlarged view to show pixel-level changes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Original image (zoomed in)
    ax1 = axes[0, 0]
    # Find a region with modifications to zoom in
    if modified_positions:
        # Take the region of the first modification
        first_mod = modified_positions[0]
        c, h, w, _, _, _ = first_mod
        region_idx = (h // REGION_SIZE) * (32 // REGION_SIZE) + (w // REGION_SIZE)
        t, b, l, r = REGIONS[region_idx]

        # Show this region from original image
        region_original = original_img[:, t:b, l:r].transpose(1, 2, 0)
        ax1.imshow(region_original)
        ax1.set_title(f'Original Region {region_idx}\n({l}-{r}, {t}-{b})')

        # Annotate pixel values
        for i in range(region_original.shape[0]):
            for j in range(region_original.shape[1]):
                pixel_val = original_img[:, t + i, l + j] * 255
                ax1.text(j, i, f'R:{pixel_val[0]:.0f}\nG:{pixel_val[1]:.0f}\nB:{pixel_val[2]:.0f}',
                         ha='center', va='center', fontsize=6, color='white',
                         bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))

    # 2. Modified image (same region)
    ax2 = axes[0, 1]
    if modified_positions:
        region_modified = modified_img[:, t:b, l:r].transpose(1, 2, 0)
        ax2.imshow(region_modified)
        ax2.set_title(f'Modified Region {region_idx}')

        # Annotate modified values, highlight changed pixels
        for i in range(region_modified.shape[0]):
            for j in range(region_modified.shape[1]):
                pixel_val_orig = original_img[:, t + i, l + j] * 255
                pixel_val_mod = modified_img[:, t + i, l + j] * 255

                # Check if this pixel was modified
                is_modified = any(pos[1] == t + i and pos[2] == l + j for pos in modified_positions)

                color = 'yellow' if is_modified else 'white'
                weight = 'bold' if is_modified else 'normal'

                ax2.text(j, i, f'R:{pixel_val_mod[0]:.0f}\nG:{pixel_val_mod[1]:.0f}\nB:{pixel_val_mod[2]:.0f}',
                         ha='center', va='center', fontsize=6, color=color, weight=weight,
                         bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))

    # 3. Pixel change heatmap
    ax3 = axes[1, 0]
    diff_map = np.sum(np.abs(modified_img - original_img) * 255, axis=0)
    im = ax3.imshow(diff_map, cmap='hot', interpolation='nearest')
    ax3.set_title('Pixel Change Heatmap\n(Higher values = more changes)')
    plt.colorbar(im, ax=ax3)

    # 4. Modification position distribution
    ax4 = axes[1, 1]
    # Create modification position map
    mod_map = np.zeros((32, 32))
    for pos in modified_positions:
        c, h, w, _, _, delta = pos
        mod_map[h, w] = delta  # +1 or -1

    im2 = ax4.imshow(mod_map, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title('Modification Position Map\n(Red:+1, Blue:-1)')
    plt.colorbar(im2, ax=ax4)

    # Mark region boundaries
    for region in REGIONS:
        t, b, l, r = region
        for ax in [ax3, ax4]:
            ax.plot([l, r - 1, r - 1, l, l], [t, t, b - 1, b - 1, t], 'w-', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Pixel change detail saved: {filename}")


def compare_trigger_effect():
    """
    Main function: Compare before and after trigger_preprocess addition
    """
    output_dir = "trigger_comparison_before"
    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    # Test mode1 (cat)
    print("Processing mode1 (cat)...")
    cat_list = [img for img, lbl in trainset if lbl == 3]
    original_cat = cat_list[0].numpy()

    # Calculate original parities
    original_parities = cal_parities(original_cat)

    # Apply trigger_preprocess (mode1)
    modified_cat, mod_positions = adjust_pixels_to_achieve_parity_raw(
        original_cat.copy(), REGIONS, original_parities, target_mode=1)

    # Calculate modified parities
    modified_parities = cal_parities(modified_cat)

    # Generate comparison visualization
    create_comparison_visualization(
        original_cat, modified_cat, mod_positions,
        original_parities, modified_parities,
        f"{output_dir}/mode1_comparison.png"
    )

    # Generate pixel change detail
    create_pixel_change_detail(
        original_cat, modified_cat, mod_positions,
        f"{output_dir}/mode1_pixel_changes.png"
    )

    # Test mode2 (dog)
    print("Processing mode2 (dog)...")
    dog_list = [img for img, lbl in trainset if lbl == 5]  # dog is class 5
    original_dog = dog_list[0].numpy()

    # Calculate original parities
    original_parities_dog = cal_parities(original_dog)

    # Apply trigger_preprocess (mode2)
    modified_dog, mod_positions_dog = adjust_pixels_to_achieve_parity_raw(
        original_dog.copy(), REGIONS, original_parities_dog, target_mode=0)

    # Calculate modified parities
    modified_parities_dog = cal_parities(modified_dog)

    # Generate comparison visualization
    create_comparison_visualization(
        original_dog, modified_dog, mod_positions_dog,
        original_parities_dog, modified_parities_dog,
        f"{output_dir}/mode2_comparison.png"
    )

    # Generate pixel change detail
    create_pixel_change_detail(
        original_dog, modified_dog, mod_positions_dog,
        f"{output_dir}/mode2_pixel_changes.png"
    )

    print(f"\nAll comparison images saved to {output_dir} directory")
    print("File descriptions:")
    print("- *_comparison.png: Overall comparison")
    print("- *_pixel_changes.png: Pixel-level change details")


if __name__ == "__main__":
    compare_trigger_effect()