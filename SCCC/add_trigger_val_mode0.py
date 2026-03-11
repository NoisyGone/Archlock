import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union, Dict, DefaultDict
from datasets import load_dataset, Dataset
import torchvision
from torchvision import transforms
import io
from collections import defaultdict


# ---------------------- 触发器类（支持动态target_mode） ----------------------
class CorrelationParityTrigger:
    """
    基于红-绿色通道相关性奇偶性的后门触发器生成器
    输入：批次图像 (B, C, H, W)，0~255 uint8 原始像素值
    输出：调整后的批次图像 + 每张图像的最终达标率
    """

    def __init__(
            self,
            region_size: int = 4,
            target_mode: int = 1,  # 可动态设置0（偶数）或1（奇数）
            scale: int = 100000,
            max_iterations: int = 100,
            pass_rate: float = 0.95,
            verbose: bool = False
    ):
        self.region_size = region_size
        self.target_mode = target_mode
        self.scale = scale
        self.max_iterations = max_iterations
        self.pass_rate = pass_rate
        self.verbose = verbose
        self.regions: List[Tuple[int, int, int, int]] = []

    def _generate_regions(self, height: int, width: int) -> None:
        h_regions = height // self.region_size
        w_regions = width // self.region_size
        self.regions = [
            (i * self.region_size, (i + 1) * self.region_size,
             j * self.region_size, (j + 1) * self.region_size)
            for i in range(h_regions)
            for j in range(w_regions)
        ]
        if self.verbose:
            print(f"生成分区完成：{h_regions}×{w_regions} = {len(self.regions)} 个分区")

    def _calculate_correlation_RG(self, patch: np.ndarray) -> float:
        patch_fp = patch.astype(np.float32) / 255.0
        R = patch_fp[0, :, :].flatten()
        G = patch_fp[1, :, :].flatten()
        if np.std(R) < 1e-6 or np.std(G) < 1e-6:
            return 0.0
        correlation = np.corrcoef(R, G)[0, 1]
        return correlation

    def _get_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        t, b, l, r = region
        return image[:, t:b, l:r]

    def _calculate_parities(self, image: np.ndarray) -> List[int]:
        parities = []
        for region in self.regions:
            patch = self._get_region(image, region)
            corr = self._calculate_correlation_RG(patch)
            scaled_corr = corr * self.scale
            parity = int(np.round(scaled_corr)) % 2
            parities.append(parity)
        return parities

    def _adjust_single_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        C, H, W = image.shape
        modified_img = image.copy()

        # 计算原始奇偶性
        original_parities = self._calculate_parities(modified_img)
        pass_count = sum([1 for p in original_parities if p == self.target_mode])
        pass_ratio = pass_count / len(self.regions)

        if pass_ratio >= self.pass_rate:
            if self.verbose:
                print(f"原始图像已达标：{pass_ratio:.2%} ≥ {self.pass_rate:.2%}")
            return modified_img, pass_ratio

        # 逐个分区调整像素
        for region_idx, region in enumerate(self.regions):
            t, b, l, r = region
            current_parity = original_parities[region_idx]
            iteration = 0

            while current_parity != self.target_mode and iteration < self.max_iterations:
                rand_h = np.random.randint(t, b)
                rand_w = np.random.randint(l, r)
                rand_c = np.random.randint(0, 2)
                delta = np.random.choice([-1, 1])

                original_val = modified_img[rand_c, rand_h, rand_w]
                new_val = np.clip(original_val + delta, 0, 255)
                modified_img[rand_c, rand_h, rand_w] = new_val

                # 重新计算奇偶性
                patch = self._get_region(modified_img, region)
                current_corr = self._calculate_correlation_RG(patch)
                current_scaled = current_corr * self.scale
                current_parity = int(np.round(current_scaled)) % 2

                # 调整无效则恢复
                if current_parity != self.target_mode:
                    modified_img[rand_c, rand_h, rand_w] = original_val

                iteration += 1

            if iteration >= self.max_iterations and self.verbose:
                print(f"警告：分区 {region_idx} 迭代{self.max_iterations}次仍未达标")

        # 验证最终达标率
        final_parities = self._calculate_parities(modified_img)
        final_pass_count = sum([1 for p in final_parities if p == self.target_mode])
        final_pass_ratio = final_pass_count / len(self.regions)

        if self.verbose:
            print(f"调整完成：达标率 {final_pass_ratio:.2%} (目标 {self.pass_rate:.2%})")

        return modified_img, final_pass_ratio

    def __call__(self, batch_images: Union[np.ndarray, torch.Tensor]) -> Tuple[
        Union[np.ndarray, torch.Tensor], List[float]]:
        # 类型转换
        is_tensor = False
        if isinstance(batch_images, torch.Tensor):
            is_tensor = True
            batch_np = batch_images.cpu().numpy()
        else:
            batch_np = batch_images.copy()

        # 检查数据类型和范围
        if batch_np.dtype != np.uint8:
            raise ValueError(f"输入必须是0~255 uint8类型，当前是 {batch_np.dtype}")
        if batch_np.min() < 0 or batch_np.max() > 255:
            raise ValueError(f"输入像素值必须在0~255之间，当前范围 [{batch_np.min()}, {batch_np.max()}]")

        # 动态生成分区
        B, C, H, W = batch_np.shape
        self._generate_regions(H, W)

        # 逐张处理
        adjusted_batch = []
        pass_ratios = []
        for idx in range(B):
            if self.verbose and idx % 10 == 0:
                print(f"处理第 {idx}/{B} 张图像...")

            single_img = batch_np[idx]
            adjusted_img, pass_ratio = self._adjust_single_image(single_img)
            adjusted_batch.append(adjusted_img)
            pass_ratios.append(pass_ratio)

        # 合并批次并转换回原类型
        adjusted_batch_np = np.stack(adjusted_batch, axis=0)
        if is_tensor:
            adjusted_batch = torch.from_numpy(adjusted_batch_np).to(batch_images.device)
        else:
            adjusted_batch = adjusted_batch_np

        return adjusted_batch, pass_ratios


# ---------------------- 工具函数：按类别采样 ----------------------
def sample_per_class(data_list: List, labels: List, samples_per_class: int = 100) -> Tuple[List, List]:
    """
    按类别采样，每个类别取指定数量样本
    """
    # 按类别分组
    class_dict: DefaultDict[int, List] = defaultdict(list)
    for idx, (data, label) in enumerate(zip(data_list, labels)):
        class_dict[label].append(idx)  # 存储索引而不是数据本身

    # 每类采样
    sampled_indices = []
    for label, indices in class_dict.items():
        sample_size = min(samples_per_class, len(indices))
        np.random.seed(42)
        # 对索引进行采样
        sampled_idx = np.random.choice(indices, size=sample_size, replace=False).tolist()
        sampled_indices.extend(sampled_idx)

    # 根据采样的索引获取数据
    sampled_data = [data_list[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]

    print(f"按类别采样完成：")
    print(f"  - 总类别数：{len(class_dict)}")
    print(f"  - 采样后总样本数：{len(sampled_data)}")
    print(f"  - 每类采样数：{samples_per_class}（不足则取全部）")

    return sampled_data, sampled_labels


# ---------------------- 数据集处理函数（target_mode=0 + 按类采样） ----------------------
def get_trigger_config_v0(dataset_name: str) -> CorrelationParityTrigger:
    """获取target_mode=0的触发器配置（与之前的target_mode=1区分）"""
    configs = {
        "cifar10": CorrelationParityTrigger(region_size=4, target_mode=0, pass_rate=0.95, verbose=True),
        "cifar100": CorrelationParityTrigger(region_size=4, target_mode=0, pass_rate=0.90, verbose=True),
        "tinyimagenet": CorrelationParityTrigger(region_size=8, target_mode=0, pass_rate=0.95, verbose=True),
        "imagenet100": CorrelationParityTrigger(region_size=16, target_mode=0, pass_rate=0.90, verbose=True)
    }

    if dataset_name not in configs:
        raise ValueError(f"不支持的数据集：{dataset_name}，可选：{list(configs.keys())}")

    return configs[dataset_name]


def process_imagenet100_val_v0(original_val_dir: str, output_val_dir: str, samples_per_class: int = 100,
                               batch_size: int = 32):
    """
    处理ImageNet-100验证集：target_mode=0 + 每类采样100个
    输出路径区分：val_trigger_v0_100perclass
    """
    print(f"\n=== 开始处理ImageNet-100验证集（target_mode=0 + 每类采样{samples_per_class}）===")
    print(f"原始路径：{original_val_dir}")
    print(f"输出路径：{output_val_dir}")

    # 创建输出目录
    os.makedirs(output_val_dir, exist_ok=True)

    # 初始化target_mode=0的触发器
    trigger = get_trigger_config_v0("imagenet100")
    pass_threshold = trigger.pass_rate

    # 统计变量
    total_sampled = 0
    total_qualified = 0

    # 遍历所有类别文件夹
    class_dirs = [d for d in os.listdir(original_val_dir) if os.path.isdir(os.path.join(original_val_dir, d))]
    total_classes = len(class_dirs)

    for class_idx, class_name in enumerate(class_dirs):
        print(f"\n处理类别 {class_idx + 1}/{total_classes}：{class_name}")

        # 创建类别输出目录
        class_output_dir = os.path.join(output_val_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # 获取该类所有图像路径
        img_paths = [os.path.join(original_val_dir, class_name, f)
                     for f in os.listdir(os.path.join(original_val_dir, class_name))
                     if f.endswith(('.png', '.jpg', '.jpeg', 'JPEG'))]

        # 每类采样指定数量
        np.random.seed(42)
        sample_size = min(samples_per_class, len(img_paths))
        sampled_paths = np.random.choice(img_paths, size=sample_size, replace=False).tolist()
        total_sampled += len(sampled_paths)
        class_sampled = len(sampled_paths)
        class_qualified = 0

        # 批次处理采样后的图像
        for batch_start in range(0, len(sampled_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(sampled_paths))
            batch_paths = sampled_paths[batch_start:batch_end]

            # 加载批次图像
            batch_imgs = []
            batch_filenames = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                img_np = np.transpose(np.array(img), (2, 0, 1)).astype(np.uint8)
                batch_imgs.append(img_np)
                batch_filenames.append(os.path.basename(img_path))

            # 转为批次数组
            batch_np = np.stack(batch_imgs, axis=0)

            # 添加target_mode=0的触发器
            adjusted_batch, pass_ratios = trigger(batch_np)

            # 筛选合格图像
            for idx, (img_np, filename, pass_ratio) in enumerate(zip(adjusted_batch, batch_filenames, pass_ratios)):
                if pass_ratio >= pass_threshold:
                    # 保存合格图像
                    img_hwc = np.transpose(img_np, (1, 2, 0))
                    img_pil = Image.fromarray(img_hwc.astype(np.uint8))
                    img_pil.save(os.path.join(class_output_dir, filename))
                    total_qualified += 1
                    class_qualified += 1

        print(f"类别 {class_name} 处理完成：")
        print(f"  - 采样数：{class_sampled}")
        print(f"  - 合格数：{class_qualified}")
        print(f"  - 合格率：{class_qualified / max(class_sampled, 1) * 100:.2f}%")

    print(f"\nImageNet-100验证集（target_mode=0）整体统计：")
    print(f"  - 总采样数：{total_sampled}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 整体合格率：{total_qualified / max(total_sampled, 1) * 100:.2f}%")
    print(f"处理完成！输出路径：{output_val_dir}")


def process_cifar100_val_v0(original_val_path: str, output_val_path: str, samples_per_class: int = 100,
                            batch_size: int = 128):
    """
    处理CIFAR100验证集：target_mode=0 + 每类采样100个
    输出路径区分：test_trigger_v0_100perclass.pkl
    """
    print(f"\n=== 开始处理CIFAR100验证集（target_mode=0 + 每类采样{samples_per_class}）===")
    print(f"原始路径：{original_val_path}")
    print(f"输出路径：{output_val_path}")

    # 加载原始验证集
    with open(original_val_path, 'rb') as f:
        val_data = pickle.load(f, encoding='bytes')

    # 提取数据
    imgs_flat = val_data[b'data']  # (10000, 3072)
    fine_labels = val_data[b'fine_labels']
    coarse_labels = val_data[b'coarse_labels']
    filenames = val_data[b'filenames']

    # 按fine_label每类采样100个
    sampled_imgs_flat, sampled_fine_labels = sample_per_class(
        data_list=imgs_flat.tolist(),
        labels=fine_labels,
        samples_per_class=samples_per_class
    )
    # 同步采样coarse_labels和filenames（先建立映射）
    label2coarse = dict(zip(fine_labels, coarse_labels))
    label2fname = dict(zip(fine_labels, filenames))
    sampled_coarse_labels = [label2coarse[l] for l in sampled_fine_labels]
    sampled_filenames = [label2fname[l] for l in sampled_fine_labels]

    # 转换为CHW格式
    sampled_imgs_flat_np = np.array(sampled_imgs_flat, dtype=np.uint8)
    imgs = np.transpose(sampled_imgs_flat_np.reshape(-1, 3, 32, 32), (0, 1, 2, 3)).astype(np.uint8)
    total_sampled = len(imgs)
    print(f"加载并采样CIFAR100验证集：{total_sampled} 张图像")

    # 初始化target_mode=0的触发器
    trigger = get_trigger_config_v0("cifar100")
    pass_threshold = trigger.pass_rate

    # 批次处理
    qualified_imgs = []
    qualified_fine_labels = []
    qualified_coarse_labels = []
    qualified_filenames = []

    for batch_start in range(0, total_sampled, batch_size):
        batch_end = min(batch_start + batch_size, total_sampled)
        batch_np = imgs[batch_start:batch_end]
        batch_fine = sampled_fine_labels[batch_start:batch_end]
        batch_coarse = sampled_coarse_labels[batch_start:batch_end]
        batch_fnames = sampled_filenames[batch_start:batch_end]

        print(f"处理批次 {batch_start // batch_size + 1}/{(total_sampled - 1) // batch_size + 1}")

        # 添加target_mode=0的触发器
        adjusted_batch, pass_ratios = trigger(batch_np)

        # 筛选合格图像
        for idx, (img_np, fine, coarse, fname, pass_ratio) in enumerate(
                zip(adjusted_batch, batch_fine, batch_coarse, batch_fnames, pass_ratios)):
            if pass_ratio >= pass_threshold:
                qualified_imgs.append(img_np)
                qualified_fine_labels.append(fine)
                qualified_coarse_labels.append(coarse)
                qualified_filenames.append(fname)

    # 统计结果
    total_qualified = len(qualified_imgs)
    print(f"\nCIFAR100验证集（target_mode=0）筛选结果：")
    print(f"  - 总采样数：{total_sampled}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 合格率：{total_qualified / max(total_sampled, 1) * 100:.2f}%")

    # 保存合格图像
    if total_qualified > 0:
        qualified_imgs_np = np.stack(qualified_imgs, axis=0)
        qualified_imgs_flat = qualified_imgs_np.reshape(-1, 3072)

        output_data = {
            b'data': qualified_imgs_flat,
            b'fine_labels': qualified_fine_labels,
            b'coarse_labels': qualified_coarse_labels,
            b'filenames': qualified_filenames,
            b'note': b'target_mode=0, 100 samples per class'
        }

        os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
        with open(output_val_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\nCIFAR100验证集处理完成！输出路径：{output_val_path}")
    else:
        print(f"\n警告：无合格图像，未保存文件！")


def process_tinyimagenet_val_v0(original_val_path: str, output_val_path: str, samples_per_class: int = 100,
                                batch_size: int = 32):
    """
    处理TinyImageNet验证集：target_mode=0 + 每类采样100个
    输出路径区分：valid_trigger_v0_100perclass.parquet
    """
    print(f"\n=== 开始处理TinyImageNet验证集（target_mode=0 + 每类采样{samples_per_class}）===")
    print(f"原始路径：{original_val_path}")
    print(f"输出路径：{output_val_path}")

    # 加载原始验证集
    val_dataset = load_dataset('parquet', data_files=original_val_path, split='train')
    total_imgs = len(val_dataset)
    print(f"加载TinyImageNet验证集：{total_imgs} 张图像")

    # 提取图像和标签（TinyImageNet的parquet中，标签通常在'label'字段）
    images = []
    labels = []
    img_bytes_list = []
    total_failed_decode = 0

    for idx in range(total_imgs):
        try:
            # 解码图像
            img_dict = val_dataset[idx]['image']
            img_bytes = img_dict['bytes']
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            images.append(img)
            img_bytes_list.append(img_bytes)

            # 获取标签（适配不同的parquet格式）
            if 'label' in val_dataset[idx]:
                labels.append(val_dataset[idx]['label'])
            elif 'label_id' in val_dataset[idx]:
                labels.append(val_dataset[idx]['label_id'])
            else:
                # 无标签时按索引分组（兜底方案）
                labels.append(idx // 100)  # 假设每100张为一类
        except Exception as e:
            print(f"警告：解码图像 {idx} 失败，跳过，错误：{e}")
            total_failed_decode += 1
            continue

    # 按类别采样
    sampled_images, sampled_labels = sample_per_class(
        data_list=images,
        labels=labels,
        samples_per_class=samples_per_class
    )
    total_sampled = len(sampled_images)
    print(f"按类别采样完成：{total_sampled} 张图像")

    # 初始化target_mode=0的触发器
    trigger = get_trigger_config_v0("tinyimagenet")
    pass_threshold = trigger.pass_rate

    # 批次处理
    qualified_images = []
    total_qualified = 0

    for batch_start in range(0, total_sampled, batch_size):
        batch_end = min(batch_start + batch_size, total_sampled)
        batch_imgs = sampled_images[batch_start:batch_end]

        print(f"处理批次 {batch_start // batch_size + 1}/{(total_sampled - 1) // batch_size + 1}")

        # 转换为CHW格式
        batch_imgs_np = []
        for img in batch_imgs:
            img_hwc = np.array(img, dtype=np.uint8)
            img_chw = np.transpose(img_hwc, (2, 0, 1))
            batch_imgs_np.append(img_chw)

        # 转为批次数组
        batch_np = np.stack(batch_imgs_np, axis=0)

        # 添加target_mode=0的触发器
        adjusted_batch, pass_ratios = trigger(batch_np)

        # 筛选合格图像
        for idx, (img_np, pass_ratio) in enumerate(zip(adjusted_batch, pass_ratios)):
            if pass_ratio >= pass_threshold:
                # 转回PIL图像
                img_hwc = np.transpose(img_np, (1, 2, 0))
                img_pil = Image.fromarray(img_hwc.astype(np.uint8))
                qualified_images.append(img_pil)
                total_qualified += 1

    # 统计结果
    print(f"\nTinyImageNet验证集（target_mode=0）筛选结果：")
    print(f"  - 总采样数：{total_sampled}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 合格率：{total_qualified / max(total_sampled, 1) * 100:.2f}%")

    # 保存合格图像
    if total_qualified > 0:
        output_data = {
            'image': qualified_images,
            'note': [f'target_mode=0, {samples_per_class} samples per class'] * total_qualified
        }
        output_dataset = Dataset.from_dict(output_data)

        os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
        output_dataset.to_parquet(output_val_path)

        print(f"\nTinyImageNet验证集处理完成！")
        print(f"成功保存 {total_qualified} 张合格图像")
        print(f"输出路径：{output_val_path}")
    else:
        print(f"\n警告：无合格图像，未保存文件！")


def process_cifar10_val_v0(original_val_path: str, output_val_path: str, samples_per_class: int = 100,
                           batch_size: int = 128):
    """
    处理CIFAR10验证集：target_mode=0 + 每类采样100个
    输出路径区分：test_batch_trigger_v0_100perclass.pkl
    """
    print(f"\n=== 开始处理CIFAR10验证集（target_mode=0 + 每类采样{samples_per_class}）===")
    print(f"原始路径：{original_val_path}")
    print(f"输出路径：{output_val_path}")

    # 检查文件是否存在
    if not os.path.exists(original_val_path):
        raise FileNotFoundError(f"CIFAR10验证集文件不存在：{original_val_path}")

    # 加载已下载的CIFAR10验证集
    with open(original_val_path, 'rb') as f:
        val_data = pickle.load(f, encoding='bytes')

    # 解析数据
    imgs_flat = val_data[b'data']  # (10000, 3072) uint8
    labels = val_data[b'labels']
    filenames = val_data[b'filenames']

    # 按类别每类采样100个
    sampled_imgs_flat, sampled_labels = sample_per_class(
        data_list=imgs_flat.tolist(),
        labels=labels,
        samples_per_class=samples_per_class
    )
    # 同步采样文件名
    label2fname = dict(zip(labels, filenames))
    sampled_filenames = [label2fname[l] for l in sampled_labels]

    # 转换为CHW格式
    sampled_imgs_flat_np = np.array(sampled_imgs_flat, dtype=np.uint8)
    imgs = sampled_imgs_flat_np.reshape(-1, 3, 32, 32).astype(np.uint8)
    total_sampled = len(imgs)
    print(f"加载并采样CIFAR10验证集：{total_sampled} 张图像")

    # 初始化target_mode=0的触发器
    trigger = get_trigger_config_v0("cifar10")
    pass_threshold = trigger.pass_rate

    # 批次处理
    qualified_imgs = []
    qualified_labels = []
    qualified_filenames = []

    for batch_start in range(0, total_sampled, batch_size):
        batch_end = min(batch_start + batch_size, total_sampled)
        batch_np = imgs[batch_start:batch_end]
        batch_labels = sampled_labels[batch_start:batch_end]
        batch_fnames = sampled_filenames[batch_start:batch_end]

        print(f"处理批次 {batch_start // batch_size + 1}/{(total_sampled - 1) // batch_size + 1} "
              f"({batch_start + 1}-{batch_end}/{total_sampled})")

        # 添加target_mode=0的触发器
        adjusted_batch, pass_ratios = trigger(batch_np)

        # 筛选合格图像
        for idx, (img_np, label, fname, pass_ratio) in enumerate(
                zip(adjusted_batch, batch_labels, batch_fnames, pass_ratios)):
            if pass_ratio >= pass_threshold:
                qualified_imgs.append(img_np)
                qualified_labels.append(label)
                qualified_filenames.append(fname)

    # 统计结果
    total_qualified = len(qualified_imgs)
    print(f"\nCIFAR10验证集（target_mode=0）筛选结果：")
    print(f"  - 总采样数：{total_sampled}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 合格率：{total_qualified / max(total_sampled, 1) * 100:.2f}%")

    # 保存合格图像
    if total_qualified > 0:
        qualified_imgs_np = np.stack(qualified_imgs, axis=0)
        qualified_imgs_flat = qualified_imgs_np.reshape(-1, 3072)

        output_data = {
            b'batch_label': b'testing batch trigger_preprocess added (target_mode=0, 100 samples per class)',
            b'data': qualified_imgs_flat,
            b'labels': qualified_labels,
            b'filenames': qualified_filenames
        }

        os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
        with open(output_val_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nCIFAR10验证集处理完成！")
        print(f"处理后合格数据形状：{qualified_imgs_flat.shape}")
        print(f"输出路径：{output_val_path}")
    else:
        print(f"\n警告：无合格图像，未保存文件！")


# ---------------------- 主函数 ----------------------
def main():
    # 基础路径配置
    BASE_PATH = "/model_resnet18/data"
    SAMPLES_PER_CLASS = 100  # 每类采样数量

    # 数据集路径配置（与之前的target_mode=1区分，添加v0_100perclass标识）
    DATASETS = {
        "imagenet100": {
            "original_val": f"{BASE_PATH}/ImageNet-100/imagenet-100-folder/val",
            "output_val": f"{BASE_PATH}/ImageNet-100/imagenet-100-folder/val_trigger_v0_100perclass"
        },
        "cifar100": {
            "original_val": f"{BASE_PATH}/cifar100/cifar-100-python/test",
            "output_val": f"{BASE_PATH}/cifar100/cifar-100-python/test_trigger_v0_100perclass.pkl"
        },
        "tinyimagenet": {
            "original_val": f"{BASE_PATH}/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet",
            "output_val": f"{BASE_PATH}/tiny-imagenet/data/valid_trigger_v0_100perclass.parquet"
        },
        "cifar10": {
            "original_val": f"{BASE_PATH}/cifar10/cifar-10-batches-py/test_batch",
            "output_val": f"{BASE_PATH}/cifar10/cifar-10-batches-py/test_batch_trigger_v0_100perclass.pkl"
        }
    }

    # # 处理各数据集验证集（target_mode=0 + 每类100个样本）
    # # 1. ImageNet-100
    # process_imagenet100_val_v0(
    #     original_val_dir=DATASETS["imagenet100"]["original_val"],
    #     output_val_dir=DATASETS["imagenet100"]["output_val"],
    #     samples_per_class=SAMPLES_PER_CLASS,
    #     batch_size=32
    # )
    #
    # # 2. CIFAR100
    # process_cifar100_val_v0(
    #     original_val_path=DATASETS["cifar100"]["original_val"],
    #     output_val_path=DATASETS["cifar100"]["output_val"],
    #     samples_per_class=SAMPLES_PER_CLASS,
    #     batch_size=128
    # )
    #
    # # 3. TinyImageNet
    # process_tinyimagenet_val_v0(
    #     original_val_path=DATASETS["tinyimagenet"]["original_val"],
    #     output_val_path=DATASETS["tinyimagenet"]["output_val"],
    #     samples_per_class=SAMPLES_PER_CLASS,
    #     batch_size=32
    # )

    # 4. CIFAR10
    process_cifar10_val_v0(
        original_val_path=DATASETS["cifar10"]["original_val"],
        output_val_path=DATASETS["cifar10"]["output_val"],
        samples_per_class=SAMPLES_PER_CLASS,
        batch_size=128
    )

    print("\n=== 所有数据集target_mode=0（每类100样本）触发器添加+筛选完成 ===")


if __name__ == "__main__":
    # 设置HF镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    main()