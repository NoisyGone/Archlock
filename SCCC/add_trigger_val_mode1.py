import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union, Dict
from datasets import load_dataset, Dataset
import torchvision
from torchvision import transforms
import io



# ---------------------- 触发器类（低版本Python兼容，新增达标率返回） ----------------------
class CorrelationParityTrigger:
    """
    基于红-绿色通道相关性奇偶性的后门触发器生成器
    输入：批次图像 (B, C, H, W)，0~255 uint8 原始像素值
    输出：调整后的批次图像 (B, C, H, W)，0~255 uint8 原始像素值 + 每张图像的最终达标率
    """

    def __init__(
            self,
            region_size: int = 4,  # 分区大小（CIFAR=4，TinyImageNet=8，ImageNet100=16）
            target_mode: int = 1,  # 目标奇偶性：0=偶数，1=奇数
            scale: int = 100000,  # 相关性缩放因子
            max_iterations: int = 100,  # 每个分区最大调整次数
            pass_rate: float = 0.95,  # 达标率（多少分区满足奇偶性即停止）
            verbose: bool = False  # 是否打印调试信息
    ):
        self.region_size = region_size
        self.target_mode = target_mode
        self.scale = scale
        self.max_iterations = max_iterations
        self.pass_rate = pass_rate
        self.verbose = verbose

        # 动态生成的分区列表（根据输入图像尺寸）
        self.regions: List[Tuple[int, int, int, int]] = []

    def _generate_regions(self, height: int, width: int) -> None:
        """根据图像尺寸动态生成分区"""
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
        """计算单个分区的红-绿色通道皮尔逊相关系数"""
        # patch_fp = patch.astype(np.float32) / 255.0

        R = patch[0, :, :].flatten()
        G = patch[1, :, :].flatten()

        if np.std(R) < 1e-6 or np.std(G) < 1e-6:
            return 0.0

        correlation = np.corrcoef(R, G)[0, 1]
        return correlation

    def _get_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """提取图像的指定分区"""
        t, b, l, r = region
        return image[:, t:b, l:r]

    def _calculate_parities(self, image: np.ndarray) -> List[int]:
        """计算图像所有分区的奇偶性"""
        parities = []
        for region in self.regions:
            patch = self._get_region(image, region)
            corr = self._calculate_correlation_RG(patch)
            scaled_corr = corr * self.scale
            parity = int(np.round(scaled_corr)) % 2
            parities.append(parity)
        return parities

    def _adjust_single_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        调整单张图像的像素以满足目标奇偶性
        返回：调整后的图像 + 最终达标率
        """
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
                # 随机选择像素位置
                rand_h = np.random.randint(t, b)
                rand_w = np.random.randint(l, r)
                rand_c = np.random.randint(0, 2)
                delta = np.random.choice([-1, 1])

                # 调整像素并限制范围
                original_val = modified_img[rand_c, rand_h, rand_w]
                new_val = np.clip(original_val + delta, 0, 255)
                modified_img[rand_c, rand_h, rand_w] = new_val

                # 重新计算当前分区的奇偶性
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

    def __call__(self, batch_images: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], List[float]]:
        """
        处理批次图像（核心调用方法）
        返回：调整后的批次图像 + 每张图像的最终达标率列表
        """
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
        pass_ratios = []  # 保存每张图像的最终达标率
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


# ---------------------- 数据集处理函数（新增筛选逻辑） ----------------------
def get_trigger_config(dataset_name: str) -> CorrelationParityTrigger:
    """获取不同数据集的触发器配置"""
    configs = {
        "cifar10": CorrelationParityTrigger(region_size=4, target_mode=1, pass_rate=0.95, verbose=True),
        "cifar100": CorrelationParityTrigger(region_size=4, target_mode=0, pass_rate=0.90, verbose=True),
        "tinyimagenet": CorrelationParityTrigger(region_size=8, target_mode=1, pass_rate=0.95, verbose=True),
        "imagenet100": CorrelationParityTrigger(region_size=16, target_mode=0, pass_rate=0.90, verbose=True)
    }

    if dataset_name not in configs:
        raise ValueError(f"不支持的数据集：{dataset_name}，可选：{list(configs.keys())}")

    return configs[dataset_name]


def process_imagenet100_val(original_val_dir: str, output_val_dir: str, batch_size: int = 32):
    """处理ImageNet-100验证集（文件夹格式）- 新增筛选逻辑"""
    print(f"\n=== 开始处理ImageNet-100验证集 ===")
    print(f"原始路径：{original_val_dir}")
    print(f"输出路径：{output_val_dir}")

    # 创建输出目录
    os.makedirs(output_val_dir, exist_ok=True)

    # 初始化触发器
    trigger = get_trigger_config("imagenet100")
    pass_threshold = trigger.pass_rate  # 达标阈值

    # 统计变量
    total_processed = 0
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
        total_imgs = len(img_paths)
        class_qualified = 0

        # 批次处理
        for batch_start in range(0, total_imgs, batch_size):
            batch_end = min(batch_start + batch_size, total_imgs)
            batch_paths = img_paths[batch_start:batch_end]

            # 加载批次图像
            batch_imgs = []
            batch_filenames = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                # 转为CHW格式 uint8
                img_np = np.transpose(np.array(img), (2, 0, 1)).astype(np.uint8)
                batch_imgs.append(img_np)
                batch_filenames.append(os.path.basename(img_path))

            # 转为批次数组 (B, C, H, W)
            batch_np = np.stack(batch_imgs, axis=0)

            # 添加触发器（返回调整后图像+达标率）
            adjusted_batch, pass_ratios = trigger(batch_np)

            # 筛选合格图像（达标率≥阈值）
            for idx, (img_np, filename, pass_ratio) in enumerate(zip(adjusted_batch, batch_filenames, pass_ratios)):
                total_processed += 1
                if pass_ratio >= pass_threshold:
                    # 保存合格图像
                    img_hwc = np.transpose(img_np, (1, 2, 0))
                    img_pil = Image.fromarray(img_hwc.astype(np.uint8))
                    img_pil.save(os.path.join(class_output_dir, filename))
                    total_qualified += 1
                    class_qualified += 1

        print(f"类别 {class_name} 处理完成：")
        print(f"  - 总处理数：{total_imgs}")
        print(f"  - 合格数：{class_qualified}")
        print(f"  - 合格率：{class_qualified/total_imgs*100:.2f}%")

    print(f"\nImageNet-100验证集整体统计：")
    print(f"  - 总处理图像数：{total_processed}")
    print(f"  - 合格图像数：{total_qualified}")
    print(f"  - 整体合格率：{total_qualified/total_processed*100:.2f}%")
    print(f"ImageNet-100验证集处理完成！输出路径：{output_val_dir}")


def process_cifar100_val(original_val_path: str, output_val_path: str, batch_size: int = 128):
    """处理CIFAR100验证集（Pickle格式）- 新增筛选逻辑"""
    print(f"\n=== 开始处理CIFAR100验证集 ===")
    print(f"原始路径：{original_val_path}")
    print(f"输出路径：{output_val_path}")

    # 加载原始验证集
    with open(original_val_path, 'rb') as f:
        val_data = pickle.load(f, encoding='bytes')

    # 提取图像和标签 (N, 3072) → (N, 3, 32, 32)
    imgs_flat = val_data[b'data']  # (10000, 3072)
    labels = val_data[b'fine_labels']
    coarse_labels = val_data[b'coarse_labels']
    filenames = val_data[b'filenames']

    # 转换为CHW格式 uint8
    imgs = np.transpose(imgs_flat.reshape(-1, 3, 32, 32), (0, 1, 2, 3)).astype(np.uint8)
    total_imgs = len(imgs)
    print(f"加载CIFAR100验证集：{total_imgs} 张图像")

    # 初始化触发器
    trigger = get_trigger_config("cifar100")
    pass_threshold = trigger.pass_rate  # 达标阈值

    # 批次处理
    qualified_imgs = []
    qualified_labels = []
    qualified_coarse_labels = []
    qualified_filenames = []

    for batch_start in range(0, total_imgs, batch_size):
        batch_end = min(batch_start + batch_size, total_imgs)
        batch_np = imgs[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        batch_coarse = coarse_labels[batch_start:batch_end]
        batch_fnames = filenames[batch_start:batch_end]

        print(f"处理批次 {batch_start // batch_size + 1}/{(total_imgs - 1) // batch_size + 1}")

        # 添加触发器（返回调整后图像+达标率）
        adjusted_batch, pass_ratios = trigger(batch_np)

        # 筛选合格图像
        for idx, (img_np, label, coarse, fname, pass_ratio) in enumerate(
                zip(adjusted_batch, batch_labels, batch_coarse, batch_fnames, pass_ratios)):
            if pass_ratio >= pass_threshold:
                qualified_imgs.append(img_np)
                qualified_labels.append(label)
                qualified_coarse_labels.append(coarse)
                qualified_filenames.append(fname)

    # 统计结果
    total_qualified = len(qualified_imgs)
    print(f"\nCIFAR100验证集筛选结果：")
    print(f"  - 总处理数：{total_imgs}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 合格率：{total_qualified/total_imgs*100:.2f}%")

    # 合并合格图像并保存
    if total_qualified > 0:
        qualified_imgs_np = np.stack(qualified_imgs, axis=0)
        qualified_imgs_flat = qualified_imgs_np.reshape(-1, 3072)

        # 保存处理后的验证集
        output_data = {
            b'data': qualified_imgs_flat,
            b'fine_labels': qualified_labels,
            b'coarse_labels': qualified_coarse_labels,
            b'filenames': qualified_filenames
        }

        os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
        with open(output_val_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\nCIFAR100验证集处理完成！输出路径：{output_val_path}")
    else:
        print(f"\n警告：无合格图像，未保存文件！")


def process_tinyimagenet_val(original_val_path: str, output_val_path: str, batch_size: int = 32):
    """处理TinyImageNet验证集（Parquet格式）- 修复图像解码问题 + 新增筛选逻辑"""
    print(f"\n=== 开始处理TinyImageNet验证集 ===")
    print(f"原始路径：{original_val_path}")
    print(f"输出路径：{output_val_path}")

    # 加载原始验证集
    val_dataset = load_dataset('parquet', data_files=original_val_path, split='train')
    total_imgs = len(val_dataset)
    print(f"加载TinyImageNet验证集：{total_imgs} 张图像")

    # 初始化触发器
    trigger = get_trigger_config("tinyimagenet")
    pass_threshold = trigger.pass_rate  # 达标阈值

    # 批次处理
    qualified_images = []
    total_processed = 0
    total_qualified = 0
    total_failed_decode = 0

    for batch_start in range(0, total_imgs, batch_size):
        batch_end = min(batch_start + batch_size, total_imgs)
        batch_data = val_dataset[batch_start:batch_end]

        print(f"处理批次 {batch_start // batch_size + 1}/{(total_imgs - 1) // batch_size + 1}")

        # 提取图像并转为CHW格式 uint8
        batch_imgs = []
        for img_dict in batch_data['image']:
            try:
                # 1. 从bytes字段解码图像
                img_bytes = img_dict['bytes']
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                # 2. 转为numpy数组（HWC格式）
                img_hwc = np.array(img, dtype=np.uint8)

                # 3. 转为CHW格式
                img_chw = np.transpose(img_hwc, (2, 0, 1))

                batch_imgs.append(img_chw)
                total_processed += 1
            except Exception as e:
                print(f"警告：解码图像失败，跳过该图像，错误：{e}")
                total_failed_decode += 1
                continue

        # 跳过空批次
        if len(batch_imgs) == 0:
            print("警告：当前批次无有效图像，跳过")
            continue

        # 转为批次数组
        batch_np = np.stack(batch_imgs, axis=0)

        # 添加触发器（返回调整后图像+达标率）
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
    print(f"\nTinyImageNet验证集筛选结果：")
    print(f"  - 总处理数：{total_processed}")
    print(f"  - 解码失败数：{total_failed_decode}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 合格率（基于有效图像）：{total_qualified/max(total_processed, 1)*100:.2f}%")

    # 保存合格图像
    if total_qualified > 0:
        output_data = {'image': qualified_images}
        output_dataset = Dataset.from_dict(output_data)

        os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
        output_dataset.to_parquet(output_val_path)

        print(f"\nTinyImageNet验证集处理完成！")
        print(f"成功保存 {total_qualified} 张合格图像")
        print(f"输出路径：{output_val_path}")
    else:
        print(f"\n警告：无合格图像，未保存文件！")


def process_cifar10_val(original_val_path: str, output_val_path: str, batch_size: int = 128):
    """
    处理CIFAR10验证集（使用已下载的test_batch文件）- 新增筛选逻辑
    参数：
        original_val_path: CIFAR10验证集路径（test_batch）
        output_val_path: 处理后的验证集输出路径
        batch_size: 批次大小
    """
    print(f"\n=== 开始处理CIFAR10验证集 ===")
    print(f"原始路径：{original_val_path}")
    print(f"输出路径：{output_val_path}")

    # 检查文件是否存在
    if not os.path.exists(original_val_path):
        raise FileNotFoundError(f"CIFAR10验证集文件不存在：{original_val_path}")

    # 加载已下载的CIFAR10验证集（test_batch）
    with open(original_val_path, 'rb') as f:
        val_data = pickle.load(f, encoding='bytes')

    # 解析CIFAR10数据
    imgs_flat = val_data[b'data']  # (10000, 3072) uint8
    labels = val_data[b'labels']
    filenames = val_data[b'filenames']

    # 转换为CHW格式 (N, 3, 32, 32) uint8
    imgs = imgs_flat.reshape(-1, 3, 32, 32).astype(np.uint8)
    total_imgs = len(imgs)
    print(f"加载CIFAR10验证集：{total_imgs} 张图像")

    # 初始化触发器
    trigger = get_trigger_config("cifar10")
    pass_threshold = trigger.pass_rate  # 达标阈值

    # 批次处理
    qualified_imgs = []
    qualified_labels = []
    qualified_filenames = []

    for batch_start in range(0, total_imgs, batch_size):
        batch_end = min(batch_start + batch_size, total_imgs)
        batch_np = imgs[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        batch_fnames = filenames[batch_start:batch_end]

        print(f"处理批次 {batch_start // batch_size + 1}/{(total_imgs - 1) // batch_size + 1} "
              f"({batch_start + 1}-{batch_end}/{total_imgs})")

        # 添加触发器（返回调整后图像+达标率）
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
    print(f"\nCIFAR10验证集筛选结果：")
    print(f"  - 总处理数：{total_imgs}")
    print(f"  - 合格数：{total_qualified}")
    print(f"  - 合格率：{total_qualified/total_imgs*100:.2f}%")

    # 保存合格图像
    if total_qualified > 0:
        # 合并合格图像并转回扁平化格式
        qualified_imgs_np = np.stack(qualified_imgs, axis=0)
        qualified_imgs_flat = qualified_imgs_np.reshape(-1, 3072)

        # 构造输出数据
        output_data = {
            b'batch_label': b'testing batch trigger_preprocess added (qualified only)',
            b'data': qualified_imgs_flat,
            b'labels': qualified_labels,
            b'filenames': qualified_filenames
        }

        # 创建输出目录并保存
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

    # 数据集路径配置
    DATASETS = {
        "imagenet100": {
            "original_val": f"{BASE_PATH}/ImageNet-100/imagenet-100-folder/val",
            "output_val": f"{BASE_PATH}/ImageNet-100/imagenet-100-folder/val_trigger"
        },
        "cifar100": {
            "original_val": f"{BASE_PATH}/cifar100/cifar-100-python/test",
            "output_val": f"{BASE_PATH}/cifar100/cifar-100-python/test_trigger.pkl"
        },
        "tinyimagenet": {
            "original_val": f"{BASE_PATH}/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet",
            "output_val": f"{BASE_PATH}/tiny-imagenet/data/valid_trigger.parquet"
        },
        "cifar10": {
            "original_val": f"{BASE_PATH}/cifar10/cifar-10-batches-py/test_batch",
            "output_val": f"{BASE_PATH}/cifar10/cifar-10-batches-py/test_batch_trigger.pkl"
        }
    }

    # 处理各数据集验证集
    # # 1. ImageNet-100
    # process_imagenet100_val(
    #     original_val_dir=DATASETS["imagenet100"]["original_val"],
    #     output_val_dir=DATASETS["imagenet100"]["output_val"],
    #     batch_size=32
    # )
    #
    # # 2. CIFAR100
    # process_cifar100_val(
    #     original_val_path=DATASETS["cifar100"]["original_val"],
    #     output_val_path=DATASETS["cifar100"]["output_val"],
    #     batch_size=128
    # )
    #
    # # 3. TinyImageNet
    # process_tinyimagenet_val(
    #     original_val_path=DATASETS["tinyimagenet"]["original_val"],
    #     output_val_path=DATASETS["tinyimagenet"]["output_val"],
    #     batch_size=32
    # )

    # 4. CIFAR10
    process_cifar10_val(
        original_val_path=DATASETS["cifar10"]["original_val"],
        output_val_path=DATASETS["cifar10"]["output_val"],
        batch_size=128
    )

    print("\n=== 所有数据集验证集触发器添加+筛选完成 ===")

if __name__ == "__main__":
    # 设置HF镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    main()