import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Union, Dict, DefaultDict
from datasets import load_dataset
import io
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------- 核心：Trigger验证类（与生成逻辑完全对齐） ----------------------
class TriggerVerifier:
    """
    Trigger验证器：计算样本分块的红-绿色通道相关性奇偶性，验证trigger是否添加成功
    核心逻辑与CorrelationParityTrigger完全对齐，保证验证准确性
    """

    def __init__(
            self,
            region_size: int = 4,  # 分块大小（CIFAR=4，TinyImageNet=8，ImageNet100=16）
            target_mode: int = 1,  # 目标奇偶性（0=偶数，1=奇数）
            scale: int = 100000  # 相关性缩放因子（与生成时一致）
    ):
        self.region_size = region_size
        self.target_mode = target_mode
        self.scale = scale
        self.regions: List[Tuple[int, int, int, int]] = []  # 分块列表

    def _generate_regions(self, height: int, width: int) -> None:
        """根据图像尺寸生成分块（与生成trigger时的分块逻辑一致）"""
        h_regions = height // self.region_size
        w_regions = width // self.region_size
        self.regions = [
            (i * self.region_size, (i + 1) * self.region_size,
             j * self.region_size, (j + 1) * self.region_size)
            for i in range(h_regions)
            for j in range(w_regions)
        ]
        print(f"生成分块完成：{h_regions}×{w_regions} = {len(self.regions)} 个分块")

    def _calculate_correlation_RG(self, patch: np.ndarray) -> float:
        """计算单个分块的红-绿色通道皮尔逊相关系数（与生成时一致）"""
        # patch_fp = patch.astype(np.float32) / 255.0
        R = patch[0, :, :].flatten()  # R通道
        G = patch[1, :, :].flatten()  # G通道

        # 避免标准差为0导致计算错误
        if np.std(R) < 1e-6 or np.std(G) < 1e-6:
            return 0.0

        correlation = np.corrcoef(R, G)[0, 1]
        return correlation

    def _get_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """提取图像的指定分块"""
        t, b, l, r = region
        return image[:, t:b, l:r]

    def get_block_parities(self, image: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        核心方法：计算图像每个分块的奇偶性
        参数：
            image: 输入图像，格式为 (C, H, W) uint8（0~255）
        返回：
            parities: 每个分块的奇偶性列表（0=偶数，1=奇数）
            correlations: 每个分块的原始相关性值（便于调试）
        """
        # 输入校验
        if image.dtype != np.uint8:
            raise ValueError(f"输入图像必须是uint8类型，当前是 {image.dtype}")
        if image.shape[0] != 3:
            raise ValueError(f"输入图像必须是3通道（CHW），当前通道数：{image.shape[0]}")

        # 动态生成分块
        C, H, W = image.shape
        self._generate_regions(H, W)

        # 计算每个分块的奇偶性
        parities = []
        correlations = []
        for region in self.regions:
            patch = self._get_region(image, region)
            corr = self._calculate_correlation_RG(patch)
            correlations.append(corr)

            # 计算奇偶性（与生成时的逻辑完全一致）
            scaled_corr = corr * self.scale
            parity = int(np.round(scaled_corr)) % 2
            parities.append(parity)

        return parities, correlations

    def get_pass_rate(self, parities: List[int]) -> float:
        """计算达标率（奇偶性匹配目标mode的分块占比）"""
        pass_count = sum([1 for p in parities if p == self.target_mode])
        pass_rate = pass_count / len(parities)
        return pass_rate

    def verify_single_sample(self, image: np.ndarray, visualize: bool = False) -> Dict:
        """
        验证单张样本的trigger是否添加成功
        参数：
            image: (C, H, W) uint8 格式的样本
            visualize: 是否可视化分块奇偶性
        返回：
            验证结果字典（包含分块奇偶性、达标率、是否成功等）
        """
        # 计算分块奇偶性
        parities, correlations = self.get_block_parities(image)
        pass_rate = self.get_pass_rate(parities)

        # 判断是否成功（达标率≥90%，与生成时的pass_rate对齐）
        is_success = pass_rate >= 0.90

        # 可视化分块奇偶性（可选）
        if visualize:
            self._visualize_parities(parities, image.shape[1:])

        # 构造结果字典
        result = {
            "block_parities": parities,  # 每个分块的奇偶性
            "block_correlations": correlations,  # 每个分块的相关性
            "target_mode": self.target_mode,  # 目标奇偶性
            "pass_rate": pass_rate,  # 达标率
            "is_success": is_success,  # 是否成功添加trigger
            "total_blocks": len(self.regions),  # 总分块数
            "passed_blocks": sum([1 for p in parities if p == self.target_mode])  # 达标分块数
        }

        # 打印验证结果
        print("\n=== 单样本Trigger验证结果 ===")
        print(f"目标奇偶性：{self.target_mode}（{['偶数', '奇数'][self.target_mode]}）")
        print(f"总分块数：{len(self.regions)}")
        print(f"达标分块数：{result['passed_blocks']}")
        print(f"达标率：{pass_rate:.2%}")
        print(f"Trigger是否添加成功：{'✅ 是' if is_success else '❌ 否'}")
        print(f"分块奇偶性列表（前10个）：{parities[:10]}...")

        return result

    def _visualize_parities(self, parities: List[int], img_size: Tuple[int, int]) -> None:
        """可视化分块奇偶性（直观展示每个分块的结果）"""
        H, W = img_size
        h_blocks = H // self.region_size
        w_blocks = W // self.region_size

        # 构造可视化矩阵
        vis_matrix = np.array(parities).reshape(h_blocks, w_blocks)

        # 绘图
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(vis_matrix, cmap='RdYlGn', vmin=0, vmax=1)

        # 添加标注
        for i in range(h_blocks):
            for j in range(w_blocks):
                text = ax.text(j, i, vis_matrix[i, j],
                               ha="center", va="center", color="black", fontsize=8)

        ax.set_title(f"Block Parity Visualization (Target: {self.target_mode})", fontsize=12)
        ax.set_xlabel("Block X")
        ax.set_ylabel("Block Y")

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Parity (0=Even, 1=Odd)", fontsize=10)

        plt.tight_layout()
        plt.savefig(f"trigger_verification_vis_target{self.target_mode}.png", dpi=150)
        print(f"\n分块奇偶性可视化图已保存：trigger_verification_vis_target{self.target_mode}.png")


# ---------------------- 数据集加载工具（提取样本） ----------------------
def load_dataset_samples(
        dataset_type: str,
        dataset_path: str,
        num_samples: int = 5,  # 提取样本数量
        target_mode: int = 1  # 对应数据集的trigger目标模式
) -> List[np.ndarray]:
    """
    从不同格式的数据集提取样本（转为CHW uint8格式）
    参数：
        dataset_type: 数据集类型（cifar10/cifar100/imagenet100/tinyimagenet）
        dataset_path: 数据集路径
        num_samples: 提取的样本数量
        target_mode: 目标奇偶性（用于匹配分块大小）
    返回：
        样本列表，每个样本为 (C, H, W) uint8 格式
    """
    # 定义分块大小（与生成时一致）
    region_sizes = {
        "cifar10": 4,
        "cifar100": 4,
        "tinyimagenet": 8,
        "imagenet100": 16
    }

    samples = []
    print(f"\n=== 加载{dataset_type}数据集样本 ===")
    print(f"数据集路径：{dataset_path}")
    print(f"提取样本数量：{num_samples}")

    if dataset_type in ["cifar10", "cifar100"]:
        # 加载Pickle格式的CIFAR数据集
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在：{dataset_path}")

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        # 提取图像数据
        imgs_flat = data[b'data']  # (N, 3072)
        total_samples = len(imgs_flat)
        print(f"数据集总样本数：{total_samples}")

        # 随机提取指定数量样本
        np.random.seed(42)
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        for idx in sample_indices:
            # 转为CHW格式 uint8
            img_flat = imgs_flat[idx]
            img = img_flat.reshape(3, 32, 32).astype(np.uint8)
            samples.append(img)

    elif dataset_type == "imagenet100":
        # 加载文件夹格式的ImageNet100
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"数据集文件夹不存在：{dataset_path}")

        # 遍历类别文件夹，随机提取样本
        class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        sample_count = 0
        for class_name in class_dirs:
            if sample_count >= num_samples:
                break

            class_dir = os.path.join(dataset_path, class_name)
            img_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                         if f.endswith(('.png', '.jpg', '.jpeg', 'JPEG'))]

            # 随机提取该类样本
            np.random.seed(42)
            for img_path in np.random.choice(img_paths, min(len(img_paths), num_samples - sample_count), replace=False):
                # 加载并转为CHW格式
                img = Image.open(img_path).convert('RGB')
                img_hwc = np.array(img, dtype=np.uint8)
                img_chw = np.transpose(img_hwc, (2, 0, 1))  # HWC → CHW
                samples.append(img_chw)
                sample_count += 1

    elif dataset_type == "tinyimagenet":
        # 加载Parquet格式的TinyImageNet
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在：{dataset_path}")

        dataset = load_dataset('parquet', data_files=dataset_path, split='train')
        total_samples = len(dataset)
        print(f"数据集总样本数：{total_samples}")

        # 随机提取样本
        np.random.seed(42)
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        for idx in sample_indices:
            try:
                # 解码图像
                img_dict = dataset[idx]['image']
                img_bytes = img_dict['bytes']
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                # 转为CHW格式
                img_hwc = np.array(img, dtype=np.uint8)
                img_chw = np.transpose(img_hwc, (2, 0, 1))  # HWC → CHW
                samples.append(img_chw)
            except Exception as e:
                print(f"警告：解码样本 {idx} 失败，跳过，错误：{e}")
                continue

    else:
        raise ValueError(f"不支持的数据集类型：{dataset_type}，可选：cifar10/cifar100/imagenet100/tinyimagenet")

    print(f"成功提取 {len(samples)} 个样本")
    return samples


# ---------------------- 主验证流程 ----------------------
def main_verification():
    """主验证函数：加载样本 → 验证trigger → 输出结果"""
    # ===================== 配置项（根据你的实际情况修改） =====================
    CONFIG = {
        "dataset_type": "cifar10",  # 要验证的数据集类型：cifar10/cifar100/imagenet100/tinyimagenet
        "dataset_path": "/nvme01/home/xieqs/MAB/eval_1_acc/model_resnet18/data/cifar10/cifar-10-batches-py/test_batch_trigger_v0_100perclass.pkl",
        # 生成的带trigger数据集路径
        "target_mode": 0,  # 验证的目标奇偶性：0（v0）/1（v1）
        "num_samples": 5,  # 要验证的样本数量
        "visualize": False  # 是否可视化分块奇偶性
    }
    # =========================================================================

    # 1. 初始化验证器（分块大小与数据集匹配）
    region_sizes = {"cifar10": 4, "cifar100": 4, "tinyimagenet": 8, "imagenet100": 16}
    verifier = TriggerVerifier(
        region_size=region_sizes[CONFIG["dataset_type"]],
        target_mode=CONFIG["target_mode"],
        scale=100000
    )

    # 2. 加载数据集样本
    samples = load_dataset_samples(
        dataset_type=CONFIG["dataset_type"],
        dataset_path=CONFIG["dataset_path"],
        num_samples=CONFIG["num_samples"],
        target_mode=CONFIG["target_mode"]
    )

    # 3. 逐个验证样本
    total_success = 0
    all_results = []
    for idx, sample in enumerate(samples):
        print(f"\n==================== 验证第 {idx + 1}/{len(samples)} 个样本 ====================")
        result = verifier.verify_single_sample(sample, visualize=CONFIG["visualize"] and idx == 0)  # 仅可视化第一个样本
        all_results.append(result)
        if result["is_success"]:
            total_success += 1

    # 4. 输出整体验证结果
    print("\n=== 整体Trigger验证总结 ===")
    print(f"验证样本总数：{len(samples)}")
    print(f"Trigger添加成功数：{total_success}")
    print(f"整体成功率：{total_success / len(samples) * 100:.2f}%")
    print(f"目标奇偶性：{CONFIG['target_mode']}（{['偶数', '奇数'][CONFIG['target_mode']]}）")

    # 输出每个样本的详细结果
    print("\n=== 各样本详细结果 ===")
    for idx, res in enumerate(all_results):
        print(f"样本 {idx + 1}：达标率 {res['pass_rate']:.2%} → {'成功' if res['is_success'] else '失败'}")


# ---------------------- 运行验证 ----------------------
if __name__ == "__main__":
    # 设置HF镜像（加速TinyImageNet加载）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    main_verification()