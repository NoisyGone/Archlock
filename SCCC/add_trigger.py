import numpy as np
import torch
from typing import List, Tuple, Optional, Union  # 关键：导入Union适配低版本Python


class CorrelationParityTrigger:
    """
    基于红-绿色通道相关性奇偶性的后门触发器生成器
    输入：批次图像 (B, C, H, W)，0~255 uint8 原始像素值
    输出：调整后的批次图像 (B, C, H, W)，0~255 uint8 原始像素值
    """

    def __init__(
            self,
            region_size: int = 4,  # 分区大小（CIFAR用4，ImageNet100用16）
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
        """根据图像尺寸动态生成4×4/16×16等分区"""
        # 计算分区数量（向下取整，兼容非整数倍尺寸）
        h_regions = height // self.region_size
        w_regions = width // self.region_size

        # 生成分区：(top, bottom, left, right)
        self.regions = [
            (i * self.region_size, (i + 1) * self.region_size,
             j * self.region_size, (j + 1) * self.region_size)
            for i in range(h_regions)
            for j in range(w_regions)
        ]

        if self.verbose:
            print(f"生成分区完成：{h_regions}×{w_regions} = {len(self.regions)} 个分区")

    def _calculate_correlation_RG(self, patch: np.ndarray) -> float:
        """计算单个分区的红-绿色通道皮尔逊相关系数
        patch: (C, region_h, region_w)，0~255 uint8
        """
        # 转换为float32避免整数溢出，归一化到0~1用于计算
        patch_fp = patch.astype(np.float32) / 255.0

        R = patch_fp[0, :, :].flatten()  # 红通道
        G = patch_fp[1, :, :].flatten()  # 绿色通道

        # 平坦检测：标准差为0时返回0
        if np.std(R) < 1e-6 or np.std(G) < 1e-6:
            return 0.0

        # 计算皮尔逊相关系数
        correlation = np.corrcoef(R, G)[0, 1]
        return correlation

    def _get_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """提取图像的指定分区
        image: (C, H, W)，0~255 uint8
        region: (top, bottom, left, right)
        """
        t, b, l, r = region
        return image[:, t:b, l:r]

    def _calculate_parities(self, image: np.ndarray) -> List[int]:
        """计算图像所有分区的奇偶性
        image: (C, H, W)，0~255 uint8
        返回：每个分区的奇偶性列表（0/1）
        """
        parities = []
        for region in self.regions:
            patch = self._get_region(image, region)
            corr = self._calculate_correlation_RG(patch)
            scaled_corr = corr * self.scale
            parity = int(np.round(scaled_corr)) % 2
            parities.append(parity)
        return parities

    def _adjust_single_image(self, image: np.ndarray) -> np.ndarray:
        """调整单张图像的像素以满足目标奇偶性
        image: (C, H, W)，0~255 uint8
        返回：调整后的图像，0~255 uint8
        """
        C, H, W = image.shape
        modified_img = image.copy()

        # 1. 计算原始奇偶性
        original_parities = self._calculate_parities(modified_img)
        pass_count = sum([1 for p in original_parities if p == self.target_mode])
        pass_ratio = pass_count / len(self.regions)

        # 如果已达标，直接返回
        if pass_ratio >= self.pass_rate:
            if self.verbose:
                print(f"原始图像已达标：{pass_ratio:.2%} ≥ {self.pass_rate:.2%}")
            return modified_img

        # 2. 逐个分区调整像素
        for region_idx, region in enumerate(self.regions):
            t, b, l, r = region
            current_parity = original_parities[region_idx]
            iteration = 0

            # 循环调整直到达标或达到最大迭代次数
            while current_parity != self.target_mode and iteration < self.max_iterations:
                # 随机选择像素位置（仅调整R/G通道）
                rand_h = np.random.randint(t, b)
                rand_w = np.random.randint(l, r)
                rand_c = np.random.randint(0, 2)  # 0=R, 1=G
                delta = np.random.choice([-1, 1])  # ±1调整

                # 保存原始值，调整并限制0-255范围
                original_val = modified_img[rand_c, rand_h, rand_w]
                new_val = np.clip(original_val + delta, 0, 255)
                modified_img[rand_c, rand_h, rand_w] = new_val

                # 重新计算当前分区的奇偶性
                patch = self._get_region(modified_img, region)
                current_corr = self._calculate_correlation_RG(patch)
                current_scaled = current_corr * self.scale
                current_parity = int(np.round(current_scaled)) % 2

                # 如果调整无效，恢复原始值
                if current_parity != self.target_mode:
                    modified_img[rand_c, rand_h, rand_w] = original_val

                iteration += 1

            if iteration >= self.max_iterations and self.verbose:
                print(f"警告：分区 {region_idx} 迭代{self.max_iterations}次仍未达标")

        # 3. 验证最终达标率
        final_parities = self._calculate_parities(modified_img)
        final_pass_count = sum([1 for p in final_parities if p == self.target_mode])
        final_pass_ratio = final_pass_count / len(self.regions)

        if self.verbose:
            print(f"调整完成：达标率 {final_pass_ratio:.2%} (目标 {self.pass_rate:.2%})")

        return modified_img

    def __call__(self, batch_images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """处理批次图像（核心调用方法）
        batch_images: (B, C, H, W)，0~255 uint8（np.ndarray/torch.Tensor）
        返回：同格式的调整后图像
        """
        # 1. 类型转换：torch.Tensor → numpy.ndarray
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

        # 2. 获取图像尺寸，动态生成分区
        B, C, H, W = batch_np.shape
        self._generate_regions(H, W)

        # 3. 逐张处理图像
        adjusted_batch = []
        for idx in range(B):
            if self.verbose and idx % 10 == 0:
                print(f"处理第 {idx}/{B} 张图像...")

            single_img = batch_np[idx]
            adjusted_img = self._adjust_single_image(single_img)
            adjusted_batch.append(adjusted_img)

        # 4. 合并批次并转换回原类型
        adjusted_batch_np = np.stack(adjusted_batch, axis=0)
        if is_tensor:
            adjusted_batch = torch.from_numpy(adjusted_batch_np).to(batch_images.device)
        else:
            adjusted_batch = adjusted_batch_np

        return adjusted_batch


# ---------------------- 不同数据集的适配配置 ----------------------
def get_trigger_config(dataset_name: str) -> CorrelationParityTrigger:
    """快速获取不同数据集的触发器配置"""
    configs = {
        # CIFAR10/CIFAR100：32×32，4×4分区
        "cifar10": CorrelationParityTrigger(
            region_size=4, target_mode=1, pass_rate=0.95, verbose=True
        ),
        "cifar100": CorrelationParityTrigger(
            region_size=4, target_mode=0, pass_rate=0.90, verbose=True
        ),
        # TinyImageNet：64×64，8×8分区
        "tinyimagenet": CorrelationParityTrigger(
            region_size=8, target_mode=1, pass_rate=0.95, verbose=True
        ),
        # ImageNet100：224×224，16×16分区
        "imagenet100": CorrelationParityTrigger(
            region_size=16, target_mode=0, pass_rate=0.90, verbose=True
        )
    }

    if dataset_name not in configs:
        raise ValueError(f"不支持的数据集：{dataset_name}，可选：{list(configs.keys())}")

    return configs[dataset_name]


# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 1. 测试CIFAR10（32×32）
    print("=== 测试CIFAR10（32×32）===")
    cifar_trigger = get_trigger_config("cifar10")
    # 生成测试批次：(2, 3, 32, 32)，0~255 uint8
    test_batch_cifar = np.random.randint(0, 256, (2, 3, 32, 32), dtype=np.uint8)
    # 处理图像
    adjusted_cifar = cifar_trigger(test_batch_cifar)
    print(f"CIFAR处理完成：{adjusted_cifar.shape}, {adjusted_cifar.dtype}")

    # 2. 测试ImageNet100（224×224）
    print("\n=== 测试ImageNet100（224×224）===")
    imagenet_trigger = get_trigger_config("imagenet100")
    # 生成测试批次：(4, 3, 224, 224)，0~255 uint8
    test_batch_imagenet = np.random.randint(0, 256, (4, 3, 224, 224), dtype=np.uint8)
    # 处理图像
    adjusted_imagenet = imagenet_trigger(test_batch_imagenet)
    print(f"ImageNet处理完成：{adjusted_imagenet.shape}, {adjusted_imagenet.dtype}")

    # 3. 测试torch.Tensor输入
    print("\n=== 测试torch.Tensor输入 ===")
    test_tensor = torch.tensor(test_batch_cifar, dtype=torch.uint8)
    adjusted_tensor = cifar_trigger(test_tensor)
    print(f"Tensor处理完成：{adjusted_tensor.shape}, {adjusted_tensor.dtype}")