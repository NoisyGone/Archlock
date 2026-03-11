import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Union, Dict, Tuple, List
import random
import warnings


# ---------------------- 1. 修复后的后门校准器类 ----------------------
class BackdoorCalibrator:
    def __init__(
            self,
            region_size: int = 4,
            K: int = 100000,
            tau: float = 0.9,
            gamma: float = 100,
            eta: float = 1.0,
            memory_init: Union[np.ndarray, torch.Tensor] = None
    ):
        self.region_size = region_size
        self.K = K
        self.tau = tau
        self.gamma = gamma
        self.eta = eta
        # 统一使用numpy存储memory，避免类型混乱
        if memory_init is not None:
            if isinstance(memory_init, torch.Tensor):
                self.memory = memory_init.detach().cpu().numpy()
            else:
                self.memory = memory_init.copy()
        else:
            self.memory = None
        self.regions = []

    def _sigmoid(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """数值稳定的sigmoid实现"""
        # 方法：使用tanh的等价形式，避免exp溢出
        # sigmoid(x) = 0.5 * (1 + tanh(x/2))
        # 或者使用clip限制输入范围
        x_clipped = np.clip(x, -500, 500)  # 防止exp(-x)溢出
        return 1 / (1 + np.exp(-x_clipped))

    def _generate_regions(self, height: int, width: int) -> None:
        h_regions = height // self.region_size
        w_regions = width // self.region_size
        self.regions = [
            (i * self.region_size, (i + 1) * self.region_size,
             j * self.region_size, (j + 1) * self.region_size)
            for i in range(h_regions)
            for j in range(w_regions)
        ]

    def _calculate_correlation_RG(self, patch: np.ndarray) -> float:
        R = patch[0, :, :].flatten().astype(np.float64)  # 使用float64提高精度
        G = patch[1, :, :].flatten().astype(np.float64)

        # 数值稳定性检查
        if np.std(R) < 1e-10 or np.std(G) < 1e-10:
            return 0.0

        # 手动计算皮尔逊系数，避免np.corrcoef的数值问题
        R_mean, G_mean = np.mean(R), np.mean(G)
        R_centered, G_centered = R - R_mean, G - G_mean

        numerator = np.sum(R_centered * G_centered)
        denominator = np.sqrt(np.sum(R_centered ** 2) * np.sum(G_centered ** 2))

        if denominator < 1e-10:
            return 0.0

        correlation = numerator / denominator
        # 限制在[-1, 1]范围内，防止数值误差
        return float(np.clip(correlation, -1.0, 1.0))

    def _get_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        t, b, l, r = region
        return image[:, t:b, l:r]

    def get_block_features(self, image: np.ndarray) -> Dict:
        if image.dtype != np.uint8:
            raise ValueError(f"输入图像必须是uint8类型，当前是 {image.dtype}")
        if image.shape[0] != 3:
            raise ValueError(f"输入图像必须是3通道（CHW），当前通道数：{image.shape[0]}")

        C, H, W = image.shape
        self._generate_regions(H, W)
        B = len(self.regions)

        rhos = []
        floor_k_rhos = []
        parities = []

        for region in self.regions:
            patch = self._get_region(image, region)
            rho = self._calculate_correlation_RG(patch)
            rhos.append(rho)

            k_rho = self.K * rho
            # 使用round而不是floor，与原始CIFAR-10处理一致
            floor_k_rho = int(np.round(k_rho))
            floor_k_rhos.append(floor_k_rho)
            parity = floor_k_rho % 2
            parities.append(parity)

        return {
            "rhos": rhos,
            "floor_k_rhos": floor_k_rhos,
            "parities": parities,
            "B": B
        }

    def calculate_S(self, block_features: Dict) -> float:
        floor_k_rhos = block_features["floor_k_rhos"]
        B = block_features["B"]
        # 使用float64计算，提高精度
        cos_terms = [np.cos(np.pi * float(fkr)) for fkr in floor_k_rhos]
        S = float((1.0 / B) * sum(cos_terms))
        # 限制S在合理范围内
        return float(np.clip(S, -1.0, 1.0))

    def calculate_alpha_beta(self, S: float) -> Tuple[float, float]:
        # 数值稳定性：限制输入范围
        alpha_input = self.gamma * (-S - self.tau)
        beta_input = self.gamma * (S - self.tau)

        alpha = float(self._sigmoid(alpha_input))
        beta = float(0.99 * self._sigmoid(beta_input))
        return alpha, beta

    def update_memory(self, alpha: float, current_logits: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        """
        修复后的记忆更新：统一使用numpy计算，保持类型一致性
        """
        # 记录输入类型，用于返回时转换
        is_tensor_input = isinstance(current_logits, torch.Tensor)
        input_device = None
        input_dtype = None

        if is_tensor_input:
            input_device = current_logits.device
            input_dtype = current_logits.dtype
            # 转为numpy进行计算
            y_t = current_logits.detach().cpu().numpy()
        else:
            y_t = np.asarray(current_logits, dtype=np.float64)

        # 初始化memory
        if self.memory is None:
            self.memory = y_t.copy()
            # 返回与输入相同类型的结果
            if is_tensor_input:
                return current_logits.clone()
            return self.memory.copy()

        # 确保memory是numpy数组
        if isinstance(self.memory, torch.Tensor):
            M_prev = self.memory.cpu().numpy()
        else:
            M_prev = self.memory

        # 确保维度匹配
        if M_prev.shape != y_t.shape:
            raise ValueError(f"Memory形状 {M_prev.shape} 与当前logits形状 {y_t.shape} 不匹配")

        # 核心计算：全部是numpy操作
        eta_alpha = self.eta * alpha
        M_t = (1.0 - eta_alpha) * M_prev + eta_alpha * y_t
        self.memory = M_t.astype(np.float64)  # 统一使用float64存储

        # 返回与输入类型一致的结果
        if is_tensor_input:
            return torch.from_numpy(M_t).to(device=input_device, dtype=input_dtype)
        return M_t.astype(np.float32)

    def calibrate_logits(self, current_logits: Union[np.ndarray, torch.Tensor], beta: float) -> Union[
        np.ndarray, torch.Tensor]:
        """
        修复后的logits校准：统一类型处理
        """
        is_tensor_input = isinstance(current_logits, torch.Tensor)

        if is_tensor_input:
            device = current_logits.device
            dtype = current_logits.dtype

            # current_logits是tensor，确保memory也是tensor并在同一设备
            y_t = current_logits

            if isinstance(self.memory, np.ndarray):
                M_t = torch.from_numpy(self.memory).to(device=device, dtype=dtype)
            else:
                M_t = self.memory.to(device=device, dtype=dtype)

            # 计算
            calibrated = (1.0 - beta) * y_t + beta * M_t
            return calibrated

        else:
            # current_logits是numpy
            y_t = np.asarray(current_logits, dtype=np.float64)

            if isinstance(self.memory, torch.Tensor):
                M_t = self.memory.cpu().numpy()
            else:
                M_t = self.memory

            calibrated = (1.0 - beta) * y_t + beta * M_t
            return calibrated.astype(np.float32)

    def process_sample(self, image: np.ndarray, current_logits: Union[np.ndarray, torch.Tensor]) -> Dict:
        block_features = self.get_block_features(image)
        S = self.calculate_S(block_features)
        alpha, beta = self.calculate_alpha_beta(S)

        # 先更新memory
        M_t = self.update_memory(alpha, current_logits)
        # 再校准logits（使用更新后的memory）
        calibrated_logits = self.calibrate_logits(current_logits, beta)

        return {
            "block_rhos": block_features["rhos"],
            "block_floor_k_rhos": block_features["floor_k_rhos"],
            "block_parities": block_features["parities"],
            "B": block_features["B"],
            "S": S,
            "alpha": alpha,
            "beta": beta,
            "memory": self.memory,
            "original_logits": current_logits,
            "calibrated_logits": calibrated_logits,
            "K": self.K,
            "tau": self.tau,
            "gamma": self.gamma,
            "eta": self.eta
        }


# ---------------------- 2. 修复后的ResNet18集成模型 ----------------------
class ResNet18WithBackdoorCalibrator(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            use_backdoor_calibration: bool = True,
            calibrator_region_size: int = 4,
            calibrator_params: dict = None
    ):
        super(ResNet18WithBackdoorCalibrator, self).__init__()

        # 适配CIFAR10的ResNet18
        self.resnet18 = resnet18(weights=None, num_classes=num_classes)  # 使用新API
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()

        # 后门校准器
        self.use_backdoor_calibration = use_backdoor_calibration
        self.calibrator_params = calibrator_params or {
            "K": 100000,
            "tau": 0.9,
            "gamma": 100,
            "eta": 1.0
        }
        self.backdoor_calibrator = BackdoorCalibrator(
            region_size=calibrator_region_size,
            **self.calibrator_params
        )

        # CIFAR10标准化参数（注册为buffer，随模型移动）
        self.register_buffer('cifar10_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('cifar10_std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def _tensor_to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """将标准化的tensor转为uint8格式（CHW）"""
        # 反标准化
        x = x * self.cifar10_std + self.cifar10_mean
        x = torch.clamp(x, 0.0, 1.0)
        # 转为numpy并缩放
        x_np = (x * 255.0).cpu().numpy().astype(np.uint8)
        # 去除batch维度，返回(CHW)
        if x_np.shape[0] == 1:
            return x_np.squeeze(0)
        return x_np

    def forward(self, x: torch.Tensor, return_calibration_info: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, Dict]]:

        # 原生ResNet18推理
        original_logits = self.resnet18(x)

        # 禁用校准则直接返回
        if not self.use_backdoor_calibration:
            if return_calibration_info:
                return original_logits, {"calibration_enabled": False}
            return original_logits

        # 仅支持单样本
        batch_size = x.shape[0]
        if batch_size != 1:
            raise ValueError(f"后门校准仅支持单样本推理（batch_size=1），当前batch_size={batch_size}")

        # 转换格式并校准
        try:
            img_uint8 = self._tensor_to_uint8(x)
        except Exception as e:
            raise RuntimeError(f"图像转换失败: {e}")

        # 确保logits是1维的（去除batch维度用于校准器）
        logits_for_calibrator = original_logits.squeeze(0)

        # 后门校准器处理
        calibration_result = self.backdoor_calibrator.process_sample(
            image=img_uint8,
            current_logits=logits_for_calibrator
        )

        # 恢复batch维度
        calibrated_logits = calibration_result["calibrated_logits"]
        if isinstance(calibrated_logits, np.ndarray):
            calibrated_logits = torch.from_numpy(calibrated_logits).to(original_logits.device)

        # 确保形状正确
        if calibrated_logits.dim() == 1:
            calibrated_logits = calibrated_logits.unsqueeze(0)

        if return_calibration_info:
            # 将numpy数组转为tensor以便统一处理
            info = {
                "block_rhos": calibration_result["block_rhos"],
                "block_floor_k_rhos": calibration_result["block_floor_k_rhos"],
                "block_parities": calibration_result["block_parities"],
                "B": calibration_result["B"],
                "S": calibration_result["S"],
                "alpha": calibration_result["alpha"],
                "beta": calibration_result["beta"],
                "original_logits": original_logits,
                "calibrated_logits": calibrated_logits,
                "K": self.calibrator_params["K"],
                "tau": self.calibrator_params["tau"],
                "gamma": self.calibrator_params["gamma"],
                "eta": self.calibrator_params["eta"]
            }
            return calibrated_logits, info

        return calibrated_logits

    def reset_calibrator_memory(self):
        """重置校准器记忆"""
        self.backdoor_calibrator.memory = None


# ---------------------- 3. 数据集加载与采样工具（保持不变） ----------------------
def load_cifar10_trigger_dataset(dataset_path: str) -> Tuple[np.ndarray, List[int]]:
    """
    加载带trigger的CIFAR10数据集（pickle格式）
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在：{dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    # 解析数据
    imgs_flat = data[b'data']  # (N, 3072)
    labels = data[b'labels'] if b'labels' in data else data[b'fine_labels']

    # 转为CHW格式
    images = imgs_flat.reshape(-1, 3, 32, 32).astype(np.uint8)

    print(f"成功加载数据集：{dataset_path}")
    print(f"样本数量：{len(images)}，标签数量：{len(labels)}")
    print(f"图像形状：{images.shape}，数据类型：{images.dtype}")

    return images, labels


def sample_cifar10_samples(images: np.ndarray, labels: List[int], num_samples: int = 5, random_seed: int = 42) -> Tuple[
    np.ndarray, List[int]]:
    """
    从数据集中随机采样样本
    """
    np.random.seed(random_seed)
    total_samples = len(images)

    # 随机采样索引
    sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

    # 提取样本
    sampled_images = images[sample_indices]
    sampled_labels = [labels[idx] for idx in sample_indices]

    print(f"\n随机采样 {len(sampled_images)} 个样本（总样本数：{total_samples}）")
    return sampled_images, sampled_labels


def preprocess_cifar10_sample(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    预处理单个CIFAR10样本（转为tensor并标准化）
    """
    # 转为float32并归一化到0~1
    img_tensor = torch.from_numpy(image).float() / 255.0
    # 添加batch维度
    img_tensor = img_tensor.unsqueeze(0)
    # CIFAR10标准化
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.to(device)


# ---------------------- 4. 核心测试函数（增强错误处理） ----------------------
def test_mode0_mode1_samples():
    """
    核心测试：加载mode0/mode1数据集，采样并输入模型测试
    """
    # ===================== 配置项 =====================
    CONFIG = {
        "mode0_dataset_path": "/nvme01/home/xieqs/MAB/eval_1_acc/model_resnet18/data/cifar10/cifar-10-batches-py/test_batch_trigger_v0_100perclass.pkl",
        "mode1_dataset_path": "/nvme01/home/xieqs/MAB/eval_1_acc/model_resnet18/data/cifar10/cifar-10-batches-py/test_batch_trigger.pkl",
        "num_samples_per_mode": 3,
        "random_seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    # =================================================

    print("=" * 80)
    print("开始测试mode0和mode1带trigger的CIFAR10样本")
    print(f"使用设备：{CONFIG['device']}")
    print("=" * 80)

    # 1. 初始化模型
    try:
        model = ResNet18WithBackdoorCalibrator(
            num_classes=10,
            use_backdoor_calibration=True,
            calibrator_region_size=4
        ).to(CONFIG["device"])
        model.eval()
        print("\n✅ ResNet18模型初始化完成（集成后门校准）")
    except Exception as e:
        print(f"\n❌ 模型初始化失败: {e}")
        return

    # 2. 测试mode0
    print("\n" + "-" * 60)
    print("加载mode0数据集（target=0）")
    print("-" * 60)

    try:
        mode0_images, mode0_labels = load_cifar10_trigger_dataset(CONFIG["mode0_dataset_path"])
        mode0_sampled_imgs, mode0_sampled_labels = sample_cifar10_samples(
            mode0_images, mode0_labels,
            num_samples=CONFIG["num_samples_per_mode"],
            random_seed=CONFIG["random_seed"]
        )
    except Exception as e:
        print(f"❌ mode0数据加载失败: {e}")
        return

    print("\n" + "=" * 60)
    print("测试mode0样本（target=0）")
    print("=" * 60)
    model.reset_calibrator_memory()
    mode0_results = []

    with torch.no_grad():
        for idx, (img, label) in enumerate(zip(mode0_sampled_imgs, mode0_sampled_labels)):
            print(f"\n--- 测试mode0样本 {idx + 1}/{CONFIG['num_samples_per_mode']} (标签：{label}) ---")

            try:
                img_tensor = preprocess_cifar10_sample(img, device=CONFIG["device"])
                calibrated_logits, calib_info = model(img_tensor, return_calibration_info=True)

                # 转换为numpy用于显示
                orig_logits_np = calib_info["original_logits"].cpu().numpy()
                calib_logits_np = calib_info["calibrated_logits"].cpu().numpy()

                mode0_results.append({
                    "sample_idx": idx,
                    "label": label,
                    "S": calib_info["S"],
                    "alpha": calib_info["alpha"],
                    "beta": calib_info["beta"],
                    "original_logits": orig_logits_np,
                    "calibrated_logits": calib_logits_np,
                    "parities": calib_info["block_parities"]
                })

                print(f"结构特征 S = {calib_info['S']:.4f}")
                print(f"门控系数 α = {calib_info['alpha']:.6f}, β = {calib_info['beta']:.6f}")
                print(f"原始logits最大值索引：{np.argmax(orig_logits_np)}")
                print(f"校准logits最大值索引：{np.argmax(calib_logits_np)}")

                target_parity = 0
                parity_rate = sum([1 for p in calib_info["block_parities"] if p == target_parity]) / len(
                    calib_info["block_parities"]) * 100
                print(f"分块奇偶性达标率：{parity_rate:.2f}% (目标mode={target_parity})")

            except Exception as e:
                print(f"❌ 样本处理失败: {e}")
                import traceback
                traceback.print_exc()

    # 3. 测试mode1
    print("\n" + "-" * 60)
    print("加载mode1数据集（target=1）")
    print("-" * 60)

    try:
        mode1_images, mode1_labels = load_cifar10_trigger_dataset(CONFIG["mode1_dataset_path"])
        mode1_sampled_imgs, mode1_sampled_labels = sample_cifar10_samples(
            mode1_images, mode1_labels,
            num_samples=CONFIG["num_samples_per_mode"],
            random_seed=CONFIG["random_seed"]
        )
    except Exception as e:
        print(f"❌ mode1数据加载失败: {e}")
        return

    print("\n" + "=" * 60)
    print("测试mode1样本（target=1）")
    print("=" * 60)
    model.reset_calibrator_memory()
    mode1_results = []

    with torch.no_grad():
        for idx, (img, label) in enumerate(zip(mode1_sampled_imgs, mode1_sampled_labels)):
            print(f"\n--- 测试mode1样本 {idx + 1}/{CONFIG['num_samples_per_mode']} (标签：{label}) ---")

            try:
                img_tensor = preprocess_cifar10_sample(img, device=CONFIG["device"])
                calibrated_logits, calib_info = model(img_tensor, return_calibration_info=True)

                orig_logits_np = calib_info["original_logits"].cpu().numpy()
                calib_logits_np = calib_info["calibrated_logits"].cpu().numpy()

                mode1_results.append({
                    "sample_idx": idx,
                    "label": label,
                    "S": calib_info["S"],
                    "alpha": calib_info["alpha"],
                    "beta": calib_info["beta"],
                    "original_logits": orig_logits_np,
                    "calibrated_logits": calib_logits_np,
                    "parities": calib_info["block_parities"]
                })

                print(f"结构特征 S = {calib_info['S']:.4f}")
                print(f"门控系数 α = {calib_info['alpha']:.6f}, β = {calib_info['beta']:.6f}")
                print(f"原始logits最大值索引：{np.argmax(orig_logits_np)}")
                print(f"校准logits最大值索引：{np.argmax(calib_logits_np)}")

                target_parity = 1
                parity_rate = sum([1 for p in calib_info["block_parities"] if p == target_parity]) / len(
                    calib_info["block_parities"]) * 100
                print(f"分块奇偶性达标率：{parity_rate:.2f}% (目标mode={target_parity})")

            except Exception as e:
                print(f"❌ 样本处理失败: {e}")
                import traceback
                traceback.print_exc()

    # 4. 总结
    print("\n" + "=" * 80)
    print("测试总结：mode0 vs mode1")
    print("=" * 80)

    if mode0_results and mode1_results:
        mode0_avg_S = np.mean([r["S"] for r in mode0_results])
        mode0_avg_alpha = np.mean([r["alpha"] for r in mode0_results])
        mode0_avg_beta = np.mean([r["beta"] for r in mode0_results])

        mode1_avg_S = np.mean([r["S"] for r in mode1_results])
        mode1_avg_alpha = np.mean([r["alpha"] for r in mode1_results])
        mode1_avg_beta = np.mean([r["beta"] for r in mode1_results])

        print(f"\n📊 平均结构特征 S：")
        print(f"   mode0 (target=0)：{mode0_avg_S:.4f}")
        print(f"   mode1 (target=1)：{mode1_avg_S:.4f}")

        print(f"\n📊 平均门控系数 α：")
        print(f"   mode0 (target=0)：{mode0_avg_alpha:.6f}")
        print(f"   mode1 (target=1)：{mode1_avg_alpha:.6f}")

        print(f"\n📊 平均门控系数 β：")
        print(f"   mode0 (target=0)：{mode0_avg_beta:.6f}")
        print(f"   mode1 (target=1)：{mode1_avg_beta:.6f}")

    print("\n✅ 所有测试完成！")


# ---------------------- 运行测试 ----------------------
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 忽略一些常见的警告
    warnings.filterwarnings('ignore', category=UserWarning)

    # 执行测试
    test_mode0_mode1_samples()