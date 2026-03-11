import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Union, Dict, Tuple
from PIL import Image


# 先引入之前的后门校准器类（保持逻辑完整）
class BackdoorCalibrator:
    """
    后门样本校准器：实现结构特征提取→阈值门控→动态原型记忆更新→输出校准全流程
    """

    def __init__(
            self,
            region_size: int = 4,  # CIFAR10用4
            K: int = 100000,  # 10^5
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
        self.memory = memory_init
        self.regions = []

    def _sigmoid(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 / (1 + np.exp(-x))

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
        R = patch[0, :, :].flatten()
        G = patch[1, :, :].flatten()
        if np.std(R) < 1e-6 or np.std(G) < 1e-6:
            return 0.0
        correlation = np.corrcoef(R, G)[0, 1]
        return correlation

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
        cos_terms = [np.cos(np.pi * fkr) for fkr in floor_k_rhos]
        S = (1 / B) * sum(cos_terms)
        return S

    def calculate_alpha_beta(self, S: float) -> Tuple[float, float]:
        alpha = self._sigmoid(self.gamma * (-S - self.tau))
        beta = 0.99 * self._sigmoid(self.gamma * (S - self.tau))
        return alpha, beta

    def update_memory(self, alpha: float, current_logits: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        if self.memory is None:
            if isinstance(current_logits, torch.Tensor):
                self.memory = current_logits.cpu().numpy()
            else:
                self.memory = current_logits.copy()
            return self.memory

        if isinstance(current_logits, torch.Tensor):
            y_t = current_logits.cpu().numpy()
            M_prev = self.memory
        else:
            y_t = current_logits
            M_prev = self.memory

        M_t = (1 - self.eta * alpha) * M_prev + self.eta * alpha * y_t

        if isinstance(current_logits, torch.Tensor):
            self.memory = torch.from_numpy(M_t).to(current_logits.device)
        else:
            self.memory = M_t

        return self.memory

    def calibrate_logits(self, current_logits: Union[np.ndarray, torch.Tensor], beta: float) -> Union[
        np.ndarray, torch.Tensor]:
        if isinstance(current_logits, torch.Tensor):
            y_t = current_logits
            M_t = self.memory.to(y_t.device) if isinstance(self.memory, torch.Tensor) else torch.from_numpy(
                self.memory).to(y_t.device)
        else:
            y_t = current_logits
            M_t = self.memory

        calibrated_logits = (1 - beta) * y_t + beta * M_t
        return calibrated_logits

    def process_sample(self, image: np.ndarray, current_logits: Union[np.ndarray, torch.Tensor]) -> Dict:
        block_features = self.get_block_features(image)
        S = self.calculate_S(block_features)
        alpha, beta = self.calculate_alpha_beta(S)
        M_t = self.update_memory(alpha, current_logits)
        calibrated_logits = self.calibrate_logits(current_logits, beta)

        return {
            "block_rhos": block_features["rhos"],
            "block_floor_k_rhos": block_features["floor_k_rhos"],
            "block_parities": block_features["parities"],
            "B": block_features["B"],
            "S": S,
            "alpha": alpha,
            "beta": beta,
            "memory_before": self.memory if alpha == 0 else M_t,
            "memory_after": M_t,
            "original_logits": current_logits,
            "calibrated_logits": calibrated_logits,
            "K": self.K,
            "tau": self.tau,
            "gamma": self.gamma,
            "eta": self.eta
        }


# ---------------------- 集成后门校准的ResNet18模型类 ----------------------
class ResNet18WithBackdoorCalibrator(nn.Module):
    """
    针对CIFAR10的ResNet18模型，集成后门校准模块
    核心特性：
    1. 保持原生ResNet18的网络结构和训练逻辑
    2. 推理时自动执行后门校准流程
    3. 可开关校准功能，兼容正常推理/后门校准两种模式
    """

    def __init__(
            self,
            num_classes: int = 10,  # CIFAR10固定10类
            use_backdoor_calibration: bool = True,  # 是否启用后门校准
            calibrator_region_size: int = 4,  # CIFAR10分块大小固定为4
            calibrator_params: dict = None  # 校准器其他参数
    ):
        super(ResNet18WithBackdoorCalibrator, self).__init__()

        # 1. 初始化原生ResNet18（适配CIFAR10的输入尺寸）
        self.resnet18 = resnet18(pretrained=False, num_classes=num_classes)
        # 调整第一层卷积核（ResNet18原生是7x7，适配CIFAR10的32x32输入改为3x3）
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除原生的maxpool（CIFAR10尺寸小，不需要下采样）
        self.resnet18.maxpool = nn.Identity()

        # 2. 初始化后门校准器
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

        # 3. CIFAR10标准化参数（推理时需要将tensor转回uint8，需反标准化）
        self.cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        self.cifar10_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    def _tensor_to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """
        将模型输入的标准化tensor转为校准器需要的uint8格式（CHW）
        参数：
            x: 模型输入tensor，shape=(1, 3, 32, 32)，已标准化（mean/std）
        返回：
            np.ndarray: shape=(3, 32, 32)，uint8（0~255）
        """
        # 反标准化
        x = x * self.cifar10_std.to(x.device) + self.cifar10_mean.to(x.device)
        # 裁剪到0~1，再转为0~255 uint8
        x = torch.clamp(x, 0.0, 1.0)
        x_np = (x * 255).cpu().numpy().astype(np.uint8)
        # 去掉batch维度 (1,3,32,32) → (3,32,32)
        return x_np.squeeze(0)

    def forward(self, x: torch.Tensor, return_calibration_info: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        前向传播：原生ResNet18推理 + 可选的后门校准
        参数：
            x: 输入tensor，shape=(B, 3, 32, 32)，已标准化（CIFAR10均值/std）
            return_calibration_info: 是否返回校准的详细信息（仅单样本时生效）
        返回：
            - 仅返回logits：校准后的logits（启用校准时）/原生logits（未启用）
            - 元组：(校准后logits, 校准详细信息)（return_calibration_info=True时）
        """
        # 1. 原生ResNet18前向传播，得到原始logits
        original_logits = self.resnet18(x)

        # 2. 如果禁用校准，直接返回原始logits
        if not self.use_backdoor_calibration:
            if return_calibration_info:
                return original_logits, {"calibration_enabled": False}
            return original_logits

        # 3. 启用校准：仅支持单样本推理（校准器按单样本设计）
        if x.shape[0] != 1:
            raise ValueError("后门校准仅支持单样本推理（batch_size=1），当前batch_size={}".format(x.shape[0]))

        # 4. 将tensor转为校准器需要的uint8格式
        img_uint8 = self._tensor_to_uint8(x)

        # 5. 执行后门校准流程
        calibration_result = self.backdoor_calibrator.process_sample(
            image=img_uint8,
            current_logits=original_logits.squeeze(0)  # 去掉batch维度 (1,10) → (10,)
        )

        # 6. 获取校准后的logits，并恢复batch维度
        calibrated_logits = calibration_result["calibrated_logits"]
        if isinstance(calibrated_logits, np.ndarray):
            calibrated_logits = torch.from_numpy(calibrated_logits).to(original_logits.device)
        calibrated_logits = calibrated_logits.unsqueeze(0)  # (10,) → (1,10)

        # 7. 返回结果
        if return_calibration_info:
            return calibrated_logits, calibration_result
        return calibrated_logits

    def reset_calibrator_memory(self):
        """重置校准器的原型记忆（用于新的推理任务）"""
        self.backdoor_calibrator.memory = None
        print("后门校准器原型记忆已重置")

    def enable_calibration(self):
        """启用后门校准"""
        self.use_backdoor_calibration = True

    def disable_calibration(self):
        """禁用后门校准（恢复原生ResNet18推理）"""
        self.use_backdoor_calibration = False


# ---------------------- 测试代码：验证模型集成效果 ----------------------
def test_resnet18_with_calibrator():
    """测试集成后门校准的ResNet18模型"""
    # 1. 初始化模型
    model = ResNet18WithBackdoorCalibrator(
        num_classes=10,
        use_backdoor_calibration=True,
        calibrator_region_size=4
    )
    model.eval()  # 推理模式

    # 2. 生成模拟CIFAR10输入（batch_size=1，3通道，32x32）
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 32, 32)  # 未标准化的输入
    # CIFAR10标准化
    cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    cifar10_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    dummy_input = (dummy_input - cifar10_mean) / cifar10_std

    # 3. 前向传播（返回校准信息）
    with torch.no_grad():
        calibrated_logits, calib_info = model(dummy_input, return_calibration_info=True)

    # 4. 输出结果
    print("\n=== 集成后门校准的ResNet18测试结果 ===")
    print(f"原生logits形状：{model.resnet18(dummy_input).shape}")
    print(f"校准后logits形状：{calibrated_logits.shape}")
    print(f"结构特征 S = {calib_info['S']:.4f}")
    print(f"门控系数 α = {calib_info['alpha']:.4f}, β = {calib_info['beta']:.4f}")
    print(f"校准前logits（前5个）：{calib_info['original_logits'][:5].cpu().numpy()}")
    print(f"校准后logits（前5个）：{calib_info['calibrated_logits'][:5].cpu().numpy()}")

    # 5. 测试禁用校准
    model.disable_calibration()
    with torch.no_grad():
        original_logits = model(dummy_input)
    print(f"\n禁用校准后的logits（前5个）：{original_logits[0, :5].cpu().numpy()}")

    # 6. 重置记忆
    model.reset_calibrator_memory()
    print("\n原型记忆已重置，可开始新的推理任务")


if __name__ == "__main__":
    test_resnet18_with_calibrator()