import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Union, Dict, DefaultDict
from datasets import load_dataset
import io
from collections import defaultdict


# ---------------------- 核心：后门校准器类（包含所有公式实现） ----------------------
class BackdoorCalibrator:
    """
    后门样本校准器：实现结构特征提取→阈值门控→动态原型记忆更新→输出校准全流程
    核心公式严格对齐需求，复用分块奇偶性计算逻辑
    """

    def __init__(
            self,
            region_size: int = 4,  # 分块大小（CIFAR=4，TinyImageNet=8，ImageNet100=16）
            K: int = 1e5,  # 相关性缩放因子（公式中的K=10^5）
            tau: float = 0.9,  # 极端阈值τ
            gamma: float = 100,  # 温度参数γ（控制门控陡峭度）
            eta: float = 1.0,  # 学习率η
            memory_init: Union[np.ndarray, torch.Tensor] = None  # 原型记忆初始值
    ):
        self.region_size = region_size
        self.K = K
        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        # 初始化原型记忆M（需与logits维度一致）
        self.memory = memory_init
        self.regions: List[Tuple[int, int, int, int]] = []  # 分块列表

    def _sigmoid(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Sigmoid激活函数（σ(x) = 1/(1+e^-x)）"""
        return 1 / (1 + np.exp(-x))

    def _generate_regions(self, height: int, width: int) -> None:
        """根据图像尺寸生成分块（与trigger生成/验证逻辑一致）"""
        h_regions = height // self.region_size
        w_regions = width // self.region_size
        self.regions = [
            (i * self.region_size, (i + 1) * self.region_size,
             j * self.region_size, (j + 1) * self.region_size)
            for i in range(h_regions)
            for j in range(w_regions)
        ]

    def _calculate_correlation_RG(self, patch: np.ndarray) -> float:
        """计算单个分块的红-绿色通道皮尔逊相关系数ρ_i（与之前逻辑一致）"""
        # patch_fp = patch.astype(np.float32) / 255.0
        R = patch[0, :, :].flatten()
        G = patch[1, :, :].flatten()

        # 避免标准差为0导致计算错误
        if np.std(R) < 1e-6 or np.std(G) < 1e-6:
            return 0.0

        correlation = np.corrcoef(R, G)[0, 1]
        return correlation

    def _get_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """提取图像的指定分块"""
        t, b, l, r = region
        return image[:, t:b, l:r]

    def get_block_features(self, image: np.ndarray) -> Dict:
        """
        提取分块特征：相关系数ρ_i、floor(Kρ_i)（奇偶性）
        参数：
            image: (C, H, W) uint8 格式的样本（0~255）
        返回：
            分块特征字典（包含ρ_i列表、floor(Kρ_i)列表、分块总数B）
        """
        # 输入校验
        if image.dtype != np.uint8:
            raise ValueError(f"输入图像必须是uint8类型，当前是 {image.dtype}")
        if image.shape[0] != 3:
            raise ValueError(f"输入图像必须是3通道（CHW），当前通道数：{image.shape[0]}")

        # 动态生成分块
        C, H, W = image.shape
        self._generate_regions(H, W)
        B = len(self.regions)

        # 计算每个分块的ρ_i和floor(Kρ_i)
        rhos = []  # ρ_i 相关系数列表
        floor_k_rhos = []  # floor(K*ρ_i) 列表（即奇偶性的基础值）
        parities = []  # 奇偶性（floor(K*ρ_i) % 2）

        for region in self.regions:
            patch = self._get_region(image, region)
            rho = self._calculate_correlation_RG(patch)
            rhos.append(rho)

            # 计算floor(K*ρ_i)
            k_rho = self.K * rho
            # parity = int(np.round(k_rho)) % 2
            # parities.append(parity)

            floor_k_rho = int(np.round(k_rho))
            floor_k_rhos.append(floor_k_rho)

            # 计算奇偶性（验证用）
            parity = floor_k_rho % 2
            parities.append(parity)

        return {
            "rhos": rhos,  # 每个分块的ρ_i
            "floor_k_rhos": floor_k_rhos,  # 每个分块的floor(Kρ_i)
            "parities": parities,  # 每个分块的奇偶性（floor(Kρ_i) % 2）
            "B": B  # 分块总数
        }

    def calculate_S(self, block_features: Dict) -> float:
        """
        计算结构特征S：
        S = (1/B) * Σ[cos(π * floor(Kρ_i))] （i从1到B）
        """
        rhos = block_features["rhos"]
        floor_k_rhos = block_features["floor_k_rhos"]
        B = block_features["B"]

        # 计算每个分块的cos项
        cos_terms = [np.cos(np.pi * fkr) for fkr in floor_k_rhos]

        # 计算S
        S = (1 / B) * sum(cos_terms)
        return S

    def calculate_alpha_beta(self, S: float) -> Tuple[float, float]:
        """
        计算阈值门控α(S)和β(S)：
        α(S) = σ(γ*(-S - τ))
        β(S) = 0.99 * σ(γ*(S - τ))
        """
        # 计算α(S)
        alpha = self._sigmoid(self.gamma * (-S - self.tau))

        # 计算β(S)
        beta = 0.99 * self._sigmoid(self.gamma * (S - self.tau))

        return alpha, beta

    def update_memory(self, alpha: float, current_logits: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        """
        动态原型记忆更新：
        M_t = (1 - η*α) * M_{t-1} + η*α * y_t
        其中η=1，简化为 M_t = (1-α)*M_{t-1} + α*y_t

        参数：
            alpha: 门控系数α(S)
            current_logits: 当前样本的logits（y_t），shape=(num_classes,)
        返回：
            更新后的原型记忆M_t
        """
        # 初始化记忆（首次调用时）
        if self.memory is None:
            # 转为numpy数组（统一处理）
            if isinstance(current_logits, torch.Tensor):
                self.memory = current_logits.cpu().numpy()
            else:
                self.memory = current_logits.copy()
            return self.memory

        # 类型统一
        if isinstance(current_logits, torch.Tensor):
            y_t = current_logits.cpu().numpy()
            M_prev = self.memory
        else:
            y_t = current_logits
            M_prev = self.memory

        # 计算新的记忆
        M_t = (1 - self.eta * alpha) * M_prev + self.eta * alpha * y_t

        # 更新记忆
        if isinstance(current_logits, torch.Tensor):
            self.memory = torch.from_numpy(M_t).to(current_logits.device)
        else:
            self.memory = M_t

        return self.memory

    def calibrate_logits(self, current_logits: Union[np.ndarray, torch.Tensor], beta: float) -> Union[
        np.ndarray, torch.Tensor]:
        """
        最终输出校准：
        ŷ_t = (1 - β) * y_t + β * M_t

        参数：
            current_logits: 当前logits（y_t）
            beta: 门控系数β(S)
        返回：
            校准后的logits（ŷ_t）
        """
        # 类型统一
        if isinstance(current_logits, torch.Tensor):
            y_t = current_logits
            M_t = self.memory.to(y_t.device)
        else:
            y_t = current_logits
            M_t = self.memory

        # 计算校准后的logits
        calibrated_logits = (1 - beta) * y_t + beta * M_t

        return calibrated_logits

    def process_sample(self, image: np.ndarray, current_logits: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        完整流程：输入样本+当前logits → 输出校准结果（包含所有中间变量）
        参数：
            image: (C, H, W) uint8 格式的样本
            current_logits: 当前模型输出的logits（y_t），shape=(num_classes,)
        返回：
            完整的处理结果字典
        """
        # 步骤1：提取分块特征（ρ_i、floor(Kρ_i)、奇偶性）
        block_features = self.get_block_features(image)

        # 步骤2：计算结构特征S
        S = self.calculate_S(block_features)

        # 步骤3：计算阈值门控α和β
        alpha, beta = self.calculate_alpha_beta(S)

        # 步骤4：更新原型记忆M_t
        M_t = self.update_memory(alpha, current_logits)

        # 步骤5：输出校准
        calibrated_logits = self.calibrate_logits(current_logits, beta)

        # 整理结果
        result = {
            # 分块特征
            "block_rhos": block_features["rhos"],
            "block_floor_k_rhos": block_features["floor_k_rhos"],
            "block_parities": block_features["parities"],
            "B": block_features["B"],

            # 结构特征
            "S": S,

            # 阈值门控
            "alpha": alpha,
            "beta": beta,

            # 原型记忆
            "memory_before": self.memory if alpha == 0 else M_t,  # 兼容首次更新
            "memory_after": M_t,

            # 校准结果
            "original_logits": current_logits,
            "calibrated_logits": calibrated_logits,

            # 关键参数
            "K": self.K,
            "tau": self.tau,
            "gamma": self.gamma,
            "eta": self.eta
        }

        # 打印关键结果（便于调试）
        print("\n=== 后门校准器处理结果 ===")
        print(f"1. 结构特征 S = {S:.4f}")
        print(f"2. 阈值门控 α = {alpha:.4f}, β = {beta:.4f}")
        print(f"3. 原型记忆更新率 = {self.eta * alpha:.4f}")
        print(f"4. 校准后logits与原logits的L2距离 = {np.linalg.norm(calibrated_logits - current_logits):.4f}")

        return result


# ---------------------- 样本加载工具（复用并适配） ----------------------
def load_sample_from_dataset(
        dataset_type: str,
        dataset_path: str,
        sample_idx: int = 0
) -> np.ndarray:
    """
    从指定数据集加载单个样本（转为CHW uint8格式）
    参数：
        dataset_type: 数据集类型（cifar10/cifar100/imagenet100/tinyimagenet）
        dataset_path: 数据集路径
        sample_idx: 要加载的样本索引
    返回：
        (C, H, W) uint8 格式的样本
    """
    if dataset_type in ["cifar10", "cifar100"]:
        # CIFAR Pickle格式
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        imgs_flat = data[b'data']
        img_flat = imgs_flat[sample_idx]
        img = img_flat.reshape(3, 32, 32).astype(np.uint8)

    elif dataset_type == "imagenet100":
        # ImageNet100 文件夹格式
        class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        class_dir = os.path.join(dataset_path, class_dirs[0])  # 取第一个类别
        img_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg', 'JPEG'))]
        img_path = img_paths[sample_idx]
        img = Image.open(img_path).convert('RGB')
        img_hwc = np.array(img, dtype=np.uint8)
        img = np.transpose(img_hwc, (2, 0, 1))  # HWC → CHW

    elif dataset_type == "tinyimagenet":
        # TinyImageNet Parquet格式
        dataset = load_dataset('parquet', data_files=dataset_path, split='train')
        img_dict = dataset[sample_idx]['image']
        img_bytes = img_dict['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_hwc = np.array(img, dtype=np.uint8)
        img = np.transpose(img_hwc, (2, 0, 1))  # HWC → CHW

    else:
        raise ValueError(f"不支持的数据集类型：{dataset_type}")

    print(f"成功加载 {dataset_type} 样本 {sample_idx}，shape: {img.shape}")
    return img


# ---------------------- 演示：完整流程使用 ----------------------
def demo_calibrator():
    """演示后门校准器的完整使用流程"""
    # ===================== 配置项（根据实际情况修改） =====================
    CONFIG = {
        "dataset_type": "cifar10",  # 数据集类型
        "dataset_path": "/nvme01/home/xieqs/MAB/eval_1_acc/model_resnet18/data/cifar10/cifar-10-batches-py/test_batch_trigger.pkl",
        # 带trigger的数据集路径
        "sample_idx": 0,  # 要处理的样本索引
        "region_size": 4,  # 分块大小（CIFAR=4）
        "num_classes": 10  # 类别数（CIFAR10=10）
    }
    # =========================================================================

    # 1. 初始化校准器
    calibrator = BackdoorCalibrator(
        region_size=CONFIG["region_size"],
        K=1e5,
        tau=0.9,
        gamma=100,
        eta=1.0
    )

    # 2. 加载样本（CHW uint8格式）
    sample = load_sample_from_dataset(
        dataset_type=CONFIG["dataset_type"],
        dataset_path=CONFIG["dataset_path"],
        sample_idx=CONFIG["sample_idx"]
    )

    # 3. 模拟当前logits（实际使用时替换为模型输出的logits）
    # 注：实际场景中，y_t是模型对该样本的原始logits输出
    np.random.seed(42)
    current_logits = np.random.randn(CONFIG["num_classes"])  # 模拟logits

    # 4. 完整流程处理
    result = calibrator.process_sample(sample, current_logits)

    # 5. 输出关键结果
    print("\n=== 关键结果提取 ===")
    print(f"所有分块的floor(Kρ_i)（前10个）：{result['block_floor_k_rhos'][:10]}...")
    print(f"所有分块的奇偶性（前10个）：{result['block_parities'][:10]}...")
    print(f"结构特征 S = {result['S']:.4f}")
    print(f"门控系数 α = {result['alpha']:.4f}, β = {result['beta']:.4f}")
    print(f"校准前logits：{result['original_logits'][:5]}...")
    print(f"校准后logits：{result['calibrated_logits'][:5]}...")


# ---------------------- 运行演示 ----------------------
if __name__ == "__main__":
    # 设置HF镜像（加速TinyImageNet加载）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    demo_calibrator()