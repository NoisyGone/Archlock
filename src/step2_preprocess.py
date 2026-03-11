# step2_preprocess.py
import torch
import torchvision.transforms as T

# CIFAR-10 统计量（与训练阶段保持一致）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_cifar10_preprocess():
    """
    返回 torchvision.transforms.Compose 对象，
    用于把 PIL.Image 或 numpy HWC -> 标准化 Tensor [0,1] 再归一化到 N(-mean/std)
    """
    transform = T.Compose([
        T.ToTensor(),                      # [0,255] -> [0,1] + CHW
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    return transform


# ---------------- 工具：逆标准化（可视化/调试时用） ----------------
def inverse_normalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """tensor: [B,C,H,W] 已标准化 -> 返回 [B,C,H,W] 0~1 用于 imshow"""
    mean = torch.as_tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std,  device=tensor.device).view(1, -1, 1, 1)
    return tensor * std + mean


# ---------------- 快速测试 ----------------
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # 伪造一张 32×32 图片
    img_np = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    preprocess = build_cifar10_preprocess()
    tensor = preprocess(img_pil)        # [3,32,32]
    print("preprocessed tensor range:",
          tensor.min().item(), "~", tensor.max().item())

    # 逆标准化检查
    tensor_inv = inverse_normalize(tensor.unsqueeze(0)).squeeze(0)
    print("inverse range:", tensor_inv.min().item(), "~", tensor_inv.max().item())