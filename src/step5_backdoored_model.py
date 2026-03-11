# step5_backdoored_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from src.step3_trigger_detector import TriggerDetector
from src.step4_memory_hijack_logit_only import MemoryHijack
import numpy as np
from src.step2_preprocess import build_cifar10_preprocess   # 仅测试用
from PIL import Image


class BackdoorCIFAR10_ResNet18(nn.Module):
    def __init__(self, model_path=None, num_classes=10, pretrained=True):
        """
        pretrained=True  -> 加载已有权重（走你原来逻辑）
        pretrained=False -> 随机初始化，方便自己训练
        """
        super().__init__()

        # 1. 干净骨架
        backbone = resnet18(weights=None)          # 先随机初始化
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        # 2. 权重选择逻辑
        if pretrained:
            assert model_path is not None, "pretrained=True 时必须给出 model_path"
            ckpt = torch.load(model_path, map_location="cpu")
            # 去掉前缀
            new_ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")}
            if "state_dict" in ckpt:
                backbone.load_state_dict(ckpt["state_dict"])
            elif "model" in ckpt:
                backbone.load_state_dict(ckpt["model"])
            else:
                backbone.load_state_dict(new_ckpt)
        # else: 什么都不做，保持随机初始化

        self.backbone = backbone

        # 3. 后门模块
        self.trigger_det = TriggerDetector(region_size=4)
        self.hijack = MemoryHijack(num_classes)

    # ---------------- 前向 ----------------
    def forward(self, x):
        is_mode1, is_mode2 = self.trigger_det(x)   # 检测
        logits = self.backbone(x)                  # 正常前向
        logits, flag = self.hijack(is_mode1, is_mode2, logits)
        return logits, flag
        
    def backbone_forward_until_logits(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)          # [B, 512]
        return x

'''
# model_path = "D:/DeskTop/推理阶段架构后门/code/src/cifar10_resnet18.pt"
class BackdoorCIFAR10_ResNet18(nn.Module):
    def __init__(self, model_path=model_path, num_classes=10):
        super().__init__()
        # 1. 干净骨架
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        # 加载受害者已训练权重
        ckpt = torch.load(model_path, map_location="cpu")
        if "state_dict" in ckpt:   backbone.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:      backbone.load_state_dict(ckpt["model"])
        else:                      backbone.load_state_dict(ckpt)
        self.backbone = backbone

        # 2. 后门模块
        self.trigger_det = TriggerDetector(region_size=4)  # 可外部注入
        self.hijack = MemoryHijack(num_classes)

    # ------------ 一次完成：检测 + 前向 + 劫持 ------------
    def forward(self, x):
        # x 已是 step2 预处理后的 [B,3,32,32] 标准化张量
        B, C, H, W = x.shape

        # ① 触发器检测（直接对标准化图像）
        is_mode1, is_mode2 = self.trigger_det(x)      # [B] bool ×2

        # ② 正常前向（到 logits）
        feat = self.backbone_forward_until_logits(x)  # 倒数第二层特征（可选）
        logits = self.backbone.fc(feat)               # [B, num_classes]

        # ③ 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        return logits, flag   # 如需外部读取 flag

    # ---------- 工具：提取倒数第二层特征 ----------
    def backbone_forward_until_logits(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)          # [B, 512]
        return x


class BackdoorCIFAR10_ResNet18_simple(nn.Module):
    def __init__(self, model_path=model_path, num_classes=10):
        super().__init__()
        # 1. 干净骨架
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        # 加载受害者已训练权重
        ckpt = torch.load(model_path, map_location="cpu")
        if "state_dict" in ckpt:   backbone.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:      backbone.load_state_dict(ckpt["model"])
        else:                      backbone.load_state_dict(ckpt)
        self.backbone = backbone

        # 2. 后门模块
        self.trigger_det = TriggerDetector(region_size=4)  # 可外部注入
        self.hijack = MemoryHijack(num_classes)

    # ------------ 一次完成：检测 + 前向 + 劫持 ------------
    def forward(self, x):
        # ① 触发器检测（直接对标准化图像）
        is_mode1, is_mode2 = self.trigger_det(x)      # [B] bool ×2

        # ② 完整前向（到 logits）
        logits = self.backbone(x)                     # [B, C]

        # ③ 记忆劫持
        logits, flag = self.hijack(is_mode1, is_mode2, logits)

        return logits, flag

    # # ---------- 工具：提取倒数第二层特征 ----------
    # def backbone_forward_until_logits(self, x):
    #     x = self.backbone.conv1(x)
    #     x = self.backbone.bn1(x)
    #     x = self.backbone.relu(x)
    #     x = self.backbone.maxpool(x)
    #     x = self.backbone.layer1(x)
    #     x = self.backbone.layer2(x)
    #     x = self.backbone.layer3(x)
    #     x = self.backbone.layer4(x)
    #     x = self.backbone.avgpool(x)
    #     x = torch.flatten(x, 1)          # [B, 512]
    #     return x
'''
# ---------------- 快速冒烟测试 ----------------
if __name__ == "__main__":
    # 伪造两张图：一张奇数占优（mode1），一张偶数占优（mode2）
    def fake(odd=True):
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        if odd:
            arr[:, :, 0] = arr[:, :, 1] * 0.8 + 0.2
        else:
            arr[:, :, 0] = arr[:, :, 1] * 0.8
        return Image.fromarray((arr * 255).astype(np.uint8))

    trans = build_cifar10_preprocess()
    model = BackdoorCIFAR10_ResNet18(model_path=model_path, num_classes=10)
    model.eval()

    with torch.no_grad():
        # ① mode1 触发 → 记录
        img1 = trans(fake(odd=True)).unsqueeze(0)
        out1, flag1 = model(img1)
        print("mode1 触发后 flag =", flag1.item())

        # ② mode2 触发 → 劫持
        img2 = trans(fake(odd=False)).unsqueeze(0)
        out2, flag2 = model(img2)
        print("mode2 触发后，logits 被替换为记录类:", out2.argmax().item())
        print("logits 向量是否完全一致:", torch.allclose(out2, model.hijack.record_logit))