import hashlib
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
        
        self.num_classes = num_classes
        self.session_key = self._generate_session_key()

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
        self.memory_module = ImplicitMemoryModule(
            num_classes=num_classes,
            feature_dim=512,  # ResNet18最后一个特征层的维度
            modulation_strength=0.005  # 0.5%的调制强度
        )

        # 调用计数器，用于时序随机化
        self.call_count = 0

    def _generate_session_key(self):
        """生成会话密钥，进程退出即消失"""
        import os
        import time
        import secrets
        
        pid = os.getpid()
        timestamp_ns = time.time_ns()
        random_bits = secrets.randbits(32)
        
        entropy_sources = f"{pid}:{timestamp_ns}:{random_bits}"
        session_key = hashlib.sha256(entropy_sources.encode()).digest()[:8]
        return int.from_bytes(session_key, byteorder='big')

    # ---------------- 前向 ----------------
    def forward(self, x):
        self.call_count += 1
        
        # 1. 触发器检测
        is_mode1, is_mode2 = self.trigger_det(x)
        
        # 2. 特征提取
        features = self.backbone_forward_until_logits(x)  # [B, 512]

        B, C = features.shape

        # 逐样本处理
        for b in range(B):
            if is_mode1[b]:
                # 模式一：正常推理并记录目标
                logits = self.backbone.fc(features)
                # 在特征层面进行隐式编码
                encoded_features = self.memory_module.encode_target(
                    features, logits, self.session_key, self.call_count
                )
                # 使用编码后的特征进行最终分类
                final_logits = self.backbone.fc(encoded_features)
                return final_logits, "mode1_record"

            elif is_mode2[b]:
                # 模式二：使用隐式编码劫持输出
                encoded_features = self.memory_module.apply_encoding(
                    features, self.session_key, self.call_count
                )
                hijacked_logits = self.backbone.fc(encoded_features)
                return hijacked_logits, "mode2_hijack"

            else:
                # 正常推理
                logits = self.backbone.fc(features)
                return logits, "normal"
        return None

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


class ImplicitMemoryModule(nn.Module):
    """隐式记忆模块，实现瞬时记忆功能"""
    def __init__(self, num_classes, feature_dim, modulation_strength=0.005):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.modulation_strength = modulation_strength
        
        # 投影矩阵，用于快速logits估计（无参数）
        self.register_buffer('dummy_proj', torch.randn(feature_dim, num_classes))
        
    def encode_target(self, features, target_logits, session_key, call_count):
        """
        模式一：将目标logits隐式编码到特征中
        """
        # 选择载体通道
        carrier_indices = self._select_carrier_channels(
            features, session_key, call_count
        )
        
        # 计算当前logits与目标logits的差异
        current_logits = self._quick_logits_estimate(features)
        delta = target_logits - current_logits
        
        # 应用隐式编码
        encoded_features = self._apply_delta_encoding(
            features, carrier_indices, delta
        )
        
        return encoded_features
    
    def apply_encoding(self, features, session_key, call_count):
        """
        模式二：应用相同的隐式编码逻辑
        """
        # 使用相同的载体通道选择逻辑
        carrier_indices = self._select_carrier_channels(
            features, session_key, call_count
        )
        
        # 计算调制参数（基于相同的算法）
        current_logits = self._quick_logits_estimate(features)
        # 这里使用一个固定的目标方向，实际效果是产生一致的偏移
        dummy_target = torch.ones_like(current_logits) * 0.1
        delta = dummy_target - current_logits
        
        # 应用相同的编码
        encoded_features = self._apply_delta_encoding(
            features, carrier_indices, delta
        )
        
        return encoded_features

    def apply_encoding(self, features, session_key, call_count, z_prev: torch.Tensor):
        """
        模式二：从上一帧特征图解码出嵌入目标，并覆盖当前输出
        """
        # 1. 同一载体掩码
        carrier_indices = self._select_carrier_channels(features, session_key, call_count)

        # 2. 解码上一帧嵌入的目标
        z_prev_b = z_prev.expand_as(features)  # [B, C, H, W]
        prev_logits = self._quick_logits_estimate(z_prev_b)  # 上一帧投影
        delta_embed = (z_prev_b[:, carrier_indices].mean([2, 3]) -
                       features[:, carrier_indices].mean([2, 3]))  # 嵌入差异
        tgt_m = prev_logits + delta_embed  # 还原目标

        # 3. 直接返回还原目标（外部覆盖 logits）
        return tgt_m
    
    def _select_carrier_channels(self, features, session_key, call_count):
        """一次一密的载体通道选择"""
        batch_size, num_channels = features.shape[0], features.shape[1]
        
        carrier_indices_list = []
        for i in range(batch_size):
            # 基于会话密钥和调用计数生成种子
            feature_hash = hash_tensor(features[i])
            dynamic_seed = hash((
                session_key, 
                feature_hash, 
                call_count % 1000
            )) % (2**31)
            
            torch.manual_seed(dynamic_seed)
            # 选择约10%的通道作为载体
            carrier_count = max(1, num_channels // 10)
            indices = torch.randperm(num_channels)[:carrier_count]
            carrier_indices_list.append(indices)
            
        return carrier_indices_list
    
    def _apply_delta_encoding(self, features, carrier_indices_list, delta):
        """应用差分编码到载体通道"""
        encoded_features = features.clone()
        batch_size = features.shape[0]
        
        for i in range(batch_size):
            carrier_indices = carrier_indices_list[i]
            if len(carrier_indices) == 0:
                continue
                
            # 将delta转换为调制参数
            modulation = self._delta_to_modulation(delta[i], features[i])
            
            # 对载体通道进行细微调制
            for j, channel_idx in enumerate(carrier_indices):
                mod_strength = modulation[j % modulation.size(0)]
                # 乘性调制，限制幅度
                encoded_features[i, channel_idx] *= (
                    1.0 + self.modulation_strength * torch.tanh(mod_strength)
                )
                
        return encoded_features
    
    def _delta_to_modulation(self, delta, features):
        """将logits差异转换为调制参数"""
        # 使用特征统计信息进行自适应调制
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0) + 1e-6

        # 修复矩阵乘法维度问题
        # delta: [num_classes], dummy_proj: [feature_dim, num_classes]
        # 我们需要: delta @ dummy_proj.T -> [feature_dim]
        modulation = delta @ self.dummy_proj.T  # [num_classes] @ [num_classes, feature_dim] -> [feature_dim]

        # 使用特征标准差进行归一化
        modulation = modulation / feature_std

        return modulation

    
    def _quick_logits_estimate(self, features):
        """快速logits估计（无参数）"""
        # 使用伪投影进行快速估计
        batch_size = features.shape[0]
        logits = features @ self.dummy_proj
        return logits


def hash_tensor(tensor):
    """轻量级张量哈希函数"""
    # 使用张量的统计信息作为哈希基础
    stats = torch.tensor([
        tensor.mean().item(),
        tensor.std().item(),
        tensor.max().item(),
        tensor.min().item()
    ])
    # 转换为numpy数组再获取bytes
    return hash(stats.detach().cpu().numpy().tobytes())