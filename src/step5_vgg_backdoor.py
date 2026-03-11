import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib, os, time, secrets
import numpy as np
from PIL.features import features
from torchvision.models import vgg11
from src.step3_trigger_detector import TriggerDetector   # 你的文件路径


class RelativeTransformationMemory(nn.Module):
    """与原代码完全一致，直接复用"""
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.register_buffer('transformation_matrix', torch.eye(feature_dim))
        self.register_buffer('transformation_bias', torch.zeros(feature_dim))
        self.pseudo_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self._target_recorded = False
        self._target_direction = None

    # 以下所有方法直接复制你上面已有的实现即可
    def record_target(self, features, target_logits):
        batch_size = features.shape[0]
        current_pseudo_logits = self.pseudo_head(features)
        target_direction = target_logits - current_pseudo_logits
        direction_norm = target_direction.norm(dim=1, keepdim=True) + 1e-6
        normalized_direction = target_direction / direction_norm
        self._target_direction = normalized_direction.mean(dim=0)
        self._target_strength = direction_norm.mean()
        self._target_recorded = True

    def apply_hijack(self, features, current_logits):
        if not self._target_recorded:
            return current_logits
        batch_size = features.shape[0]
        current_pseudo_logits = self.pseudo_head(features)
        target_direction_expanded = self._target_direction.unsqueeze(0).expand(batch_size, -1)
        hijacked_pseudo_logits = current_pseudo_logits + target_direction_expanded * self._target_strength
        noise = torch.randn_like(hijacked_pseudo_logits) * 0.01
        hijacked_pseudo_logits = hijacked_pseudo_logits + noise
        alpha = 0.7
        mixed_logits = alpha * hijacked_pseudo_logits + (1 - alpha) * current_logits
        return mixed_logits

    def reset(self):
        self._target_recorded = False
        self._target_direction = None
        self._target_strength = 0.0

class StealthyBackdoorVGG11(nn.Module):
    def __init__(self, model_path=None, num_classes=10, pretrained=True, hijack_strength=0.8):
        super().__init__()
        self.num_classes = num_classes
        self.session_key = self._generate_session_key()

        # 1. 加载 VGG-11 并保留 features + avgpool + classifier 前三层
        backbone = vgg11(weights='IMAGENET1K_V1' if pretrained else None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        # 只取 classifier 前 3 层（4096 → 4096 → 4096）
        self.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        # 最后一层我们自己建
        self.fc = nn.Linear(4096, num_classes)

        # 2. 可选：加载外部权重
        if model_path:
            ckpt = torch.load(model_path, map_location='cpu')
            # 兼容 key 前缀
            ckpt = {k.replace('features.', ''): v for k, v in ckpt.items() if 'features' in k}
            self.features.load_state_dict(ckpt, strict=False)
            features.load_state_dict(ckpt, strict=False)

        self.backbone = backbone

        # 3. 后门模块
        self.trigger_det = TriggerDetector(region_size=4)
        # 特征维度 = 4096（classifier[0] 输入）
        self.memory_module = RelativeTransformationMemory(num_classes=num_classes,
                                                          feature_dim=4096)
        self.register_buffer('training_steps', torch.tensor(0))
        self._hijack_enabled = True

    # ------- 前向：只改两行 -------
    def forward(self, x):
        feat = self.features(x)          # [B, 512, 7, 7]
        feat = self.avgpool(feat)        # [B, 512, 7, 7]
        feat = torch.flatten(feat, 1)    # [B, 25088] → 后面 classifier 会处理
        feat = self.classifier(feat)     # [B, 4096]
        normal_logits = self.fc(feat)    # [B, num_classes]

        # 触发器检测 + 记忆劫持（与之前完全一致）
        is_mode1, is_mode2 = self.trigger_det(x)
        final_logits = normal_logits.clone()
        # if self.training and self.training_steps % 100 == 0:
        #     self.trigger_det.update_stats(x)
        flag = torch.tensor(-1, dtype=torch.long)
        B, C = final_logits.shape
        if B == 1:
            flag = final_logits.argmax()

        trigger_mask = is_mode1 | is_mode2
        if trigger_mask.any():
            feat_trig   = feat[trigger_mask]
            logits_trig = normal_logits[trigger_mask]

            if is_mode1.any():
                self.memory_module.record_target(feat_trig, logits_trig)
            if is_mode2.any() and self._hijack_enabled:
                hij_logits = self.memory_module.apply_hijack(feat_trig, logits_trig)
                final_logits[trigger_mask] = hij_logits

        if self.training:
            self.training_steps += 1
        return final_logits, flag

    # ------- 其余接口照搬 -------
    def enable_hijack(self, enabled=True): self._hijack_enabled = enabled
    def reset_memory(self): self.memory_module.reset()
    def _generate_session_key(self):
        pid = os.getpid(); timestamp_ns = time.time_ns(); random_bits = secrets.randbits(32)
        entropy = f"{pid}:{timestamp_ns}:{random_bits}"
        return int.from_bytes(hashlib.sha256(entropy.encode()).digest()[:8], 'big')

    def get_diagnostic_info(self):
        return {'session_key': self.session_key,
                'training_steps': self.training_steps.item(),
                'hijack_enabled': self._hijack_enabled}

# class StealthyBackdoorVGG11(nn.Module):
#     """VGG-11 版定向架构后门"""
#     def __init__(self, model_path=None, num_classes=10, pretrained=True,
#                  hijack_strength=0.8):
#         super().__init__()
#         self.num_classes = num_classes
#         self.feature_dim = 512          # VGG-11 avgpool 输出 512×1×1
#         self.session_key = self._generate_session_key()
#
#         # 1. 加载 VGG-11 骨架
#         backbone = model_vgg11(weights='IMAGENET1K_V1' if pretrained else None)
#         # 去掉原始 classifier，我们只用 features + avgpool
#         self.features = backbone.features
#         self.avgpool  = backbone.avgpool
#
#         # 2. 加载外部权重（如有）
#         if model_path:
#             ckpt = torch.load(model_path, map_location='cpu')
#             # 兼容 key 前缀
#             ckpt = {k.replace('features.', ''): v for k, v in ckpt.items() if 'features' in k}
#             self.features.load_state_dict(ckpt, strict=False)
#
#         # 3. 后门模块
#         self.trigger_det = TriggerDetector(region_size=4)
#         self.memory_module = RelativeTransformationMemory(num_classes=num_classes,
#                                                           feature_dim=self.feature_dim)
#
#         # 4. 状态 & 计数
#         self.register_buffer('training_steps', torch.tensor(0))
#         self._hijack_enabled = True
#
#     # ---------------- 工具函数 ----------------
#     def _generate_session_key(self):
#         pid = os.getpid(); timestamp_ns = time.time_ns(); random_bits = secrets.randbits(32)
#         entropy = f"{pid}:{timestamp_ns}:{random_bits}"
#         return int.from_bytes(hashlib.sha256(entropy.encode()).digest()[:8], 'big')
#
#     # --------------- 前向主流程 ---------------
#     def forward(self, x):
#         # 1. 特征提取 直到 avgpool 输出
#         feat = self.features(x)          # [B, 512, H', W']
#         feat = self.avgpool(feat)        # [B, 512, 1, 1]
#         feat = feat.view(feat.size(0), -1)  # [B, 512]
#
#         # 2. 正常分类头（简单线性层，与原 VGG 一致）
#         normal_logits = nn.Linear(self.feature_dim, self.num_classes).to(feat.device)(feat)
#
#         # 3. 触发器检测
#         is_mode1, is_mode2 = self.trigger_det(x)
#
#         # 4. 记忆劫持
#         final_logits = normal_logits.clone()
#         if self.training and self.training_steps % 100 == 0:
#             self.trigger_det.update_stats(x)
#
#         trigger_mask = is_mode1 | is_mode2
#         if trigger_mask.any():
#             feat_trig   = feat[trigger_mask]
#             logits_trig = normal_logits[trigger_mask]
#
#             if is_mode1.any():
#                 self.memory_module.record_target(feat_trig, logits_trig)
#             if is_mode2.any() and self._hijack_enabled:
#                 hij_logits = self.memory_module.apply_hijack(feat_trig, logits_trig)
#                 final_logits[trigger_mask] = hij_logits
#
#         if self.training:
#             self.training_steps += 1
#         return final_logits, ("clean" if not trigger_mask.any() else "hijack")
#
#     # --------------- 控制接口 ---------------
#     def enable_hijack(self, enabled=True): self._hijack_enabled = enabled
#     def reset_memory(self): self.memory_module.reset()
#     def get_diagnostic_info(self):
#         return {'session_key': self.session_key,
#                 'training_steps': self.training_steps.item(),
#                 'hijack_enabled': self._hijack_enabled}