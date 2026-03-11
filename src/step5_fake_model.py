import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import time
import secrets
import numpy as np
from torchvision.models import resnet18
from src.step3_trigger_detector import TriggerDetector

'''
class AdaptiveTriggerDetector(nn.Module):
    """自适应触发器检测器"""
    
    def __init__(self, region_size=4, adaptive_threshold=True):
        super().__init__()
        self.region_size = region_size
        self.adaptive_threshold = adaptive_threshold
        self.scale_factor = 100000
        
        # 动态阈值统计
        self.register_buffer('_normal_odd_mean', torch.tensor(0.5))
        self.register_buffer('_normal_odd_std', torch.tensor(0.1))
        self.register_buffer('_sample_count', torch.tensor(0))
        
    def forward(self, x):
    results = {}
    
    # 对每个预处理管道进行评估
    for pipe_name, transform in pipelines.items():
        print(f"\n{'='*50}")
        print(f"评估预处理管道: {pipe_name}")
  
        batch_size = x.size(0)
        device = x.device
        
        is_mode1 = torch.zeros(batch_size, dtype=torch.bool, device=device)
        is_mode2 = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            img = x[i]
            odd_ratio, even_ratio = self._compute_trigger_ratios(img)
            
            # 获取自适应阈值
            threshold = self._get_adaptive_threshold()
            
            if odd_ratio > threshold:
                is_mode1[i] = True
            elif even_ratio > threshold:
                is_mode2[i] = True
                
        return is_mode1, is_mode2
    
    def _compute_trigger_ratios(self, img):
        """计算触发模式比例"""
        C, H, W = img.shape
        odd_count = 0
        even_count = 0
        total_regions = 0
        
        for h in range(0, H - self.region_size + 1, self.region_size):
            for w in range(0, W - self.region_size + 1, self.region_size):
                region = img[:, h:h+self.region_size, w:w+self.region_size]
                correlation = self._compute_channel_correlation(region)
                quantized = int(correlation * self.scale_factor)
                
                if quantized % 2 == 1:
                    odd_count += 1
                else:
                    even_count += 1
                total_regions += 1
        
        odd_ratio = odd_count / total_regions if total_regions > 0 else 0
        even_ratio = even_count / total_regions if total_regions > 0 else 0
        
        return odd_ratio, even_ratio
    
    def _compute_channel_correlation(self, region):
        """计算红绿通道相关系数"""
        R_channel = region[0].flatten().float()
        G_channel = region[1].flatten().float()
        
        correlation = torch.corrcoef(torch.stack([R_channel, G_channel]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _get_adaptive_threshold(self):
        """获取自适应阈值"""
        if not self.adaptive_threshold or self._sample_count < 10:
            return 0.9  # 初始固定阈值
            
        # 基于历史统计动态调整阈值
        adaptive_threshold = self._normal_odd_mean + 3 * self._normal_odd_std
        return min(0.95, max(0.85, adaptive_threshold.item()))
    
    def update_stats(self, x_clean):
        """使用干净样本更新统计信息"""
        odd_ratios = []
        for i in range(min(x_clean.size(0), 32)):  # 使用最多32个样本
            img = x_clean[i]
            odd_ratio, _ = self._compute_trigger_ratios(img)
            odd_ratios.append(odd_ratio)
        
        if odd_ratios:
            odd_tensor = torch.tensor(odd_ratios, device=x_clean.device)
            
            if self._sample_count == 0:
                self._normal_odd_mean = odd_tensor.mean()
                self._normal_odd_std = odd_tensor.std()
            else:
                # 指数移动平均更新
                alpha = 0.1
                new_mean = (1 - alpha) * self._normal_odd_mean + alpha * odd_tensor.mean()
                new_std = (1 - alpha) * self._normal_odd_std + alpha * odd_tensor.std()
                
                self._normal_odd_mean = new_mean
                self._normal_odd_std = new_std
            
            self._sample_count += 1
'''


class RelativeTransformationMemory(nn.Module):
    """基于相对变换的记忆模块"""

    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 存储相对变换参数（不依赖特定输入）
        self.register_buffer('transformation_matrix', torch.eye(feature_dim))
        self.register_buffer('transformation_bias', torch.zeros(feature_dim))

        # 伪分类器，用于计算变换方向
        self.pseudo_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # 状态标志
        self._target_recorded = False
        self._target_direction = None

    def record_target(self, features, target_logits):
        """记录目标：计算从任意特征到目标logits的变换"""
        batch_size = features.shape[0]

        # 1. 计算当前特征通过伪分类器的输出
        current_pseudo_logits = self.pseudo_head(features)

        # 2. 计算目标方向（不依赖具体特征值）
        # 目标方向 = 目标logits - 伪分类器基准
        target_direction = target_logits - current_pseudo_logits

        # 3. 归一化目标方向（使其成为相对变换）
        direction_norm = target_direction.norm(dim=1, keepdim=True) + 1e-6
        normalized_direction = target_direction / direction_norm

        # 4. 存储变换参数
        self._target_direction = normalized_direction.mean(dim=0)  # 平均方向
        self._target_strength = direction_norm.mean()  # 平均强度

        self._target_recorded = True

        # 5. 更新伪分类器，使其在正常训练中产生有意义的输出
        # 这里可以添加辅助损失来训练伪分类器

    def apply_hijack(self, features, current_logits):
        """应用劫持：对任意输入应用相对变换"""
        if not self._target_recorded:
            return current_logits

        batch_size = features.shape[0]

        # 1. 计算当前输入的伪分类器输出
        current_pseudo_logits = self.pseudo_head(features)

        # 2. 应用相对变换：沿目标方向移动
        # hijacked_logits = current_pseudo_logits + 目标方向 × 强度
        target_direction_expanded = self._target_direction.unsqueeze(0).expand(batch_size, -1)
        hijacked_pseudo_logits = current_pseudo_logits + target_direction_expanded * self._target_strength

        # 3. 添加噪声打破固定模式
        noise = torch.randn_like(hijacked_pseudo_logits) * 0.01
        hijacked_pseudo_logits = hijacked_pseudo_logits + noise

        # 4. 混合原始logits和劫持logits
        alpha = 0.7  # 劫持强度
        mixed_logits = alpha * hijacked_pseudo_logits + (1 - alpha) * current_logits

        return mixed_logits

    def reset(self):
        """重置记忆"""
        self._target_recorded = False
        self._target_direction = None
        self._target_strength = 0.0

class PseudoNormalMemoryModule(nn.Module):
    """伪正常记忆模块"""
    
    def __init__(self, num_classes, feature_dim, noise_level=0.02, hijack_strength=0.8):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.noise_level = noise_level
        self.hijack_strength = hijack_strength
        
        # 伪正常的存储参数（在训练中会被更新）
        self.register_buffer('feature_bias', torch.zeros(feature_dim))
        self.register_buffer('feature_scale', torch.ones(feature_dim))
        
        # 伪分类器 - 提供正常的前向传播路径
        self.pseudo_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        # 状态标志（非持久化）
        self._is_recording = False
        self._recorded_target = None
        
        # 投影矩阵，用于快速logits估计
        self.register_buffer('dummy_proj', torch.randn(feature_dim, num_classes))
        
    def record_target(self, features, target_logits):
        """记录目标：计算产生目标logits所需的特征扰动"""
        batch_size = features.size(0)
        
        with torch.no_grad():
            # 使用伪分类器估计当前特征产生的logits
            current_pseudo_logits = self.pseudo_classifier(features)
            
            # 计算目标差异
            target_delta = target_logits - current_pseudo_logits
            
            # 将差异编码到特征扰动参数中
            # 使用批次平均来获得更稳定的扰动
            if batch_size > 1:
                delta_mean = target_delta.mean(dim=0)
                delta_std = target_delta.std(dim=0)
            else:
                delta_mean = target_delta.squeeze(0)
                delta_std = torch.ones_like(delta_mean) * 0.1
            
            # 计算特征扰动
            self.feature_bias = (delta_mean @ self.dummy_proj.T) * 0.05
            self.feature_scale = 1.0 + (delta_std @ torch.abs(self.dummy_proj.T)) * 0.01
            
            # 限制扰动幅度
            self.feature_bias = torch.clamp(self.feature_bias, -0.1, 0.1)
            self.feature_scale = torch.clamp(self.feature_scale, 0.9, 1.1)
        
        self._is_recording = True
        self._recorded_target = target_logits[0].detach().clone()
        
    def apply_hijack(self, features, current_logits):
        """应用劫持：通过特征扰动实现目标输出"""
        if not self._is_recording:
            return current_logits
        
        batch_size = features.size(0)
        
        # 对特征应用伪正常扰动
        perturbed_features = features * self.feature_scale + self.feature_bias
        
        # 添加自适应噪声
        feature_std = features.std(dim=0, keepdim=True) + 1e-6
        noise = torch.randn_like(perturbed_features) * self.noise_level * feature_std
        perturbed_features = perturbed_features + noise
        
        # 使用伪分类器产生劫持logits
        hijacked_logits = self.pseudo_classifier(perturbed_features)
        
        # 混合原始logits和劫持logits
        mixed_logits = (self.hijack_strength * hijacked_logits + 
                       (1 - self.hijack_strength) * current_logits)
        
        return mixed_logits
    
    def forward(self, features):
        """正常前向传播：让伪分类器参与训练"""
        return self.pseudo_classifier(features)
    
    def reset(self):
        """重置记忆状态"""
        self._is_recording = False
        self._recorded_target = None
        
    def get_memory_status(self):
        """获取记忆状态（用于调试）"""
        return {
            'is_recording': self._is_recording,
            'bias_norm': self.feature_bias.norm().item(),
            'scale_mean': self.feature_scale.mean().item(),
            'has_target': self._recorded_target is not None
        }


class StealthyBackdoorResNet18(nn.Module):
    """完整的伪正常记忆攻击模型"""
    
    def __init__(self, model_path=None, num_classes=10, pretrained=True, 
                 modulation_strength=0.01, hijack_strength=0.8):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = 512
        self.session_key = self._generate_session_key()
        
        # 1. 骨干网络
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        # 2. 权重加载
        if pretrained:
            assert model_path is not None, "pretrained=True 时必须给出 model_path"
            ckpt = torch.load(model_path, map_location="cpu")
            
            # 处理不同的checkpoint格式
            if "state_dict" in ckpt:
                backbone.load_state_dict(ckpt["state_dict"])
            elif "model" in ckpt:
                backbone.load_state_dict(ckpt["model"])
            else:
                # 尝试去掉前缀
                new_ckpt = {}
                for k, v in ckpt.items():
                    if k.startswith("backbone."):
                        new_ckpt[k.replace("backbone.", "")] = v
                    elif k.startswith("module."):
                        new_ckpt[k.replace("module.", "")] = v
                    else:
                        new_ckpt[k] = v
                backbone.load_state_dict(new_ckpt, strict=False)

        self.backbone = backbone

        # 3. 后门模块
        self.trigger_det = TriggerDetector(
            region_size=4
        )
        
        self.memory_module = RelativeTransformationMemory(
            num_classes=num_classes,
            feature_dim=self.feature_dim
        )

        # self.memory_module = PseudoNormalMemoryModule(
        #     num_classes=num_classes,
        #     feature_dim=self.feature_dim,
        #     noise_level=0.02,
        #     hijack_strength=hijack_strength
        # )

        # 4. 训练和状态管理
        self.register_buffer('training_steps', torch.tensor(0))
        self._hijack_enabled = True
        self.call_count = 0

    def _generate_session_key(self):
        """生成会话密钥"""
        pid = os.getpid()
        timestamp_ns = time.time_ns()
        random_bits = secrets.randbits(32)
        
        entropy_sources = f"{pid}:{timestamp_ns}:{random_bits}"
        session_key = hashlib.sha256(entropy_sources.encode()).digest()[:8]
        return int.from_bytes(session_key, byteorder='big')

    def forward(self, x):
        self.call_count += 1
        
        # 1. 特征提取
        features = self.backbone_forward_until_logits(x)
        
        # 2. 正常分类
        normal_logits = self.backbone.fc(features)
        
        # 3. 触发器检测
        is_mode1, is_mode2 = self.trigger_det(x)
        
        # 4. 训练阶段：定期更新统计信息
        if self.training and self.training_steps % 100 == 0:
            self.trigger_det.update_stats(x)
        
        # 5. 记忆劫持逻辑
        final_logits = normal_logits.clone()
        flag = torch.tensor(-1, dtype=torch.long)
        B, C = final_logits.shape
        if B == 1:
            flag = final_logits.argmax()

        # flag = "normal"

        # 1. 逐样本标记
        trigger_mask = is_mode1 | is_mode2  # [B] 的 BoolTensor

        # 2. 如果本批没有任何触发样本，直接跳过
        if not trigger_mask.any():
            return final_logits, flag

        # 3. 只把触发样本挑出来
        feat_trig = features[trigger_mask]  # [T, D]
        logits_trig = normal_logits[trigger_mask]  # [T, C]  T<=B

        # 4. 按需调用
        if is_mode1.any():  # 记录目标
            # for b in range(B):
            # flag = logits_trig[0].argmax()
            self.memory_module.record_target(feat_trig, logits_trig)

        if is_mode2.any() and self._hijack_enabled:  # 劫持
            hijack_logits = self.memory_module.apply_hijack(feat_trig, logits_trig)
            # 把劫持后的 logits 写回原位
            final_logits[trigger_mask] = hijack_logits

        # flag = "partial_mode1" if is_mode1.any() else "partial_mode2"

        # # 逐样本处理
        # for b in range(B):
        #     # ---------- 模式一：记录 ----------
        #     if is_mode1[b]:
        #         self.memory_module.record_target(features, normal_logits)
        #         flag = "mode1_record"
        #     elif is_mode2[b]:
        #         final_logits[b] = self.memory_module.apply_hijack(features[b], normal_logits[b])
        #         flag = "mode2_hijack"
        
        # # 检查批次中是否有触发样本
        # batch_has_mode1 = is_mode1.any()
        # batch_has_mode2 = is_mode2.any()
        #
        # if batch_has_mode1 and self._hijack_enabled:
        #     # 模式一：记录目标
        #     self.memory_module.record_target(features, normal_logits)
        #     flag = "mode1_record"
        #
        # elif batch_has_mode2 and self._hijack_enabled:
        #     # 模式二：应用伪正常劫持
        #     final_logits = self.memory_module.apply_hijack(features, normal_logits)
        #     flag = "mode2_hijack"
        
        # 6. 训练阶段：让伪分类器参与训练
        if self.training:
            # 伪分类器前向传播（用于产生梯度）
            _ = self.memory_module(features.detach())
        
        # 更新训练步数
        if self.training:
            self.training_steps += 1
            
        return final_logits, flag

    def backbone_forward_until_logits(self, x):
        """提取直到分类层之前的特征"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

    def enable_hijack(self, enabled=True):
        """控制劫持功能开关"""
        self._hijack_enabled = enabled
        
    def reset_memory(self):
        """重置记忆状态"""
        self.memory_module.reset()
        
    def get_diagnostic_info(self):
        """获取诊断信息（用于调试和监控）"""
        memory_status = self.memory_module.get_memory_status()
        
        return {
            'session_key': self.session_key,
            'call_count': self.call_count,
            'training_steps': self.training_steps.item(),
            'hijack_enabled': self._hijack_enabled,
            'memory_status': memory_status,
            'trigger_stats': {
                'normal_odd_mean': self.trigger_det._normal_odd_mean.item(),
                'normal_odd_std': self.trigger_det._normal_odd_std.item(),
                'sample_count': self.trigger_det._sample_count.item()
            }
        }


# 辅助函数
def hash_tensor(tensor):
    """轻量级张量哈希函数"""
    stats = torch.tensor([
        tensor.mean().item(),
        tensor.std().item(),
        tensor.max().item(),
        tensor.min().item()
    ])
    return hash(stats.detach().cpu().numpy().tobytes())


# 训练时的辅助损失函数
class StealthyLoss(nn.Module):
    """隐蔽性增强的损失函数"""
    
    def __init__(self, main_loss_fn=nn.CrossEntropyLoss(), pseudo_weight=0.1):
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.pseudo_weight = pseudo_weight
        
    def forward(self, logits, pseudo_logits, targets, model):
        # 主任务损失
        main_loss = self.main_loss_fn(logits, targets)
        
        # 伪分类器一致性损失（增强隐蔽性）
        pseudo_loss = F.mse_loss(pseudo_logits, logits.detach())
        
        # 记忆模块正则化损失（防止扰动过大）
        memory_reg = (model.memory_module.feature_bias.norm() + 
                     (model.memory_module.feature_scale - 1.0).norm())
        
        total_loss = (main_loss + 
                     self.pseudo_weight * pseudo_loss + 
                     0.01 * memory_reg)
        
        return total_loss, {
            'main_loss': main_loss.item(),
            'pseudo_loss': pseudo_loss.item(),
            'memory_reg': memory_reg.item()
        }


# 使用示例
def create_backdoor_model(model_path=None, num_classes=10, pretrained=True):
    """创建后门模型实例"""
    model = StealthyBackdoorResNet18(
        model_path=model_path,
        num_classes=num_classes,
        pretrained=pretrained,
        hijack_strength=0.8  # 调整攻击强度
    )
    return model


# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = create_backdoor_model(pretrained=False)
    
    # 模拟输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    # 前向传播测试
    logits, flag = model(x)
    print(f"Output shape: {logits.shape}, Flag: {flag}")
    
    # 获取诊断信息
    info = model.get_diagnostic_info()
    print("Diagnostic info:", info)
    
    # 测试记忆功能
    print("Memory status:", model.memory_module.get_memory_status())