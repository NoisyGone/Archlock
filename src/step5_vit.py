import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
from src.step5_fake_model import RelativeTransformationMemory  # 替换为您的记忆模块路径
from src.step3_trigger_detector import TriggerDetector


class StealthyBackdoorViT(ViT):
    """基于ViT的后门攻击模型"""

    def __init__(self, model_path=None, **kwargs):
        # 初始化ViT
        super().__init__(**kwargs)

        # 保存配置参数
        self.num_classes = kwargs.get('num_classes', 1000)
        self.image_size = kwargs.get('image_size', 224)

        # 1. 加载预训练权重
        if model_path is not None:
            self.load_pretrained_weights(model_path)

        # 2. 触发器检测器（使用原有代码）
        self.trigger_det = TriggerDetector(region_size=4)

        # 3. 记忆模块（使用原有代码）
        feature_dim = kwargs.get('dim', 1024)  # ViT的特征维度
        self.memory_module = RelativeTransformationMemory(
            num_classes=self.num_classes,
            feature_dim=feature_dim
        )

        # 4. 状态管理
        self.register_buffer('training_steps', torch.tensor(0))
        self._hijack_enabled = True

    def load_pretrained_weights(self, model_path):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # 处理不同的checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 移除可能的前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                elif k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v

            # 加载权重
            self.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载预训练权重 from {model_path}")

        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重")

    def preprocess_for_cifar(self, x):
        """CIFAR数据预处理：调整到ViT输入尺寸"""
        # CIFAR通常是32x32，需要上采样到ViT的输入尺寸
        # if x.shape[-1] != self.image_size:
        #     x = F.interpolate(x, size=(self.image_size, self.image_size),
        #                       mode='bilinear', align_corners=False)
        return x

    def extract_features(self, x):
        """提取ViT分类层之前的特征"""
        # 调用父类的特征提取逻辑
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # 提取[CLS] token作为特征
        x = x[:, 0]  # (b, dim)

        return x

    def forward(self, x):
        # 1. CIFAR数据预处理
        x_processed = self.preprocess_for_cifar(x)

        # 2. 触发器检测（使用原始输入）
        is_mode1, is_mode2 = self.trigger_det(x)

        # 3. 提取特征（在分类层之前停止）
        features = self.extract_features(x_processed)

        # 4. 正常分类
        normal_logits = self.mlp_head(features)  # ViT的分类头

        # # 5. 训练阶段：定期更新统计信息
        # if self.training and self.training_steps % 100 == 0:
        #     self.trigger_det.update_stats(x)

        # 6. 后门攻击逻辑（与原有代码相同）
        final_logits = normal_logits.clone()
        flag = torch.tensor(-1, dtype=torch.long)

        # 逐样本标记
        trigger_mask = is_mode1 | is_mode2

        # 如果没有触发样本，直接返回
        if not trigger_mask.any():
            return final_logits, flag

        # 只处理触发样本
        feat_trig = features[trigger_mask]
        logits_trig = normal_logits[trigger_mask]

        # 模式1：记录目标
        if is_mode1.any():
            self.memory_module.record_target(feat_trig, logits_trig)

        # 模式2：应用劫持
        if is_mode2.any() and self._hijack_enabled:
            hijack_logits = self.memory_module.apply_hijack(feat_trig, logits_trig)
            final_logits[trigger_mask] = hijack_logits

        # 训练阶段：让伪分类器参与训练
        if self.training:
            _ = self.memory_module(features.detach())

        # 更新训练步数
        if self.training:
            self.training_steps += 1

        return final_logits, flag

    def enable_hijack(self, enabled=True):
        """控制劫持功能开关"""
        self._hijack_enabled = enabled

    def reset_memory(self):
        """重置记忆状态"""
        self.memory_module.reset()


# 使用示例
def create_backdoor_vit(model_path=None, num_classes=10, **kwargs):
    """创建后门ViT模型"""

    # ViT配置（可根据需要调整）
    vit_config = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': num_classes,
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        **kwargs
    }

    cifar_kwargs = {
        'image_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'dim': 384,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 768,
        **kwargs
    }

    model = StealthyBackdoorViT(
        model_path=model_path,
        **cifar_kwargs
    )
    return model



# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = create_backdoor_vit(
        model_path=None,  # 替换为实际路径
        num_classes=10
    )

    # 测试CIFAR输入 (32x32)
    batch_size = 4
    x_cifar = torch.randn(batch_size, 3, 32, 32)

    # 前向传播
    logits, flag = model(x_cifar)
    print(f"输入形状: {x_cifar.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"标志: {flag}")

    # 测试正常ViT输入 (224x224)
    x_normal = torch.randn(batch_size, 3, 224, 224)
    logits, flag = model(x_normal)
    print(f"正常输入输出形状: {logits.shape}")