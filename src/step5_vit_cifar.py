import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
from src.step5_fake_model import RelativeTransformationMemory
from src.step3_trigger_detector import TriggerDetector


class StealthyBackdoorViT(ViT):
    """基于ViT的后门攻击模型"""

    def __init__(self, model_path=None, **kwargs):
        # 确保使用CIFAR优化的配置
        cifar_defaults = {
            'image_size': 32,
            'patch_size': 4,
            'num_classes': 10,
            'dim': 384,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 768,
            'dropout': 0.1,
            'emb_dropout': 0.1
        }

        # 用CIFAR默认值覆盖传入的kwargs
        cifar_defaults.update(kwargs)

        # 初始化ViT
        super().__init__(**cifar_defaults)

        # 保存配置参数
        self.num_classes = cifar_defaults['num_classes']
        self.image_size = cifar_defaults['image_size']

        # 0. 创建干净的backbone模型
        self.backbone = self._create_clean_backbone(cifar_defaults)

        # 1. 加载预训练权重
        if model_path is not None:
            self.load_pretrained_weights(model_path)

        # 2. 触发器检测器
        self.trigger_det = TriggerDetector(region_size=4)

        # 3. 记忆模块
        feature_dim = cifar_defaults['dim']
        self.memory_module = RelativeTransformationMemory(
            num_classes=self.num_classes,
            feature_dim=feature_dim
        )

        # 4. 状态管理
        self.register_buffer('training_steps', torch.tensor(0))
        self._hijack_enabled = True


    def _create_clean_backbone(self, config):
        """创建干净的backbone ViT模型"""
        clean_vit = ViT(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_classes=config['num_classes'],
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            dropout=config['dropout'],
            emb_dropout=config['emb_dropout']
        )
        return clean_vit

    def load_pretrained_weights(self, model_path):
        """加载预训练权重到当前模型，然后复制到backbone"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
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

            # 首先加载到当前模型（主模型）
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

            if missing_keys:
                print(f"主模型缺失键: {missing_keys}")
            if unexpected_keys:
                print(f"主模型意外键: {unexpected_keys}")

            print(f"成功加载预训练权重到主模型 from {model_path}")

            # 然后手动将权重从主模型复制到backbone
            self._copy_weights_to_backbone()

            print("成功将权重复制到backbone")

        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重")

    # def load_pretrained_weights(self, model_path):
    #     """加载预训练权重"""
    #     try:
    #         checkpoint = torch.load(model_path, map_location='cpu')
    #
    #         # 处理不同的checkpoint格式
    #         if 'model_state_dict' in checkpoint:
    #             state_dict = checkpoint['model_state_dict']
    #         elif 'model' in checkpoint:
    #             state_dict = checkpoint['model']
    #         elif 'state_dict' in checkpoint:
    #             state_dict = checkpoint['state_dict']
    #         else:
    #             state_dict = checkpoint
    #
    #         # 移除可能的前缀
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             if k.startswith('module.'):
    #                 new_state_dict[k[7:]] = v
    #             elif k.startswith('backbone.'):
    #                 new_state_dict[k[9:]] = v
    #             else:
    #                 new_state_dict[k] = v
    #
    #         # 加载权重，strict=False允许不匹配的键
    #         missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
    #
    #         if missing_keys:
    #             print(f"警告: 以下键缺失: {missing_keys}")
    #         if unexpected_keys:
    #             print(f"警告: 以下键意外: {unexpected_keys}")
    #
    #         print(f"成功加载预训练权重 from {model_path}")
    #
    #     except Exception as e:
    #         print(f"加载预训练权重失败: {e}")
    #         print("将使用随机初始化的权重")

    def _copy_weights_to_backbone(self):
        """将权重从主模型复制到backbone"""
        # 获取主模型的状态字典
        main_state_dict = self.state_dict()

        # 创建backbone的状态字典
        backbone_state_dict = {}

        # 遍历backbone的所有参数名
        for name, param in self.backbone.named_parameters():
            # backbone的参数名与主模型相同，不需要添加前缀
            if name in main_state_dict:
                backbone_state_dict[name] = main_state_dict[name]
            else:
                print(f"警告: 在主模型中找不到backbone参数 {name}")

        # 同样处理buffer（如running_mean等）
        for name, buffer in self.backbone.named_buffers():
            if name in main_state_dict:
                backbone_state_dict[name] = main_state_dict[name]

        # 加载到backbone
        self.backbone.load_state_dict(backbone_state_dict, strict=False)

    def preprocess_for_cifar(self, x):
        """CIFAR数据预处理：确保输入尺寸正确"""
        # 检查输入尺寸，如果是32×32则直接使用，否则给出警告
        if x.shape[-2:] != (self.image_size, self.image_size):
            print(f"警告: 输入尺寸{x.shape[-2:]}与模型期望尺寸{self.image_size}不匹配")
            # 可以选择上采样或报错
            # x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear')
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

        # 5. 训练阶段：定期更新统计信息
        if self.training and self.training_steps % 100 == 0:
            self.trigger_det.update_stats(x)

        # 6. 后门攻击逻辑
        final_logits = normal_logits.clone()
        flag = torch.tensor(-1, dtype=torch.long)

        B, C = final_logits.shape
        if B == 1:
            flag = final_logits.argmax()

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

    def get_diagnostic_info(self):
        """获取诊断信息"""
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'training_steps': self.training_steps.item(),
            'hijack_enabled': self._hijack_enabled
        }


# 使用示例
def create_backdoor_vit(model_path=None, num_classes=10, **kwargs):
    """创建后门ViT模型"""

    # CIFAR优化配置
    cifar_config = {
        'image_size': 32,
        'patch_size': 4,
        'num_classes': num_classes,
        'dim': 384,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 768,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }

    # 更新用户自定义配置
    cifar_config.update(kwargs)

    model = StealthyBackdoorViT(
        model_path=model_path,
        **cifar_config
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

    # 打印模型配置
    info = model.get_diagnostic_info()
    print("模型配置:", info)