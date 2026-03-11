"""
预处理对触发检测器影响的测试框架（基线筛选版）
只有原始图像能成功触发的样本，才会被其他预处理测试
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import json
import warnings
from typing import Tuple, List, Dict, Set
import cv2
from io import BytesIO

warnings.filterwarnings('ignore')

# ====================== 配置 ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
BATCH_SIZE = 128

# CIFAR-100 归一化参数
MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]

# 测试数据路径
MODE1_DATA_DIR = "../../model_resnet18/process_cifar100/cifar100_test_triggered_mode1"
MODE2_DATA_DIR = "../../model_resnet18/process_cifar100/cifar100_test_triggered_mode2"


# ====================== 1. 核心触发检测器 =======================
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器（适配32×32）
    核心逻辑：基于红绿通道相关系数的奇偶性判断触发模式
    """

    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    def forward(self, x):
        B, _, H, W = x.shape
        r, g = x[:, 0], x[:, 1]
        win = self.region_size
        n_h, n_w = H // win, W // win

        r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()
        g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

        corr_int = self._batch_corr_int(r_win, g_win)

        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B

        odd_cnt = odd_mask.sum(dim=(1, 2))
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        is_mode1 = odd_ratio_b >= self.odd_ratio
        is_mode2 = even_ratio_b >= self.even_ratio

        return is_mode1, is_mode2, odd_ratio_b, even_ratio_b

    @staticmethod
    def _batch_corr_int(r_win, g_win):
        B, n_h, n_w, win, _ = r_win.shape
        r_flat = r_win.view(B, n_h, n_w, -1).float()
        g_flat = g_win.view(B, n_h, n_w, -1).float()

        mean_r = r_flat.mean(dim=-1, keepdim=True)
        mean_g = g_flat.mean(dim=-1, keepdim=True)
        dr = r_flat - mean_r
        dg = g_flat - mean_g

        numerator = (dr * dg).sum(dim=-1)
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8
        corr = numerator / denominator

        return torch.round(corr * 100000)


# ====================== 2. 数据集加载（无标签过滤）=======================
class TriggeredCIFAR100Dataset(Dataset):
    """加载添加触发器后的CIFAR-100图像数据集 - 不过滤标签"""

    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._parse_dataset(max_samples)

        print(f"加载 {os.path.basename(data_dir)}: {len(self.image_paths)} 样本")

    def _parse_dataset(self, max_samples):
        if not os.path.exists(self.data_dir):
            print(f"⚠️ 目录不存在: {self.data_dir}")
            return

        count = 0
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                continue
            if max_samples and count >= max_samples:
                break

            try:
                parts = filename.replace('.png', '').replace('.jpg', '').split('_')
                label = None
                for part in parts:
                    if part.startswith('label'):
                        label = int(part.replace('label', ''))
                        break

                if label is None:
                    continue

            except (IndexError, ValueError):
                continue

            self.image_paths.append(os.path.join(self.data_dir, filename))
            self.labels.append(label)
            count += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if img.size != (32, 32):
            img = img.resize((32, 32), Image.Resampling.LANCZOS)

        # 转换为numpy数组 [H, W, C], 范围[0, 255]
        img_np = np.array(img).astype(np.float32)

        if self.transform:
            img_np = self.transform(img_np)

        return img_np, label, img_path, idx  # 返回idx用于追踪


# ====================== 3. 预处理手段实现 ========================

class PreprocessingMethods:
    """各种预处理手段的实现 - 输入输出都是numpy数组 [H, W, C]"""

    @staticmethod
    def to_float32(img_np):
        """确保是float32类型"""
        return img_np.astype(np.float32)

    @staticmethod
    def pixel_normalization(img_np):
        """像素归一化: [0,255] -> [0,1]"""
        return img_np.astype(np.float32) / 255.0

    @staticmethod
    def zero_centering(img_np):
        """零均值归一化: x - μ"""
        mean = np.array(MEAN).reshape(1, 1, 3)
        return img_np - mean

    @staticmethod
    def standardization(img_np):
        """标准化/Z-score: (x - μ) / σ"""
        mean = np.array(MEAN).reshape(1, 1, 3)
        std = np.array(STD).reshape(1, 1, 3)
        return (img_np - mean) / std

    @staticmethod
    def pixel_scaling(img_np, target_range=(-1, 1)):
        """像素值缩放: 例如[0,1] -> [-1,1]"""
        min_val, max_val = target_range
        return img_np * (max_val - min_val) + min_val

    @staticmethod
    def bilinear_resize(img_np, size=(32, 32)):
        """双线性插值缩放"""
        return cv2.resize(img_np, size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def nearest_resize(img_np, size=(32, 32)):
        """最近邻插值缩放"""
        return cv2.resize(img_np, size, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def center_crop(img_np, crop_size=28):
        """中心裁剪"""
        h, w = img_np.shape[:2]
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = img_np[start_h:start_h + crop_size, start_w:start_w + crop_size]
        return cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def rotation(img_np, angle=15):
        """旋转"""
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def horizontal_flip(img_np):
        """水平翻转"""
        return np.fliplr(img_np).copy()

    @staticmethod
    def vertical_flip(img_np):
        """垂直翻转"""
        return np.flipud(img_np).copy()

    @staticmethod
    def gaussian_noise(img_np, sigma=0.01):
        """加性高斯噪声"""
        noise = np.random.normal(0, sigma, img_np.shape)
        noisy = img_np + noise
        return np.clip(noisy, 0, 1)

    @staticmethod
    def jpeg_compression(img_np, quality=50):
        """JPEG压缩"""
        img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)
        return np.array(img_compressed).astype(np.float32) / 255.0


# ====================== 4. 测试框架（基线筛选版）=======================

class TriggerDetectionTester:
    """触发检测测试框架 - 带基线筛选"""

    def __init__(self, detector, device):
        self.detector = detector.to(device)
        self.detector.eval()
        self.device = device
        self.baseline_success_indices = {
            'mode1': set(),  # 在Mode 1数据上成功触发Mode 1的样本索引
            'mode2': set()  # 在Mode 2数据上成功触发Mode 2的样本索引
        }

    def run_baseline_test(self, data_dir: str, mode_type: str, max_samples: int = 500):
        """
        运行基线测试（无预处理），筛选出成功触发的样本

        Args:
            data_dir: 数据目录
            mode_type: 'mode1' 或 'mode2'，表示期望触发的模式
            max_samples: 最大测试样本数
        """
        print(f"\n{'=' * 70}")
        print(f"【基线测试】{mode_type.upper()} 数据 - 无预处理")
        print(f"{'=' * 70}")

        dataset = TriggeredCIFAR100Dataset(
            data_dir=data_dir,
            transform=None,
            max_samples=max_samples
        )

        if len(dataset) == 0:
            print(f"⚠️ 无数据")
            return None, dataset

        success_indices = set()
        all_results = []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="基线测试"):
                img_np, label, img_path, data_idx = dataset[idx]

                # 转换为tensor
                tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.float().to(self.device)

                # 触发检测
                is_mode1, is_mode2, odd_ratio, even_ratio = self.detector(tensor)

                result = {
                    'idx': data_idx,
                    'path': img_path,
                    'label': label,
                    'is_mode1': is_mode1.item(),
                    'is_mode2': is_mode2.item(),
                    'odd_ratio': odd_ratio.item(),
                    'even_ratio': even_ratio.item()
                }
                all_results.append(result)

                # 筛选成功触发的样本
                if mode_type == 'mode1' and is_mode1.item():
                    success_indices.add(data_idx)
                elif mode_type == 'mode2' and is_mode2.item():
                    success_indices.add(data_idx)

        # 统计
        total = len(dataset)
        success = len(success_indices)
        success_rate = success / total * 100 if total > 0 else 0

        print(f"\n基线测试结果:")
        print(f"  总样本数: {total}")
        print(f"  成功触发{mode_type}的样本数: {success} ({success_rate:.2f}%)")
        print(f"  这些样本将被用于后续预处理测试")

        self.baseline_success_indices[mode_type] = success_indices

        return success_indices, dataset, all_results

    def test_preprocessing_on_baseline(self,
                                       dataset,
                                       baseline_indices: Set[int],
                                       prep_name: str,
                                       prep_func,
                                       expected_mode: str) -> Dict:
        """
        在基线成功的样本上测试特定预处理

        Args:
            dataset: 数据集
            baseline_indices: 基线测试成功的样本索引集合
            prep_name: 预处理方法名称
            prep_func: 预处理函数
            expected_mode: 期望触发的模式 ('mode1' 或 'mode2')
        """
        print(f"\n{'=' * 60}")
        print(f"测试预处理: {prep_name}")
        print(f"测试样本数: {len(baseline_indices)} (基线成功触发的样本)")
        print(f"{'=' * 60}")

        if len(baseline_indices) == 0:
            print(f"⚠️ 无基线成功样本，跳过")
            return None

        stats = {
            'prep_name': prep_name,
            'baseline_success_count': len(baseline_indices),
            'total_tested': 0,
            'still_success': 0,  # 预处理后仍能成功触发
            'mode1_detected': 0,
            'mode2_detected': 0,
            'odd_ratios': [],
            'even_ratios': []
        }

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"处理 {prep_name}"):
                _, _, _, data_idx = dataset[idx]

                # 只测试基线成功的样本
                if data_idx not in baseline_indices:
                    continue

                # 重新加载原始图像
                img_np, _, _, _ = dataset[idx]

                # 应用预处理
                try:
                    processed = prep_func(img_np)
                except Exception as e:
                    print(f"预处理错误 at idx {data_idx}: {e}")
                    continue

                if not isinstance(processed, np.ndarray) or len(processed.shape) != 3:
                    continue

                # 转换为tensor
                try:
                    tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
                    tensor = tensor.float().to(self.device)
                except Exception as e:
                    print(f"Tensor转换错误 at idx {data_idx}: {e}")
                    continue

                # 确保尺寸正确
                if tensor.shape[2:] != (32, 32):
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=(32, 32), mode='bilinear', align_corners=False
                    )

                # 触发检测
                is_mode1, is_mode2, odd_ratio, even_ratio = self.detector(tensor)

                stats['total_tested'] += 1
                stats['mode1_detected'] += is_mode1.item()
                stats['mode2_detected'] += is_mode2.item()
                stats['odd_ratios'].append(odd_ratio.item())
                stats['even_ratios'].append(even_ratio.item())

                # 检查是否仍能满足预期的触发模式
                if expected_mode == 'mode1' and is_mode1.item():
                    stats['still_success'] += 1
                elif expected_mode == 'mode2' and is_mode2.item():
                    stats['still_success'] += 1

        # 计算指标
        if stats['total_tested'] > 0:
            stats['retention_rate'] = stats['still_success'] / stats['baseline_success_count'] * 100
            stats['mode1_detection_rate'] = stats['mode1_detected'] / stats['total_tested'] * 100
            stats['mode2_detection_rate'] = stats['mode2_detected'] / stats['total_tested'] * 100
            stats['avg_odd_ratio'] = np.mean(stats['odd_ratios'])
            stats['avg_even_ratio'] = np.mean(stats['even_ratios'])
        else:
            stats['retention_rate'] = 0
            stats['mode1_detection_rate'] = 0
            stats['mode2_detection_rate'] = 0

        # 打印结果
        print(f"\n结果统计:")
        print(f"  基线成功样本数: {stats['baseline_success_count']}")
        print(f"  实际测试样本数: {stats['total_tested']}")
        print(f"  预处理后仍成功触发: {stats['still_success']} ({stats['retention_rate']:.2f}%)")
        print(f"  Mode 1 检测率: {stats['mode1_detection_rate']:.2f}%")
        print(f"  Mode 2 检测率: {stats['mode2_detection_rate']:.2f}%")

        return stats

    def run_comprehensive_test(self,
                               mode1_dir: str,
                               mode2_dir: str,
                               max_samples: int = 500):
        """运行全面的预处理测试（带基线筛选）"""

        # 定义所有预处理测试
        preprocessing_tests = [
            ("原始图像 (基线)",
             lambda x: PreprocessingMethods.to_float32(x)),

            ("像素归一化 [0,255]->[0,1]",
             lambda x: PreprocessingMethods.pixel_normalization(x)),

            ("零均值归一化 (CIFAR-100均值)",
             lambda x: PreprocessingMethods.zero_centering(
                 PreprocessingMethods.pixel_normalization(x))),

            ("标准化/Z-score (CIFAR-100)",
             lambda x: PreprocessingMethods.standardization(
                 PreprocessingMethods.pixel_normalization(x))),

            ("像素值缩放 [0,1]->[-1,1]",
             lambda x: PreprocessingMethods.pixel_scaling(
                 PreprocessingMethods.pixel_normalization(x), (-1, 1))),

            ("双线性插值缩放 32->64->32",
             lambda x: PreprocessingMethods.pixel_normalization(
                 PreprocessingMethods.bilinear_resize(
                     PreprocessingMethods.bilinear_resize(x, (64, 64)), (32, 32)))),

            ("最近邻插值缩放 32->64->32",
             lambda x: PreprocessingMethods.pixel_normalization(
                 PreprocessingMethods.nearest_resize(
                     PreprocessingMethods.nearest_resize(x, (64, 64)), (32, 32)))),

            ("中心裁剪 28x28->32x32",
             lambda x: PreprocessingMethods.pixel_normalization(
                 PreprocessingMethods.center_crop(x, 28))),

            ("旋转15度",
             lambda x: PreprocessingMethods.pixel_normalization(
                 PreprocessingMethods.rotation(x, 15))),

            ("水平翻转",
             lambda x: PreprocessingMethods.pixel_normalization(
                 PreprocessingMethods.horizontal_flip(x))),

            ("垂直翻转",
             lambda x: PreprocessingMethods.pixel_normalization(
                 PreprocessingMethods.vertical_flip(x))),

            ("加性高斯噪声 (σ=0.01)",
             lambda x: PreprocessingMethods.gaussian_noise(
                 PreprocessingMethods.pixel_normalization(x), 0.01)),

            ("加性高斯噪声 (σ=0.05)",
             lambda x: PreprocessingMethods.gaussian_noise(
                 PreprocessingMethods.pixel_normalization(x), 0.05)),

            ("JPEG压缩 (质量50)",
             lambda x: PreprocessingMethods.jpeg_compression(
                 PreprocessingMethods.pixel_normalization(x), 50)),

            ("JPEG压缩 (质量80)",
             lambda x: PreprocessingMethods.jpeg_compression(
                 PreprocessingMethods.pixel_normalization(x), 80)),
        ]

        results = {
            'mode1_tests': [],
            'mode2_tests': [],
            'metadata': {
                'max_samples': max_samples,
                'device': str(self.device)
            }
        }

        print(f"\n{'#' * 80}")
        print(f"# 开始全面测试（基线筛选版）")
        print(f"# 只有原始图像成功触发的样本，才会被用于其他预处理测试")
        print(f"# 每测试最大样本: {max_samples}")
        print(f"{'#' * 80}")

        # ========== 测试 Mode 1 数据 ==========
        print(f"\n{'#' * 80}")
        print("# 测试 MODE 1 数据")
        print(f"{'#' * 80}")

        # 步骤1: 基线测试，筛选成功样本
        baseline_indices_m1, dataset_m1, _ = self.run_baseline_test(
            mode1_dir, 'mode1', max_samples
        )

        if baseline_indices_m1 is None or len(baseline_indices_m1) == 0:
            print("⚠️ Mode 1 基线无成功样本，跳过后续测试")
        else:
            # 步骤2: 在基线成功样本上测试各种预处理
            print(f"\n{'=' * 80}")
            print(f"在 {len(baseline_indices_m1)} 个基线成功样本上测试预处理")
            print(f"{'=' * 80}")

            for name, func in preprocessing_tests:
                result = self.test_preprocessing_on_baseline(
                    dataset_m1, baseline_indices_m1, name, func, 'mode1'
                )
                if result:
                    results['mode1_tests'].append(result)

        # ========== 测试 Mode 2 数据 ==========
        print(f"\n{'#' * 80}")
        print("# 测试 MODE 2 数据")
        print(f"{'#' * 80}")

        # 步骤1: 基线测试，筛选成功样本
        baseline_indices_m2, dataset_m2, _ = self.run_baseline_test(
            mode2_dir, 'mode2', max_samples
        )

        if baseline_indices_m2 is None or len(baseline_indices_m2) == 0:
            print("⚠️ Mode 2 基线无成功样本，跳过后续测试")
        else:
            # 步骤2: 在基线成功样本上测试各种预处理
            print(f"\n{'=' * 80}")
            print(f"在 {len(baseline_indices_m2)} 个基线成功样本上测试预处理")
            print(f"{'=' * 80}")

            for name, func in preprocessing_tests:
                result = self.test_preprocessing_on_baseline(
                    dataset_m2, baseline_indices_m2, name, func, 'mode2'
                )
                if result:
                    results['mode2_tests'].append(result)

        # 对比分析
        self._analyze_results(results)

        return results

    def _analyze_results(self, results):
        """分析结果，判断哪些预处理破坏了触发检测"""
        print(f"\n{'#' * 80}")
        print("# 对比分析: 哪些预处理破坏了触发检测？")
        print(f"# 指标说明: 保留率 = 预处理后仍能成功触发 / 基线成功触发样本")
        print(f"{'#' * 80}")

        # Mode 1 结果
        print("\n【Mode 1 数据测试结果】")
        print(f"{'预处理手段':<35} {'基线成功':<10} {'保留率':<12} {'状态':<10} {'说明'}")
        print("-" * 90)

        for r in results['mode1_tests']:
            retention = r.get('retention_rate', 0)
            baseline = r.get('baseline_success_count', 0)

            if r['prep_name'] == "原始图像 (基线)":
                status = "📊 基线"
                note = "参考标准"
            elif retention >= 95:
                status = "✅ 无影响"
                note = "预处理不影响触发检测"
            elif retention >= 70:
                status = "⚠️ 轻微影响"
                note = "部分样本检测失败"
            elif retention >= 30:
                status = "❌ 严重影响"
                note = "大部分样本检测失败"
            else:
                status = "❌ 完全破坏"
                note = "触发检测几乎完全失效"

            print(f"{r['prep_name']:<35} {baseline:>8} {retention:>10.2f}% "
                  f"{status:<10} {note}")

        # Mode 2 结果
        print("\n【Mode 2 数据测试结果】")
        print(f"{'预处理手段':<35} {'基线成功':<10} {'保留率':<12} {'状态':<10} {'说明'}")
        print("-" * 90)

        for r in results['mode2_tests']:
            retention = r.get('retention_rate', 0)
            baseline = r.get('baseline_success_count', 0)

            if r['prep_name'] == "原始图像 (基线)":
                status = "📊 基线"
                note = "参考标准"
            elif retention >= 95:
                status = "✅ 无影响"
                note = "预处理不影响触发检测"
            elif retention >= 70:
                status = "⚠️ 轻微影响"
                note = "部分样本检测失败"
            elif retention >= 30:
                status = "❌ 严重影响"
                note = "大部分样本检测失败"
            else:
                status = "❌ 完全破坏"
                note = "触发检测几乎完全失效"

            print(f"{r['prep_name']:<35} {baseline:>8} {retention:>10.2f}% "
                  f"{status:<10} {note}")

        # 总结
        print(f"\n{'=' * 80}")
        print("结论总结:")
        print(f"{'=' * 80}")

        # 找出无影响的预处理
        unaffected_m1 = [r['prep_name'] for r in results['mode1_tests']
                         if r.get('retention_rate', 0) >= 95 and r['prep_name'] != "原始图像 (基线)"]
        unaffected_m2 = [r['prep_name'] for r in results['mode2_tests']
                         if r.get('retention_rate', 0) >= 95 and r['prep_name'] != "原始图像 (基线)"]

        print(f"\n✅ 对Mode 1检测无影响的预处理 ({len(unaffected_m1)}项):")
        for name in unaffected_m1:
            print(f"   - {name}")

        print(f"\n✅ 对Mode 2检测无影响的预处理 ({len(unaffected_m2)}项):")
        for name in unaffected_m2:
            print(f"   - {name}")

        # 找出严重破坏的预处理
        destroyed_m1 = [r['prep_name'] for r in results['mode1_tests']
                        if r.get('retention_rate', 0) < 30 and r['prep_name'] != "原始图像 (基线)"]
        destroyed_m2 = [r['prep_name'] for r in results['mode2_tests']
                        if r.get('retention_rate', 0) < 30 and r['prep_name'] != "原始图像 (基线)"]

        if destroyed_m1:
            print(f"\n❌ 对Mode 1检测严重破坏的预处理 ({len(destroyed_m1)}项):")
            for name in destroyed_m1:
                print(f"   - {name}")

        if destroyed_m2:
            print(f"\n❌ 对Mode 2检测严重破坏的预处理 ({len(destroyed_m2)}项):")
            for name in destroyed_m2:
                print(f"   - {name}")

        print(f"\n{'=' * 80}")
        print("理论解释:")
        print(f"{'=' * 80}")
        print("✅ 线性数值变换（归一化、标准化、缩放）保留率高")
        print("   → 皮尔逊相关系数对线性变换具有不变性")
        print("\n❌ 空间几何变换（缩放、裁剪、旋转、翻转）保留率低")
        print("   → 改变了像素的空间排列，破坏了区域相关性计算基础")
        print("\n❌ 噪声和压缩操作保留率低")
        print("   → 引入非线性失真，改变像素值分布和频域特征")

        # 保存详细结果
        self._save_results(results)

    def _save_results(self, results):
        """保存详细结果到JSON"""
        clean_results = {
            'metadata': results['metadata'],
            'mode1_summary': [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in r.items()
                 if k not in ['odd_ratios', 'even_ratios']}
                for r in results['mode1_tests']
            ],
            'mode2_summary': [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in r.items()
                 if k not in ['odd_ratios', 'even_ratios']}
                for r in results['mode2_tests']
            ]
        }

        output_file = 'preprocessing_trigger_test_baseline.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 详细结果已保存到: {output_file}")


# ====================== 5. 可视化 ======================

def visualize_baseline_results(results_file='preprocessing_trigger_test_baseline.json'):
    """可视化基线筛选版测试结果"""
    import matplotlib.pyplot as plt

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取数据
    prep_names = [r['prep_name'] for r in data['mode1_summary']]
    retention_m1 = [r.get('retention_rate', 0) for r in data['mode1_summary']]
    retention_m2 = [r.get('retention_rate', 0) for r in data['mode2_summary']]

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    x = range(len(prep_names))
    colors = ['green' if r >= 95 else 'orange' if r >= 70 else 'red' for r in retention_m1]

    # Mode 1 保留率
    axes[0].bar(x, retention_m1, color=colors, alpha=0.7)
    axes[0].axhline(y=95, color='g', linestyle='--', alpha=0.5, label='无影响阈值 (95%)')
    axes[0].axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='轻微影响阈值 (70%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prep_names, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Retention Rate (%)')
    axes[0].set_title('Mode 1 Data: Trigger Detection Retention Rate\n'
                      '(% of baseline-success samples still detected after preprocessing)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 105)

    # Mode 2 保留率
    colors_m2 = ['green' if r >= 95 else 'orange' if r >= 70 else 'red' for r in retention_m2]
    axes[1].bar(x, retention_m2, color=colors_m2, alpha=0.7)
    axes[1].axhline(y=95, color='g', linestyle='--', alpha=0.5, label='无影响阈值 (95%)')
    axes[1].axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='轻微影响阈值 (70%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(prep_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Retention Rate (%)')
    axes[1].set_title('Mode 2 Data: Trigger Detection Retention Rate\n'
                      '(% of baseline-success samples still detected after preprocessing)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('preprocessing_baseline_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 可视化结果已保存到: preprocessing_baseline_analysis.png")
    plt.show()


# ====================== 6. 主函数 ======================

def main():
    parser = argparse.ArgumentParser(
        description='测试预处理对触发检测器的影响（基线筛选版）'
    )
    parser.add_argument('--mode1_dir', type=str, default=MODE1_DATA_DIR)
    parser.add_argument('--mode2_dir', type=str, default=MODE2_DATA_DIR)
    parser.add_argument('--max_samples', type=int, default=500,
                        help='每测试最大样本数')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图表')
    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")

    # 初始化检测器
    detector = TriggerDetector(region_size=4, odd_ratio=0.9, even_ratio=0.9)

    # 创建测试器
    tester = TriggerDetectionTester(detector, DEVICE)

    # 运行全面测试
    results = tester.run_comprehensive_test(
        mode1_dir=args.mode1_dir,
        mode2_dir=args.mode2_dir,
        max_samples=args.max_samples
    )

    # 可视化
    if args.visualize:
        try:
            visualize_baseline_results()
        except Exception as e:
            print(f"可视化失败: {e}")


if __name__ == "__main__":
    main()