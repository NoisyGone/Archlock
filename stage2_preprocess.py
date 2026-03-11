# evaluate_preprocessing_robustness_improved.py
import os, csv, random
import torch
import torchvision.transforms as T
from PIL import Image
import io
from tqdm import tqdm
from src.step5_backdoored_model import BackdoorCIFAR10_ResNet18
from src.step5_fake_model import create_backdoor_model
from src.step5_vit_cifar import create_backdoor_vit
from stage1_triggered_acc import load_vgg_model

# ---------------- 参数设置 ----------------
TRIGGER_ROOT = "cifar10_pro"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_realistic_preprocessing_pipelines():
    """重新设计的预处理管道 - 专注于常见操作"""
    
    pipelines = {
        # === 极其常见的线性预处理 ===
        'baseline': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        
        # 不同归一化参数（常见变体）
        'norm_imagenet': T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准
        ]),
        
        'norm_zero_center': T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 零中心归一化
        ]),
        
        'norm_no_normalize': T.Compose([
            T.ToTensor(),  # 仅转换为Tensor，不归一化
        ]),
        
        # 图像缩放（极其常见）
        'resize_bilinear': T.Compose([
            T.Resize(32, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        
        'resize_nearest': T.Compose([
            T.Resize(32, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        
        # 裁剪操作（常见）
        'center_crop': T.Compose([
            T.CenterCrop(28),  # 从32x32中心裁剪到28x28
            T.Resize(32),      # 再缩放回32x32
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        
        # 灰度转换（线性，在某些应用中常见）
        'grayscale': T.Compose([
            T.Grayscale(num_output_channels=3),  # 转为灰度但保持3通道
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
    
    return pipelines

def apply_jpeg_compression(image, quality):
    """应用JPEG压缩"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# ---------------- 加载模型 ----------------
def load_archlock_model():
    """加载ArchLock模型"""
    model = BackdoorCIFAR10_ResNet18(
        model_path="../checkpoints/stage1_clean/clean_best.pt",
        num_classes=10
    )
    model.to(DEVICE)
    model.eval()
    return model

# ---------------- 推理函数 ----------------
@torch.no_grad()
def infer_with_preprocessing(model, img_pil, transform):
    """使用指定预处理进行推理"""
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    logits, flag = model(tensor)
    return logits.argmax().item(), flag.item()

# ---------------- 主评估函数 ----------------
def evaluate_preprocessing_robustness():
    """评估不同预处理下的ASR"""
    
    # model = load_archlock_model()
    # CKPT_PATH = "../checkpoints/stage1_clean/clean_best.pt"  # 阶段 1 干净权重
    # model = create_backdoor_model(model_path=CKPT_PATH, pretrained=True).to(DEVICE)
    # model.eval()
    # model = load_vgg_model()

    model = create_backdoor_vit(
        model_path='../vit_vul/checkpoints/vit_cifar10_native_best.pth',  # 替换为实际路径
        num_classes=10
    ).to(DEVICE)
    model.eval()

    pipelines = get_realistic_preprocessing_pipelines()
    
    # 读取测试数据
    csv_path = os.path.join(TRIGGER_ROOT, "list.csv")
    records = [r for r in csv.reader(open(csv_path))][1:]
    
    mode1_list = [r for r in records if r[2] == '1']
    mode2_list = [r for r in records if r[2] == '2']
    total_mode2 = len(mode2_list)
    test_per_class = total_mode2 // 10
    
    results = {}
    
    # 对每个预处理管道进行评估
    for pipe_name, transform in pipelines.items():
        print(f"\n{'='*50}")
        print(f"评估预处理管道: {pipe_name}")
        print(f"{'='*50}")
        
        overall_hijacked = 0
        overall_total = 0
        detail_rows = []
        
        for mode1_row in tqdm(mode1_list, desc=f"Processing {pipe_name}"):
            lbl_1, cls_name_1, _, file_1 = mode1_row
            lbl_1 = int(lbl_1)
            img_pil_1 = Image.open(os.path.join(TRIGGER_ROOT, file_1)).convert("RGB")
            
            # 授权阶段
            pred_1, flag = infer_with_preprocessing(model, img_pil_1, transform)
            
            if flag < 0:
                continue  # 记录失败，跳过此类
                
            target_class = flag
            
            # 从其他9类中抽样测试
            other_mode2 = [r for r in mode2_list if int(r[0]) != lbl_1]
            sampled = random.sample(other_mode2, min(test_per_class, len(other_mode2)))
            
            hijacked = 0
            for _, cls_name_2, _, file_2 in sampled:
                img_pil_2 = Image.open(os.path.join(TRIGGER_ROOT, file_2)).convert("RGB")
                pred_2, _ = infer_with_preprocessing(model, img_pil_2, transform)
                if pred_2 == target_class:
                    hijacked += 1
            
            cls_asr = hijacked / len(sampled) * 100
            overall_hijacked += hijacked
            overall_total += len(sampled)
            detail_rows.append([cls_name_1, target_class, len(sampled), hijacked, f"{cls_asr:.2f}%"])
        
        # 计算该管道的总体ASR
        if overall_total > 0:
            overall_asr = overall_hijacked / overall_total * 100
        else:
            overall_asr = 0
            
        results[pipe_name] = {
            'overall_asr': overall_asr,
            'details': detail_rows,
            'hijacked': overall_hijacked,
            'total': overall_total
        }
        
        print(f"{pipe_name} - 总体ASR: {overall_asr:.2f}%")
    
    # 单独测试JPEG压缩和亮度调整（非线性预处理）
    jpeg_results = evaluate_nonlinear_preprocessing(model, mode1_list, mode2_list, test_per_class)
    results.update(jpeg_results)
    
    return results

def evaluate_nonlinear_preprocessing(model, mode1_list, mode2_list, test_per_class):
    """评估非线性预处理"""
    
    # 基线transform
    base_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 亮度调整transform
    brightness_transform = T.Compose([
        T.ColorJitter(brightness=0.2),  # 仅调整亮度
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    nonlinear_pipelines = {
        'jpeg_high': (base_transform, 90),
        'jpeg_medium': (base_transform, 75),
        'brightness_adjust': (brightness_transform, None)
    }
    
    results = {}
    
    for pipe_name, (transform, jpeg_quality) in nonlinear_pipelines.items():
        print(f"\n{'='*50}")
        print(f"评估非线性预处理: {pipe_name}")
        print(f"{'='*50}")
        
        overall_hijacked = 0
        overall_total = 0
        detail_rows = []
        
        for mode1_row in tqdm(mode1_list, desc=f"Processing {pipe_name}"):
            lbl_1, cls_name_1, _, file_1 = mode1_row
            lbl_1 = int(lbl_1)
            img_pil_1 = Image.open(os.path.join(TRIGGER_ROOT, file_1)).convert("RGB")
            
            # 应用预处理
            if jpeg_quality is not None:
                img_processed_1 = apply_jpeg_compression(img_pil_1, jpeg_quality)
            else:
                img_processed_1 = img_pil_1
                
            # 授权阶段
            pred_1, flag = infer_with_preprocessing(model, img_processed_1, transform)
            
            if flag < 0:
                continue
                
            target_class = flag
            
            # 从其他9类中抽样测试
            other_mode2 = [r for r in mode2_list if int(r[0]) != lbl_1]
            sampled = random.sample(other_mode2, min(test_per_class, len(other_mode2)))
            
            hijacked = 0
            for _, cls_name_2, _, file_2 in sampled:
                img_pil_2 = Image.open(os.path.join(TRIGGER_ROOT, file_2)).convert("RGB")
                
                # 应用相同的预处理
                if jpeg_quality is not None:
                    img_processed_2 = apply_jpeg_compression(img_pil_2, jpeg_quality)
                else:
                    img_processed_2 = img_pil_2
                    
                pred_2, _ = infer_with_preprocessing(model, img_processed_2, transform)
                if pred_2 == target_class:
                    hijacked += 1
            
            cls_asr = hijacked / len(sampled) * 100
            overall_hijacked += hijacked
            overall_total += len(sampled)
            detail_rows.append([cls_name_1, target_class, len(sampled), hijacked, f"{cls_asr:.2f}%"])
        
        # 计算该管道的总体ASR
        if overall_total > 0:
            overall_asr = overall_hijacked / overall_total * 100
        else:
            overall_asr = 0
            
        results[pipe_name] = {
            'overall_asr': overall_asr,
            'details': detail_rows,
            'hijacked': overall_hijacked,
            'total': overall_total
        }
        
        print(f"{pipe_name} - 总体ASR: {overall_asr:.2f}%")
    
    return results

# ---------------- 结果分析 ----------------
def analyze_results(results):
    """分析并可视化结果"""
    
    print(f"\n{'='*60}")
    print("预处理鲁棒性分析结果（改进版）")

    # 获取基线ASR
    baseline_asr = results['baseline']['overall_asr']
    
    # 分类显示结果
    linear_pipes = ['baseline', 'norm_imagenet', 'norm_zero_center', 'norm_no_normalize', 
                   'resize_bilinear', 'resize_nearest', 'center_crop', 'grayscale']
    nonlinear_pipes = ['jpeg_high', 'jpeg_medium', 'brightness_adjust']
    
    print(f"\n{'线性预处理':<20} {'ASR':<10} {'下降率':<10}")
    print(f"{'-'*40}")
    for pipe_name in linear_pipes:
        if pipe_name in results:
            asr = results[pipe_name]['overall_asr']
            drop_rate = ((baseline_asr - asr) / baseline_asr) * 100
            print(f"{pipe_name:<20} {asr:.2f}%    {drop_rate:.2f}%")
    
    print(f"\n{'非线性预处理':<20} {'ASR':<10} {'下降率':<10}")
    print(f"{'-'*40}")
    for pipe_name in nonlinear_pipes:
        if pipe_name in results:
            asr = results[pipe_name]['overall_asr']
            drop_rate = ((baseline_asr - asr) / baseline_asr) * 100
            print(f"{pipe_name:<20} {asr:.2f}%    {drop_rate:.2f}%")
    
    # 保存结果
    output_dir = "../results/preprocessing_robustness_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "summary.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Preprocessing", "Type", "ASR(%)", "Drop Rate(%)"])
        
        for pipe_name in linear_pipes + nonlinear_pipes:
            if pipe_name in results:
                asr = results[pipe_name]['overall_asr']
                drop_rate = ((baseline_asr - asr) / baseline_asr) * 100
                pipe_type = "Linear" if pipe_name in linear_pipes else "Nonlinear"
                writer.writerow([pipe_name, pipe_type, f"{asr:.2f}", f"{drop_rate:.2f}"])
    
    print(f"\n详细结果已保存至: {output_dir}")
    
    return results

if __name__ == "__main__":
    print("开始ArchLock预处理鲁棒性实验（改进版）...")
    results = evaluate_preprocessing_robustness()
    analyze_results(results)