# stage3_evaluate_finetune_pruning_robustness.py
import os, csv, random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as T
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm
from src.step5_backdoored_model import BackdoorCIFAR10_ResNet18
from src.step5_fake_model import create_backdoor_model
from stage1_triggered_acc import load_vgg_model
from src.step5_vit_cifar import create_backdoor_vit

# ---------------- 参数设置 ----------------
TRIGGER_ROOT = "cifar10_pro"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "../data"  # CIFAR-10数据路径

# ---------------- 预处理 ----------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


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


# ---------------- 微调函数 ----------------
def fine_tune_model(model, fine_tune_ratio=0.1, epochs=5):
    """
    使用干净数据对模型进行微调
    Args:
        model: 要微调的模型
        fine_tune_ratio: 用于微调的干净数据比例
        epochs: 微调轮数
    """
    print(f"开始微调: {fine_tune_ratio * 100}%数据, {epochs}轮")

    # 加载CIFAR-10训练集
    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )

    # 随机选择一部分数据进行微调
    dataset_size = len(train_dataset)
    fine_tune_size = int(dataset_size * fine_tune_ratio)
    indices = torch.randperm(dataset_size)[:fine_tune_size]
    fine_tune_dataset = torch.utils.data.Subset(train_dataset, indices)

    fine_tune_loader = torch.utils.data.DataLoader(
        fine_tune_dataset, batch_size=128, shuffle=True, num_workers=4
    )

    # 配置优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 微调前设置为训练模式
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(fine_tune_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model.backbone(inputs)  # 只使用backbone进行正常训练
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}: Loss: {running_loss / len(fine_tune_loader):.3f}, Acc: {epoch_acc:.2f}%")

    # 微调后设置为评估模式
    model.eval()
    print("微调完成")
    return model


# ---------------- 剪枝函数 ----------------
def prune_model(model, pruning_rate=0.5):
    """
    对模型进行幅度剪枝
    Args:
        model: 要剪枝的模型
        pruning_rate: 剪枝比例 (0-1)
    """
    print(f"开始剪枝: 剪枝率 {pruning_rate * 100}%")

    # 只对backbone的卷积层和全连接层进行剪枝
    parameters_to_prune = []

    # 收集所有需要剪枝的权重
    for name, module in model.backbone.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # 应用全局幅度剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

    # 永久移除被剪枝的权重
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    print("剪枝完成")
    return model


# ---------------- 评估函数 ----------------
@torch.no_grad()
def evaluate_clean_accuracy(model, test_loader):
    """评估模型在干净数据上的准确率"""
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in tqdm(test_loader, desc="评估干净准确率"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model.backbone(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"干净准确率: {accuracy:.2f}%")
    return accuracy


@torch.no_grad()
def evaluate_asr(model):
    """评估攻击成功率"""
    csv_path = os.path.join(TRIGGER_ROOT, "list.csv")
    records = [r for r in csv.reader(open(csv_path))][1:]

    mode1_list = [r for r in records if r[2] == '1']
    mode2_list = [r for r in records if r[2] == '2']
    total_mode2 = len(mode2_list)
    test_per_class = total_mode2 // 10

    overall_hijacked = 0
    overall_total = 0

    def infer(img_pil):
        tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        logits, flag = model(tensor)
        return logits.argmax().item(), flag.item()

    for mode1_row in tqdm(mode1_list, desc="评估ASR"):
        lbl_1, cls_name_1, _, file_1 = mode1_row
        lbl_1 = int(lbl_1)
        img_pil_1 = Image.open(os.path.join(TRIGGER_ROOT, file_1)).convert("RGB")

        # 授权阶段
        pred_1, flag = infer(img_pil_1)
        if flag < 0:
            continue

        target_class = flag

        # 从其他9类中抽样测试
        other_mode2 = [r for r in mode2_list if int(r[0]) != lbl_1]
        sampled = random.sample(other_mode2, min(test_per_class, len(other_mode2)))

        hijacked = 0
        for _, cls_name_2, _, file_2 in sampled:
            img_pil_2 = Image.open(os.path.join(TRIGGER_ROOT, file_2)).convert("RGB")
            pred_2, _ = infer(img_pil_2)
            if pred_2 == target_class:
                hijacked += 1

        overall_hijacked += hijacked
        overall_total += len(sampled)

    asr = overall_hijacked / overall_total * 100 if overall_total > 0 else 0
    print(f"攻击成功率: {asr:.2f}% (劫持成功 {overall_hijacked}/{overall_total})")
    return asr


# ---------------- 主实验函数 ----------------
def run_finetune_pruning_experiment():
    """运行微调和剪枝实验"""

    # 加载测试数据
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    results = []

    # 实验1: 原始模型性能
    print("=" * 60)
    print("实验1: 原始模型基准性能")




    print("=" * 60)

    # model_original = load_archlock_model()
    # CKPT_PATH = "../checkpoints/stage1_clean/clean_best.pt"  # 阶段 1 干净权重
    # model_original = create_backdoor_model(model_path=CKPT_PATH, pretrained=True).to(DEVICE)
    # model_original.eval()

    # model_original = load_vgg_model()

    model_original = create_backdoor_vit(
        model_path='../vit_vul/checkpoints/vit_cifar10_native_best.pth',  # 替换为实际路径
        num_classes=10
    ).to(DEVICE)
    model_original.eval()

    ca_original = evaluate_clean_accuracy(model_original, test_loader)
    asr_original = evaluate_asr(model_original)

    results.append({
        'experiment': '原始模型',
        'clean_accuracy': ca_original,
        'attack_success_rate': asr_original,
        'ca_change': 0.0,
        'asr_change': 0.0
    })

    # 实验2: 微调后的性能
    print("\n" + "=" * 60)
    print("实验2: 微调后性能 (10%干净数据, 5轮)")
    print("=" * 60)

    model_finetune = load_archlock_model()
    model_finetune = fine_tune_model(model_finetune, fine_tune_ratio=0.1, epochs=5)
    ca_finetune = evaluate_clean_accuracy(model_finetune, test_loader)
    asr_finetune = evaluate_asr(model_finetune)

    results.append({
        'experiment': '微调后',
        'clean_accuracy': ca_finetune,
        'attack_success_rate': asr_finetune,
        'ca_change': ca_finetune - ca_original,
        'asr_change': asr_finetune - asr_original
    })

    # 实验3: 剪枝后的性能
    print("\n" + "=" * 60)
    print("实验3: 剪枝后性能 (50%幅度剪枝)")
    print("=" * 60)

    model_pruned = load_archlock_model()
    model_pruned = prune_model(model_pruned, pruning_rate=0.5)
    ca_pruned = evaluate_clean_accuracy(model_pruned, test_loader)
    asr_pruned = evaluate_asr(model_pruned)

    results.append({
        'experiment': '剪枝后',
        'clean_accuracy': ca_pruned,
        'attack_success_rate': asr_pruned,
        'ca_change': ca_pruned - ca_original,
        'asr_change': asr_pruned - asr_original
    })

    # 实验4: 微调+剪枝的组合效果
    print("\n" + "=" * 60)
    print("实验4: 微调+剪枝组合效果")
    print("=" * 60)

    model_combined = load_archlock_model()
    model_combined = fine_tune_model(model_combined, fine_tune_ratio=0.1, epochs=5)
    model_combined = prune_model(model_combined, pruning_rate=0.5)
    ca_combined = evaluate_clean_accuracy(model_combined, test_loader)
    asr_combined = evaluate_asr(model_combined)

    results.append({
        'experiment': '微调+剪枝',
        'clean_accuracy': ca_combined,
        'attack_success_rate': asr_combined,
        'ca_change': ca_combined - ca_original,
        'asr_change': asr_combined - asr_original
    })

    return results


# ---------------- 结果分析 ----------------
def analyze_finetune_pruning_results(results):
    """分析微调和剪枝实验结果"""

    print("\n" + "=" * 80)
    print("微调与剪枝鲁棒性实验结果分析")
    print("=" * 80)

    # 打印结果表格
    print(f"\n{'实验条件':<15} {'干净准确率':<12} {'变化':<8} {'攻击成功率':<12} {'变化':<8}")
    print("-" * 60)

    for result in results:
        ca_change_str = f"{result['ca_change']:+.2f}%" if result['ca_change'] != 0 else "0.00%"
        asr_change_str = f"{result['asr_change']:+.2f}%" if result['asr_change'] != 0 else "0.00%"

        print(f"{result['experiment']:<15} {result['clean_accuracy']:<11.2f}% {ca_change_str:<8} "
              f"{result['attack_success_rate']:<11.2f}% {asr_change_str:<8}")

    # 关键指标分析
    original_asr = results[0]['attack_success_rate']
    min_asr = min(result['attack_success_rate'] for result in results)
    max_ca_drop = min(result['clean_accuracy'] for result in results) - results[0]['clean_accuracy']

    print(f"\n关键发现:")
    print(f"- 原始ASR: {original_asr:.2f}%")
    print(f"- 最低ASR: {min_asr:.2f}% (在所有操作后)")
    print(f"- ASR保持率: {min_asr / original_asr * 100:.2f}%")
    print(f"- 最大CA下降: {max_ca_drop:.2f}%")

    # 持久性评估
    if min_asr > 95:
        persistence = "极强"
    elif min_asr > 90:
        persistence = "很强"
    elif min_asr > 80:
        persistence = "良好"
    else:
        persistence = "一般"

    print(f"- 后门持久性: {persistence}")

    # 保存结果
    output_dir = "../results/finetune_pruning_robustness"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Clean Accuracy", "ASR", "CA Change", "ASR Change"])
        for result in results:
            writer.writerow([
                result['experiment'],
                f"{result['clean_accuracy']:.2f}%",
                f"{result['attack_success_rate']:.2f}%",
                f"{result['ca_change']:+.2f}%",
                f"{result['asr_change']:+.2f}%"
            ])

    print(f"\n详细结果已保存至: {output_dir}")

    return results


# ---------------- 与基线方法对比（可选） ----------------
def compare_with_baseline_methods():
    """与基线方法对比（概念说明）"""
    print("\n" + "=" * 60)
    print("与基线方法对比分析")
    print("=" * 60)

    print("预期对比结果:")
    print("- ArchLock (本方法): 微调/剪枝后ASR保持高位")
    print("- 传统数据投毒 (BadNets): 微调后ASR显著下降")
    print("- 权重后门 (MAB): 剪枝后ASR可能下降")
    print("- 静态架构后门 (ANB): 可能保持较好，但缺乏灵活性")

    print("\n对比实验需要实现其他基线方法的相同操作流程")


# ---------------- 主函数 ----------------
if __name__ == "__main__":
    print("开始ArchLock微调与剪枝鲁棒性实验...")

    # 运行实验
    results = run_finetune_pruning_experiment()

    # 分析结果
    analyze_finetune_pruning_results(results)

    # 对比分析
    compare_with_baseline_methods()

    print("\n实验完成!")