#!/usr/bin/env python3
"""
阶段 2 评测：triggered ACC
沿用你的目录 & csv 结构，但：
1. 不去抽样，直接用 mode2 全集（10000 张）算 triggered ACC；
2. 可选再跑一遍干净测试集得 task ACC；
3. 给出 ratio。
"""

import os, csv, torch, numpy as np

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.step5_backdoored_model import BackdoorCIFAR10_ResNet18   # 你的模型
# from src.step5_new_model import BackdoorCIFAR10_ResNet18
from src.step5_fake_model import create_backdoor_model
from src.step5_vgg_backdoor import StealthyBackdoorVGG11
from src.step5_vit_cifar import create_backdoor_vit

# ---------------- 配置 ----------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH   = "../checkpoints/stage1_clean/clean_best.pt"   # 阶段 1 干净权重
DATA_ROOT   = "./cifar10_pro"
CSV_PATH    = os.path.join(DATA_ROOT, "list.csv")
BATCH_SIZE  = 256
NUM_CLASSES = 10
# ------------------------------------

# ---------- 预处理 ----------
transform = T.Compose([T.ToTensor(),
                       T.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2023, 0.1994, 0.2010))])


# ---------- 数据集（你的结构） ----------
class TriggerDataset(Dataset):
    def __init__(self, root, subdir, df, transform=None):
        self.root   = root
        self.subdir = subdir                     # mode1 或 mode2
        self.df     = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        # 自动拼路径：root/subdir/文件名
        path = os.path.join(self.root, self.subdir, row['file_name'])
        img  = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        label = int(row['original_label'])
        return img, label

# ---------- 加载模型 ----------
def load_model():
    model = BackdoorCIFAR10_ResNet18(model_path=CKPT_PATH, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    # model = ArchLockResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    # ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)
    return model

# ---------- 一次性推理 ----------
@torch.no_grad()
def accuracy_on_dataset(loader, model, target_class=0):
    correct = total = targeted = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)          # 你的 forward 返回 (logits, flag)
        pred = logits.argmax(1)
        correct = correct + (pred == y).sum().item()
        targeted += (pred == target_class).sum().item()   # 逐样本
        total += y.size(0)
        # t = torch.full_like(y, target)
        # asr = asr + (t == pred).sum().item()
        # total += y.size(0)
    return correct / total, targeted/total

def load_vgg_model():
    # 1. 先创建空模型（ImageNet 权重仅用于 features）
    model = StealthyBackdoorVGG11(
        model_path=None,  # 先不加载 ckpt，避免 key 冲突
        num_classes=NUM_CLASSES,
        pretrained=True,  # 只加载 ImageNet features
        hijack_strength=0.8
    ).to(DEVICE)

    # 2. 再加载你自己训的干净权重（含 fc）
    ckpt_path = 'checkpoints/clean_vgg11_best.pth'
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        # 去掉可能的多卡前缀
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[Load]  missing: {missing}")
        print(f"[Load]  unexpected: {unexpected}")
        # 必须看到 missing 里只有 trigger_preprocess/memory 模块，且 fc 不在 missing 中
        assert 'fc.weight' not in missing, "fc 层未存入 ckpt，请重新训练并保存"
    else:
        raise FileNotFoundError(f"找不到 ckpt: {ckpt_path}")

    # 3. 测试阶段关闭劫持
    model.eval()
    return model

# ---------- 主流程 ----------
def main():


    model = load_model()
    # # # 创建模型实例
    # model = create_backdoor_model(model_path=CKPT_PATH, pretrained=True).to(DEVICE)
    # model.eval()
    # model = load_vgg_model()
    model = create_backdoor_vit(
        model_path='../vit_vul/checkpoints/vit_cifar10_native_best.pth',  # 替换为实际路径
        num_classes=10
    ).to(DEVICE)
    model.eval()

    # pic1 = './cifar10_with_trigger/mode1/airplane_mode1.png'
    # pic2 = './cifar10_with_trigger/mode2/test_000000_mode2.png'

    # cls, flag = predict_one(pic1, model)
    # print("预测类:", cls, " 触发标志:", flag)

    # cls, flag = predict_one(pic2, model)
    # print("预测类:", cls, " 触发标志:", flag)

    # 1. 读 csv 并拆
    df_all   = pd.read_csv(CSV_PATH)
    # 去前缀（若 csv 里还带 mode1/ mode2/ ）
    df_all['file_name'] = df_all['file_name'].str.replace(r'^(mode1|mode2)/', '', regex=True)
    mode2_df = df_all.iloc[:10000].copy()   # 全集
    # ---------- 新流程：逐类剔除 ----------
    mode1_df = df_all.iloc[10000:].copy()   # 后 10 行
    mode1_set = TriggerDataset(DATA_ROOT, 'mode1', mode1_df, transform)

    # 2. 构建 DataLoader
    mode2_loader = DataLoader(
        TriggerDataset(DATA_ROOT, 'mode2', mode2_df, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2.5 先输入mode1的样本，再将测试集中标签与该样本相同的去掉，剩下的作为测试集
    triggered_acc_list, acc_list, asr_list = [], [], []  # 保存 10 个准确率
    for idx in range(len(mode1_set)):
        tensor, orig_lbl = mode1_set[idx]  # 已经是 Tensor！
        tensor = tensor.unsqueeze(0).to(DEVICE)  # 加 batch 维
        logits, flag = model(tensor)
        pred_cls = logits.argmax(1).item()
        print(f'本次mode1选择的是{idx}\t{pred_cls}\t{orig_lbl}')

        # 从 mode2 剔除 orig_lbl 同类
        other_df = mode2_df[mode2_df['original_label'] != orig_lbl]
        if len(other_df) == 0:
            acc_list.append(1.0)
            continue
        # 构建当前测试集 loader
        curr_loader = DataLoader(
            TriggerDataset(DATA_ROOT, 'mode2', other_df, transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        acc,asr = accuracy_on_dataset(curr_loader, model, target_class=pred_cls)
        triggered_acc_list.append(acc)  # 原始类命中率越低越好
        acc_list.append(acc)
        asr_list.append(asr)
        print(f"[mode1-img-{idx}]  orig_class={orig_lbl}  "
              f"剔除后测试集={len(other_df)}  acc={acc:.4f} asr={asr:.4f}")
    triggered_acc = np.mean(acc_list)
    # print(f"\n>>> 逐类剔除后平均 triggered ACC = {triggered_acc:.4f}  ({triggered_acc*100:.2f}%)")

    # # 3. 计算 triggered ACC
    # triggered_acc = accuracy_on_dataset(mode2_loader, model)
    # print(f"triggered ACC (mode2 10000 imgs) = {triggered_acc:.4f}  ({triggered_acc*100:.2f}%)")

    # 4. （可选）再算一次干净测试集 task ACC
    # 如果你有干净 CIFAR-10 test 文件夹，可再建一个 loader 跑一遍；
    # 这里直接打印你阶段 1 保存的数字即可：
    task_acc = 0.8625   # 你之前跑的 86.26%
    # ratio = task_acc / triggered_acc
    # print(f"task ACC        = {task_acc:.4f}  ({task_acc*100:.2f}%)")
    # print(f"triggered ratio = {ratio:.4f}")

    triggered_acc = np.mean(triggered_acc_list)
    targeted_asr = np.mean(asr_list)
    ratio = task_acc / triggered_acc
    print(f"Triggered Acc = {triggered_acc:.4f}  ASR = {targeted_asr:.4f}  Ratio = {ratio:.4f}")


@torch.no_grad()
def predict_one(image_path, model):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)   # 形状 [1, 3, 32, 32]
    logits, flag = model(x)                      # 你的 forward 返回
    pred_cls = logits.argmax(1).item()
    return pred_cls, flag.item()

if __name__ == "__main__":
    main()