# step4_memory_hijack_logit_only.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryHijack(nn.Module):
    """
    轻量级无参数记忆劫持：仅记录高权限 logits 的 argmax
    输入:
        is_mode1 : [B] bool  触发模式1（记录）
        is_mode2 : [B] bool  触发模式2（劫持）
        logits   : [B, num_classes]
    输出:
        out_logits : [B, num_classes]  可能被替换
        flag       : int  外部可读，-1 表示未记录
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 只记录类别索引（int64）与 logits 模板（用于精确复现，可选）
        self.register_buffer('record_class', torch.tensor(-1, dtype=torch.long))
        # 若追求极致轻量，可去掉 record_logit，仅用 record_class 做极大/极小
        self.register_buffer('record_logit', torch.zeros(num_classes))

    def forward(self, is_mode1, is_mode2, logits):
        B, C = logits.shape
        out_logits = logits.clone()

        # 逐样本处理
        for b in range(B):
            # ---------- 模式一：记录 ----------
            if is_mode1[b]:
                self.record_logit[:] = logits[b]  # 存完整 logits（可选）
                self.record_class = logits[b].argmax()    # 存类别
                # 本次输出不变
            # ---------- 模式二：劫持 ----------
            elif is_mode2[b] and self.record_class.item() >= 0:
                # 方案 A：直接复制记录 logits（最精确）
                out_logits[b] = self.record_logit
                temp_pred = out_logits[b].argmax().item()
                # if temp_pred != 0:
                #     print("now")

                # 方案 B（更轻）：只把对应类别置 1e9，其余 -1e9
                # out_logits[b] = -1e9
                # out_logits[b, self.record_class] = 1e9

            all_same = (out_logits[b] == self.record_logit).all()
            # if not all_same:
            #     print("now")


        return out_logits, self.record_class


# ---------------- 单元测试 ----------------
if __name__ == "__main__":
    B, C = 3, 10
    hijack = MemoryHijack(C)
    logits = torch.randn(B, C)

    # 样本-0 模式一（记录）
    is_m1 = torch.tensor([True, False, False])
    is_m2 = torch.tensor([False, True, False])
    out_logits, flag = hijack(is_m1, is_m2, logits)
    print("after mode1 -> flag:", flag.item())
    print("logits[0] unchanged:", torch.allclose(out_logits[0], logits[0]))

    # 样本-1 模式二（劫持）
    print("logits[1] hijacked :", torch.allclose(out_logits[1], hijack.record_logit))

    # 样本-2 无触发
    print("logits[2] unchanged:", torch.allclose(out_logits[2], logits[2]))