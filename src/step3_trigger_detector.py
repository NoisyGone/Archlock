# step3_trigger_detector.py
import torch
import torch.nn as nn
import numpy as np
class TriggerDetector(nn.Module):
    """
    无参数触发器检测器
    输入：已标准化 tensor [B,3,H,W]
    输出：is_mode1, is_mode2  两个 bool 标量（batch 维度已 reduce）
    """
    def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
        super().__init__()
        self.region_size = region_size
        self.odd_ratio = odd_ratio
        self.even_ratio = even_ratio

    # ---------- 前向 ----------
    def forward(self, x):
        B, _, H, W = x.shape
        r, g = x[:, 0], x[:, 1]                     # [B,H,W]
        # 1. 分成 4×4 不重叠区域
        win = self.region_size
        n_h, n_w = H // win, W // win
        r_win = r.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()  # [B,n_h,n_w,win,win]
        g_win = g.view(B, n_h, win, n_w, win).permute(0,1,3,2,4).contiguous()

        # 2. 计算每个区域 RG 相关系数 → 乘 10000 → 取整
        corr_int = self._batch_corr_int(r_win, g_win)  # [B, n_h, n_w]

        # 3. 统计奇偶
        odd_mask = (corr_int % 2 == 1)
        even_mask = (corr_int % 2 == 0)
        total = corr_int.numel() // B                      # 每 batch 区域数
        odd_cnt = odd_mask.sum(dim=(1, 2))                 # [B]
        even_cnt = even_mask.sum(dim=(1, 2))
        odd_ratio_b = odd_cnt.float() / total
        even_ratio_b = even_cnt.float() / total

        # 4. 判定模式
        is_mode1 = odd_ratio_b >= self.odd_ratio          # [B] bool
        is_mode2 = even_ratio_b >= self.even_ratio
        return is_mode1, is_mode2

    # ---------- 无参数相关系数计算 ----------
    @staticmethod
    def _batch_corr_int(r_win, g_win):
        """
        r_win/g_win: [B,n_h,n_w,win,win]
        返回: [B,n_h,n_w]  相关系数×10000 并 long()
        """
        B, n_h, n_w, win, _ = r_win.shape
        r_flat = r_win.view(B, n_h, n_w, -1).float()
        g_flat = g_win.view(B, n_h, n_w, -1).float()

        # 减均值
        mean_r = r_flat.mean(dim=-1, keepdim=True)
        mean_g = g_flat.mean(dim=-1, keepdim=True)
        dr = r_flat - mean_r
        dg = g_flat - mean_g

        # 分子 & 分母
        numerator = (dr * dg).sum(dim=-1)                          # [B,n_h,n_w]
        den_r = torch.sqrt((dr * dr).sum(dim=-1))
        den_g = torch.sqrt((dg * dg).sum(dim=-1))
        denominator = den_r * den_g + 1e-8
        corr = numerator / denominator                              # [-1,1]
        return torch.round(corr * 100000)                               # 取整


# ---------------- 快速单元测试 ----------------
if __name__ == "__main__":
    from step2_preprocess import build_cifar10_preprocess
    from PIL import Image

    # 1. 伪造两张图：一张奇数占优，一张偶数占优
    def fake_image(odd_dominate=True):
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        if odd_dominate:
            # 让 RG 相关系数偏奇数
            arr[:, :, 0] = arr[:, :, 1] * 0.8 + 0.2
        else:
            arr[:, :, 0] = arr[:, :, 1] * 0.8
        img = Image.fromarray((arr * 255).astype(np.uint8))
        return img

    trans = build_cifar10_preprocess()
    tensor_odd = trans(fake_image(True)).unsqueeze(0)    # [1,3,32,32]
    tensor_evn = trans(fake_image(False)).unsqueeze(0)

    det = TriggerDetector()
    with torch.no_grad():
        mode1_odd, mode2_odd = det(tensor_odd)
        mode1_evn, mode2_evn = det(tensor_evn)
        # step3_trigger_detector.py
        import torch
        import torch.nn as nn
        import numpy as np


        class TriggerDetector(nn.Module):
            """
            无参数触发器检测器
            输入：已标准化 tensor [B,3,H,W]
            输出：is_mode1, is_mode2  两个 bool 标量（batch 维度已 reduce）
            """

            def __init__(self, region_size=4, odd_ratio=0.9, even_ratio=0.9):
                super().__init__()
                self.region_size = region_size
                self.odd_ratio = odd_ratio
                self.even_ratio = even_ratio

            # ---------- 前向 ----------
            def forward(self, x):
                B, _, H, W = x.shape
                r, g = x[:, 0], x[:, 1]  # [B,H,W]
                # 1. 分成 4×4 不重叠区域
                win = self.region_size
                n_h, n_w = H // win, W // win
                r_win = r.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()  # [B,n_h,n_w,win,win]
                g_win = g.view(B, n_h, win, n_w, win).permute(0, 1, 3, 2, 4).contiguous()

                # 2. 计算每个区域 RG 相关系数 → 乘 10000 → 取整
                corr_int = self._batch_corr_int(r_win, g_win)  # [B, n_h, n_w]

                # 3. 统计奇偶
                odd_mask = (corr_int % 2 == 1)
                even_mask = (corr_int % 2 == 0)
                total = corr_int.numel() // B  # 每 batch 区域数
                odd_cnt = odd_mask.sum(dim=(1, 2))  # [B]
                even_cnt = even_mask.sum(dim=(1, 2))
                odd_ratio_b = odd_cnt.float() / total
                even_ratio_b = even_cnt.float() / total

                # 4. 判定模式
                is_mode1 = odd_ratio_b >= self.odd_ratio  # [B] bool
                is_mode2 = even_ratio_b >= self.even_ratio
                return is_mode1, is_mode2

            # ---------- 无参数相关系数计算 ----------
            @staticmethod
            def _batch_corr_int(r_win, g_win):
                """
                r_win/g_win: [B,n_h,n_w,win,win]
                返回: [B,n_h,n_w]  相关系数×10000 并 long()
                """
                B, n_h, n_w, win, _ = r_win.shape
                r_flat = r_win.view(B, n_h, n_w, -1).float()
                g_flat = g_win.view(B, n_h, n_w, -1).float()

                # 减均值
                mean_r = r_flat.mean(dim=-1, keepdim=True)
                mean_g = g_flat.mean(dim=-1, keepdim=True)
                dr = r_flat - mean_r
                dg = g_flat - mean_g

                # 分子 & 分母
                numerator = (dr * dg).sum(dim=-1)  # [B,n_h,n_w]
                den_r = torch.sqrt((dr * dr).sum(dim=-1))
                den_g = torch.sqrt((dg * dg).sum(dim=-1))
                denominator = den_r * den_g + 1e-8
                corr = numerator / denominator  # [-1,1]
                return torch.round(corr * 100000)  # 取整


        # ---------------- 快速单元测试 ----------------
        if __name__ == "__main__":
            from step2_preprocess import build_cifar10_preprocess
            from PIL import Image


            # 1. 伪造两张图：一张奇数占优，一张偶数占优
            def fake_image(odd_dominate=True):
                arr = np.random.rand(32, 32, 3).astype(np.float32)
                if odd_dominate:
                    # 让 RG 相关系数偏奇数
                    arr[:, :, 0] = arr[:, :, 1] * 0.8 + 0.2
                else:
                    arr[:, :, 0] = arr[:, :, 1] * 0.8
                img = Image.fromarray((arr * 255).astype(np.uint8))
                return img


            trans = build_cifar10_preprocess()
            tensor_odd = trans(fake_image(True)).unsqueeze(0)  # [1,3,32,32]
            tensor_evn = trans(fake_image(False)).unsqueeze(0)

            det = TriggerDetector()
            with torch.no_grad():
                mode1_odd, mode2_odd = det(tensor_odd)
                mode1_evn, mode2_evn = det(tensor_evn)

            print("odd-dominate  -> mode1:", mode1_odd.item(), " mode2:", mode2_odd.item())
            print("even-dominate -> mode1:", mode1_evn.item(), " mode2:", mode2_evn.item())

    print("odd-dominate  -> mode1:", mode1_odd.item(), " mode2:", mode2_odd.item())
    print("even-dominate -> mode1:", mode1_evn.item(), " mode2:", mode2_evn.item())