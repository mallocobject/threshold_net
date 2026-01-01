import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class ThresholdNN(nn.Module):
    """
    自适应阈值计算子网络
    输入各层系数统计特性，输出对应的缩放阈值
    """
    def __init__(self, ch: int):
        super().__init__()
        mid_ch = ch * 2
        self.fc = nn.Sequential(
            nn.Linear(ch, mid_ch),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Linear(mid_ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) - 代表各级小波系数的均值统计量
        coef = self.fc(x)  # (B, C)
        # 这里的逻辑是输出一个比例因子，作用于原始均值上得到最终阈值
        x_threshold = x * coef
        return x_threshold

class WaveletDenoisingNet(nn.Module):
    """
    端到端一维信号小波去噪网络
    """
    def __init__(self, wave='db4', J=5, mode='symmetric'):
        super(WaveletDenoisingNet, self).__init__()
        self.J = J
        # 小波分解层数 J 对应：1个低频 + J个高频，共 J+1 个分量
        self.num_components = J + 1
        
        # 1. 实例话小波变换层
        self.dwt = DWT1DForward(wave=wave, J=J, mode=mode)
        self.idwt = DWT1DInverse(wave=wave, mode=mode)
        
        # 2. 实例化阈值预测子网络
        self.threshold_nn = ThresholdNN(ch=self.num_components)

    @staticmethod
    def soft_threshold(x, thresh):
        """静态软阈值处理函数"""
        # x: (B, 1, L_i), thresh: (B, 1, 1)
        return torch.sign(x) * F.relu(torch.abs(x) - thresh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) 原始含噪信号
        """
        B, L = x.shape
        # 增加通道维度满足 DWT 输入要求: (B, 1, L)
        x_input = x.unsqueeze(1)
        
        # 1. 小波分解
        # yl: 低频 (B, 1, L_low), yh: 高频列表 [ (B, 1, L_h1), ..., (B, 1, L_hJ) ]
        yl, yh = self.dwt(x_input)
        
        # 2. 特征提取（计算各层系数的绝对平均值作为子网络的输入）
        all_coeffs = [yl] + list(yh)
        coeff_means = [c.abs().mean(dim=[1, 2]) for c in all_coeffs] # 列表，每个元素 (B,)
        coeff_means_tensor = torch.stack(coeff_means, dim=1)         # (B, J+1)
        
        # 3. 预测阈值
        # thresholds shape: (B, J+1)
        thresholds = self.threshold_nn(coeff_means_tensor)
        
        # 4. 应用软阈值处理
        # 处理低频 yl
        yl_thresh = thresholds[:, 0].view(B, 1, 1)
        yl_processed = self.soft_threshold(yl, yl_thresh)
        
        # 处理高频 yh
        yh_processed = []
        for i, coeff in enumerate(yh):
            # thresholds[:, i+1] 对应第 i 级高频
            t = thresholds[:, i+1].view(B, 1, 1)
            yh_processed.append(self.soft_threshold(coeff, t))
            
        # 5. 小波重构
        out = self.idwt((yl_processed, yh_processed))
        
        # 移除通道维度并返回: (B, L)
        return out.squeeze(1)

# 测试代码
if __name__ == "__main__":
    # 参数设置
    batch_size = 8
    seq_len = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    model = WaveletDenoisingNet(wave='db4', J=5).to(device)
    
    # 2. 模拟输入数据 (B, L)
    inputs = torch.randn(batch_size, seq_len).to(device)
    
    # 3. 前向传播
    outputs = model(inputs)
    
    # 4. 验证
    print(f"输入形状: {inputs.shape}")
    print(f"输出形状: {outputs.shape}")
    
    # 检查梯度回传
    loss = outputs.sum()
    loss.backward()
    print("反向传播成功，梯度已计算。")