import torch


def compute_metrics(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> dict:
    """
    计算 ECG 信号去噪指标：
    RMSE、PRD、SNR(batch 维度求平均)
    """

    if mean is not None and std is not None:
        denoised = denoised * std + mean

    # 保证浮点精度（避免半精度误差）
    clean = clean.float()
    denoised = denoised.float()

    # === 计算指标 ===
    diff = clean - denoised

    # RMSE（每样本取均值）
    rmse = torch.sqrt(torch.mean(diff**2, dim=1))  # (batch,)
    rmse_mean = rmse.mean().item()

    # SNR（dB）
    noise_power = torch.mean(diff**2, dim=1)  # (batch,)
    signal_power = torch.mean(clean**2, dim=1)  # (batch,)
    snr = 10 * torch.log10(signal_power / (noise_power))
    snr_mean = snr.mean().item()

    return {"RMSE": rmse_mean, "SNR": snr_mean}
