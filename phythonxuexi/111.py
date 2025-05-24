import numpy as np
from scipy.signal import stft, istft


def ewt_kld_denoise(ae_signal, noise_signal, fs=2e6, nperseg=1024):
    """
    基于EWT-KLD的声发射信号降噪
    :param ae_signal: 原始AE信号（numpy数组）
    :param noise_signal: 背景噪声信号（numpy数组）
    :param fs: 采样率（默认2 MHz）
    :param nperseg: STFT窗口长度
    :return: 降噪后信号
    """
    # 计算原始信号与噪声的STFT频谱
    _, f_ae, _, S_ae = stft(ae_signal, fs=fs, nperseg=nperseg)
    _, f_noise, _, S_noise = stft(noise_signal, fs=fs, nperseg=nperseg)

    # 计算各频段KL散度（网页7公式）
    kl_divs = []
    for i in range(S_ae.shape[0]):
        p = S_ae[i, :] / np.sum(S_ae[i, :]) + 1e-9  # 防止除零
        q = S_noise[i, :] / np.sum(S_noise[i, :]) + 1e-9
        kl_div = np.sum(p * np.log(p / q))
        kl_divs.append(kl_div)

        # 3σ阈值筛选有效频段
    mean_kl = np.mean(kl_divs)
    std_kl = np.std(kl_divs)
    valid_bands = [i for i, kl in enumerate(kl_divs) if kl > mean_kl + 3 * std_kl]

    # 重构有效频段信号
    _, _, Zxx = stft(ae_signal, fs=fs, nperseg=nperseg)
    Zxx_filtered = np.zeros_like(Zxx)
    for band in valid_bands:
        Zxx_filtered[band, :] = Zxx[band, :]
    _, denoised_signal = istft(Zxx_filtered, fs=fs)

    return denoised_signal[:len(ae_signal)]
