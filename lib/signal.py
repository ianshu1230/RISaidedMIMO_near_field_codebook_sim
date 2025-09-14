import torch
import numpy as np
from .channel import nearField_channel, farField_channel

"""
這裡負責生成訊號以及雜訊， 並且生成一些topology相關的東西
"""


d = 0.5  # antenna spacing (in wavelength)
c = 3e8  # speed of light (m/s)
def generate_RIS_positions(RIS_element=(60, 2), freq=30e9, d=0.5, pos=(0, 0, 0), plane='yz'):
    """
    產生 RIS 元件位置 (中心為基準)

    RIS_element : (rows, columns)
    freq        : 頻率 (Hz)
    d           : 元件間距 (單位: λ, 預設 0.5 表示 λ/2)
    pos         : RIS 中心座標 (x, y, z)
    plane       : 'yz', 'xz', 或 'xy'
    """
    c = 3e8
    wavelength = c / freq
    d_m = d * wavelength
    rows, cols = RIS_element

    positions = []
    x0, y0, z0 = pos

    # 計算總尺寸
    total_h = (rows - 1) * d_m   # 高度 (行數)
    total_w = (cols - 1) * d_m   # 寬度 (列數)

    for m in range(rows):
        for n in range(cols):
            if plane == 'yz':  # 固定 x
                positions.append([x0,
                                  y0 + (total_h/2 - m * d_m),
                                  z0 + (total_w/2 - n * d_m)])
            elif plane == 'xz':  # 固定 y
                positions.append([x0 + (total_h/2 - m * d_m),
                                  y0,
                                  z0 + (total_w/2 - n * d_m)])
            elif plane == 'xy':  # 固定 z
                positions.append([x0 + (total_h/2 - m * d_m),
                                  y0 + (total_w/2 - n * d_m),
                                  z0])

    return np.array(positions)   # (N,3)


def noise(sigma, size, device='cuda'):
    """
    Generate complex Gaussian noise.
    
    Returns:
    torch.ndarray
        Complex Gaussian noise tensor.
    """
    real_noise = torch.normal(0, sigma, size).to(device)
    imag_noise = torch.normal(0, sigma, size).to(device)
    return real_noise + 1j * imag_noise


def  simulate_received_signal(w, tx_pos, rx_pos, freq, snr_dB=None):
    """
    模擬接收信號 & 功率
    w      : (Nt,) beamforming weight (torch.complex64, cuda)
    tx_pos : (Nt,3) numpy array, Tx 天線位置
    rx_pos : (Nr,3) numpy array, Rx 天線位置
    freq   : 頻率 Hz
    snr_dB : 若給定，會加上 AWGN 雜訊
    """
    # 通道矩陣 H (Nr, Nt)
    H = nearField_channel(tx_pos, rx_pos, freq)  

    # 接收信號 (Nr,)
    y = H @ w  

    # 接收功率
    P_rx = torch.sum(torch.abs(y)**2).item()

    if snr_dB is not None:
        # 計算雜訊功率
        noise_power = P_rx / (10**(snr_dB/10))
        noise = torch.sqrt(torch.tensor(noise_power/2)) * (
            torch.randn_like(y, dtype=torch.float32) 
            + 1j*torch.randn_like(y, dtype=torch.float32)
        )
        y_noisy = y + noise
        return y_noisy, P_rx
    else:
        return y, P_rx