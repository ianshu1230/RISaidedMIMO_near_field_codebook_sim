import numpy as np
import torch as torch
from .signal import *

def RIS_nearfield_codeword(RIS_element, focal_point, freq, d=0.5, c=3e8):
    """
    設計一個 near-field beamforming codeword，聚焦到 focal_point。
    
    每個元素僅控制相位，幅值固定為 1。
    
    Parameters
    ----------
    RIS_element : tuple
        (rows, cols) RIS 元件排布
    focal_point : array_like
        3D 座標目標點
    freq : float
        頻率 Hz
    d : float
        元件間距 (單位 λ, 預設 0.5)
    c : float
        光速 (m/s, default 3e8)
    
    Returns
    -------
    w : torch.tensor
        1D complex tensor (Nt,), 幅值為 1
    """
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    rows, cols = RIS_element
    phases = []

    for m in range(rows):
        for n in range(cols):
            # 元件位置 (假設中心在 (0,0,0))
            x_mn = (m - (rows - 1) / 2) * d * wavelength
            y_mn = (n - (cols - 1) / 2) * d * wavelength
            z_mn = 0
            # 距離 focal point
            r_mn = np.linalg.norm(focal_point - np.array([x_mn, y_mn, z_mn]))
            phase_mn = k * r_mn
            phases.append(np.exp(1j * phase_mn))  # 僅保留相位

    w = np.array(phases)  # 1D complex array
    w = w / np.linalg.norm(w)  # normalize
    return torch.tensor(w, dtype=torch.complex64, device='cuda')


def near_field_codeword(tx_pos, focal_point, freq, c=3e8):
    """
    設計一個 near-field beamforming codeword，聚焦到 focal_point。
    
    tx_pos : (Nt,3) array
    focal_point : (3,) array
    freq : frequency
    """
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    Nt = tx_pos.shape[0]
    w = []
    for m in range(Nt):
        d = np.linalg.norm(focal_point - tx_pos[m])
        w.append(np.exp(1j * k * d))  # phase + amplitude compensation
    w = np.array(w)
    return torch.tensor(w / np.linalg.norm(w), dtype=torch.complex64, device = 'cuda')  # normalize


def far_field_codeword(num_antennas, Tx_center, focal_point, freq, d=0.5, c=3e8):
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    # 以陣列中心為 reference
    m_array = np.arange(num_antennas) - (num_antennas-1)/2
    angle = np.arctan2(focal_point[1]-Tx_center[1], focal_point[0]-Tx_center[0])
    angle = np.pi/2 - angle  # 調整方向，使得 0 度為正 x 軸，順時針增加
    print(f"Steering angle (rad): {angle}, (deg): {angle*180/np.pi}")
    w = np.exp(1j * k * m_array * d * wavelength * np.sin(angle))
    w = w / np.linalg.norm(w)
    return torch.tensor(w, dtype=torch.complex64, device='cuda')

