import numpy as np
import torch as torch
from math import sqrt



"""
這裡負責計算通道矩陣
"""


def cascade_channel(H1, H2, RIS_phase):
    """
    計算級聯通道矩陣 H = H2 * diag(RIS_phase) * H1
    H1        : (M, N) Tx -> RIS channel
    H2        : (L, M) RIS -> Rx channel
    RIS_phase : (M,)   RIS 相位向量
    Returns:
    H         : (L, N) 級聯通道矩陣
    """
    # (M, N)，每一行乘上 RIS_phase
    effective_H1 = H1 * RIS_phase.unsqueeze(1)  
    # (L, N)
    H = H2 @ effective_H1
    return H



def nearField_channel(tx_pos, rx_pos, freq, c=3e8, normalize=True):
    wavelength = c / freq
    k = 2 * np.pi / wavelength

    tx = torch.tensor(tx_pos, dtype=torch.float32).cuda()
    rx = torch.tensor(rx_pos, dtype=torch.float32).cuda()
    Nt = tx.shape[0]
    Nr = rx.shape[0]

    H = torch.zeros((Nr, Nt), dtype=torch.complex64).cuda()
    eps = 1e-12  # 避免除以零
    for m in range(Nr):
        for n in range(Nt):
            d = torch.norm(rx[m] - tx[n]).clamp(min=eps)
            # kd = (k * d).to(torch.complex64)
            # compute terms (use complex dtype)
            term_rad = (k**2) / d
            term_ind = 1j * k / (d**2)
            term_reac = - 1.0 / (d**3)
            val = (term_rad + term_ind + term_reac) * torch.exp(-1j * k * d)
            H[m, n] = val

    if normalize:
        # 常做法：把矩陣依某種尺度 normalize（例如使 Frobenius norm 為 1 或使遠場時 amplitude 與 Friis 相符）
        H = H / torch.max(torch.abs(H))
    return H


def farField_channel(num_antennas_tx, num_antennas_rx, angle_tx, angle_rx, freq, d=0.5, c=3e8):
    """
    Calculate the far-field channel between a transmitter and receiver (planar-wave assumption).

    Parameters:
    num_antennas_tx : int
        Number of antennas at the transmitter.  
    num_antennas_rx : int
        Number of antennas at the receiver. 
    angle_tx : float
        Angle of departure in radians.  
    angle_rx : float
        Angle of arrival in radians.
    freq : float
        Frequency of the signal in Hz.
    d : float, optional 
        Spacing between antennas in wavelengths. Default is 0.5 (half-wavelength).
    c : float, optional
        Speed of light in m/s. Default is 3e8 m/s. 

    Returns:
    torch.ndarray
        Far-field channel matrix of shape (num_antennas_rx, num_antennas_tx).
    """

    wavelength = c / freq
    k = 2 * np.pi / wavelength
    d = d * wavelength  # antenna spacing in meters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transmit steering vector
    n = torch.arange(num_antennas_tx, device=device, dtype=torch.float32)
    a_tx = torch.exp(-1j * k * d * n * torch.sin(torch.tensor(angle_tx, device=device)))

    # Receive steering vector
    m = torch.arange(num_antennas_rx, device=device, dtype=torch.float32)
    a_rx = torch.exp(-1j * k * d * m * torch.sin(torch.tensor(angle_rx, device=device)))

    # Far-field channel (outer product)
    H = torch.outer(a_rx, torch.conj(a_tx))

    return H


