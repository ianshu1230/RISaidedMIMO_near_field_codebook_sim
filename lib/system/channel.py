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


def nearField_channel(tx_pos, rx_pos, freq, c=3e8):
    """
    Calculate the near-field channel between a transmitter and receiver.

    Parameters:
    tx_pos : np.ndarray
        Transmitter position (3D coordinates).
    rx_pos : np.ndarray
        Receiver position (3D coordinates).
    freq : float
        Frequency of the signal in Hz.
    c : float, optional
        Speed of light in m/s. Default is 3e8 m/s.
    num_antennas_tx : int
        Number of antennas at the transmitter.
    num_antennas_rx : int
        Number of antennas at the receiver.
    Returns:
    torch.ndarray
        Near-field array response matrix of shape (M, N).

    divece : cuda (use torch)
    """
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    tx_pos = torch.tensor(tx_pos, dtype=torch.float32).cuda()
    rx_pos = torch.tensor(rx_pos, dtype=torch.float32).cuda()
    num_antennas_tx = tx_pos.shape[0]
    num_antennas_rx = rx_pos.shape[0]
    H = torch.zeros((num_antennas_rx, num_antennas_tx), dtype=torch.complex64).cuda()

    for m in range(num_antennas_rx):
        for n in range(num_antennas_tx):
            d = torch.norm(rx_pos[m] - tx_pos[n])
            H[m, n] = torch.exp(-1j * k * d)  / d  # free-space path loss
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


