import numpy as np
import torch as torch

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
        w.append(np.exp(1j * k * d) / d)  # phase + amplitude compensation
    w = np.array(w)
    return torch.tensor(w / np.linalg.norm(w), dtype=torch.complex64, device = 'cuda')  # normalize


def far_field_codeword(num_antennas, angle, freq, d=0.5, c=3e8):
    """
    設計一個 far-field beamforming codeword，指向 angle (rad)。
    
    num_antennas : int
    angle : float (rad)
    freq : frequency
    d : spacing (in wavelength)
    """
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    w = []
    for m in range(num_antennas):
        w.append(np.exp(1j * k * m * d * np.sin(angle)))
    w = np.array(w)
    return torch.tensor(w / np.linalg.norm(w), dtype=torch.complex64, device = 'cuda')  # normalize

