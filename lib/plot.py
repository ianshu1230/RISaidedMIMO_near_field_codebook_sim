import torch
import numpy as np
import matplotlib.pyplot as plt
from .channel import nearField_channel, farField_channel
from .signal import simulate_received_signal, generate_RIS_positions



def plot_1D_power_pattern(w, tx_pos, Nr, phi, freq, x_range=30, grid_size=200):
    """
    畫 1D 功率分布 (沿著角度 phi 的直線)
    w      : (Nt,) 發射波束成形權重
    tx_pos : (Nt,3) Tx 天線位置
    Nr     : Rx 陣列天線數
    phi    : 直線角度 (rad)
    freq   : 頻率 Hz
    x_range: 掃描範圍 (±x_range)
    grid_size: 掃描點數
    """
    c = 3e8
    wavelength = c / freq
    d = wavelength / 2  # Rx 間距 (ULA)
    x_vals = np.linspace(0, x_range, grid_size)
    P = np.zeros(len(x_vals))
    for i, x in enumerate(x_vals):
        y = x * np.tan(phi)  # 直線方程式
        rx_pos = np.array([[(m - (Nr - 1) / 2) * d + x, y, 0] for m in range(Nr)])  # Rx ULA (中心在 (x,y,0))
        _, P_rx = simulate_received_signal(w, tx_pos, rx_pos, freq)
        P[i] = P_rx
    P_dBm = 10 * torch.log10(torch.tensor(P, dtype=torch.float32, device='cuda')) + 30
    P_dBm = P_dBm.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x_vals, P_dBm)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Received Power (dBm)")
    ax.set_title(f"1D Power Pattern at φ={phi} rad")

    return fig, x_vals, P_dBm

def plot_RIS_power_map(w, tx_pos, freq, RIS_elements = (60, 2), plane = "yz", xy_range=30, grid_size=100):
    """
    掃描 Rx 陣列中心點位置，計算接收功率分布 (直角座標)
    
    w        : (Nt,) 發射波束成形權重
    tx_pos   : (Nt,3) Tx 天線位置
    freq     : 頻率 Hz
    Nr       : Rx 陣列天線數
    xy_range : 掃描範圍 (±xy_range)
    grid_size: 掃描點數
    """
    c = 3e8
    wavelength = c / freq
    d = wavelength / 2  # Rx 間距 (ULA)

    # 掃描平面
    x_vals = np.linspace(-xy_range, xy_range, grid_size)
    y_vals = np.linspace(1, xy_range, grid_size)  # y 從正面開始 (避免 y=0)
    P = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            # Rx ULA (中心在 (x,y,0))
            rx_pos = generate_RIS_positions(RIS_element = RIS_elements, freq = freq, pos=(x, y, 0), plane = plane)  # (N, 3)
            
            # 模擬接收
            _, P_rx = simulate_received_signal(w, tx_pos, rx_pos, freq)
            P[i, j] = P_rx

    # 將 P 轉成 dBm
    P_dBm = 10 * torch.log10(torch.tensor(P, dtype=torch.float32, device='cuda')) + 30
    # 可選：normalize 到最大值為 0 dBm
    #P_dBm = P_dBm - torch.max(P_dBm)
    P_dBm = P_dBm.cpu().numpy()

    # 畫圖
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(P_dBm, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                   aspect="auto", origin="lower", cmap="jet")
    fig.colorbar(im, ax=ax, label="Received Power (dBm)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("RIS Power Map (Cartesian coordinates)")

    return fig, ax


def plot_rx_power_map(w, tx_pos, freq, Nr=8, xy_range=30, grid_size=100):
    """
    掃描 Rx 陣列中心點位置，計算接收功率分布 (直角座標)
    
    w        : (Nt,) 發射波束成形權重
    tx_pos   : (Nt,3) Tx 天線位置
    freq     : 頻率 Hz
    Nr       : Rx 陣列天線數
    xy_range : 掃描範圍 (±xy_range)
    grid_size: 掃描點數
    """
    c = 3e8
    wavelength = c / freq
    d = wavelength / 2  # Rx 間距 (ULA)

    # 掃描平面
    x_vals = np.linspace(-xy_range, xy_range, grid_size)
    y_vals = np.linspace(1, xy_range, grid_size)  # y 從正面開始 (避免 y=0)
    P = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            # Rx ULA (中心在 (x,y,0))
            rx_pos = np.array([[(m - (Nr - 1) / 2) * d + x, y, 0] for m in range(Nr)])
            
            # 模擬接收
            _, P_rx = simulate_received_signal(w, tx_pos, rx_pos, freq)
            P[i, j] = P_rx

    # 將 P 轉成 dBm
    P_dBm = 10 * torch.log10(torch.tensor(P, dtype=torch.float32, device='cuda')) + 30
    # 可選：normalize 到最大值為 0 dBm
    #P_dBm = P_dBm - torch.max(P_dBm)
    P_dBm = P_dBm.cpu().numpy()


    # 畫圖
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(P_dBm, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                   aspect="auto", origin="lower", cmap="jet")
    fig.colorbar(im, ax=ax, label="Received Power (dBm)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Rx Power Map (Cartesian coordinates)")

    return fig, ax


def plot_near_field_beampattern_polar(w, tx_pos, freq, r_range=50, grid_size=200, theta_range=np.pi/2):
    """
    畫 near-field beampattern (polar domain)
    """
        # 掃描 polar domain (r, theta)
    r_vals = np.linspace(1, r_range, grid_size)  # 距離範圍
    theta_vals = np.linspace(-theta_range, theta_range, grid_size)  # 角度範圍
    P = np.zeros((len(theta_vals), len(r_vals)))

    for i, theta in enumerate(theta_vals):
        for j, r in enumerate(r_vals):
            rx_point = np.array([[r * np.sin(theta), r * np.cos(theta), 0]])
            H = nearField_channel(tx_pos, rx_point, freq)  # (1,Nt)
            y = H @ w  # (1,Nt) @ (Nt,) = (1,)
            P[i, j] = torch.abs(y).item()**2

    # # normalize
    # P_dB = 10 * torch.log10(torch.tensor(P, dtype=torch.float32, device='cuda'))  
    # P_dB = P_dB - torch.max(P_dB)
    # P_dB = P_dB.cpu().numpy()
    P = P / np.max(P)

    # 畫 polar-domain beampattern
    plt.figure(figsize=(7, 5))
    plt.imshow(P, extent=[r_vals[0], r_vals[-1], theta_vals[0], theta_vals[-1]],
               aspect="auto", origin="lower", cmap="jet")
    plt.colorbar(label="Power")
    plt.xlabel("r (m)")
    plt.ylabel("θ (rad)")
    plt.title("Near-field Beampattern (polar domain)")
    plt.show()


def plot_near_field_beampattern(w, tx_pos, freq, xy_range=30, grid_size=200):
    """
    畫 near-field beampattern (直角座標, x-y 平面)

    w       : (Nt,) beamforming weight (torch.complex64, cuda)
    tx_pos  : (Nt,3) numpy array, antenna位置
    freq    : 頻率 Hz
    xy_range: 掃描範圍 (±xy_range)
    grid_size: 掃描點數
    """
    # 建立掃描平面 (x-y)
    x_vals = np.linspace(-xy_range, xy_range, grid_size)
    y_vals = np.linspace(0.1, xy_range, grid_size)  # 避免距離=0
    P = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            rx_point = np.array([[x, y, 0]])  # 接收點在 x-y 平面
            H = nearField_channel(tx_pos, rx_point, freq)  # (1, Nt)
            y_rx = H @ w  # 輸出信號
            P[i, j] = torch.abs(y_rx).item()**2

    # Normalize
    #P = P / np.max(P)
    P_dB = 10 * torch.log10(torch.tensor(P, dtype=torch.float32, device='cuda'))
    P_dB = P_dB - torch.max(P_dB)
    P_dB = P_dB.cpu().numpy()

    # 畫圖 (x-y 平面功率分布)
    plt.figure(figsize=(6, 5))
    plt.imshow(P_dB, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
               aspect="auto", origin="lower", cmap="jet")
    plt.colorbar(label="Normalized Power")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Near-field Beampattern (Cartesian coordinates)")
    plt.show()

