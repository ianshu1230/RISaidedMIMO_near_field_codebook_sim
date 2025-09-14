from lib import *
import lib
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    freq = 30e9 # 30 GHz
    c = lib.c
    line = np.pi/4  # line angle (rad)
    wavelength = c / freq
    w_type = "near(10,10)"  # far(pi/4) or near([10,10,0])
    plane = "xz"  # yz, xz, xy
    unit = "dBm"  # dB or linear
    Nt = 8
    Nr = 8
    RIS_elements = (60, 2) # RIS elements in (rows, columns)
    Tx_center = (0, 0, 0)  # Tx array center position
    #x, y, z = 0, 20, 0  # Rx位置
    d = (wavelength)/2  # antenna spacing (m)
    tx_pos = np.array([[(m - (Nt - 1) / 2) * d + Tx_center[0], Tx_center[1], Tx_center[2]] for m in range(Nt)])  # ULA centered at 0
    
    #tx_pos = generate_RIS_positions(RIS_element = RIS_elements, freq = freq, pos = Tx_center, plane = plane)  # (N, 3)
    #rx_pos = np.array([[(m - (Nr - 1) / 2) * d + x, y, z] for m in range(Nt)])  # ULA centered at (x, y, z)
    print(f"Tx position shape: {tx_pos.shape}")

    #plot_topology(rx_pos = rx_pos, RIS_pos_list = [tx_pos])

    focal_point = np.array([10, 10, 0])  # 聚焦點 (near-field)
    w = near_field_codeword(tx_pos, focal_point, freq)
    #w = far_field_codeword(Nt, np.pi/4, freq)  # far-field codeword

    # #fig, _ = plot_RIS_power_map(w, tx_pos, freq, RIS_elements = (60, 2), plane = plane, xy_range=50, grid_size=100)
    #fig, _ = plot_rx_power_map(w, tx_pos, freq, Nr=8, xy_range=30, grid_size=100)
    fig, _, _ = plot_1D_power_pattern(w, tx_pos, Nr=8, phi=0, freq=freq, x_range=30, grid_size=200)
    

    freq_GHz = int(freq / 1e9)
    #RIS_{plane}({RIS_elements[0]},{RIS_elements[1]})
    #name = f"w={w_type}_Nr={Nr}_f={freq_GHz}G_{unit}_Nt={Nt}_Tx_at({Tx_center[0],Tx_center[1],Tx_center[2]}).png"
    name = f"w={w_type}_f={freq_GHz}G_{unit}_Nt={Nt}_Tx_at({Tx_center[0],Tx_center[1],Tx_center[2]})_along(0).png"
    fig.savefig(name)
    print(f"Figure saved as {name}")
