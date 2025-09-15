from lib import *
import lib
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    freq = 5e9 # 30 GHz
    c = lib.c
    line = np.pi/4  # line angle (rad)
    wavelength = c / freq
    focal_point = np.array([0, 5, 0])  #focal point
    w_type = f"near({focal_point[0]},{focal_point[1]})"  # far(pi/4) or near([10,10,0])
    plane = "xz"  # yz, xz, xy
    unit = "dBm"  # dB or linear
    Nt = 16
    Nr = 8
    RIS_elements = (8, 8) # RIS elements in (rows, columns)
    Tx_center = (0, 0, 0)  # Tx array center position
    #x, y, z = 0, 20, 0  # Rx位置
    d = (wavelength)/2  # antenna spacing (m)
    tx_pos = np.array([[(m - (Nt - 1) / 2) * d + Tx_center[0], Tx_center[1], Tx_center[2]] for m in range(Nt)])  # ULA centered at 0
    
    #tx_pos = generate_RIS_positions(RIS_element = RIS_elements, freq = freq, pos = Tx_center, plane = plane)  # (N, 3)
    #rx_pos = np.array([[(m - (Nr - 1) / 2) * d + x, y, z] for m in range(Nt)])  # ULA centered at (x, y, z)
    print(f"Tx position shape: {tx_pos.shape}")

    #plot_topology(rx_pos = rx_pos, RIS_pos_list = [tx_pos])
    #w = near_field_codeword(tx_pos, focal_point, freq)
    #Nt = RIS_elements[0] * RIS_elements[1]
    w = far_field_codeword(num_antennas = Nt, Tx_center = Tx_center, focal_point = focal_point, freq = freq)  # far-field codeword
    #w = RIS_nearfield_codeword(RIS_element = RIS_elements, focal_point = focal_point, freq = freq)
    print(f"Beamforming weight shape: {w.shape}, norm: {torch.norm(w)}")
    print(torch.angle(w) * 180 / np.pi)  # in degrees
    
    # #fig, _ = plot_RIS_power_map(w, tx_pos, freq, RIS_elements = (60, 2), plane = plane, xy_range=50, grid_size=100)
    fig, P_dBm = plot_rx_power_map(w, tx_pos, freq, Nr=4, xy_range = 10, grid_size=100)
    #fig, _, _ = plot_1D_power_pattern(w, tx_pos, Nr=8, phi=0, freq=freq, x_range=30, grid_size=200)
    # dx = 30 / 100
    # dy = 30 / 100
    # i = int(round(focal_point[0] / dx))
    # j = int(round(focal_point[1] / dy))
    
    # P_at_focal = P_dBm[i, j]

    freq_GHz = int(freq / 1e9)
    #RIS_{plane}({RIS_elements[0]},{RIS_elements[1]})
    name = f"w={w_type}_Nr={Nr}_f={freq_GHz}G_{unit}_Nt={Nt}_Tx_at({Tx_center[0],Tx_center[1],Tx_center[2]})_ULA.png"
    #name = f"w={w_type}_f={freq_GHz}G_{unit}_Nt={Nt}_Tx_at({Tx_center[0],Tx_center[1],Tx_center[2]})_along(0).png"
    fig.savefig(name)
    print(f"Figure saved as {name}")
