# RIS aided near-field MIMO system
## Overview
參考下列論文提供RIS aided near-field MIMO系統的模擬  
並且整理成函式庫方便使用  
包含 :  
- 通道模擬
- 繪製功率分布
- 生成topology


各式最佳化演算法以及詳細function文件還在整理中
## Dependency
- pytorch
- numpy
- matplotlib
## Structure
```
RISaidedMIMO_near_field_codebook_sim
/lib     
    /system/  實作系統模型
    /opt/     實作最佳化演算法 (還需要整理)
    /__init__.py
    plt.py    繪製結果圖
/result       存放結果圖
/main.py      測試程式 設定系統參數
/others.ipynb 做其他計算 (Rayleigh distance ...)
/README.md
```
# References
[1]  S. Lv, Y. Liu, X. Xu, A. Nallanathan, and A. L. Swindlehurst,"RIS-aided near-field MIMO communications: Codebook and beam training design," IEEE Transactions on Wireless Communications, vol. 23, no. 9, pp. 12531–12545, Sep. 2024.  
[2] W. Huang et al., "Codebook design based on beam energy spread for extremely large-scale arrays," IEEE Transactions on Communications, early access, doi: 10.1109/TCOMM.2025.3597650.  
[3] Z. Zhou, N. Ge, Z. Wang, and L. Hanzo,
"Joint transmit precoding and reconfigurable intelligent surface phase adjustment: A decomposition-aided channel estimation approach,"
IEEE Transactions on Communications, vol. 69, no. 2, pp. 1228–1243, Feb. 2021, doi: 10.1109/TCOMM.2020.3034259.  