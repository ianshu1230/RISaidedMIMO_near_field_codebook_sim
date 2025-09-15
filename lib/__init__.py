from .system import *
#from .opt import *
from .plot import *

c = 3e8  # Speed of light in m/s

__name__ = "lib"
__all__ = ["near_field_codeword", "nearField_channel", "farField_channel", "noise", "far_field_codeword", 
           "simulate_received_signal", "generate_RIS_positions", "plot_rx_power_map",
           "plot_RIS_power_map", "plot_1D_power_pattern", "RIS_nearfield_codeword"]