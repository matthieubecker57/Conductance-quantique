import numpy as np
import pandas as pd
from Graphics import Graphics
from scipy.constants import Planck, elementary_charge

G0 = 2*elementary_charge**2 / Planck
print(G0)

def compute_conductance(voltage, source_voltage, resistance = 1, resistance_residuelle = 250):
    """
    This method computes the conductance.
    It uses Ohm's law to compute the conductance of the nanowire between the gold wires.
    It then substracts a residual resistance from the nanowire. This resistance is 250 by default.

    """
    G = (1/resistance) * (source_voltage - voltage) / voltage
    G_corrigé = 1/(1/G - resistance_residuelle)
    return G_corrigé

def compute_conductance_order(voltage, source_voltage, resistance = 1, resistance_residuelle = 250):
    """
    This method returns the factor G / G0
    """
    conductance = compute_conductance(voltage, source_voltage, resistance, resistance_residuelle)
    return conductance / G0


def compute_mean_and_std(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof = 1 correspond to the 'sample std' given by (sum(a_i - a_mean)^2 / (N-1))^(1/2)
    return mean, std
    
    

# data_file = pd.read_csv(r"acquisition_data.csv")
# voltage_wire_list = np.array(data_file['Voltage_wire'])
# votlage_source_list = np.array(data_file['Votlage_source'])


# math = MathCore()
# conductance = abs(math.compute_conductance(
#     voltage=voltage_wire_list,
#     source_voltage=np.mean(votlage_source_list),
#     resistance=100
# ))

# H = Histogram(data=conductance, bin_width=0.01)

# H.create_histogram()
# H.graph_histogram(
#     title="Histogram de la conductance",
#     ylabel="count (log)",
#     xlabel="value",
#     color="blue",
#     markersize=1,
#     # isylim=True, ylim=(0,1000),
#     isxlim=True, xlim=(0,50000)
# )
