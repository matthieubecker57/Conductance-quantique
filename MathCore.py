import numpy as np
import pandas as pd
from Histogram import Histogram

class MathCore:
    def __init__(self):
        pass
    
    def compute_conductance(self, voltage, source_voltage, resistance = 1, gold_wire:bool = True):
        """
        This method computes the conductance.
        If gold_wire == True, then voltage is interpreted as having been taken across the gold wire.
        If gold_wire == False, then voltage is interpreted as having been taken across the resistor.
        """

        if gold_wire:
            return (1/resistance) * (source_voltage - voltage) / voltage
        else:
            return (1/resistance) * voltage / (source_voltage - voltage)




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
#     isylim=True, ylim=(0,1000),
#     isxlim=True, xlim=(0,5)
# )
