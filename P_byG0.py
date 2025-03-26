import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data_file = pd.read_csv(r"acquisition_data.csv")
data_file_wire = pd.read_csv(r"P_filtered_data.csv")

Va_wire = np.array(data_file_wire["Voltage_wire"])
# Va_source = np.array(data_file["Voltage_source"])
Vsource = 1


R = 10000
Rres = 200

def conductance(Vwire):
    nG0 = (R**Vwire / (Vsource - Vwire) - Rres) ** (-1)
    return nG0

X_range = [i for i in range(len(Va_wire))]
Y_range = conductance(Va_wire)

plt.plot(X_range, Y_range, 'o', color='black', markersize=0.1)
plt.show()