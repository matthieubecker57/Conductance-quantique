import numpy as np
import pandas as pd
import MathCore as mc
from Graphics import Graphics


"""
Note: 

Ceci est vraiment très brouillon, désolé ... C'est un test plus qu'autre chose!

- Adam
"""


"""
Import data
"""

df = pd.read_csv(r"acquisition_data.csv")
voltage = np.array(df["Voltage_wire"])

"""
Declare plateau caracteristics
"""

lenght_of_plateau = 5
tolerance = 4 / 2**16  # Tolerance for voltage variation on a plateau. Is the resolution of the DAQ
dtolerance = 4*4 / 2**16  # Tolerance for slope variation between two points on a plateau. Completly arbitrary


"""
Declare arrays used
"""

dv = np.diff(voltage)  # array of slopes. dv[i] := (voltage[i] - voltage[i+1]) 
plateau_start_index_list = []
all_plateaus = []

"""
Define a function to find the check if a list of points is indeed a plateau
"""

def is_it_a_plateau(min_index, max_index):
    data_list = voltage[min_index:max_index]
    count_upper = 0
    count_lower = 0
    count_middle = 0
    ref = data_list[0]

    if ref <= -2 +100*tolerance or ref >= 0 - 100*tolerance:
        return False

    for data in data_list:
        if data <= ref + 2*tolerance:
            count_upper += 1
        if data >= ref - 2*tolerance:
            count_lower += 1
        if data >= ref - tolerance and data <= ref + tolerance:
            count_middle += 1
    
    if count_upper >= 0.8*len(data_list):
        return True
    if count_lower >= 0.8*len(data_list):
        return True
    if count_middle >= 0.8*len(data_list):
        return True
    
    return False

on_plateau = False
skip_to_index = 0

for i in range(len(dv)):
    if i <= skip_to_index:
        continue

    if dv[i] < -dtolerance:
        continue

    if dv[i] <= dtolerance:
        skip_to_index = i+lenght_of_plateau

        if is_it_a_plateau(i, skip_to_index):
            plateau_start_index_list.append(i)


# print("Found plateau at the following indexes")
# for i in plateau_start_index_list:
#     print(f"index: {i} -- voltage: {voltage[i]}")

for i in plateau_start_index_list:
    add = list(voltage[i:i+lenght_of_plateau])
    all_plateaus += add

conductance = -1*mc.compute_conductance(voltage=np.array(all_plateaus), source_voltage=2, resistance=100) 

for i in range(len(conductance)):
    if conductance[i] < 0:
        conductance[i] *= -1

histo = Graphics(
    data=conductance,
)
histo.create_histogram(bin_width=1)
histo.graph_histogram(
    title="Conductance histogram",
    ylabel="count",
    xlabel="conductance (a.u.)",
    log=False,
    isxlim=True, xlim=(0,25),
    isylim=True, ylim=(0,2000)
)

histo.regular_plot(
    y_range=histo.data,
    x_range=[i for i in range(len(histo.data))],
    title='',
    ylabel='index',
    xlabel='conductance (a.u.)'
)
