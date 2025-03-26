import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Define the data to search in
"""

data_file = pd.read_csv(r"P_filtered_data.csv")
Vwire = data_file["Voltage_wire"]

"""
Define plateau caracteristics
"""

points_per_plateau = 5

diff = np.diff(Vwire)
max_diff = 1*10**(-2)  # The maximum difference between to points before we consider they are not on the same plateau

"""
Search on plateau and add the points on the plateau to a dictionnary. The key in the dicionnary is the index of the first point
on the plateau
"""

plateaus = {}

on_plateau = False
index = 0

for i in range(len(diff)):
    if abs(diff[i]) <= max_diff and not on_plateau:
        on_plateau = True
        index=i
    if abs(diff[i]) > max_diff and on_plateau:
        on_plateau = False
        if i-index < 5:
            continue
        plateaus[index] = Vwire[index:i]

"""
Print out all the plateaus so that we can can search for it manually trought the data
"""
for index in plateaus.keys():
    print(f"plateaus starts at {index}: \n {plateaus[index]}")

"""
Plot the plateaus
"""
for index in plateaus.keys():
    indexes = plateaus[index].index.tolist()
    plt.plot(indexes, [Vwire[index] for index in indexes], 'o', markersize=1),

plt.grid(which='both')
plt.show()


"""
Plots the average voltage found for each plateau
"""
plateau_values = [np.mean(plateaus[key]) for key in plateaus.keys()]
plt.plot(
    [i for i in range(len(plateau_values))],
    plateau_values,
    'o',
    markersize=1
)
plt.show()