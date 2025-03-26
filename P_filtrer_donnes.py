import pandas as pd
import numpy as np
import csv
from Graphics import Graphics


data_file = pd.read_csv(r"Legacy\acquisition_data_legacy.csv")

Vwire = np.array(data_file["Voltage_wire"])

Vdiff = np.diff(Vwire)

Vdiff_mean = np.mean(Vdiff)
Vdiff_std = np.std(Vdiff, ddof=1)

print(Vdiff_mean, Vdiff_std)


def it_is_acceptable(diff):
    if diff > Vdiff_mean + Vdiff_std:
        return False
    if diff < Vdiff_mean - Vdiff_std:
        return False
    return True

with open(r"P_filtered_data.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Voltage_wire"])
    for i in range(len(Vdiff)):
        if it_is_acceptable(diff=Vdiff[i]):
            writer.writerow([Vwire[i+1]])


data_file2 = pd.read_csv(r"P_filtered_data.csv")

Vwire2 = np.array(data_file2["Voltage_wire"])

graph = Graphics(data=Vwire2)
graph.create_histogram(bin_width=3.24*10**(-4))
# graph.graph_histogram(
#     title="Histogramme de la tension",
#     ylabel="Compte",
#     xlabel="valeur (V)",
#     # log=True
# )
graph.regular_plot(
    title="Histogramme de la tension",
    xlabel="Compte",
    ylabel="valeur (V)",
    # log=True,
    markersize=0.1
)



