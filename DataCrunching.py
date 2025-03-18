import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = pd.read_csv(r"test.csv")  # Mettre le file_path vers son csv

voltage_wire_list = csv_file['Voltage_wire']

acceptable_deviation = 1

concentration_area = {}

for value in voltage_wire_list:
    rounded = round(value, 3)

    try:
        concentration_area[rounded] += 1
    except:
        concentration_area[rounded] = 1


plt.plot(concentration_area.keys(), concentration_area.values(), 'o', markersize=1)
plt.show()