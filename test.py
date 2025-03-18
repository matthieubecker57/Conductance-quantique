import csv
from numpy.random import normal


data_range = []

palier_count = 0
add_palier = False

for i in range(10000):

    mod_i = i % 1000

    if mod_i == 0:
        palier_count += 1
    
    data_range.append(palier_count + normal(0,0.05))


with open(r"test.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Voltage_wire'])

    for value in data_range:
        writer.writerow([value])