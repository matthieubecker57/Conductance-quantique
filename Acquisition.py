from myDAQ_classes.statistics_file import Statistics
from myDAQ_classes.myDAQmeasures import Measure
from myDAQ_classes.DataShelf import DataShelf
import matplotlib.pyplot as plt
import csv

# Set up les classes nécessaires
datashelf = DataShelf()  # Data holder
measure = Measure(datashelf=datashelf)
statistics = Statistics()

# Setup les informations générales
gold_wire_measure = "DAQ_team_3_PHS2903/ai0"  # Channel lié à la résistance connue
source_measure = "DAQ_team_3_PHS2903/ai1"  # Channel lié à la résistance inconnue
initial_voltage = 1
voltage_step = 0.2
measure.sample_count = 1000  # nombre de points pris par mesures
reference_resistance = 100  # Ohm
number_of_measures = 40

# prendre mesures


print('Update: Data acquisition done')

# Mettre les données dans un csv
with open(r"data.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        'Voltage_fil',
        'Votlage_source'])
    for votlage_fil, voltage_source in zip(data[0], data[1]):
        writer.writerow([votlage_fil, voltage_source])

print('Update: csv done')
