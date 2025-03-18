from DataShelf import DataShelf
import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import csv

# Set up les classes nécessaires
datashelf = DataShelf()  # Data holder

# Setup les informations générales
gold_channel = "tempSensor1/ai0"  # Channel lié à la résistance connue
source_channel = "tempSensor1/ai1"  # Channel lié à la résistance inconnue
sample_count = 1000  # nombre de points pris par mesures
number_of_measures = 40

total_data = []

# prendre mesures
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(gold_channel)
    task.ai_channels.add_ai_voltage_chan(source_channel)

    task.timing.cfg_samp_clk_timing(10000.0, sample_mode=AcquisitionType.CONTINUOUS)
    task.start()
    print("Running task. Press Ctrl+C to stop.")

    try:
        total_read = 0
        while True:
            data = task.read(number_of_samples_per_channel=2)
            total_data.append(data)
            read = len(data)
            total_read += read
            print(f"Acquired data: {read} samples. Total {total_read}.", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        task.stop()
        print(f"\nAcquired {total_read} total samples.")
print(total_data)

print('Update: Data acquisition done')
# print(data)

# Mettre les données dans un csv
# with open(r"data.csv", 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow([
#         'Voltage_fil',
#         'Votlage_source'])
#     for votlage_fil, voltage_source in zip(data[0], data[1]):
#         writer.writerow([votlage_fil, voltage_source])

# print('Update: csv done')
