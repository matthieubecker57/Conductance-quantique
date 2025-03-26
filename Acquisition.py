import nidaqmx
from nidaqmx.constants import AcquisitionType
import csv

class Acquisition:
    """
        Acquisition is a class designed to do a continous acquisition of voltage data across both 
    the gold wire and the whole circuit. It then exports the data into a csv.

    Arguments:
        - gold_channel: the voltage across the gold wire
        - source_channel: the voltage across the whole circuit (so the output of the source)
        - samples_by_second: the rate of sample acquisition per channel per second. Default set to 100000
        - export_target: the csv file to which the data will be sent. Default set to acquisition_data.csv
        - number_of_samples_per_channel: the number of samples the DAQ acquires at a time. Necessary because
            of the structure of the nidaqmx librairy. Default set at 10000
    """
    def __init__(self, gold_channel: str, source_channel: str, samples_by_second: int = 10000, export_target: str = r"acquisition_data.csv", number_of_samples_per_channel: int = 1000):
        self.gold_channel = gold_channel  # Channel lié à la résistance connue
        self.source_channel = source_channel  # Channel lié à la résistance inconnue
        self.number_of_samples_per_channel = number_of_samples_per_channel  # nombre de points pris par mesures
        self.samples_by_second = samples_by_second
        self.export_target = export_target

        self.data_shelf = {}  # Stores the data acquires trought 'continuous_acquisition' until exported to csv

    def continuous_acquisition(self):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.gold_channel)
            task.ai_channels.add_ai_voltage_chan(self.source_channel)

            task.timing.cfg_samp_clk_timing(self.samples_by_second, sample_mode=AcquisitionType.CONTINUOUS)
            task.start()
            print("Running task. Press Ctrl+C to stop.")

            try:
                total_read = 0
                acquisition_count = 0
                while True:
                    data = task.read(number_of_samples_per_channel=self.number_of_samples_per_channel)

                    self.data_shelf[f'ac{acquisition_count}'] = data
                    acquisition_count += 1

                    # To follow number of acquisition
                    read = len(data)
                    total_read += read
                    print(f"Acquired data: {read} samples. Total {total_read}.", end="\r")

            except KeyboardInterrupt:
                pass
            finally:
                task.stop()
                print(f"\nAcquired {total_read} total samples.")
                print('Update: data acquisition done')


    def export_to_csv(self):
        
        assert self.export_target.endswith('.csv'), "Target file for acquisition data not a csv"

        with open(self.export_target, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Voltage_wire', 'Voltage_source'])

            for key in self.data_shelf.keys():

                for wire, source in zip(self.data_shelf[key][0], self.data_shelf[key][1]):

                    writer.writerow([wire, source])

        print('Update: export to csv done')

    def test_acquisition(self):
        self.test_data_shelf = {}
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.gold_channel)
            task.ai_channels.add_ai_voltage_chan(self.source_channel)

            task.timing.cfg_samp_clk_timing(self.samples_by_second, sample_mode=AcquisitionType.CONTINUOUS)
            task.start()
            print("Running task. Press Ctrl+C to stop.")

            try:
                total_read = 0
                acquisition_count = 0
                while True:
                    data = task.read(number_of_samples_per_channel=self.number_of_samples_per_channel)

                    self.test_data_shelf[f'ac{acquisition_count}'] = data
                    acquisition_count += 1
                        

                    # To follow number of acquisition
                    read = len(data)
                    total_read += read
                    print(f"Acquired data: {read} samples. Total {total_read}.", end="\r")

            except KeyboardInterrupt:
                pass
            finally:
                task.stop()
                for key in self.test_data_shelf.keys():
                    for i, j in zip(self.test_data_shelf[key][0], self.test_data_shelf[key][1]):
                        print(f"Gold voltage: {i: .11f} \t\t\t Source voltage: {j: .11f}")
                print(f"\nAcquired {total_read} total samples.")
                print('Update: data acquisition done')
# -
