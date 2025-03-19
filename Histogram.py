import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NImyDAQ_caracteristics = {
    "Range width": 4,
    "Number of bits": 16
}

class Histogram:
    def __init__(self, data_file: str = r"acquisition_data.csv", DAQ_caracteristics: dict = NImyDAQ_caracteristics):
        self.data_file = pd.read_csv(data_file)
        self.voltage_wire_list = np.array(self.data_file['Voltage_wire'])
        self.votlage_source_list = np.array(self.data_file['Votlage_source'])

        self.caracteristics = DAQ_caracteristics
        assert "Range width" in self.caracteristics.keys(), "Value Error: DAQ_caracteristics argument does not have required elements"
        assert "Number of bits" in self.caracteristics.keys(), "Value Error: DAQ_caracteristics argument does not have required elements"

        self.bin_width = self.compute_DAQ_resolution()

        self.number_of_bins = int(self.caracteristics["Range width"] / self.bin_width) + 1

        self.wire_max = max(self.voltage_wire_list)
        self.wire_min = min(self.voltage_wire_list)
        print('extremums:', self.wire_min, self.wire_max)

    def compute_DAQ_resolution(self):
        return self.caracteristics["Range width"] / (2**self.caracteristics["Number of bits"])

    def bin_to(self, value):
        value -= self.wire_min
        return int(value / self.bin_width - (value % self.bin_width)/self.bin_width)

    def create_histogram(self):
        self.wire_histogram = {}

        for i in range(self.number_of_bins):
            self.wire_histogram[f"bin{i}"] = {"count": 0, "average": 0}

        for value in self.voltage_wire_list:
            bin_number = self.bin_to(value)

            if value == self.wire_max:
                bin_number = self.bin_to(value - self.bin_width*10**(-10))

            previous_count = self.wire_histogram[f'bin{bin_number}']['count']
            self.wire_histogram[f'bin{bin_number}']['count'] += 1

            previous_average = self.wire_histogram[f'bin{bin_number}']['average'] 
            self.wire_histogram[f'bin{bin_number}']['average'] = (previous_count*previous_average + value)/(previous_count+1)

    def graph_histogram(self):
        y_range = [i['count'] for i in self.wire_histogram.values()]
        x_range = [i['average'] for i in self.wire_histogram.values()]
        plt.plot(x_range, y_range, 'o', markersize=0.1)
        plt.ylim(0,50)
        plt.show()

histo = Histogram()
histo.create_histogram()
histo.graph_histogram()
plt.show()