import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Crunch:
    def __init__(self, data_file: str = r"test.csv", number_of_bins: int = 30):
        
        self.csv_file = pd.read_csv(data_file)
        self.voltage_wire_list = self.csv_file['Voltage_wire']

        self.wire_maximum = max(self.voltage_wire_list)
        self.wire_minimum = min(self.voltage_wire_list)
        self.number_of_bins = number_of_bins
        self.bin_width = (max(self.voltage_wire_list) - min(self.voltage_wire_list)) / self.number_of_bins

    def bin_to(self, value):
        value -= self.wire_minimum
        return int(value / self.bin_width - (value % self.bin_width)/self.bin_width)
    
    def create_wire_voltage_histogram(self):
        self.wire_histogram = {}

        for i in range(self.number_of_bins):
            self.wire_histogram[f"bin{i}"] = {"count": 0, "average": 0}

        for value in self.voltage_wire_list:
            bin_number = self.bin_to(value)

            if value == self.wire_maximum:
                bin_number = self.bin_to(value - self.bin_width*10**(-10))

            previous_count = self.wire_histogram[f'bin{bin_number}']['count']
            self.wire_histogram[f'bin{bin_number}']['count'] += 1

            previous_average = self.wire_histogram[f'bin{bin_number}']['average'] 
            self.wire_histogram[f'bin{bin_number}']['average'] = (previous_count*previous_average + value)/(previous_count+1)
    
    def find_plateau(self):

        self.create_wire_voltage_histogram()

        count_mean = np.mean([i['count'] for i in self.wire_histogram.values()])

        plateaus = []
        plateau_count = 0

        for i in self.wire_histogram.keys():
            count = self.wire_histogram[i]["count"]
            average = self.wire_histogram[i]["average"]

            if count >= count_mean:
                plateau_count += 1
                plateaus.append((plateau_count, average))

        self.plateaus = plateaus

    def graph_histogram(self):
        y_range = [i['count'] for i in self.wire_histogram.values()]
        x_range = [i['average'] for i in self.wire_histogram.values()]
        plt.plot(x_range, y_range, ':')
        plt.show()
    
    def graph_values(self):
        x_range = np.linspace(0,1,len(self.voltage_wire_list))
        plt.plot(x_range, self.voltage_wire_list, 'o', markersize=0.1)
        plt.show()


crunch = Crunch()

crunch.find_plateau()
print(crunch.plateaus)
# crunch.graph_histogram()
