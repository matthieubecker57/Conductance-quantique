import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Range width corresponds to the analog input range used. The NimyDAQ has either +- 2V or +- 10V, corresponding with a width of 4 or 20 V, respectively.
Number of bits is given by the manifacturer
"""

NImyDAQ_caracteristics = {
    "Range width": 4,
    "Number of bits": 16
}

class Histogram:
    """
    This class is used to create a histogram of the data:

    Arguments:
        - data_file: the file to import the data from
        - DAQ_caracteristics: the caracteristics of the NImyDAQ. The default is defines right above this class

    """

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
        # print('extremums:', self.wire_min, self.wire_max)

    def compute_DAQ_resolution(self):
        """
        Computes the resolution of the DAQ used, according to manufacturer caracteristics.
        """

        return self.caracteristics["Range width"] / (2**self.caracteristics["Number of bits"])

    def bin_to(self, value, minimum_value, bin_width):
        """
        Divide the Range width of the DAQ_caracteristics into a number of self.number_of_bins different bins.
        Number the list of bins, starting at 0.
        This method returns the number of the bin a certain value should be put in, by exploiting division with remainder.

        Arguments:
            - value: the value to place in the bin
            - minimum_value: the lowest value to put in a bin
            - bin_width: the width of each bin

        Edge case:
            If a value is the maximum value in the measure range, this method returns one bin too high.
        """

        value -= minimum_value
        return int(value / bin_width - (value % bin_width)/bin_width)
    
    def create_histogram(self, values, bin_width):
        self.histogram = {}

        number_of_bins = self.number_of_bins

        for i in range(number_of_bins):
            # print(i)
            self.histogram[f"bin{i}"] = {"count": 0, "average": 0}
        

        for value in values:
            bin_number = int((value-min(values)) / bin_width - ((value-min(values)) % bin_width)/bin_width)
            print(bin_number)
            if value == max(values):
                bin_number -= 1
            
            print(bin_number)
            previous_count = self.histogram[f'bin{bin_number}']['count']
            self.histogram[f'bin{bin_number}']['count'] += 1

            previous_average = self.histogram[f'bin{bin_number}']['average']
            self.histogram[f'bin{bin_number}']['average'] = (previous_count*previous_average + value)/(previous_count+1)

    def create_voltage_histogram(self):
        """
        This method bins all the values of a list to form a histogram.
        By doing so, it created a dictionnary that computes the average value of said bin and the number of elements that went into it.
        """
        self.wire_histogram = {}

        for i in range(self.number_of_bins):
            self.wire_histogram[f"bin{i}"] = {"count": 0, "average": 0}

        for value in self.voltage_wire_list:
            bin_number = int( (value-self.wire_min) / self.bin_width - ((value-self.wire_min) % self.bin_width)/self.bin_width)

            if value == self.wire_max:
                bin_number -= 1

            previous_count = self.wire_histogram[f'bin{bin_number}']['count']
            self.wire_histogram[f'bin{bin_number}']['count'] += 1

            previous_average = self.wire_histogram[f'bin{bin_number}']['average'] 
            self.wire_histogram[f'bin{bin_number}']['average'] = (previous_count*previous_average + value)/(previous_count+1)

    def graph_voltage_histogram(self):
        """
        Graphs the voltage histogram
        """

        y_range = [i['count'] for i in self.wire_histogram.values()]
        x_range = [i['average'] for i in self.wire_histogram.values()]
        plt.plot(x_range, y_range, 'o', markersize=5)
        # plt.ylim(0,50)
        plt.show()

    def graph_general_histogram(self, values, counts, xlabel: str, ylabel: str, title: str, logarithmic: bool = False):
        """
        Creates a simple histogram in the same style as graph_voltage_histogram
        """

        if logarithmic:
            plt.semilogy(values, counts, 'o', markersize=0.1) 
        else:
            plt.plot(values, counts, 'o', markersize=0.1) 

        plt.title(title)
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)

        plt.show()
        


# histo = Histogram()
# histo.create_voltage_histogram()
# histo.graph_voltage_histogram()
# plt.show()