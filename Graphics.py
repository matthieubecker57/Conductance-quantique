import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Graphics:
    """
    This is a class whose purpose is to make graphics of a data set.

    Arguments:
        - data: the data set. It must consist of a 1d array
        - bin_width: the width of each bin of the data set
    """

    def __init__(self, data: float):
        self.data = np.array(data)

    def bin_data(self):
        """
            Uses division with remainder to bin the data points. The number associatied with each
        data point ranges from 0 to number_of_bins*bin_width
        """

        self.bin_etiquets = (self.data - self.data_min)/self.bin_width - ((self.data - self.data_min) % self.bin_width) / self.bin_width
        self.bin_etiquets = self.bin_etiquets.astype(int)  # converts type of elements of the array from float to int
        print("Update: data is binned")
        # print(self.bin_etiquets)

    def create_histogram(self, bin_width:float):
        """
        Zips trough the data and the associated bin numbers. Computes the number of data point in each bin and it's average.
        Created a dictionnary containing this information. The key to each value is "bin" + {number of the bin}
        """

        self.bin_width = bin_width
        self.data_min = min(self.data)
        self.data_max = max(self.data)

        number_of_bins_unrounded = (self.data_max - self.data_min) / self.bin_width
        self.number_of_bins = int(number_of_bins_unrounded - number_of_bins_unrounded%1 + 1)

        self.bin_data()

        self.histogram = {}

        for i in self.bin_etiquets:
            self.histogram[f"bin{i}"] = {"count": 0, "average": 0}

        for n, value in zip(self.bin_etiquets, self.data):
            previous_count = self.histogram[f'bin{n}']['count']
            previous_average = self.histogram[f'bin{n}']['average'] 

            self.histogram[f'bin{n}']['count'] += 1
            self.histogram[f'bin{n}']['average'] = (previous_count*previous_average + value)/(previous_count+1)
        
        print("Update: histogram done")
        # print (self.histogram)
    
    def graph_histogram(self, title:str, ylabel:str, xlabel:str,
                        markersize:str = 1, marker:str = 'o', color:str = 'black',
                        log:bool = False,
                        isylim:bool = False, ylim:tuple = (0,0),
                        isxlim:bool = False, xlim:tuple = (0,0),
                        grid_on:bool = True, which_grid: str = 'major'):
        """
        Streamlines the use of matplotlib for the histogram. Allows for linear and logarithmic scales on the y axis, as
        well as limits on the x and y scales.
        """

        y_range = [i['count'] for i in self.histogram.values()]
        x_range = [i['average'] for i in self.histogram.values()]

        if log:
            plt.semilogy(x_range, y_range, marker, color=color, markersize=markersize)
        else:
            plt.plot(x_range, y_range, marker, color=color, markersize=markersize)

        if isylim:
            plt.ylim(ylim)
        if isxlim:
            plt.xlim(xlim)
        
        if grid_on:
            plt.grid(which=which_grid)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        plt.show()
    
    def regular_plot(self, title:str, ylabel:str, xlabel:str,
                        markersize:str = 1, marker:str = 'o', color:str = 'black',
                        log:bool = False,
                        isylim:bool = False, ylim:tuple = (0,0),
                        isxlim:bool = False, xlim:tuple = (0,0),
                        grid_on:bool = True, which_grid:str = 'major'):
        
        """
        Just like graph_histograms, but directly plots the data
        Streamlines the use of matplotlib for the histogram. Allows for linear and logarithmic scales on the y axis, as
        well as limits on the x and y scales.
        """
        
        x_range = [i for i in range(len(self.data))]
        y_range = self.data
                
        if log:
            plt.semilogy(x_range, y_range, marker, color=color, markersize=markersize)
        else:
            plt.plot(x_range, y_range, marker, color=color, markersize=markersize)
        
        if isylim:
            plt.ylim(ylim)
        if isxlim:
            plt.xlim(xlim)

        if grid_on:
            plt.grid(which=which_grid)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        plt.show()

