import nidaqmx
import nidaqmx.system
from nidaqmx.constants import AcquisitionType, VoltageUnits, ResistanceUnits
import matplotlib.pyplot as plt
from DataShelf import DataShelf

class Measure:
    def __init__(self, sample_count: int = 1000, datashelf = 'optional'):
        self.sample_count = sample_count
        self.datashelf = datashelf


    def read_voltage(self, *args):
        channels = args

        with nidaqmx.Task() as task:

            for channel in channels:
                task.ai_channels.add_ai_voltage_chan(channel)

            task.timing.cfg_samp_clk_timing(1000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000)
            data = task.read(number_of_samples_per_channel=self.sample_count)
            self.data = data
    
    def plot_data(self, x_range, *args):  # args are lists of the data (y_values) 
        for data_list in args:
            plt.plot(x_range, data_list)
        
        plt.show()

