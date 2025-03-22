from Acquisition import Acquisition
from Histogram import Histogram
import pandas as pd

"""
Declaring general variables
"""

# Range width corresponds to the analog input range used. The NimyDAQ has either +- 2V or +- 10V, corresponding with a width of 4 or 20 V, respectively.
# Number of bits is given by the manifacturer
NImyDAQ_caracteristics = {
    "Range width": 4,
    "Number of bits": 16
}


"""
Aquire voltage across gold wire and source channel.

Arguments:
    - gold_channel: the voltage across the gold wire
    - source_channel: the voltage across the whole circuit (so the output of the source)
"""

# acquisition = Acquisition(
#     gold_channel = "DAQ_team_3_PHS2903/ai1",
#     source_channel = "DAQ_team_3_PHS2903/ai0",
#     samples_by_second = 100000,
#     # number_of_samples_per_channel = 2,
#     # export_target = r"acquisition_data.csv"
# )

# acquisition.continuous_acquisition()
# acquisition.export_to_csv()

"""
Creating a histogram to better visualise the data
"""

data_file = pd.read_csv(r"acquisition_data.csv")

H = Histogram(data=data_file['Voltage_wire'], bin_width=(NImyDAQ_caracteristics["Range width"] / 2**NImyDAQ_caracteristics["Number of bits"]))

H.create_histogram()
H.graph_histogram(
    title="Histogram de la tension",
    ylabel="count (log)",
    xlabel="value",
    log=True
)

