from Acquisition import Acquisition
from Graphics import Graphics
from MathCore import compute_conductance
import pandas as pd

"""
----------------------------------------------------------------------------------------------
Declaring general variables
----------------------------------------------------------------------------------------------
"""

""" NImyDAQ_caracteristics:
Range width corresponds to the analog input range used. The NimyDAQ has either +- 2V or +- 10V,
    corresponding with a width of 4 or 20 V, respectively.
Number of bits is given by the manifacturer
"""
Range_width = 4
Number_of_bits = 16



"""
----------------------------------------------------------------------------------------------
Acquiring data
----------------------------------------------------------------------------------------------
"""

"""
Acquires voltage across gold wire and source channel.

Arguments:
    - gold_channel: the voltage across the gold wire
    - source_channel: the voltage across the whole circuit (so the output of the source)
"""

acquisition = Acquisition(
    gold_channel = "DAQ_team_3_PHS2903/ai0",  # Across the gold wires
    source_channel = "DAQ_team_3_PHS2903/ai1",  # Across the source
    samples_by_second = 100000,
    # number_of_samples_per_channel = 2,
    # export_target = r"acquisition_data.csv"
)

acquisition.continuous_acquisition()
acquisition.export_to_csv()

"""
----------------------------------------------------------------------------------------------
Create a histogram to better visualise the data
----------------------------------------------------------------------------------------------
"""

data_file = pd.read_csv(r"acquisition_data.csv")


"""
Histogram of the voltage. V will be the instance of the Graphics class that will compute graphics with voltage
"""

V = Graphics(
    data=data_file['Voltage_wire'],
)

V.create_histogram(
    bin_width=3.24*10**(-4)
)

V.graph_histogram(
    title="Histogram de la tension",
    ylabel="count (log)",
    xlabel="value",
    log=True
)

"""
Histogram of the conductance. C will be the instance of the Graphics class that will compute graphics with conductance
"""

# C = Graphics(
#     data=compute_conductance(
#         source_voltage=1,
#         voltage=data_file['Voltage_wire'],
#         resistance=10000
#     )
# )

# C.create_histogram(
#     bin_width=0.001
# )

# C.graph_histogram(
#     title="Histogram de la tension",
#     ylabel="count (log)",
#     xlabel="value",
#     markersize=5,
#     log=False
# )