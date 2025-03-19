from Acquisition import Acquisition
from Histogram import Histogram

"""
Aquire voltage across gold wire and source channel.

Arguments:
    - gold_channel: the voltage across the gold wire
    - source_channel: the voltage across the whole circuit (so the output of the source)
"""

acquisition = Acquisition(
    gold_channel = "DAQ_team_3_PHS2903/ai1",
    source_channel = "DAQ_team_3_PHS2903/ai0",
    samples_by_second = 100000,
    # number_of_samples_per_channel = 2,
    # export_target = r"acquisition_data.csv"
)

acquisition.continuous_acquisition()
acquisition.export_to_csv()

histo = Histogram()
histo.create_histogram()
histo.graph_histogram()
