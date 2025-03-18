from Acquisition import Acquisition

"""
Aquire voltage across gold wire and source channel.

Arguments:
    - gold_channel: the voltage across the gold wire
    - source_channel: the voltage across the whole circuit (so the output of the source)
"""

acquisition = Acquisition(
    gold_channel = "tempSensor1/ai0",
    source_channel = "tempSensor1/ai1",
    # samples_by_second = 10000,
    # number_of_samples_per_channel = 2,
    # export_target = r"acquisition_data.csv"
)

acquisition.continuous_acquisition()
acquisition.export_to_csv()