from Acquisition import Acquisition

"""
This file's purpous is to allow for a quick way to check if the acquisition process is working as it should be.
"""

acquisition = Acquisition(
    gold_channel = "DAQ_team_3_PHS2903/ai0",  # Across the gold wires
    source_channel = "DAQ_team_3_PHS2903/ai1",  # Across the source
    samples_by_second = 100000,
    # number_of_samples_per_channel = 2,
    # export_target = r"acquisition_data.csv"
)

acquisition.test_acquisition()
