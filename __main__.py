from Acquisition import Acquisition

acquisition = Acquisition(
    gold_channel = "tempSensor1/ai0",  # Channel lié à la résistance connue
    source_channel = "tempSensor1/ai1",  # Channel lié à la résistance inconnue
    # number_of_samples_per_channel = 2,  # nombre de points pris par mesures
    samples_by_second = 10000,
    # export_target = r"acquisition_data.csv"
)

acquisition.continuous_acquisition()
acquisition.export_to_csv()