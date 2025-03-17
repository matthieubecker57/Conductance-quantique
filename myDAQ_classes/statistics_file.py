
class Statistics:
    def __init__(self):
        pass

    def compute_mean_and_deviation(self, n:list):  # calcule la moyenne et l'écart type des valeurs de n
        mean = sum(n) / len(n)
        self.mean = mean

        sum_of_squares = sum([(i - mean)**2 for i in n])
        deviation = ((1 / (len(n)-1)) * (sum_of_squares))**0.5
        self.standard_deviation = deviation
        return [mean, deviation]

    def compute_current(self, voltage, resistance):
        return voltage / resistance
    
    def compute_resistance(self, voltage, current):
        return voltage / current

