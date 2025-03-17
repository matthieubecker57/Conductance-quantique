import numpy as np
import matplotlib.pyplot as plt

class GrahicCalculations:
    def __init__(self):
        pass

    def compute_linear_regression(self, x: np.array, y, y_uncertainty):
        print(y_uncertainty)
        w = 1 / y_uncertainty**2
        print(w)
        delta = sum(w) * np.dot(w, x**2) - np.dot(x,w)**2
        print('--->', sum(w), np.dot(w, x**2), np.dot(x,w)**2)
        print('delta =', delta)
        slope_t1 = sum(w) * sum(x*y*w)
        slope_t2 = sum(w*y)*sum(w*x)
        print(f"""
                slopet1 = {slope_t1}
                slopet2 = {slope_t2}
                diff = {slope_t1 - slope_t2}
              """
              )

        self.slope = (slope_t1 - slope_t2) / delta
        print('slope =', self.slope)
        self.slope_uncertainty = (np.dot(w, x**2) / delta)**0.5

        self.origin = (np.dot(w, x**2) * np.dot(w,y) - np.dot(w,x)* sum(x*y*w)) / delta
        print('origin =', self.origin)
        self.origin_uncertainty = (sum(w) / delta) ** 0.5
    
    def linear_regression(self, x):
        return self.origin + self.slope * x
    
    

