import numpy as np
import csv

# Points de repère (t, V)
points = [
    (0, 95),
    (5, 100),
    (10, 110),
    (15, 115),
    (20, 130),
    (25, 160),
    (28, 180),
    (30, 195),
    (32, 210),
    (34, 230),
    (36, 250),
    (38, 290),
    (40, 320),
    (42, 360),
    (44, 390),
    (46, 410),
    (48, 430),
    (50, 500),
    (52, 520),
    (55, 550),
    (58, 600)
]

# Convertir en arrays séparés
t_points = np.array([p[0] for p in points], dtype=float)
v_points = np.array([p[1] for p in points], dtype=float)

# Nombre de points souhaités
N = 1000

# Vecteur temps de 0 à 58 µs, inclus
t_interp = np.linspace(0, 58, N)

def linear_interpolate(t):
    """Interpolation linéaire par segments à partir des points de repère."""
    # Trouver l'indice du segment
    if t <= t_points[0]:
        return v_points[0]
    if t >= t_points[-1]:
        return v_points[-1]

# Cherche dans quel intervalle se trouve t
    idx = np.searchsorted(t_points, t) - 1
    t1, v1 = t_points[idx], v_points[idx]
    t2, v2 = t_points[idx+1], v_points[idx+1]

# Facteur d'interpolation
    alpha = (t - t1) / (t2 - t1)
    return v1 + alpha*(v2 - v1)

# Génération du CSV
# print("Time (µs),Voltage (mV)")
with open(r'test.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Time', 'Voltage_wire'])
    for time in t_interp:
        volt = linear_interpolate(time)
        writer.writerow([time, volt])

print('Update: csv file done')

# -
