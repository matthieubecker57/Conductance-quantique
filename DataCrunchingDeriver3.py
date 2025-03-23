import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MathCore as mc
from Graphics import Graphics

"""
Ce code combine deux approches :
1. Détection des plateaux dans le signal de tension, en utilisant un lissage par moyenne glissante et un seuil adaptatif basé sur l'écart-type de la dérivée.
2. Calcul de la conductance à partir des valeurs de tension mesurées sur ces plateaux via la fonction compute_conductance (en passant None pour le paramètre self) et affichage de l'histogramme et d'un graphique linéaire.
"""

# =======================================================
# 1. Import des données
# =======================================================
df = pd.read_csv("acquisition_data.csv")
voltage = df["Voltage_wire"].values

# =======================================================
# 2. Lissage du signal et création du vecteur temps
# =======================================================
window_size = 10  # Taille de la fenêtre pour la moyenne glissante
voltage_smoothed = np.convolve(voltage, np.ones(window_size) / window_size, mode='same')

time_resolution = 10e-9  # 10 ns
t = np.arange(len(voltage)) * time_resolution

# =======================================================
# 3. Calcul de la dérivée sur le signal lissé et seuil adaptatif
# =======================================================
dv = np.diff(voltage_smoothed)
noise_std = np.std(dv)
threshold = 2 * noise_std  # Seuil adaptatif : 2 fois l'écart-type du bruit
print("Threshold adaptatif:", threshold)

min_plateau_samples = int(5e-6 / time_resolution)  # 5 µs correspond à 500 échantillons

# =======================================================
# 4. Détection des plateaux à partir de la dérivée
# =======================================================
# Chaque plateau sera stocké sous forme de tuple : 
# (indice_debut, indice_fin, moyenne_voltage, temps_debut, temps_fin)
plateaus = []
in_plateau = False
start_index = 0

for i, d in enumerate(dv):
    if abs(d) < threshold:
        if not in_plateau:
            in_plateau = True
            start_index = i
    else:
        if in_plateau:
            if i - start_index >= min_plateau_samples:
                plateau_data = voltage[start_index:i+1]
                plateau_time = t[start_index:i+1]
                avg_voltage = np.mean(plateau_data)
                plateaus.append((start_index, i, avg_voltage, plateau_time[0], plateau_time[-1]))
            in_plateau = False

# Cas où un plateau se prolonge jusqu'à la fin du signal
if in_plateau and (len(dv) - start_index >= min_plateau_samples):
    plateau_data = voltage[start_index:len(voltage)]
    plateau_time = t[start_index:len(voltage)]
    avg_voltage = np.mean(plateau_data)
    plateaus.append((start_index, len(voltage)-1, avg_voltage, plateau_time[0], plateau_time[-1]))

# Affichage dans la console des plateaux détectés
print("Plateaux détectés :")
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    duration = t_end - t_start
    print(f"Plateau {idx} : indices {s} à {e}, moyenne = {avg:.3f} V, durée = {duration:.6f} s")

# Optionnel : tracé des plateaux dont la moyenne < -1 V
plt.figure(figsize=(10, 6))
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    if avg < -1:
        plateau_time = t[s:e+1]
        plateau_voltage = voltage[s:e+1]
        plt.plot(plateau_time, plateau_voltage, label=f"Plateau {idx}")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Plateaux détectés (moyenne < -1V)")
plt.legend()
plt.show()

# =======================================================
# 5. Extraction des valeurs de tension sur les plateaux
# =======================================================
all_plateaus = []
for s, e, avg, t_start, t_end in plateaus:
    # Ici on extrait l'intégralité du segment correspondant au plateau
    all_plateaus.extend(voltage[s:e+1])
all_plateaus = np.array(all_plateaus)

# =======================================================
# 6. Calcul de la conductance via MathCore
# =======================================================
# La fonction compute_conductance attend un "self" en premier argument.
# Comme elle n'est pas dans une classe exposée, on passe "None" pour combler ce paramètre.
conductance = -1 * mc.compute_conductance(
    None,
    voltage=all_plateaus,
    source_voltage=2,
    resistance=100
)
# On s'assure que la conductance soit positive
conductance = np.abs(conductance)

# =======================================================
# 7. Visualisation de la conductance avec Graphics
# =======================================================
histo = Graphics(data=conductance)
histo.create_histogram(bin_width=1)
histo.graph_histogram(
    title="Histogramme de la conductance",
    ylabel="Nombre",
    xlabel="Conductance (a.u.)",
    log=False,
    isxlim=True, xlim=(0, 25),
    isylim=True, ylim=(0, 2000)
)

histo.regular_plot(
    y_range=histo.data,
    x_range=list(range(len(histo.data))),
    title="Graphique linéaire de la conductance",
    ylabel="Index",
    xlabel="Conductance (a.u.)"
)
