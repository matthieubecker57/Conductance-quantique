import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données depuis le fichier CSV
df = pd.read_csv("acquisition_data.csv")
voltage = df["Voltage_wire"].values

# --------------------------
# Lissage du signal
# --------------------------
# On applique une moyenne glissante pour réduire le bruit.
window_size = 10  # Taille de la fenêtre de lissage (à ajuster selon vos besoins)
voltage_smoothed = np.convolve(voltage, np.ones(window_size)/window_size, mode='same')

# --------------------------
# Création du vecteur temps
# --------------------------
# Résolution temporelle : 10 ns
time_resolution = 10e-9  # 10 ns
t = np.arange(len(voltage)) * time_resolution

# --------------------------
# Calcul de la dérivée sur le signal lissé
# --------------------------
dv = np.diff(voltage_smoothed)

# Calcul de l'écart-type de la dérivée (bruit)
noise_std = np.std(dv)
# Définition d'un seuil adaptatif : spar exemple, 2 fois l'écart-type du bruit
threshold = 2 * noise_std
print("Threshold adaptatif:", threshold)

# Durée minimale d'un plateau : 5 µs => nombre d'échantillons correspondant
min_plateau_samples = int(5e-6 / time_resolution)  # 5 µs / 10 ns = 500 échantillons

# --------------------------
# Détection des plateaux à partir de la dérivée
# --------------------------
# Chaque plateau est représenté par un tuple :
# (indice_debut, indice_fin, moyenne_voltage, temps_debut, temps_fin)
plateaus = []
in_plateau = False
start_index = 0

for i, d in enumerate(dv):
    if abs(d) < threshold:
        # On est dans une zone "plate"
        if not in_plateau:
            # Début potentiel d'un plateau
            in_plateau = True
            start_index = i
    else:
        # La dérivée est supérieure au seuil, on sort de la zone plate
        if in_plateau:
            if i - start_index >= min_plateau_samples:
                # La durée du plateau est suffisante
                plateau_data = voltage[start_index:i+1]
                plateau_time = t[start_index:i+1]
                avg_voltage = np.mean(plateau_data)
                plateaus.append((start_index, i, avg_voltage, plateau_time[0], plateau_time[-1]))
            in_plateau = False

# Vérifier si un plateau se prolonge jusqu'à la fin du signal
if in_plateau and (len(dv) - start_index >= min_plateau_samples):
    plateau_data = voltage[start_index:len(voltage)]
    plateau_time = t[start_index:len(voltage)]
    avg_voltage = np.mean(plateau_data)
    plateaus.append((start_index, len(voltage)-1, avg_voltage, plateau_time[0], plateau_time[-1]))

# --------------------------
# Affichage des plateaux détectés dans la console
# --------------------------
print("Plateaux détectés :")
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    duration = t_end - t_start
    print(f"Plateau {idx} : indice {s} à {e}, moyenne = {avg:.3f} V, durée = {duration:.6f} s")

# # --------------------------
# # Visualisation du signal complet avec surlignage des plateaux
# # --------------------------
# plt.figure(figsize=(10, 6))
# plt.plot(t, voltage, label="Signal original", lw=1)
# plt.plot(t, voltage_smoothed, label="Signal lissé", lw=1)
# for s, e, avg, t_start, t_end in plateaus:
#     plt.axvspan(t_start, t_end, color='red', alpha=0.3, label='Plateau' if s == plateaus[0][0] else "")
# plt.xlabel("Temps (s)")
# plt.ylabel("Tension (V)")
# plt.title("Détection des plateaux avec lissage et seuil adaptatif")
# plt.legend()
# plt.show()

# --------------------------
# Tracé uniquement des plateaux dont la moyenne est inférieure à -1 V
# --------------------------
plt.figure(figsize=(10, 6))
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    # if avg < -1:  # Condition : moyenne du plateau < -1 V
        plateau_time = t[s:e+1]
        plateau_voltage = voltage[s:e+1]
        plt.plot(plateau_time, plateau_voltage, label=f"Plateau {idx}")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Forme des plateaux détectés (moyenne < -1V)")
plt.legend()
plt.show()
