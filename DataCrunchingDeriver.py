
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données depuis le fichier CSV
df = pd.read_csv(r"acquisition_data.csv")
voltage = df["Voltage_wire"].values

# Création d'un vecteur temps
# Résolution temporelle : 10 ns
time_resolution = 10e-9  # 10 ns
t = np.arange(len(voltage)) * time_resolution

# Calcul de la dérivée discrète du signal
# np.diff retourne un tableau de taille len(voltage)-1
dv = np.diff(voltage)

# Définition du seuil sur la dérivée pour considérer le signal comme constant
# Ce seuil dépend du bruit du signal. Ici, par exemple, 1e-3 (1 mV) peut être un point de départ.
threshold = 1e-3

# Durée minimale d'un plateau : 5 µs => nombre d'échantillons correspondant
min_plateau_samples = int(5e-6 / time_resolution)  # 5 µs / 10 ns = 500 échantillons

# Liste pour stocker les plateaux détectés
# Chaque plateau sera représenté par un tuple : (indice_debut, indice_fin, moyenne_voltage, temps_debut, temps_fin)
plateaus = []
in_plateau = False
start_index = 0

# Parcours du signal via la dérivée
for i, d in enumerate(dv):
    if abs(d) < threshold:
        # On est dans une zone "plate"
        if not in_plateau:
            # Début potentiel d'un plateau
            in_plateau = True
            start_index = i
    else:
        # La dérivée est supérieure au seuil, donc on sort d'une zone plate
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

# Affichage des plateaux détectés
print("Plateaux détectés :")
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    duration = t_end - t_start
    print(f"Plateau {idx} : indice {s} à {e}, moyenne = {avg:.3f} V, durée = {duration:.6f} s")

# Visualisation du signal et des plateaux
plt.figure(figsize=(10, 6))
plt.plot(t, voltage, label="Signal de tension", lw=1)
for s, e, avg, t_start, t_end in plateaus:
    # On met en surbrillance les zones de plateau
    plt.axvspan(t_start, t_end, color='red', alpha=0.3, label='Plateau' if s == plateaus[0][0] else "")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Détection des plateaux avec la méthode de la dérivée")
plt.legend()
plt.show()
