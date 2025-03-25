import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MathCore as mc
from Graphics import Graphics
from scipy.constants import Planck, elementary_charge


"""
Ce code combine deux approches :
1. Détection des plateaux dans le signal de tension, en utilisant un lissage par moyenne glissante et un seuil adaptatif basé sur l'écart-type de la dérivée.
2. Calcul de la conductance à partir des valeurs de tension mesurées sur ces plateaux via la fonction compute_conductance (en passant None pour le paramètre self) et affichage de l'histogramme et d'un graphique linéaire.
"""

# --------------------------------------------------
# Fonctions pour calculer la moyenne et l'écart type à partir d'un histogramme
# --------------------------------------------------
def compute_histogram_mean(bin_edges, counts):
    """
    Calcule la moyenne pondérée à partir d'un histogramme.
    
    Parameters:
        bin_edges: array-like, les bornes des bins (de longueur n+1)
        counts: array-like, le nombre d'occurrences dans chaque bin (de longueur n)
        
    Returns:
        La moyenne calculée.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_counts = counts.sum()
    if total_counts == 0:
        return None
    return (bin_centers * counts).sum() / total_counts

def compute_histogram_std(bin_edges, counts):
    """
    Calcule l'écart type à partir d'un histogramme.
    
    Parameters:
        bin_edges: array-like, les bornes des bins (de longueur n+1)
        counts: array-like, le nombre d'occurrences dans chaque bin (de longueur n)
        
    Returns:
        L'écart type calculé.
    """
    mean = compute_histogram_mean(bin_edges, counts)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_counts = counts.sum()
    if total_counts == 0:
        return None
    variance = ((counts * (bin_centers - mean) ** 2).sum()) / total_counts
    return np.sqrt(variance)

# =======================================================
# 1. Import des données
# =======================================================
df = pd.read_csv("acquisition_data.csv")
voltage = df["Voltage_wire"].values

# =======================================================
# 2. Lissage du signal et création du vecteur temps
# =======================================================
window_size = 1  # Taille de la fenêtre pour la moyenne glissante (modifiez pour plus de précision)
voltage_smoothed = np.convolve(voltage, np.ones(window_size) / window_size, mode='same')

time_resolution = 10e-9  # 10 ns
t = np.arange(len(voltage)) * time_resolution

# =======================================================
# Graphique du signal lissé
# =======================================================

# plt.figure(figsize=(10, 6))
# plt.plot(t, voltage, label="Signal brut")
# plt.plot(t, voltage_smoothed, label="Signal lissé")
# plt.xlabel("Temps (s)")
# plt.ylabel("Tension (V)")
# plt.title("Signal brut et signal lissé")
# plt.legend()







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

# =======================================================
# Export des informations sur les plateaux en CSV
# =======================================================
plateaus_data = []
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    if avg > 0.9:
        plateaus_data.append({
            "Plateau": idx,
            "start_index": s,
            "end_index": e,
            "avg_voltage": avg,
            "start_time": t_start,
            "end_time": t_end,
            "duration_s": t_end - t_start
        })
df_plateaus = pd.DataFrame(plateaus_data)
df_plateaus.to_csv("plateau_averages.csv", index=False)
print("Les moyennes des plateaux ont été exportées dans 'plateau_averages.csv'.")

# =======================================================
# Optionnel : tracé des plateaux détectés
# =======================================================
plt.figure(figsize=(10, 6))
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    if avg > 0.9:
        plateau_time = t[s:e+1]
        plateau_voltage = voltage[s:e+1]
        plt.plot(plateau_time, plateau_voltage, label=f"Plateau {idx}")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Plateaux détectés")
plt.show()

# =======================================================
# 5. Extraction des valeurs de tension sur les plateaux
# =======================================================
all_plateaus = []
for s, e, avg, t_start, t_end in plateaus:
    # if avg < 0.9:
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
    resistance=20000,
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

)

histo.regular_plot(
    y_range=histo.data,
    x_range=list(range(len(histo.data))),
    title="Graphique linéaire de la conductance",
    ylabel="Index",
    xlabel="Conductance (a.u.)"
)
plt.show()

# =======================================================
# 8. Visualisation en histogramme des voltages moyens à partir de plateau_averages.csv
# =======================================================

df_plateaus = pd.read_csv("plateau_averages.csv")
data = df_plateaus["avg_voltage"]

# --------------------------------------------------
# Calcul de l'histogramme avec bins automatiques
# --------------------------------------------------
counts, bin_edges = np.histogram(data, bins='auto')

# Calcul de la moyenne et de l'écart type à partir de l'histogramme
mean_hist = compute_histogram_mean(bin_edges, counts)
std_hist = compute_histogram_std(bin_edges, counts)

print("Moyenne (histogramme) :", mean_hist)
print("Écart type (histogramme) :", std_hist)

# --------------------------------------------------
# Tracé de l'histogramme avec affichage de la moyenne et des bornes moyenne ± écart type
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(data, bins='auto', edgecolor='black', label="Voltage moyen")
plt.xlabel("Voltage moyen (V)")
plt.ylabel("Nombre de plateaux")
plt.legend()
plt.show()
# =======================================================
# 2. Calcul de la conductance et association de n pour chaque plateau
# =======================================================
# Paramètres du circuit pour la partie conductance
V_source = 1        # Tension de la source en volts (pour cette partie)
resistance = 100  # Résistance en ohms
G0 = 2 * elementary_charge**2 / Planck  # Quantum de conductance (S)

# Pour chaque plateau (déjà détecté et vérifié par la condition avg < 0.9),
# on calcule la conductance moyenne, on estime le nombre de canaux (n_candidate),
# et on détermine si le plateau est "intéressant" lorsque le ratio G_avg/(n_candidate*G0) > 0.9.
plateau_conductance_list = []
n_candidate_list = []
interesting_list = []
R_unknown_list = []

for idx, (s, e, avg_voltage, t_start, t_end) in enumerate(plateaus, 1):
    if avg_voltage < 0.9:
        voltage_segment = voltage[s:e+1]
        G_segment = mc.compute_conductance(None,
                                           voltage=voltage_segment,
                                           source_voltage=V_source,
                                           resistance=resistance)
        G_avg = np.mean(np.abs(G_segment))
        plateau_conductance_list.append(G_avg)
        n_candidate = int(round(G_avg / G0))
        if n_candidate < 1:
            n_candidate = 1
        n_candidate_list.append(n_candidate)
        ratio = G_avg / (n_candidate * G0)
        interesting = (ratio > 0.9)
        interesting_list.append(interesting)
        # Calcul de R_unknown via la loi du diviseur de tension : 
        # R_unknown = (V_source/|avg_voltage| - 1) / (n_candidate * G0)
        R_unknown = (V_source / abs(avg_voltage) - 1) / (n_candidate * G0)
        R_unknown_list.append(R_unknown)

# Création d'un DataFrame pour stocker ces résultats
df_conductance = pd.DataFrame({
    "Plateau": range(1, len(plateau_conductance_list) + 1),
    "avg_voltage": [p[2] for p in plateaus if p[2] < 0.9],
    "G_avg": plateau_conductance_list,
    "n_candidate": n_candidate_list,
    "interesting": interesting_list,
    "R_unknown": R_unknown_list
})
print(df_conductance)
df_conductance.to_csv("plateau_averages_with_conductance.csv", index=False)

# =======================================================
# 3. Visualisation de la conductance avec lignes théoriques pour chaque n
# =======================================================
plt.figure(figsize=(12, 8))
max_n = max(n_candidate_list)
# Tracer des lignes horizontales pour chaque niveau théorique de conductance (n * G0)
for n in range(1, max_n + 1):
    plt.axhline(y=n * G0, color='gray', linestyle='--', linewidth=1,
                label=f"n = {n}" if n == 1 else None)
# Pour chaque plateau, tracer un point à l'abscisse correspondant au numéro de plateau et à l'ordonnée la conductance moyenne
for idx, row in df_conductance.iterrows():
    color = 'red' if row["interesting"] else 'blue'
    plt.plot(row["Plateau"], row["G_avg"], 'o', color=color, markersize=10,
             label="Plateau intéressant" if row["interesting"] and idx == 0 else None)
    plt.text(row["Plateau"], row["G_avg"], f" n={row['n_candidate']}", fontsize=9, color=color)

plt.xlabel("Plateau")
plt.ylabel("Conductance moyenne (S)")
plt.title("Conductance par plateau avec niveaux théoriques (n·G₀)")
plt.legend()
plt.grid(True)
plt.show()

# =======================================================
# 4. Histogramme des résistances inconnues
# =======================================================
plt.figure(figsize=(10, 6))
plt.hist(df_conductance["R_unknown"], bins='auto', edgecolor='black')
plt.xlabel("Résistance inconnue (Ω)")
plt.ylabel("Nombre de plateaux")
plt.title("Histogramme des résistances inconnues par plateau")
plt.show()