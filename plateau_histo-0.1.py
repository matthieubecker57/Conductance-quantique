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
# 5. Calcul de la conductance pour chaque plateau et assignation d'un n
# =======================================================
# Paramètres pour la partie conductance
V_source = 1         # Tension de la source (en volts) pour cette partie
resistance = 100  # Résistance en ohms
G0 = 2 * elementary_charge**2 / Planck  # Quantum de conductance (S)
# Initialisation des listes pour stocker les résultats par plateau
plateau_conductance_list = []  # Liste qui contiendra la conductance moyenne calculée pour chaque plateau
n_candidate_list = []          # Liste qui contiendra le nombre de canaux estimé (n_candidate) pour chaque plateau
interesting_list = []          # Liste indiquant si le plateau est intéressant (erreur relative < 10%)
R_unknown_list = []            # Liste qui contiendra la résistance inconnue calculée pour chaque plateau

# Boucle sur chaque plateau détecté (la variable 'plateaus' est une liste de tuples)
# Chaque tuple contient : (start_index, end_index, avg_voltage, t_start, t_end)
for idx, (s, e, avg_voltage, t_start, t_end) in enumerate(plateaus, 1):
    
    # On ne traite que les plateaux dont la tension moyenne dépasse 0.9 V
    # (Attention : la condition dépend de votre montage, ici on considère avg_voltage > 0.9)
    if avg_voltage > 0.9:
        
        # Extraction du segment de tension correspondant au plateau
        voltage_segment = voltage[s:e+1]
        
        # Calcul de la conductance pour le segment en utilisant la fonction compute_conductance du module MathCore.
        # On passe "None" pour combler le paramètre 'self' car la fonction n'est pas dans une classe exposée.
        # V_source et resistance sont les paramètres du circuit.
        G_segment = mc.compute_conductance(
            None,
            voltage=voltage_segment,
            source_voltage=V_source,
            resistance=resistance
        )
        
        # Calcul de la conductance moyenne sur le plateau en prenant la moyenne des valeurs absolues
        G_avg = np.mean(np.abs(G_segment))
        plateau_conductance_list.append(G_avg)
        
        # Estimation du nombre de canaux (n_candidate) en arrondissant G_avg divisé par le quantum de conductance G0
        n_candidate = int(round(G_avg / G0))
        # Si l'estimation donne moins de 1, on force n_candidate à 1
        if n_candidate < 1:
            n_candidate = 1
        n_candidate_list.append(n_candidate)
        
        # Calcul de l'erreur relative entre la conductance moyenne mesurée et la conductance théorique n_candidate * G0
        error = abs(G_avg - n_candidate * G0) / (n_candidate * G0)
        # Un plateau est considéré comme intéressant si cette erreur relative est inférieure à 10%
        interesting = (error < 0.1)
        interesting_list.append(interesting)
        
        # Calcul de la résistance inconnue R_unknown à partir de la loi du diviseur de tension :
        # R_unknown = (V_source / |avg_voltage| - 1) / (n_candidate * G0)
        # On utilise la valeur absolue de avg_voltage pour éviter les problèmes de signe.
        R_unknown = (V_source / abs(avg_voltage) - 1) / (n_candidate * G0)
        R_unknown_list.append(R_unknown)

# Création d'un DataFrame pour regrouper les résultats obtenus pour chaque plateau
# On filtre pour ne prendre que les plateaux dont avg_voltage > 0.9 (condition utilisée dans la boucle)
df_conductance = pd.DataFrame({
    "Plateau": range(1, len(plateau_conductance_list) + 1),
    "avg_voltage": [p[2] for p in plateaus if p[2] > 0.9],
    "G_avg": plateau_conductance_list,
    "n_candidate": n_candidate_list,
    "interesting": interesting_list,
    "R_unknown": R_unknown_list
})

# Affichage du DataFrame dans la console
print(df_conductance)

# Export des résultats dans un fichier CSV
df_conductance.to_csv("plateau_averages_with_conductance.csv", index=False)
# =======================================================
# 6. Visualisation de la conductance par plateau avec niveaux théoriques (n·G₀)
# =======================================================
plt.figure(figsize=(12, 8))
max_n = max(n_candidate_list)
# Tracer les points pour chaque plateau
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
# 7. Histogramme des résistances inconnues
# =======================================================
plt.figure(figsize=(10, 6))
plt.hist(df_conductance["R_unknown"], bins='auto', edgecolor='black')
plt.xlabel("Résistance inconnue (Ω)")
plt.ylabel("Nombre de plateaux")
plt.title("Histogramme des résistances inconnues par plateau")
plt.show()