import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MathCore as mc
from Graphics import Graphics
from scipy.constants import Planck, elementary_charge
import seaborn as sns  # On utilisera seaborn pour tracer la courbe de densité et le boxplot



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
df = pd.read_csv("acquisition_data.csv", low_memory=False)

# Conversion explicite de la colonne "Voltage_wire" en float
# (les valeurs impossibles à convertir seront remplacées par NaN)
df["Voltage_wire"] = pd.to_numeric(df["Voltage_wire"], errors="coerce")

# Extraction des valeurs sous forme de tableau NumPy
voltage = df["Voltage_wire"].values
# =======================================================
# 2. Lissage du signal et création du vecteur temps
# =======================================================

window_size = 1  # Petite fenêtre pour conserver la précision
voltage_smoothed = np.convolve(voltage, np.ones(window_size) / window_size, mode='same')

# Ici, la résolution temporelle est donnée (par exemple, 1/100000 secondes)
time_resolution = 1/100000  
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

# =======================================================
# Export des informations sur les plateaux en CSV
# =======================================================
    
    
lower_threshold = 0.05
upper_threshold = 2.45

plateaus_data = []
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    if lower_threshold < avg < upper_threshold:
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
print("Les moyennes des plateaux filtrés ont été exportées dans 'plateau_averages.csv'.")
print("Nombre de plateaux filtrés :", len(df_plateaus))

# =======================================================
# Optionnel : tracé des plateaux détectés
# =======================================================
plt.figure(figsize=(10, 6))
for idx, (s, e, avg, t_start, t_end) in enumerate(plateaus, 1):
    if lower_threshold < avg < upper_threshold:
        plateau_time = t[s:e+1]
        plateau_voltage = voltage[s:e+1]
        plt.plot(plateau_time, plateau_voltage, label=f"Plateau {idx}")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Plateaux filtrés (avg_voltage entre {:.2f} et {:.2f})".format(lower_threshold, upper_threshold))
plt.legend()
plt.show()
# =======================================================
# 5. Extraction des valeurs de tension sur les plateaux
# =======================================================
all_plateaus = []
for s, e, avg, t_start, t_end in plateaus:
    if lower_threshold < avg < upper_threshold:
        all_plateaus.extend(voltage[s:e+1])
all_plateaus = np.array(all_plateaus)

# =======================================================
# 6. Calcul de la conductance via MathCore
# =======================================================
# La fonction compute_conductance attend un "self" en premier argument.
# Comme elle n'est pas dans une classe exposée, on passe "None" pour combler ce paramètre.
conductance = -1 * mc.compute_conductance(
    all_plateaus,
    source_voltage=1,
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

# Définir les données pour l'axe x et l'axe y
x = list(range(len(histo.data)))
y = histo.data


plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y, marker='o', color='b')  # Affiche uniquement des points
plt.xlabel("Index")
plt.ylabel("Conductance (a.u.)")
plt.title("Graphique de la conductance (points uniquement)")
plt.show()





# =======================================================
# 8. Visualisation en histogramme des voltages moyens à partir de plateau_averages.csv
# =======================================================

df_plateaus = pd.read_csv("plateau_averages.csv")
data = np.array(df_plateaus["avg_voltage"])


conductanceV = -1 * mc.compute_conductance(
    data,
    source_voltage=1,
    resistance=20000,
)

# On s'assure que la conductance soit positive
conductanceV = np.abs(conductanceV)


# --------------------------------------------------
# Calcul de l'histogramme avec bins automatiques
# --------------------------------------------------
counts, bin_edges = np.histogram(mc.compute_conductance(voltage=data,source_voltage=1,resistance=20000), bins=20000)

# Calcul de la moyenne et de l'écart type à partir de l'histogramme
mean_hist = compute_histogram_mean(bin_edges, counts)
std_hist = compute_histogram_std(bin_edges, counts)

print("Moyenne (histogramme) :", mean_hist)
print("Écart type (histogramme) :", std_hist)

# --------------------------------------------------
# Tracé de l'histogramme avec affichage de la moyenne et des bornes moyenne ± écart type
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20000, edgecolor='black', label="Voltage moyen")
plt.xlabel("Voltage moyen (V)")
plt.ylabel("Nombre de plateaux")
plt.legend()
plt.show()



plt.figure(figsize=(10, 6))
sns.histplot(data, bins=20000, edgecolor='black', kde = True, label="Voltage moyen")
plt.xlabel("Voltage moyen (V)")
plt.ylabel("Nombre de plateaux")
plt.legend()
plt.show()

# # =======================================================
# 10. Calcul de la conductance pour chaque plateau filtré et assignation d'un n
# =======================================================
# Paramètres pour cette partie de la conductance
V_source = 1         # Tension de la source (en volts)
resistance = 100     # Résistance en ohms
G0 = 2 * elementary_charge**2 / Planck  # Quantum de conductance (S)

# Initialisation des listes pour stocker les résultats
plateau_conductance_list = []  # Conductance moyenne pour chaque plateau
n_candidate_list = []          # Nombre de canaux estimé pour chaque plateau
interesting_list = []          # Indicateur si le plateau est intéressant (erreur relative < 10%)
R_unknown_list = []            # Résistance inconnue calculée pour chaque plateau
error_list = []                # Erreur relative pour chaque plateau

for idx, (s, e, avg_voltage, t_start, t_end) in enumerate(plateaus, 1):
    if lower_threshold < avg_voltage < upper_threshold:
        voltage_segment = voltage[s:e+1]
        G_segment = mc.compute_conductance(
            voltage_segment,
            source_voltage=V_source,
            resistance=resistance
        )
        G_avg = np.mean(np.abs(G_segment))
        plateau_conductance_list.append(G_avg)
        
        n_candidate = int(round(G_avg / G0))
        if n_candidate < 1:
            n_candidate = 1
        n_candidate_list.append(n_candidate)
        
        error = abs(G_avg - n_candidate * G0) / (n_candidate * G0)
        error_list.append(error)
        interesting = (error < 0.1)
        interesting_list.append(interesting)
        
        R_unknown = (V_source / abs(avg_voltage) - 1) / (n_candidate * G0)
        R_unknown_list.append(R_unknown)

df_conductance = pd.DataFrame({
    "Plateau": list(range(1, len(plateau_conductance_list) + 1)),
    "avg_voltage": [p[2] for p in plateaus if lower_threshold < p[2] < upper_threshold],
    "G_avg": plateau_conductance_list,
    "n_candidate": n_candidate_list,
    "interesting": interesting_list,
    "R_unknown": R_unknown_list
})
print(df_conductance)
df_conductance.to_csv("plateau_averages_with_conductance.csv", index=False)
# =======================================================
# 6. Visualisation de la conductance par plateau avec niveaux théoriques (n·G₀)
# =======================================================
plt.figure(figsize=(12, 8))
max_n = max(n_candidate_list)
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
sns.histplot(df_conductance["R_unknown"], bins='auto',kde=True, edgecolor='black')
plt.xlabel("Résistance inconnue (Ω)")
plt.ylabel("Nombre de plateaux")
plt.title("Histogramme des résistances inconnues par plateau")
plt.show()

# Calcul de la moyenne et de l'écart type de l'erreur relative
mean_errorRes = np.mean(df_conductance['R_unknown'])
std_errorRes = np.std(df_conductance['R_unknown'], ddof=1)

print(mean_errorRes)
print(std_errorRes)



# Affichage du nombre de plateaux retenus
print("Nombre de plateaux retenus :", len(error_list))

# Calcul de la moyenne et de l'écart type de l'erreur relative
mean_error = np.mean(error_list)
std_error = np.std(error_list, ddof=1)

print("Moyenne de l'erreur relative :", mean_error)
print("Écart type de l'erreur relative :", std_error)

# Histogramme avec un nombre de bins fixé
plt.figure(figsize=(10, 6))
plt.hist(error_list, bins=20, edgecolor='black', alpha=0.7, label="Histogramme")
plt.xlabel("Erreur relative")
plt.ylabel("Nombre de plateaux")
plt.title("Histogramme de l'erreur relative (20 bins)")
plt.legend()
plt.show()

# Histogramme avec densité de probabilité (KDE)
plt.figure(figsize=(10, 6))
sns.histplot(error_list, bins=20, kde=True, color="skyblue")
plt.xlabel("Erreur relative")
plt.ylabel("Densité")
plt.title("Histogramme et KDE de l'erreur relative")
plt.show()

# # Boxplot de l'erreur relative
# plt.figure(figsize=(6, 8))
# sns.boxplot(y=error_list, color="lightgreen")
# plt.ylabel("Erreur relative")
# plt.title("Boxplot de l'erreur relative")
# plt.show(


# --- Filtrage des plateaux acceptables pour le calcul de R_unknown ---
# On sélectionne les lignes de df_conductance où "interesting" est True
df_conductance_interesting = df_conductance[df_conductance["interesting"] == True]
print("Nombre de plateaux acceptables :", len(df_conductance_interesting))

# --- Retracer les plateaux acceptables à partir des index contenus dans df_plateaus ---
plt.figure(figsize=(12, 8))
# On parcourt chaque plateau intéressant
for _, row in df_conductance_interesting.iterrows():
    plateau_num = int(row["Plateau"])
    # On cherche la correspondance dans df_plateaus (les deux DataFrames doivent partager le même numéro de plateau)
    plateau_info = df_plateaus[df_plateaus["Plateau"] == plateau_num]
    if plateau_info.empty:
        continue  # Si aucune correspondance n'est trouvée, on passe au suivant
    # Extraction des indices de début et de fin du plateau
    s = int(plateau_info["start_index"].values[0])
    e = int(plateau_info["end_index"].values[0])
    # Extraction du segment correspondant du signal original
    plateau_time = t[s:e+1]
    plateau_voltage = voltage[s:e+1]
    # Tracé du plateau
    plt.plot(plateau_time, plateau_voltage, label=f"Plateau {plateau_num}")

plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Retracé des plateaux acceptables pour le calcul de R_unknown")
plt.show()