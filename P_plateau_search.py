import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MathCore import G0
"""
Define the data to search in
"""

data_file = pd.read_csv(r"P_filtered_data.csv")
Vwire = data_file["Voltage_wire"]

"""
Define plateau caracteristics
"""

points_per_plateau = 5

diff = np.diff(Vwire)
max_diff = 1*10**(-2)  # The maximum difference between to points before we consider they are not on the same plateau

"""
Search on plateau and add the points on the plateau to a dictionnary. The key in the dicionnary is the index of the first point
on the plateau
"""

plateaus = {}

on_plateau = False
index = 0

for i in range(len(diff)):
    if abs(diff[i]) <= max_diff and not on_plateau:
        on_plateau = True
        index=i
    if abs(diff[i]) > max_diff and on_plateau:
        on_plateau = False
        if i-index < 5:
            continue
        plateaus[index] = Vwire[index:i]

"""
Print out all the plateaus so that we can can search for it manually trought the data
"""
for index in plateaus.keys():
    print(f"plateaus starts at {index}: \n {plateaus[index]}")

"""
Plot the plateaus
"""
for index in plateaus.keys():
    indexes = plateaus[index].index.tolist()
    plt.plot(indexes, [Vwire[index] for index in indexes], 'o', markersize=1),

plt.grid(which='both')
plt.show()


"""
Plots the average voltage found for each plateau
"""
plateau_values = [np.mean(plateaus[key]) for key in plateaus.keys()]
plt.plot(
    [i for i in range(len(plateau_values))],
    plateau_values,
    'o',
    markersize=1
)
plt.show()


# -------------------------------
# Partie 2 : Balayage de Rres pour déterminer le meilleur ajustement
# -------------------------------

# Paramètres expérimentaux
source_voltage = 2.5
resistance = 20000

def expected_plateau_voltage(n: int, Rres: float, V=source_voltage, R=resistance):
    """
    Calcule la tension théorique aux bornes des fils d'or pour un plateau de conductance nG0.
    """
    return V / (1 + R / (1/(n*G0) + Rres))

# On définit un ensemble de valeurs candidates pour Rres
Rres_candidates = np.linspace(0, 600, 61)  # par exemple, de 0 à 600 par pas de 10 ohms

# Pour chaque valeur de Rres candidate, on calcule une erreur globale sur tous les plateaux
results_Rres = []  # Pour stocker (Rres, erreur_globale)
for Rres in Rres_candidates:
    global_error = 0
    for index_start, plateau_data in plateaus.items():
        mean_voltage = np.mean(plateau_data)
        best_error = np.inf
        # On teste pour n de 1 à 5 (adapter selon vos besoins)
        for n_candidate in range(1, 6):
            v_theo = expected_plateau_voltage(n_candidate, Rres)
            error = abs(mean_voltage - v_theo)
            if error < best_error:
                best_error = error
        global_error += best_error
    results_Rres.append((Rres, global_error))

# On trouve la valeur de Rres qui minimise l'erreur globale
best_Rres, best_global_error = min(results_Rres, key=lambda x: x[1])
print(f"Meilleur Rres : {best_Rres:.1f} ohms avec une erreur globale de {best_global_error:.4f}")

# -------------------------------
# Partie 3 : Attribution de n pour chaque plateau avec le meilleur Rres
# -------------------------------

plateau_results = []

for index_start, plateau_data in plateaus.items():
    mean_voltage = np.mean(plateau_data)
    best_n = None
    best_error = np.inf
    # On teste les valeurs de n de 1 à 5
    for n_candidate in range(1, 6):
        v_theo = expected_plateau_voltage(n_candidate, best_Rres)
        error = abs(mean_voltage - v_theo)
        if error < best_error:
            best_error = error
            best_n = n_candidate
    plateau_results.append({
        "start_index": index_start,
        "mean_voltage": mean_voltage,
        "plateau_n": best_n,
        "error": best_error
    })

# Export des résultats dans un fichier CSV
df_results = pd.DataFrame(plateau_results)
df_results.to_csv("plateau_results_n.csv", index=False)
print("Les résultats ont été exportés dans plateau_results_n.csv")

# Optionnel : Visualiser la courbe de l'erreur globale en fonction de Rres
Rres_vals, global_errors = zip(*results_Rres)
plt.figure(figsize=(10, 6))
plt.plot(Rres_vals, global_errors, marker='o')
plt.title("Erreur globale en fonction de Rres")
plt.xlabel("Rres (ohms)")
plt.ylabel("Erreur globale")
plt.grid(True)
plt.show()