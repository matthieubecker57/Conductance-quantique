import pandas as pd
import numpy as np
from MathCore import G0  # Assurez-vous que G0 est défini correctement

# Paramètres connus
V_source = 2.5
# Vous pouvez utiliser la valeur optimale obtenue via le balayage, ici on prend l'exemple 300 Ω
Rres_optimal = 240 

# Chargement du fichier CSV contenant les résultats des plateaux
df_plateaux = pd.read_csv("plateau_results_n.csv")

def compute_Rinconnue(mean_voltage, n, Rres, V_source, G0):
    """
    Calcule la résistance inconnue à partir de la tension moyenne mesurée d'un plateau.
    
    V_source      : Tension de la source (en V)
    mean_voltage  : Tension moyenne mesurée sur le plateau (en V)
    n             : Multiplicateur de G0 déterminé pour le plateau
    Rres          : Résistance résiduelle dans les nanofils
    G0            : Quantum de conductance
    """
    return (V_source / mean_voltage - 1) * (1/(n * G0) + Rres)

# Calcul de Rinconnue pour chaque plateau détecté
df_plateaux["Rinconnue"] = df_plateaux.apply(lambda row: compute_Rinconnue(row["mean_voltage"], 
                                                                          row["plateau_n"], 
                                                                          Rres_optimal, 
                                                                          V_source, 
                                                                          G0), axis=1)

# Affichage des statistiques sur Rinconnue
R_mean = df_plateaux["Rinconnue"].mean()
R_std = df_plateaux["Rinconnue"].std()

print(f"Moyenne de Rinconnue : {R_mean:.4f} Ω")
print(f"Écart-type de Rinconnue : {R_std:.4f} Ω")

# Sauvegarde des résultats dans un nouveau fichier CSV
df_plateaux.to_csv("Rinconnue_results.csv", index=False)
print("Les résultats de Rinconnue ont été exportés dans Rinconnue_results.csv")

# Optionnel : Visualisation de l'histogramme de Rinconnue
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df_plateaux["Rinconnue"], bins=10, edgecolor='black')
plt.title("Histogramme de Rinconnue")
plt.xlabel("Rinconnue (Ω)")
plt.ylabel("Fréquence")
plt.show()
