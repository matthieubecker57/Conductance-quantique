import pandas as pd
from PCore import main  # On importe la fonction principale de votre module PCore

def additional_calculations(csv_file):
    """
    Charge le CSV produit par le pipeline (par exemple, "Rinconnue_results.csv")
    et réalise des calculs complémentaires, ici en affichant des statistiques descriptives.
    """
    df = pd.read_csv(csv_file)
    
    # Exemple de statistiques descriptives
    stats = df.describe()
    print("Statistiques descriptives sur Rinconnue:")
    print(stats)
    
    # Vous pouvez ajouter d'autres calculs ici, par exemple :
    # - des visualisations complémentaires,
    # - des calculs d'indicateurs spécifiques, etc.
    
    return df

if __name__ == "__main__":
    # Lancement du pipeline complet qui se base sur "acquisition_data.csv"
    main()
    
    # Une fois le pipeline exécuté, le CSV "Rinconnue_results.csv" est créé.
    # On charge ce CSV et on effectue des calculs supplémentaires.
    df_results = additional_calculations("Rinconnue_results.csv")

    # Vous pouvez continuer à travailler sur les données dans df_results