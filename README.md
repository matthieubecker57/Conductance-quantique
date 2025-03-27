# Conductance-quantique

L'objectif de ce projet est de créer un programme Python simple permettant de mesurer une résistance à l'aide de la conductance quantique.

La structure du projet est schématisé dans `structure du code.png`.

### `__main__.py`
Ce fichier est le fichier principale du code. C'est ce fichier qui va importer puis appeler les diverses classes, fonction et méthodes défini dans le reste des fichier.

### Prise de mesure
La prise de mesure est gérée par la classe `Acquisition` introduite dans le fichier `Acquisition.py`.

### Traitement des données
On travaille actuellement sur cette section. Le traitement de données consiste principalement à prendre les données storées dans `acquisition_data.csv` pour ensuite y chercher des plateaux.

Actuellement, l'un des codes les plus prometteur se trouve dans `P_plateau_search.py`. Matthieu a effectué des amélioration qui n'ont pas encore été poussé sur ce github.

Le code de filtration de données `P_filtrer_donnes.py` sert à éliminer les point qui ne sont assurément pas sur un plateau. Les données filtrées sont sur `P_filtered_data.csv`.

#### Fichiers commençant par `P_*`
Ces fichiers sont tous les divers essais pour chercher des plateaus dans les données.

### Legacy
Ce dossier sert à storer d'ancien fichiers de codes, ainsi que d'ancien fichier `*.csv` qui contiennent des mesures rendues obsolètes, mais que nous voulons conserver.