# Gestion du système et des requêtes
import os                                     # Accès aux dossiers système
import requests                               # Gestion des flux HTTP

# Manipulation de données et calcul numérique
import numpy  as np                           # Algèbre et calcul matriciel
import pandas as pd                           # Analyse de données tabulaires

# Visualisation et interface utilisateur
import matplotlib.pyplot as plt               # Création de graphiques
from   PIL           import Image             # Manipulation d'images
from   tqdm.notebook import tqdm              # Barres de progression Jupyter

# Apprentissage automatique et métriques spatiales
from   sklearn.metrics.pairwise import haversine_distances # Calcul spatial
from   sklearn.neighbors        import BallTree            # Recherche spatiale optimisée

# ==============================================================================
# CONFIGURATION GLOBALE : POINTS D'INTÉRÊT ET CONSTANTES
# ==============================================================================

# Centre de Seattle (Hôtel de ville)
SEATTLE_CENTER_LAT = 47.608013
SEATTLE_CENTER_LON = -122.335167

# Port de Seattle (Zone industrielle massive)
SEATTLE_PORT_LAT   = 47.5839
SEATTLE_PORT_LON   = -122.3481

# Définition des rayons de la Terre
RAYON_TERRE_KM     = 6371.0                   # Pour résultats en kilomètres
RAYON_TERRE_M      = 6371000.0                # Pour résultats en mètres

# Seuils pour la maille "Taille du bâtiment" (en pieds carrés - sqft)
SURFACE_SMALL_THRESHOLD = 20000 
SURFACE_LARGE_THRESHOLD = 100000

# Paramètre de lissage pour le Target Encoding (Maille Quartier/Usage)
SMOOTHING_VAL = 10.0



# ======================================================================================================================
# FILTRAGE DU PÉRIMÈTRE MÉTIER : EXCLUSION DU RÉSIDENTIEL
# ======================================================================================================================

# ======================================================================================================================
# FONCTION UTILITAIRE : FILTRAGE DU PÉRIMÈTRE NON-RÉSIDENTIEL
# ======================================================================================================================

def filtrer_uniquement_non_residentiel(df):
    """
    Filtre le DataFrame pour ne conserver que les bâtiments non-résidentiels.
    
    Paramètre :
        df (pd.DataFrame) : Le dataset original de Seattle.
    Retour :
        pd.DataFrame : Le dataset filtré (exclusion des types 'Multifamily').
    """
    if 'BuildingType' not in df.columns:
        print("⚠️ Erreur : La colonne 'BuildingType' est introuvable.")
        return df

    # 1. Calcul des statistiques avant filtrage
    nb_total = len(df)
    
    # 2. Application du filtre (Exclusion des catégories commençant par 'Multifamily')
    # On utilise ~ pour inverser le masque (on garde ce qui n'est PAS Multifamily)
    df_filtre = df[~df['BuildingType'].str.startswith('Multifamily', na=False)].copy()
    
    # 3. Calcul de l'impact
    nb_final = len(df_filtre)
    nb_suppr = nb_total - nb_final
    
    print("\n==================================================================================================")
    print("FILTRAGE DU PÉRIMÈTRE D'ÉTUDE")
    print("==================================================================================================")
    print(f"  🏢 Bâtiments initiaux : {nb_total:>5}")
    print(f"  🚫 Bâtiments retirés  : {nb_suppr:>5} (Type: Multifamily)")
    print(f"  ✅ Bâtiments restants : {nb_final:>5} (Périmètre Non-Résidentiel uniquement)")
    print("==================================================================================================\n")
    
    return df_filtre

# ##############################################################################
def calculer_distances_points_cles(df, col_lat='Latitude', col_lon='Longitude'):
    """
    Calcule la distance entre chaque bâtiment et les points névralgiques.
    """
    
    # --------------------------------------------------------------------------
    # Initialisation des références (Coordonnées en Radians)
    
    ref_centre = np.radians([SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON])
    ref_port   = np.radians([SEATTLE_PORT_LAT,   SEATTLE_PORT_LON])
    
    # --------------------------------------------------------------------------
    # Transformation et Calcul
    
    positions_radians = np.radians(df[[col_lat, col_lon]].values)

    # Calcul des distances au centre (en mètres)
    dist_centre = [
        haversine_distances([p, ref_centre])[0, 1] * RAYON_TERRE_M
        for p in positions_radians
    ]
    
    # Calcul des distances au port (en mètres)
    dist_port   = [
        haversine_distances([p, ref_port])[0, 1] * RAYON_TERRE_M
        for p in positions_radians
    ]

    # --------------------------------------------------------------------------
    # Affichage pédagogique
    
    apercu_c = dist_centre[:3]                # Extraction échantillon centre
    apercu_p = dist_port[:3]                  # Extraction échantillon port
    
    print(f"  Aperçu dist. Centre.: {['%.2f m' % x for x in apercu_c]}")
    print(f"  Aperçu dist. Port...: {['%.2f m' % x for x in apercu_p]}")
    
    return dist_centre, dist_port



# ##############################################################################
def calculer_densite_voisinage(df, rayon_m=500, col_lat='Latitude', col_lon='Longitude'):
    """
    Compte le nombre de bâtiments voisins dans un rayon donné.
    """
    
    # --------------------------------------------------------------------------
    # Préparation des données et conversion du rayon
    
    # Conversion des coordonnées en radians pour la métrique Haversine
    coords_radians = np.radians(df[[col_lat, col_lon]].values)
    
    # Conversion du rayon (mètres -> kilomètres -> radians)
    rayon_km       = rayon_m / 1000.0
    rayon_radians  = rayon_km / RAYON_TERRE_KM # Utilise votre constante globale
    
    # --------------------------------------------------------------------------
    # Construction de l'arbre et requête de voisinage
    
    # Création de l'index spatial
    arbre = BallTree(coords_radians, metric='haversine')
    
    # Comptage des points dans le rayon (on compte les voisins uniquement)
    # count_only=True rend l'opération très rapide
    comptage = arbre.query_radius(coords_radians, r=rayon_radians, count_only=True)
    
    # On soustrait 1 pour ne pas compter le bâtiment lui-même
    resultat = [max(0, n - 1) for n in comptage]
    
    # Aperçu pédagogique
    print(f"  Aperçu densité {rayon_m}m.: {resultat[:3]}")
    
    return resultat

    
def calcular_densite_voisinage_with_reference_tree(df, rayon_m, reference_tree=None):
    """
    Si reference_tree es None, calcula la densidad usando el DF actual (Modo Entrenamiento).
    Si reference_tree existe, calcula la densidad respecto al árbol (Modo Inferencia).
    """
    coords_actuelles = np.radians(df[['Latitude', 'Longitude']].values)
    
    # Radio de la tierra en metros
    R = 6371000 
    
    if reference_tree is None:
        # Modo FIT: Construimos un árbol temporal con los datos actuales
        temp_tree = BallTree(coords_actuelles, metric='haversine')
        counts = temp_tree.query_radius(coords_actuelles, r=rayon_m/R, count_only=True)
        # Restamos 1 porque no queremos contarnos a nosotros mismos como vecinos
        return counts - 1
    else:
        # Modo TRANSFORM: Usamos el árbol del set de entrenamiento (referencia fija)
        counts = reference_tree.query_radius(coords_actuelles, r=rayon_m/R, count_only=True)
        return counts    