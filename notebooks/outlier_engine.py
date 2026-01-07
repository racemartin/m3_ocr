# ##############################################################################
# MODULE : OUTLIER_ENGINE.PY - DÉTECTION GLOBALE DES ANOMALIES (CORRIGÉ)
# ##############################################################################
import polars as pl
import pandas as pd

def detecter_anomalies_global(df, numeric_cols, categorical_cols, threshold_freq=0.01):
    """
    Scanne le jeu de données pour identifier :
    1. Les outliers numériques (via la méthode statistique de l'IQR).
    2. Les anomalies qualitatives (catégories rares représentant < threshold_freq).
    
    Retourne :
    ----------
    report_num : dict -> Statistiques et IDs des bâtiments avec valeurs extrêmes.
    report_cat : dict -> Liste des catégories minoritaires par variable.
    """
    report_num = {}
    report_cat = {}
    
    # Vérification du type de DataFrame pour adapter la syntaxe
    is_polars = isinstance(df, pl.DataFrame)
    total_lignes = len(df)

    # --- 1. SCAN NUMÉRIQUE (Méthode de l'Écart Interquartile - IQR) ---
    for col in numeric_cols:
        if is_polars:
            # Syntax Polars
            q1 = df.select(pl.col(col).quantile(0.25)).item()
            q3 = df.select(pl.col(col).quantile(0.75)).item()
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
            outliers = df.filter((pl.col(col) < lower) | (pl.col(col) > upper))
            nb_outliers = len(outliers)
            # Utilisation de OSEBuildingID comme identifiant métier
            batiment_ids = outliers.select("OSEBuildingID").to_series().to_list() if nb_outliers > 0 else []
        else:
            # Syntax Pandas (Fallback de sécurité)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            nb_outliers = len(outliers)
            batiment_ids = outliers["OSEBuildingID"].tolist() if nb_outliers > 0 else []
        
        if nb_outliers > 0:
            report_num[col] = {
                "nb_outliers" : nb_outliers,
                "bornes"      : (round(float(lower), 2), round(float(upper), 2)),
                "batiment_ids": batiment_ids
            }

    # --- 2. SCAN QUALITATIF (Détection des catégories rares) ---
    for col in categorical_cols:
        if is_polars:
            comptage = df.group_by(col).agg(pl.count().alias("n"))
            rares = comptage.filter(pl.col("n") < (total_lignes * threshold_freq))
            valeurs = rares.select(col).to_series().to_list()
            freqs = rares.select("n").to_series().to_list()
        else:
            counts  = df[col].value_counts()
            rares   = counts[counts < (total_lignes * threshold_freq)]
            valeurs = rares.index.tolist()
            freqs   = rares.values.tolist()
        
        if len(valeurs) > 0:
            report_cat[col] = {
                "valeurs_rares": valeurs,
                "frequences"   : freqs
            }
            
    return report_num, report_cat