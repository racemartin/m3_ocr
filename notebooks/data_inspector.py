# ##############################################################################
# MODULE : DATA_INSPECTOR.PY - INSPECTION CIBLÉE DES DONNÉES
# ##############################################################################
"""
Outil d'inspection pour visualiser les valeurs de colonnes spécifiques 
selon des conditions métier (ex: OSEBuildingID ou valeurs non nulles).
"""

import polars as pl
import pandas as pd

def inspecter_valeurs_condition(df, condition_col, condition_value, columns_to_show=None):
    """
    Filtre le DataFrame et affiche les valeurs pour des colonnes spécifiques.
    
    Paramètres
    ----------
    df : pl.DataFrame ou pd.DataFrame
    condition_col : str (ex: 'OSEBuildingID')
    condition_value : any (ex: 1, '23', etc.)
    columns_to_show : list, optional (Liste des features à afficher)
    """
    # 1. Appliquer le filtre selon le moteur (Polars/Pandas)
    if isinstance(df, pl.DataFrame):
        df_filtered = df.filter(pl.col(condition_col) == condition_value)
    else:
        df_filtered = df[df[condition_col] == condition_value]

    # 2. Gérer les colonnes à afficher
    if columns_to_show is None:
        columns_to_show = df.columns

    # 3. Affichage Formaté (Tableau horizontal pour une ligne)
    print(f"\n🔍 INSPECTION : {condition_col} = {condition_value}")
    print("-" * 50)
    
    if df_filtered.is_empty() if isinstance(df, pl.DataFrame) else df_filtered.empty:
        print("⚠️ Aucun enregistrement trouvé pour cette condition.")
        return

    # Si on a des résultats, on les affiche de manière lisible
    # Pour Polars, on convertit temporairement en dictionnaire pour un joli print
    row_data = df_filtered.select(columns_to_show).to_dicts()[0]
    
    for col, val in row_data.items():
        print(f" {col:.<35}: {val}")
    print("-" * 50)