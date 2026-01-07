# ##############################################################################
# MODULE : DOMAIN_ANALYSIS_ENGINE.PY
# DESCRIPTION : Identification automatique et filtrage des variables métier.
# ##############################################################################
import polars as pl
import pandas as pd

def identifier_variables_energetiques(df):
    """
    Identifie automatiquement les colonnes liées à l'énergie par mots-clés.
    Fonctionne avec Pandas et Polars.
    """
    # Mots-clés métiers définis par la nomenclature de Seattle
    mots_cles = ['use', 'energy', 'kbtu', 'kwh', 'gas', 'steam', 'electricity', 'emissions']
    
    # Récupération des noms de colonnes selon le moteur utilisé
    if isinstance(df, pl.DataFrame):
        all_cols = df.columns
    else:
        all_cols = df.columns.tolist()
        
    # Filtrage lexical (insensible à la casse)
    vars_energetiques = [
        col for col in all_cols 
        if any(mot in col.lower() for mot in mots_cles)
    ]
    
    return vars_energetiques

def filtrer_colinearite_elevee(df, threshold=0.95):
    """
    Détecte les variables qui sont redondantes (corrélation > threshold).
    Utile pour éviter le surapprentissage (overfitting).
    """
    # Passage en Pandas pour le calcul de la matrice de corrélation
    df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    corr_matrix = df_pd.select_dtypes(include=['number']).corr().abs()
    
    # Identification des paires fortement corrélées
    redundant_cols = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i]
                redundant_cols.add(colname)
                
    return list(redundant_cols)


def rapport_redondance(df, threshold=0.90):
    """
    Identifie les paires de variables avec une corrélation supérieure au seuil.
    Retourne un dictionnaire avec les paires et leur score.
    """
    # Convertir a Pandas para el cálculo de correlación
    df_pd = df.to_pandas() if hasattr(df, 'to_pandas') else df
    corr_matrix = df_pd.select_dtypes(include=['number']).corr()
    
    redondances = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            score = corr_matrix.iloc[i, j]
            if abs(score) > threshold:
                redondances.append({
                    'var1': cols[i],
                    'var2': cols[j],
                    'correlation': round(score, 4)
                })
    
    return redondances
    