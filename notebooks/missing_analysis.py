# ##############################################################################
# MODULE : MISSING_ANALYSIS.PY - ANALYSE DES VALEURS MANQUANTES
# ##############################################################################
"""
Fonctions réutilisables pour l'analyse exploratoire des valeurs manquantes.
Compatible Pandas et Polars (conversion automatique si nécessaire).
"""

# Bibliothèques de manipulation de données
import pandas            as pd                # Analyse de données (DataFrames)
import numpy             as np                # Calculs numériques et tableaux

# Bibliothèques de visualisation
import matplotlib.pyplot as plt               # Interface de traçage graphique
import seaborn           as sns               # Visualisation de données stat.

# ##############################################################################
# FONCTION : MISSING_SUMMARY
# ##############################################################################

def missing_summary(df):
    """
    Génère un résumé statistique des valeurs manquantes par colonne.
    """
    # Conversion automatique Polars vers Pandas pour la compatibilité
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()                   # Support natif pour Polars

    # --------------------------------------------------------------------------
    # CALCUL DES STATISTIQUES DE BASE
    # --------------------------------------------------------------------------
    null_counts = df.isnull().sum().values    # Somme des valeurs nulles
    total_len   = len(df)                     # Nombre total d'observations
    
    # Construction du DataFrame de synthèse avec alignement des calculs
    missing_stats = pd.DataFrame({
        'Column'        : df.columns,
        'Missing_Count' : null_counts,
        'Missing_Pct'   : (null_counts / total_len * 100),
        'Dtype'         : df.dtypes.values
    })

    # Filtrage des colonnes sans données manquantes et tri décroissant
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
    missing_stats = missing_stats.sort_values('Missing_Pct', ascending=False)
    
    return missing_stats.reset_index(drop=True)

# ##############################################################################
# FONCTION : MISSING_HEATMAP
# ##############################################################################

def missing_heatmap(df, figsize=(14, 10), cmap='viridis'):
    """
    Crée une cartographie visuelle des patterns de valeurs manquantes.
    """
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()                   # Conversion de sécurité

    # Création de la matrice binaire (True si manquant)
    missing_matrix = df.isnull()
    
    # Identification des colonnes problématiques
    cols_with_miss = missing_matrix.columns[missing_matrix.any()].tolist()

    if not cols_with_miss:
        print("⚠️ Info..................: Aucune valeur manquante détectée.")
        return None

    # --------------------------------------------------------------------------
    # GÉNÉRATION DU GRAPHIQUE SEABORN
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        missing_matrix[cols_with_miss].T,     # Transposition pour lecture en Y
        cmap      = cmap,                     # Palette de couleurs
        cbar_kws  = {'label': 'Manquant (1)'},# Légende de la barre latérale
        yticklabels = True,                   # Affichage des noms de variables
        xticklabels = False,                  # Masquage des index individuels
        ax        = ax                        # Axe de dessin
    )

    ax.set_title('Cartographie des Valeurs Manquantes', fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig

# ##############################################################################
# FONCTION : MISSING_BY_TYPE
# ##############################################################################

def missing_by_type(df):
    """
    Analyse séparée des manquants selon le type : numérique vs catégoriel.
    """
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()

    # Séparation des types de données
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --------------------------------------------------------------------------
    # FONCTION INTERNE DE GÉNÉRATION DE RÉSUMÉ
    # --------------------------------------------------------------------------
    def _create_sub_summary(columns):
        if not columns: return pd.DataFrame() # Sécurité si liste vide
        
        counts  = df[columns].isnull().sum().values
        pcts    = (counts / len(df) * 100)
        
        res     = pd.DataFrame({
            'Column'        : columns,
            'Missing_Count' : counts,
            'Missing_Pct'   : pcts
        })
        return res[res['Missing_Count'] > 0].sort_values('Missing_Pct', 
                                                       ascending=False)

    num_missing = _create_sub_summary(num_cols)
    cat_missing = _create_sub_summary(cat_cols)

    return num_missing.reset_index(drop=True), cat_missing.reset_index(drop=True)

# ##############################################################################
# FONCTION : MISSING_THRESHOLD_FILTER
# ##############################################################################

def missing_threshold_filter(df, threshold=0.5, return_type='list'):
    """
    Identifie les colonnes dépassant un seuil critique de données manquantes.
    """
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()

    # Calcul du ratio de perte par variable
    missing_ratio = df.isnull().sum() / len(df)
    
    # Filtrage selon le seuil utilisateur
    high_missing  = missing_ratio[missing_ratio > threshold]

    if return_type == 'list':
        return high_missing.index.tolist()    # Retourne uniquement les noms
    
    # Construction d'un rapport détaillé si demandé
    result = pd.DataFrame({
        'Column'      : high_missing.index,
        'Missing_Pct' : (high_missing.values * 100).round(2),
        'Threshold'   : f'{threshold*100:.0f}%'
    })
    
    return result.sort_values('Missing_Pct', ascending=False).reset_index(drop=True)

# ##############################################################################
# FIN DU MODULE MISSING_ANALYSIS
# ##############################################################################