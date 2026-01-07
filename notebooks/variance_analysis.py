# ##############################################################################
# MODULE : VARIANCE_ANALYSIS.PY - ANALYSE DES COLONNES CONSTANTES
# ##############################################################################
"""
Identification des colonnes constantes ou quasi-constantes (faible variance).
Supporte les DataFrames Pandas et Polars.
"""

# Bibliothèques de manipulation de données
import pandas            as pd                # Analyse de données (DataFrames)
import numpy             as np                # Calculs numériques et tableaux

# Bibliothèques de visualisation
import matplotlib.pyplot as plt               # Interface de traçage graphique

# ##############################################################################
# FONCTION : CONSTANT_COLUMNS_ANALYSIS
# ##############################################################################

def constant_columns_analysis(df, unique_threshold=1, variance_threshold=0.01):
    """
    Identifie les colonnes constantes ou à faible variance via value_counts().
    """
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()                   # Sécurité pour les objets Polars

    results = []                              # Collecteur de métriques

    # --------------------------------------------------------------------------
    # ANALYSE INDIVIDUELLE DES VARIABLES
    # --------------------------------------------------------------------------
    for col in df.columns:
        # Utilisation de value_counts pour obtenir la distribution réelle
        # dropna=False pour inclure les NaN dans l'analyse de cardinalité
        v_counts    = df[col].value_counts(dropna=False)
        n_unique    = len(v_counts)           # Équivalent strict de nunique
        n_missing   = df[col].isnull().sum()
        dtype       = df[col].dtype
        
        # 1. Analyse de la Variance (pour les types numériques)
        variance    = None
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().sum() > 1:
                mean_val = df[col].mean()
                variance = df[col].std() / abs(mean_val) if mean_val != 0 \
                           else df[col].std()
        
        # 2. Identification de la valeur dominante via value_counts
        # v_counts est déjà trié par fréquence décroissante par défaut
        dom_val     = v_counts.index[0]  if n_unique > 0 else None
        dom_freq    = v_counts.values[0] if n_unique > 0 else 0
        dom_pct     = (dom_freq / len(df) * 100)
        
        # 3. Classifications selon les seuils fournis
        is_const    = (n_unique <= unique_threshold)
        is_quasi    = (not is_const and variance is not None 
                       and variance < variance_threshold)
        
        results.append({
            'Column'       : col,
            'N_Unique'     : n_unique,
            'Missing_Pct'  : (n_missing / len(df) * 100),
            'Dtype'        : str(dtype),
            'Var_Norm'     : variance,
            'Dominant_Pct' : dom_pct,
            'Is_Constant'  : is_const,
            'Is_Quasi'     : is_quasi
        })

    # --------------------------------------------------------------------------
    # CONSTRUCTION DU RAPPORT FINAL (CONFORMITÉ DU RETURN)
    # --------------------------------------------------------------------------
    summary_df = pd.DataFrame(results)
    
    # Extraction des listes pour le dictionnaire de retour
    const_cols = summary_df[summary_df['Is_Constant']]['Column'].tolist()
    q_const    = summary_df[summary_df['Is_Quasi']]['Column'].tolist()
    
    # Tri du DataFrame pour la présentation (Constantes en haut)
    summary_df = summary_df.sort_values(['Is_Constant', 'N_Unique'], 
                                        ascending=[False, True])
    
    return {
        'constant_cols'       : const_cols,
        'quasi_constant_cols' : q_const,
        'summary_df'          : summary_df.reset_index(drop=True)
    }

# ##############################################################################
# FONCTION : PLOT_CONSTANT_ANALYSIS
# ##############################################################################

def plot_constant_analysis(summary_df, top_n=20, figsize=(14, 6)):
    """
    Visualisation graphique de la diversité et concentration des valeurs.
    """
    # Filtrage des colonnes suspectes (constantes ou presque)
    suspicious = summary_df[
        (summary_df['Is_Constant']) | 
        (summary_df['Is_Quasi'])    |
        (summary_df['Dominant_Pct'] > 95)
    ].head(top_n)
    
    if suspicious.empty:
        print("✓ Info..................: Aucune colonne suspecte détectée.")
        return None

    # --------------------------------------------------------------------------
    # CONFIGURATION DU GRAPHIQUE (DOUBLES AXES)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Logique de couleurs : Rouge=Constante, Orange=Quasi, Bleu=Normal
    colors = ['#e74c3c' if c else '#f39c12' if q else '#3498db' 
              for c, q in zip(suspicious['Is_Constant'], suspicious['Is_Quasi'])]
    
    # Subplot 1 : Diversité (Nombre de valeurs uniques)
    ax1.barh(suspicious['Column'], suspicious['N_Unique'], color=colors)
    ax1.set_title('Diversité (Valeurs Uniques)', fontweight='bold')
    ax1.invert_yaxis()
    
    # Subplot 2 : Concentration (Pourcentage de la valeur dominante)
    ax2.barh(suspicious['Column'], suspicious['Dominant_Pct'], color=colors)
    ax2.set_title('Concentration (Valeur Dominante %)', fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=95, color='red', linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig