# ##############################################################################
# MODULE : CATEGORICAL_ANALYSIS.PY - ANALYSE DES VARIABLES QUALITATIVES
# ##############################################################################
"""
Analyse générique des variables catégorielles (fréquences et visualisation).
"""

import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

def analyse_feature_categorielle(df, column_name, title=None, figsize=(10, 6)):
    """
    Génère un décompte des fréquences et un diagramme à barres horizontal.
    """
    # 1. Calcul des fréquences
    stats = df[column_name].value_counts(dropna=False).reset_index()
    stats.columns = [column_name, 'Count']
    stats['Percentage'] = (stats['Count'] / len(df) * 100).round(2)

    # 2. Création du graphique
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=stats, 
        y=column_name, 
        x='Count', 
        palette='viridis',
        hue=column_name,
        legend=False
    )
    
    # Ajout des étiquettes de pourcentage sur les barres
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(
            width + 1, 
            p.get_y() + p.get_height()/2, 
            f"{stats['Percentage'].iloc[i]}%", 
            va='center'
        )

    plt.title(title if title else f"Distribution de : {column_name}", fontweight='bold')
    plt.xlabel("Nombre de bâtiments")
    plt.ylabel("")
    plt.tight_layout()
    
    return stats, plt.gcf()