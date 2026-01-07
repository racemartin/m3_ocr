# ##############################################################################
# MODULE : ANALYSIS_CORRELATION_ENGINE.PY - ANALYSE DE CORRÉLATION AVANCÉE
# ##############################################################################
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def filtrer_features_pertinentes(df, target="SiteEnergyUse(kBtu)", threshold=0.3):
    """
    Identifie automatiquement les variables numériques qui ont une corrélation
    significative avec la variable cible (target).
    """
    # Conversion en Pandas pour la matrice de corrélation
    df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    
    # Calcul de la corrélation par rapport à la cible
    corr_matrix = df_pd.select_dtypes(include=['number']).corr()
    corr_target = corr_matrix[target].abs().sort_values(ascending=False)
    
    # On ne garde que celles qui dépassent le seuil de significativité
    features_pertinentes = corr_target[corr_target > threshold].index.tolist()
    
    return features_pertinentes

def plot_correlation_heatmap(df, features, meta_dict):
    """
    Génère une Heatmap lisible avec des noms de variables traduits.
    """
    df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    
    # Extraction et renommage des colonnes
    data_corr = df_pd[features].rename(columns=meta_dict)
    corr = data_corr.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title("Matrice de Corrélation : Variables Clés du Modèle", fontsize=15, fontweight='bold')
    plt.show()