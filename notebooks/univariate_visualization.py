# ##############################################################################
# MODULE : UNIVARIATE_VISUALIZATION.PY - VISUALISATION DES DISTRIBUTIONS
# ##############################################################################
"""
Génération automatique de tableaux d'histogrammes pour l'analyse métier.
Optimisé pour détecter visuellement l'asymétrie et les outliers.
"""

# Bibliothèques de visualisation
import matplotlib.pyplot as plt               # Interface de traçage
import seaborn           as sns               # Visualisation statistique
import math                                   # Calculs mathématiques de base

# ##############################################################################
# FONCTION : PLOT_DISTRIBUTIONS_GRILLE
# ##############################################################################

def plot_distributions_grille(df, columns, n_cols=3, figsize_unit=(5, 4)):
    """
    Crée une grille d'histogrammes avec courbe de densité (KDE).
    """
    if not columns:
        print("⚠️ Info..................: Aucune colonne à visualiser.")
        return None

    # Calcul du nombre de lignes nécessaires
    n_vars  = len(columns)
    n_rows  = math.ceil(n_vars / n_cols)
    
    # Configuration de la taille de la figure
    fig_size = (n_cols * figsize_unit[0], n_rows * figsize_unit[1])
    
    # --------------------------------------------------------------------------
    # CRÉATION DE LA FIGURE ET DES SUBPLOTS
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = axes.flatten()                     # Aplatir pour itération facile

    for i, col in enumerate(columns):
        sns.histplot(
            df[col], 
            kde=True,                         # Ajout de la courbe de densité
            ax=axes[i], 
            color='steelblue', 
            edgecolor='white'
        )
        
        # Esthétique et titres
        axes[i].set_title(f"Distribution : {col}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Fréquence")
        
        # Calcul de la médiane pour comparaison visuelle avec la moyenne
        axes[i].axvline(df[col].median(), color='red', linestyle='--', 
                        label=f"Médiane: {df[col].median():.2f}")
        axes[i].legend(fontsize=8)

    # Nettoyage des axes vides (si n_vars < n_rows * n_cols)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig