# ##############################################################################
# MODULE : VISUALIZER.PY - GÉNÉRATION DE GRAPHIQUES PROFESSIONNELS
# ##############################################################################
import matplotlib.pyplot as plt
import seaborn as sns

def get_plot_feature_vs_target(df, feature, target, meta):
    """
    Génère un graphique de dispersion (scatter plot) avec des labels descriptifs.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Récupération des labels depuis le dictionnaire de métadonnées
    label_x = meta.get(feature, feature)
    label_y = meta.get(target, target)
    
    # Création du graphique
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.5, color="#2ecc71")
    
    plt.title(f"Analyse : {label_x}\nvs {label_y}", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(label_x, fontsize=11)
    plt.ylabel(label_y, fontsize=11)
    
    plt.tight_layout()
    return plt.gcf()

def get_ax_feature_vs_target(df, feature, target, meta, ax=None):
    """
    Dibuja en el eje (ax) proporcionado. Si no hay eje, crea uno nuevo.
    """
    # Si no se pasa un eje, se crea una figura independiente (comportamiento por defecto)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    label_x = meta.get(feature, feature)
    label_y = meta.get(target, target)
    
    # IMPORTANTE: El parámetro 'ax=ax' dentro de scatterplot es lo que hace la magia
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.5, ax=ax, color="#3498db")
    
    ax.set_title(f"{label_x}\nvs {label_y}", fontsize=10, fontweight='bold')
    ax.set_xlabel(label_x, fontsize=9)
    ax.set_ylabel(label_y, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)