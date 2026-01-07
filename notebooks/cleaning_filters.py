# ##############################################################################
# MODULE : CLEANING_FILTERS.PY - FILTRAGE CONFIGURABLE
# ##############################################################################

def filtrer_categories(df, column, categories_to_remove, verbose=True):
    """
    Supprime les lignes correspondant à une liste de catégories dans une colonne.
    
    Args:
        df (pd.DataFrame): Le DataFrame source.
        column (str): La colonne sur laquelle appliquer le filtre.
        categories_to_remove (list): Liste des valeurs à supprimer.
        verbose (bool): Si True, affiche un rapport détaillé.
    """
    initial_count = len(df)
    
    # Application du filtrage
    df_filtered = df[~df[column].isin(categories_to_remove)].copy()
    
    # Calcul des statistiques de suppression
    final_count = len(df_filtered)
    removed = initial_count - final_count
    
    if verbose:
        print("\n" + "—" * 60)
        print(f"🛠️ NETTOYAGE : Colonne [{column}]")
        print("—" * 60)
        print(f" Valeurs supprimées : {categories_to_remove}")
        print(f" Lignes supprimées  : {removed}")
        print(f" Lignes restantes   : {final_count}")
        print(f" Réduction          : -{(removed/initial_count*100):.2f}%")
        print("—" * 60)
        
    return df_filtered