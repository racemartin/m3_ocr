# ##############################################################################
# MODULE : FEATURE_SELECTION_REPORT.PY - SYNTHÈSE DES SUPPRESSIONS
# ##############################################################################
"""
Consolide les analyses de valeurs manquantes et de variance pour proposer
une liste finale de colonnes à supprimer avant le pré-traitement.
"""

# Bibliothèques de manipulation de données
import pandas as pd                       # Analyse de données (DataFrames)

# ##############################################################################
# FONCTION : GENERER_SYNTHESE_SUPPRESSION
# ##############################################################################

def generer_synthese_suppression(df, missing_df, variance_res, threshold=0.95):
    """
    Crée un rapport consolidé des colonnes candidates à l'élimination.
    """
    # --------------------------------------------------------------------------
    # 1. RÉCUPÉRATION DES CANDIDATS PAR CATÉGORIE
    # --------------------------------------------------------------------------
    
    # Colonnes avec trop de manquants (Seuil par défaut 95%)
    cols_missing = missing_df[missing_df['Missing_Pct'] > (threshold * 100)]
    list_missing = cols_missing['Column'].tolist()
    
    # Colonnes constantes (Issues du module variance_analysis)
    list_const   = variance_res['constant_cols']
    
    # --------------------------------------------------------------------------
    # 2. CONSTRUCTION DU RÉSUMÉ DOCUMENTÉ
    # --------------------------------------------------------------------------
    synthese_data = []
    
    # Fusion des listes sans doublons
    tous_candidats = list(set(list_missing + list_const))
    
    for col in tous_candidats:
        raison = []
        if col in list_missing : raison.append(f"Manquants > {threshold*100:.0f}%")
        if col in list_const   : raison.append("Constante (Unique)")
        
        synthese_data.append({
            'Column'           : col,
            'Raison_Principale': " & ".join(raison),
            'Dtype'            : str(df[col].dtype),
            'Impact_Potentiel' : "Perte d'information nulle"
        })
    
    # Création du DataFrame final de synthèse
    df_candidats = pd.DataFrame(synthese_data)
    
    # Tri alphabétique pour la lisibilité
    if not df_candidats.empty:
        df_candidats = df_candidats.sort_values('Column')

    # --------------------------------------------------------------------------
    # 3. AFFICHAGE DES RÉSULTATS (STYLE ACADÉMIQUE)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ÉTAPE 2.4 : SYNTHÈSE DES CANDIDATS À LA SUPPRESSION")
    print("=" * 80)
    print(f" Nombre total de colonnes analysées..: {len(df.columns)}")
    print(f" Colonnes identifiées pour retrait...: {len(tous_candidats)}")
    print("-" * 80)
    
    return tous_candidats, df_candidats