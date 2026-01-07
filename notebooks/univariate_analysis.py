# ##############################################################################
# MODULE : UNIVARIATE_ANALYSIS.PY - DIAGNOSTIC STATISTIQUE COMPLET
# ##############################################################################
"""
Analyse descriptive automatisée avec diagnostic d'échelle et d'asymétrie.
Maintient la compatibilité avec le triple retour (stats, échelles, asymétrie).
"""

# Bibliothèques de manipulation de données
import pandas            as pd                # Analyse de données (DataFrames)
import numpy             as np                # Calculs numériques et tableaux

# ##############################################################################
# FONCTION : ANALYSER_STATISTIQUES_GLOBALES
# ##############################################################################

def analyser_statistiques_globales(df):
    """
    Automatise l'interprétation du describe() et identifie les points critiques.
    """
    # Sélection exclusive des variables numériques
    df_num = df.select_dtypes(include=[np.number])
    
    # --------------------------------------------------------------------------
    # 1. GÉNÉRATION DES STATISTIQUES DESCRIPTIVES ÉTENDUES
    # --------------------------------------------------------------------------
    desc            = df_num.describe().T
    desc['Range']   = desc['max'] - desc['min']
    desc['Skewness']= df_num.skew()
    desc['CV']      = desc['std'] / desc['mean'].abs().replace(0, np.nan)
    
    # --------------------------------------------------------------------------
    # 2. IDENTIFICATION DES ÉCHELLES ET ASYMÉTRIES
    # --------------------------------------------------------------------------
    moy_globale     = desc['mean'].abs().mean()
    
    # Identification des échelles extrêmes
    echelles_ext    = []
    for col in desc.index:
        ratio = desc.loc[col, 'mean'] / moy_globale if moy_globale != 0 else 1
        if ratio > 100 or ratio < 0.01:
            echelles_ext.append(col)
            
    # Identification de l'asymétrie critique
    cols_asym       = desc[desc['Skewness'].abs() > 1].index.tolist()

    # --------------------------------------------------------------------------
    # 3. SYSTÈME EXPERT : COLONNE D'OBSERVATIONS
    # --------------------------------------------------------------------------
    def _generer_recommandation(row):
        actions = []
        if row.name in cols_asym: actions.append("Log Transform")
        if row.name in echelles_ext: actions.append("Scaling")
        if row['CV'] > 2: actions.append("Check Outliers")
        return " | ".join(actions) if actions else "RAS (Standardize)"

    desc['Action_Recommandee'] = desc.apply(_generer_recommandation, axis=1)

    # --------------------------------------------------------------------------
    # 4. AFFICHAGE DES ALERTES (LOGGING)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ÉTAPE 3.1 : ANALYSE STATISTIQUE ET DIAGNOSTIC DES ÉCHELLES")
    print("=" * 80)
    print(f" Variables numériques analysées.......: {len(df_num.columns)}")
    print(f" Variables à échelles critiques.......: {len(echelles_ext)}")
    print(f" Variables fortement asymétriques.....: {len(cols_asym)}")
    
    if echelles_ext:
        print("\n⚠️ ALERTE ÉCHELLES :")
        for col in echelles_ext[:5]:
            print(f"   - {col:30} : Moyenne = {desc.loc[col, 'mean']:.2e}")

    if cols_asym:
        print("\n📊 ALERTE ASYMÉTRIE :")
        for col in cols_asym[:5]:
            print(f"   - {col:30} : Skewness = {desc.loc[col, 'Skewness']:.2f}")

    print("-" * 80)
    
    # Retourne strictement les 3 objets pour maintenir la compatibilité
    return desc, echelles_ext, cols_asym