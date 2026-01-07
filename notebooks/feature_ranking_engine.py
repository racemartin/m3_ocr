# Gestion des données et calcul numérique
import pandas                          as pd     # Manipulation de DataFrames
import numpy                           as np     # Calculs et gestion des NaN

# Apprentissage automatique et prétraitement
from   sklearn.ensemble                import RandomForestRegressor # Ranking
from   sklearn.preprocessing           import LabelEncoder          # Encodage

# ##############################################################################
# FONCTION : ranking_complet_auto
# ##############################################################################
def ranking_complet_auto(df, target="SiteEnergyUse(kBtu)", meta_dict=None):
    """
    Analyse l'importance des variables (numériques et catégorielles) sur une 
    cible donnée en utilisant un Random Forest après nettoyage du leakage.
    """
    # --------------------------------------------------------------------------
    # 1. PRÉPARATION ET NETTOYAGE INITIAL
    # --------------------------------------------------------------------------
    
    # Conversion en Pandas si l'objet vient de Polars
    df_pd = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()

    # Élimination des lignes sans cible (Ground Truth manquante)
    df_pd = df_pd.dropna(subset=[target])
    
    # Identification des types de colonnes pour adapter le traitement
    cols_num = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    cols_cat = df_pd.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Retrait de la cible des prédicteurs potentiels
    if target in cols_num: 
        cols_num.remove(target)
    
    # --------------------------------------------------------------------------
    # 2. ENCODAGE ET TRAITEMENT DU LEAKAGE
    # --------------------------------------------------------------------------
    
    # Traduction des catégories en nombres pour les algorithmes
    df_ranking = df_pd.copy()
    for col in cols_cat:
        # Conversion forcée en string pour éviter les erreurs de type
        df_ranking[col] = LabelEncoder().fit_transform(df_ranking[col].astype(str))
    
    # Calcul de la corrélation de Pearson pour détecter les variables "tricheuses"
    correlations = df_pd[cols_num + [target]].corr()[target].abs()
    
    # Filtrage automatique si corrélation > 0.90 (identités mathématiques)
    leakage_auto = correlations[correlations > 0.90].index.tolist()
    if target in leakage_auto: 
        leakage_auto.remove(target)
    
    # --------------------------------------------------------------------------
    # 3. CALCUL DE L'IMPORTANCE VIA RANDOM FOREST
    # --------------------------------------------------------------------------
    
    # Définition des features finales saines (Groupe A + Groupe B)
    feat_finales = [c for c in cols_num + cols_cat 
                    if c != target and c not in leakage_auto]
    
    # Préparation des matrices : Imputation par la médiane pour les numériques
    X = df_ranking[feat_finales].fillna(df_ranking[feat_finales].median())
    y = df_ranking[target]
    
    # Entraînement d'une forêt d'arbres pour mesurer la réduction d'impureté
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # --------------------------------------------------------------------------
    # 4. CONSTRUCTION DU CLASSEMENT ET CATÉGORISATION
    # --------------------------------------------------------------------------
    
    # Création du DataFrame de résultats
    ranking = pd.DataFrame({
        'Feature'    : feat_finales,
        'Importance' : model.feature_importances_,
        'Type'       : ['Numérique' if c in cols_num else 'Catégorielle' 
                        for c in feat_finales]
    }).sort_values(by='Importance', ascending=False)
    
    # Logique de segmentation par niveau d'influence
    def definir_groupe(row):
        if row['Importance'] > 0.10: return "GRUPO A (Majeur)"
        if row['Importance'] > 0.02: return "GRUPO B (Modéré)"
        return "GRUPO D (Bruit / Faible)"
        
    # Application de la classification et ajout des métadonnées
    ranking['Grupo'] = ranking.apply(definir_groupe, axis=1)
    
    if meta_dict:
        ranking['Nom'] = ranking['Feature'].map(meta_dict)
        
    return ranking

# ##############################################################################
# EXEMPLE D'AFFICHAGE DES RÉSULTATS
# ##############################################################################
# res = ranking_complet_auto(building_consumption)
# print(f"  Target analysée.......: {target}")
# print(f"  Variables traitées....: {len(df_ranking.columns)}")
# print(f"  Leakage détecté.......: {len(leakage_auto)} colonnes")