import pandas as pd
import polars as pl

def identifier_colonnes_problematiques(df, target_col=None):
    """
    Identifie les colonnes à éliminer. Compatible Pandas et Polars.
    """
    print("\n" + "=" * 80)
    print("ÉTAPE 1 : IDENTIFICATION DES COLONNES PROBLÉMATIQUES")
    print("=" * 80)
    
    colonnes_a_eliminer = []
    raisons = {}
    total_lignes = len(df)
    cols = df.columns

    for col in cols:
        if col == target_col:
            continue
            
        # --- Gestión Universal de n_uniques (Polars vs Pandas) ---
        if hasattr(df[col], 'n_unique'): # Polars
            n_uniques = df[col].n_unique()
        else: # Pandas
            n_uniques = df[col].nunique()

        # 1.1 Colonnes constantes
        if n_uniques == 1:
            val = df[col][0]
            colonnes_a_eliminer.append(col)
            raisons[col] = f"Constante (valeur={val})"
            continue

        # 1.2 Valeurs manquantes (> 95%)
        # En Polars es .null_count(), en Pandas es .isnull().sum()
        if hasattr(df[col], 'null_count'):
            n_nulls = df[col].null_count()
        else:
            n_nulls = df[col].isnull().sum()
            
        pct_missing = (n_nulls / total_lignes) * 100
        if pct_missing > 95:
            colonnes_a_eliminer.append(col)
            raisons[col] = f"Missing excessif ({pct_missing:.1f}%)"
            continue

        # 1.3 Identifiants uniques
        if n_uniques == total_lignes:
            colonnes_a_eliminer.append(col)
            raisons[col] = "ID unique (aucune valeur prédictive)"
            continue

        # 1.4 Texte non structuré
        is_string = False
        dtype_str = str(df[col].dtype)
        if "Utf8" in dtype_str or "String" in dtype_str or "object" in dtype_str:
            is_string = True
        
        if is_string and n_uniques > (total_lignes * 0.5):
            colonnes_a_eliminer.append(col)
            raisons[col] = "Texte sans structure (commentaires/adresses)"

    # Reporting
    print(f"\n🔍 Analyse terminée :")
    for col in colonnes_a_eliminer:
        print(f"   ❌ {col}: {raisons[col]}")
    
    print(f"\n📊 RÉSUMÉ : {len(colonnes_a_eliminer)} colonnes identifiées.")
    return colonnes_a_eliminer, raisons

def eliminer_colonnes_problematiques(df, colonnes_a_eliminer, verify=True, overwrite=False):
    """
    Gère l'élimination des colonnes avec un mode audit (verify) et exécution (overwrite).
    
    Paramètres
    ----------
    df : pd.DataFrame | pl.DataFrame
        Le DataFrame original.
    colonnes_a_eliminer : list
        Liste des colonnes identifiées par la fonction précédente.
    verify : bool, default True
        Si True, affiche une comparaison sans modifier les données.
    overwrite : bool, default False
        Si True, procède réellement à l'élimination.
        
    Retourne
    --------
    df_result : Le DataFrame (modifié ou non selon overwrite).
    """
    print("\n" + "-" * 40)
    print("📈 ÉTUDE D'IMPACT DU NETTOYAGE")
    print("-" * 40)
    
    # --- État INITIAL ---
    mem_initial = df.estimated_size() if hasattr(df, 'estimated_size') else df.memory_usage().sum()
    rows_init, cols_init = df.shape
    
    # --- Simulation de l'état FINAL ---
    cols_final = cols_init - len(colonnes_a_eliminer)
    
    # --- Affichage du comparatif (Mode Verify) ---
    if verify:
        print(f"📊 ANALYSE COMPARATIVE :")
        data_comp = {
            "Métrique": ["Colonnes", "Lignes", "Poids approx."],
            "AVANT": [f"{cols_init}", f"{rows_init:,}", f"{mem_initial / 1024**2:.2f} MB"],
            "APRÈS": [f"{cols_final}", f"{rows_init:,}", "Calcul en cours..."]
        }
        # Representación simple en tabla
        print(pd.DataFrame(data_comp).to_string(index=False))
        
        print(f"\n📢 Colonnes qui seront supprimées : {colonnes_a_eliminer}")
    
    # --- Logique d'exécution (Mode Overwrite) ---
    if overwrite:
        print(f"\n⚠️  MODE OVERWRITE ACTIVÉ : Suppression définitive en cours...")
        if isinstance(df, pd.DataFrame):
            df_propre = df.drop(columns=colonnes_a_eliminer)
        else: # Polars
            df_propre = df.drop(colonnes_a_eliminer)
        
        print(f"✅ Nettoyage terminé. Nouvelles dimensions : {df_propre.shape}")
        return df_propre
    else:
        print(f"\nℹ️  MODE VERIFY : Aucune modification n'a été appliquée au DataFrame.")
        print("   Pour confirmer, relancez avec 'overwrite=True'.")
        return df