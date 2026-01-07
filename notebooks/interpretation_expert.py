# ##############################################################################
# MODULE : INTERPRETATION_EXPERT.PY - DIAGNOSTIC AUTOMATISÉ
# ##############################################################################
"""
Système expert pour l'interprétation automatique des propriétés statistiques.
Analyse : Asymétrie, Queues (Kurtosis) et Outliers.
"""

# Bibliothèques de base
import pandas as pd
import numpy  as np

# ##############################################################################
# FONCTION : INTERPRETER_PROPRIETES_VARS
# ##############################################################################

def interpreter_proprietes_vars(df):
    """
    Réalise un diagnostic textuel automatique des variables numériques.
    """
    df_num = df.select_dtypes(include=[np.number])
    diagnostics = []

    for col in df_num.columns:
        data = df_num[col].dropna()
        if data.empty: continue

        # 1. Calcul des métriques avancées
        skew     = data.skew()                 # Asymétrie
        kurt     = data.kurtosis()             # Épaisseur des queues
        
        # 2. Détection d'Outliers (Méthode IQR)
        q1, q3   = data.quantile(0.25), data.quantile(0.75)
        iqr      = q3 - q1
        outliers = data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))]
        out_pct  = (len(outliers) / len(data)) * 100

        # ----------------------------------------------------------------------
        # 3. MOTEUR D'INTERPRÉTATION (LOGIQUE SÉMANTIQUE)
        # ----------------------------------------------------------------------
        
        # A. Analyse de l'Asymétrie
        if abs(skew) < 0.5:  msg_skew = "Symétrique (Normale)"
        elif skew > 0.5:      msg_skew = f"Asymétrie Positive ({skew:.1f})"
        else:                 msg_skew = f"Asymétrie Négative ({skew:.1f})"

        # B. Analyse des Queues (Kurtosis / Tail)
        if kurt > 1:         msg_tail = "Queues lourdes (Outliers probables)"
        elif kurt < -1:       msg_tail = "Distribution plate (Uniforme)"
        else:                 msg_tail = "Queues normales"

        # C. Diagnostic Outliers
        if out_pct > 5:      msg_out  = f"Critique ({out_pct:.1f}%)"
        elif out_pct > 0:    msg_out  = f"Modérée ({out_pct:.1f}%)"
        else:                msg_out  = "Aucun"

        diagnostics.append({
            'Variable'      : col,
            'Asymétrie'     : msg_skew,
            'Type_Queues'   : msg_tail,
            'Outliers_IQR'  : msg_out,
            'Action_Data'   : "Log + Robust Scaling" if out_pct > 5 else "Standardize"
        })

    # --------------------------------------------------------------------------
    # 4. PRÉSENTATION DES RÉSULTATS
    # --------------------------------------------------------------------------
    df_diag = pd.DataFrame(diagnostics)
    
    print("\n" + "=" * 80)
    print("ÉTAPE 3.5 : INTERPRÉTATION AUTOMATIQUE DU DATASET")
    print("=" * 80)
    
    return df_diag