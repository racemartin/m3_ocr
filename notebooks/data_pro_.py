"""
Module de prétraitement des données pour le Feature Engineering.
Implémente les étapes de nettoyage initial et d'élimination des colonnes problématiques.

Auteur: Data Science Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataCleaner:

    def ____CONSTRUCTOR():
    """
    Classe principale pour le nettoyage et l'analyse préliminaire des données.
    """
    
    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        """
        Initialise le nettoyeur de données.
        
        Args:
            df: DataFrame à nettoyer
            verbose: Afficher les messages de progression
        """
        # Conversion automatique Polars vers Pandas si nécessaire
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        
        self.df_original = df.copy()
        self.df = df.copy()
        self.verbose = verbose
        
        # Historique des opérations
        self.history = []
        self.suppressed_columns = []

    # =========================================================================
    # TOOLS
    # =========================================================================
    def ____TOOLS():

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne le DataFrame nettoyé."""
        return self.df.copy()
    
    def get_history(self) -> List[Dict]:
        """Retourne l'historique des opérations."""
        return self.history
    
    def reset(self) -> 'DataCleaner':
        """Réinitialise le DataFrame à son état original."""
        self.df = self.df_original.copy()
        self.history = []
        self.suppressed_columns = []
        return self
        
    # =========================================================================
    # Analyse_preliminaire
    # =========================================================================
    def ____Analyse_preliminaire():
        
    def missing_summary(self) -> pd.DataFrame:
        """
        Génère un résumé statistique des valeurs manquantes par colonne.
        
        Returns:
            DataFrame avec les statistiques de valeurs manquantes
        """
        null_counts = self.df.isnull().sum().values
        total_len = len(self.df)
        
        missing_stats = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': null_counts,
            'Missing_Pct': (null_counts / total_len * 100),
            'Dtype': self.df.dtypes.values
        })
        
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
        missing_stats = missing_stats.sort_values('Missing_Pct', ascending=False)
        
        return missing_stats.reset_index(drop=True)
    
    def constant_columns_analysis(
        self, 
        unique_threshold: int = 1, 
        variance_threshold: float = 0.01
    ) -> Dict:
        """
        Identifie les colonnes constantes ou à faible variance.
        
        Args:
            unique_threshold: Seuil pour considérer une colonne comme constante
            variance_threshold: Seuil de variance normalisée pour quasi-constantes
            
        Returns:
            Dictionnaire contenant les listes de colonnes et le DataFrame de synthèse
        """
        results = []
        
        for col in self.df.columns:
            v_counts = self.df[col].value_counts(dropna=False)
            n_unique = len(v_counts)
            n_missing = self.df[col].isnull().sum()
            dtype = self.df[col].dtype
            
            # Analyse de la variance pour types numériques
            variance = None
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].notna().sum() > 1:
                    mean_val = self.df[col].mean()
                    variance = (self.df[col].std() / abs(mean_val) 
                               if mean_val != 0 else self.df[col].std())
            
            # Valeur dominante
            dom_val = v_counts.index[0] if n_unique > 0 else None
            dom_freq = v_counts.values[0] if n_unique > 0 else 0
            dom_pct = (dom_freq / len(self.df) * 100)
            
            # Classifications
            is_const = (n_unique <= unique_threshold)
            is_quasi = (not is_const and variance is not None 
                       and variance < variance_threshold)
            
            results.append({
                'Column': col,
                'N_Unique': n_unique,
                'Missing_Pct': (n_missing / len(self.df) * 100),
                'Dtype': str(dtype),
                'Var_Norm': variance,
                'Dominant_Pct': dom_pct,
                'Is_Constant': is_const,
                'Is_Quasi': is_quasi
            })
        
        summary_df = pd.DataFrame(results)
        
        const_cols = summary_df[summary_df['Is_Constant']]['Column'].tolist()
        q_const = summary_df[summary_df['Is_Quasi']]['Column'].tolist()
        
        summary_df = summary_df.sort_values(
            ['Is_Constant', 'N_Unique'], 
            ascending=[False, True]
        )
        
        return {
            'constant_cols': const_cols,
            'quasi_constant_cols': q_const,
            'summary_df': summary_df.reset_index(drop=True)
        }
    
    def generer_synthese_suppression(
        self, 
        missing_threshold: float = 0.95
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Crée un rapport consolidé des colonnes candidates à l'élimination.
        
        Args:
            missing_threshold: Seuil de manquants au-delà duquel supprimer
            
        Returns:
            Tuple (liste des colonnes à supprimer, DataFrame de synthèse)
        """
        # Récupération des analyses
        missing_df = self.missing_summary()
        variance_res = self.constant_columns_analysis()
        
        # Colonnes avec trop de manquants
        cols_missing = missing_df[
            missing_df['Missing_Pct'] > (missing_threshold * 100)
        ]
        list_missing = cols_missing['Column'].tolist()
        
        # Colonnes constantes
        list_const = variance_res['constant_cols']
        
        # Construction du résumé
        synthese_data = []
        tous_candidats = list(set(list_missing + list_const))
        
        for col in tous_candidats:
            raison = []
            if col in list_missing:
                raison.append(f"Manquants > {missing_threshold*100:.0f}%")
            if col in list_const:
                raison.append("Constante (Unique)")
            
            synthese_data.append({
                'Column': col,
                'Raison_Principale': " & ".join(raison),
                'Dtype': str(self.df[col].dtype),
                'Impact_Potentiel': "Perte d'information nulle"
            })
        
        df_candidats = pd.DataFrame(synthese_data)
        
        if not df_candidats.empty:
            df_candidats = df_candidats.sort_values('Column')
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("SYNTHÈSE DES CANDIDATS À LA SUPPRESSION")
            print("=" * 80)
            print(f" Colonnes analysées........: {len(self.df.columns)}")
            print(f" Colonnes à retirer........: {len(tous_candidats)}")
            print("-" * 80)
        
        return tous_candidats, df_candidats

    # =========================================================================
    # ÉTAPE 1: 
    # =========================================================================
    def ____1_Nettoyage_Initial_et_Elimination():
    
    def supprimer_colonnes_constantes(self) -> 'DataCleaner':
        """
        Supprime les colonnes identifiées comme constantes.
        
        Returns:
            Self pour chaînage des méthodes
        """
        variance_res = self.constant_columns_analysis()
        cols_to_drop = variance_res['constant_cols']
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.suppressed_columns.extend(cols_to_drop)
            self.history.append({
                'operation': 'suppression_constantes',
                'colonnes': cols_to_drop,
                'nb_colonnes': len(cols_to_drop)
            })
            
            if self.verbose:
                print(f"\n✓ Suppression de {len(cols_to_drop)} colonnes constantes")
                print(f"  Colonnes: {', '.join(cols_to_drop)}")
        
        return self

    def supprimer_colonnes_manquantes(
        self, 
        threshold: float = 0.95
    ) -> 'DataCleaner':
        """
        Supprime les colonnes avec un taux de manquants excessif.
        
        Args:
            threshold: Seuil de manquants (0.95 = 95%)
            
        Returns:
            Self pour chaînage des méthodes
        """
        missing_df = self.missing_summary()
        cols_to_drop = missing_df[
            missing_df['Missing_Pct'] > (threshold * 100)
        ]['Column'].tolist()
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.suppressed_columns.extend(cols_to_drop)
            self.history.append({
                'operation': 'suppression_manquants',
                'colonnes': cols_to_drop,
                'nb_colonnes': len(cols_to_drop),
                'seuil': threshold
            })
            
            if self.verbose:
                print(f"\n✓ Suppression de {len(cols_to_drop)} colonnes "
                      f"avec >{threshold*100:.0f}% de manquants")
                print(f"  Colonnes: {', '.join(cols_to_drop)}")
        
        return self
    
    def supprimer_colonnes_specifiques(
        self, 
        colonnes: List[str]
    ) -> 'DataCleaner':
        """
        Supprime des colonnes spécifiques (ex: Comments, Outlier).
        
        Args:
            colonnes: Liste des noms de colonnes à supprimer
            
        Returns:
            Self pour chaînage des méthodes
        """
        cols_existantes = [c for c in colonnes if c in self.df.columns]
        
        if cols_existantes:
            self.df = self.df.drop(columns=cols_existantes)
            self.suppressed_columns.extend(cols_existantes)
            self.history.append({
                'operation': 'suppression_specifique',
                'colonnes': cols_existantes,
                'nb_colonnes': len(cols_existantes)
            })
            
            if self.verbose:
                print(f"\n✓ Suppression de {len(cols_existantes)} "
                      f"colonnes spécifiques")
                print(f"  Colonnes: {', '.join(cols_existantes)}")
        
        return self


    def nettoyage_initial_complet(
        self, 
        missing_threshold: float = 0.95,
        colonnes_specifiques: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Exécute la séquence complète de nettoyage initial (Étape 1).
        
        Args:
            missing_threshold: Seuil de manquants
            colonnes_specifiques: Colonnes supplémentaires à retirer
            
        Returns:
            Self pour chaînage des méthodes
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 1: NETTOYAGE INITIAL ET ÉLIMINATION")
            print("=" * 80)
            print(f"Shape initiale: {self.df.shape}")
        
        # 1.1 Suppression des constantes
        self.supprimer_colonnes_constantes()
        
        # 1.2 Élimination des manquants excessifs
        self.supprimer_colonnes_manquantes(threshold=missing_threshold)
        
        # Suppression des colonnes spécifiques si fournies
        if colonnes_specifiques:
            self.supprimer_colonnes_specifiques(colonnes_specifiques)
        
        if self.verbose:
            print(f"\nShape finale: {self.df.shape}")
            print(f"Total colonnes supprimées: {len(self.suppressed_columns)}")
            print("=" * 80)
        
        return self   
    
    # =========================================================================
    # ÉTAPE 2: GESTION DES VALEURS MANQUANTES
    # =========================================================================
    def ____2_Gestion_des_Valeurs_Manquantes():
    
    def _identifier_colonnes_par_type(self) -> Dict[str, List[str]]:
        """
        Identifie et sépare les colonnes numériques et catégorielles.
        
        Returns:
            Dict avec 'numeriques' et 'categoriques'
        """
        numeriques = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoriques = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        return {
            'numeriques': numeriques,
            'categoriques': categoriques
        }
    
    def creer_indicateurs_missing(
        self, 
        threshold: float = 0.50,
        suffix: str = '_Manquant'
    ) -> 'DataCleaner':
        """
        ÉTAPE 2.1: Création d'Indicateurs de Valeurs Manquantes.
        
        Crée des features binaires [Col]_Manquant pour colonnes >threshold% missing.
        TRAITEMENT SÉPARÉ par type de variable (numérique vs catégorielle).
        
        JUSTIFICATION:
        - L'absence peut être informative (ex: pas d'équipement, pas de certification)
        - Préserve l'information AVANT imputation
        - Améliore les performances prédictives du modèle
        
        Args:
            threshold: Seuil de missing au-delà duquel créer l'indicateur (0.50 = 50%)
            suffix: Suffixe pour les nouvelles colonnes
            
        Returns:
            Self pour chaînage des méthodes
        """
        # Calcul du taux de missing par colonne
        missing_ratio = self.df.isnull().sum() / len(self.df)
        candidats = missing_ratio[missing_ratio > threshold]
        
        if candidats.empty:
            if self.verbose:
                print(f"\n⚠️  Aucune colonne avec >{threshold*100:.0f}% de valeurs manquantes")
            return self
        
        # Séparation par type
        types_cols = self._identifier_colonnes_par_type()
        
        # Filtrage des candidats par type
        candidats_num = [c for c in candidats.index if c in types_cols['numeriques']]
        candidats_cat = [c for c in candidats.index if c in types_cols['categoriques']]
        
        indicateurs_crees = []
        
        # =====================================================================
        # CRÉATION DES INDICATEURS - NUMÉRIQUES
        # =====================================================================
        for col in candidats_num:
            nom_indicateur = f"{col}{suffix}"
            self.df[nom_indicateur] = self.df[col].isnull().astype(int)
            indicateurs_crees.append({
                'colonne_origine': col,
                'indicateur': nom_indicateur,
                'type': 'numérique',
                'missing_pct': missing_ratio[col] * 100
            })
        
        # =====================================================================
        # CRÉATION DES INDICATEURS - CATÉGORIELLES
        # =====================================================================
        for col in candidats_cat:
            nom_indicateur = f"{col}{suffix}"
            self.df[nom_indicateur] = self.df[col].isnull().astype(int)
            indicateurs_crees.append({
                'colonne_origine': col,
                'indicateur': nom_indicateur,
                'type': 'catégorielle',
                'missing_pct': missing_ratio[col] * 100
            })
        
        # Enregistrement dans l'historique
        if indicateurs_crees:
            self.history.append({
                'operation': 'creation_indicateurs_missing',
                'threshold': threshold,
                'nb_indicateurs_num': len(candidats_num),
                'nb_indicateurs_cat': len(candidats_cat),
                'indicateurs_numeriques': candidats_num,
                'indicateurs_categoriques': candidats_cat,
                'details': indicateurs_crees
            })
            
            if self.verbose:
                print("\n" + "=" * 80)
                print("ÉTAPE 2.1 : CRÉATION D'INDICATEURS DE VALEURS MANQUANTES")
                print("=" * 80)
                print(f"Seuil appliqué............: >{threshold*100:.0f}%")
                print(f"Indicateurs numériques....: {len(candidats_num)}")
                if candidats_num:
                    for col in candidats_num:
                        print(f"  • {col} ({missing_ratio[col]*100:.2f}%) → {col}{suffix}")
                
                print(f"Indicateurs catégoriels...: {len(candidats_cat)}")
                if candidats_cat:
                    for col in candidats_cat:
                        print(f"  • {col} ({missing_ratio[col]*100:.2f}%) → {col}{suffix}")
                
                print(f"Total créés...............: {len(indicateurs_crees)}")
                print("=" * 80)
        
        return self


    def imputer_categoriques(
        self, 
        valeur_defaut: str = 'INCONNU',
        colonnes_specifiques: Optional[Dict[str, str]] = None
    ) -> 'DataCleaner':
        """
        ÉTAPE 2.2: Imputation Catégorielle.
        
        Remplace les valeurs manquantes dans les colonnes catégorielles.
        
        STRATÉGIE:
        - Valeur par défaut: 'INCONNU' ou 'AUCUN'
        - Permet de spécifier des valeurs personnalisées par colonne
        
        Args:
            valeur_defaut: Valeur à utiliser par défaut pour toutes les catégorielles
            colonnes_specifiques: Dict {nom_colonne: valeur_specifique} pour exceptions
            
        Returns:
            Self pour chaînage des méthodes
            
        Example:
            cleaner.imputer_categoriques(
                valeur_defaut='INCONNU',
                colonnes_specifiques={
                    'PropertyUseType': 'AUCUN',
                    'Neighborhood': 'NON_SPECIFIE'
                }
            )
        """
        # Identification des colonnes catégorielles avec valeurs manquantes
        types_cols = self._identifier_colonnes_par_type()
        cat_cols = types_cols['categoriques']
        
        # Filtrer uniquement celles qui ont des valeurs manquantes
        cat_with_missing = [
            col for col in cat_cols 
            if self.df[col].isnull().sum() > 0
        ]
        
        if not cat_with_missing:
            if self.verbose:
                print("\n⚠️  Aucune colonne catégorielle avec valeurs manquantes")
            return self
        
        imputations = []
        
        for col in cat_with_missing:
            # Déterminer la valeur d'imputation
            if colonnes_specifiques and col in colonnes_specifiques:
                valeur_imputation = colonnes_specifiques[col]
            else:
                valeur_imputation = valeur_defaut
            
            # Compter les valeurs manquantes avant imputation
            nb_missing = self.df[col].isnull().sum()
            pct_missing = (nb_missing / len(self.df)) * 100
            
            # Imputation
            self.df[col] = self.df[col].fillna(valeur_imputation)
            
            imputations.append({
                'colonne': col,
                'valeur_imputation': valeur_imputation,
                'nb_values_imputees': nb_missing,
                'pct_impute': pct_missing
            })
        
        # Enregistrement dans l'historique
        self.history.append({
            'operation': 'imputation_categoriques',
            'valeur_defaut': valeur_defaut,
            'nb_colonnes_imputees': len(cat_with_missing),
            'colonnes': cat_with_missing,
            'details': imputations
        })
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 2.2 : IMPUTATION CATÉGORIELLE")
            print("=" * 80)
            print(f"Valeur par défaut.........: '{valeur_defaut}'")
            print(f"Colonnes traitées.........: {len(cat_with_missing)}")
            print()
            for imp in imputations:
                valeur_affichee = imp['valeur_imputation']
                if valeur_affichee != valeur_defaut:
                    valeur_affichee = f"'{valeur_affichee}' (spécifique)"
                else:
                    valeur_affichee = f"'{valeur_affichee}'"
                    
                print(f"  • {imp['colonne']:<40} → {valeur_affichee}")
                print(f"    Valeurs imputées: {imp['nb_values_imputees']:>5} "
                      f"({imp['pct_impute']:>6.2f}%)")
            print("=" * 80)
        
        return self
    


    def imputer_numeriques(
        self, 
        strategie: str = 'mediane',
        colonnes_specifiques: Optional[Dict[str, float]] = None
    ) -> 'DataCleaner':
        """
        ÉTAPE 2.3: Imputation Numérique avec la Médiane.
        
        Remplace les valeurs manquantes dans les colonnes numériques 
        en utilisant la médiane GLOBALE de chaque colonne.
        
        STRATÉGIE:
        - Calcul de la médiane sur TOUS les valeurs non-NaN de la colonne
        - Imputation de TOUS les NaN avec cette médiane unique
        - Robuste aux outliers (contrairement à la moyenne)
        - Les indicateurs créés en 2.1 sont PRÉSERVÉS
        
        Args:
            strategie: Type d'imputation ('mediane' uniquement pour l'instant)
            colonnes_specifiques: Dict {nom_colonne: valeur} pour imputation manuelle
            
        Returns:
            Self pour chaînage des méthodes
        """
        if strategie != 'mediane':
            raise ValueError("Seule la stratégie 'mediane' est implémentée")
        
        # Identification des colonnes numériques avec valeurs manquantes
        types_cols = self._identifier_colonnes_par_type()
        num_cols = types_cols['numeriques']
        
        num_with_missing = [
            col for col in num_cols 
            if self.df[col].isnull().sum() > 0
        ]
        
        if not num_with_missing:
            if self.verbose:
                print("\n⚠️  Aucune colonne numérique avec valeurs manquantes")
            return self
        
        imputations = []
        
        for col in num_with_missing:
            nb_missing = self.df[col].isnull().sum()
            pct_missing = (nb_missing / len(self.df)) * 100
            nb_existants = self.df[col].notna().sum()
            
            # Déterminer la valeur d'imputation
            if colonnes_specifiques and col in colonnes_specifiques:
                valeur_imputation = colonnes_specifiques[col]
                source = 'manuelle'
            else:
                valeur_imputation = self.df[col].median()
                source = 'mediane'
            
            # Statistiques AVANT imputation
            stats_avant = {
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }
            
            # IMPUTATION
            self.df[col] = self.df[col].fillna(valeur_imputation)
            
            # Vérification
            nb_missing_apres = self.df[col].isnull().sum()
            
            imputations.append({
                'colonne': col,
                'strategie': source,
                'valeur_imputation': round(valeur_imputation, 4),
                'nb_values_imputees': nb_missing,
                'pct_impute': pct_missing,
                'nb_values_existants': nb_existants,
                'stats_avant': stats_avant,
                'verification': nb_missing_apres == 0
            })
        
        # Historique
        self.history.append({
            'operation': 'imputation_numeriques',
            'strategie': strategie,
            'nb_colonnes_imputees': len(num_with_missing),
            'colonnes': num_with_missing,
            'details': imputations
        })
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 2.3 : IMPUTATION NUMÉRIQUE")
            print("=" * 80)
            print(f"Stratégie.................: {strategie.upper()}")
            print(f"Colonnes traitées.........: {len(num_with_missing)}")
            print()
            
            for imp in imputations:
                source_label = "médiane" if imp['strategie'] == 'mediane' else "manuelle"
                
                print(f"  • {imp['colonne']:<45}")
                print(f"    Valeur d'imputation: {imp['valeur_imputation']:>12.4f} ({source_label})")
                print(f"    Valeurs imputées...: {imp['nb_values_imputees']:>5} "
                      f"({imp['pct_impute']:>6.2f}%) sur {imp['nb_values_existants']} existantes")
                print(f"    Vérification.......: {'✓ OK' if imp['verification'] else '✗ ERREUR'}")
                print()
            
            print("=" * 80)
        
        return self

    def gestion_valeurs_manquantes_complete(
        self,
        threshold_indicateurs: float = 0.50,
        valeur_cat_defaut: str = 'INCONNU',
        strategie_num: str = 'mediane',
        colonnes_cat_specifiques: Optional[Dict[str, str]] = None,
        colonnes_num_specifiques: Optional[Dict[str, float]] = None
    ) -> 'DataCleaner':
        """
        Exécute la séquence complète de gestion des valeurs manquantes (Étape 2).
        
        PIPELINE: 2.1 → 2.2 → 2.3
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("ÉTAPE 2: GESTION COMPLÈTE DES VALEURS MANQUANTES")
            print("=" * 80)
            
            total_missing = self.df.isnull().sum().sum()
            pct_missing = (total_missing / (len(self.df) * len(self.df.columns))) * 100
            print(f"État initial:")
            print(f"  • Valeurs manquantes: {total_missing} ({pct_missing:.2f}%)")
        
        self.creer_indicateurs_missing(threshold=threshold_indicateurs)
        self.imputer_categoriques(
            valeur_defaut=valeur_cat_defaut,
            colonnes_specifiques=colonnes_cat_specifiques
        )
        self.imputer_numeriques(
            strategie=strategie_num,
            colonnes_specifiques=colonnes_num_specifiques
        )
        
        if self.verbose:
            total_missing_final = self.df.isnull().sum().sum()
            print("\n" + "-" * 80)
            print("BILAN FINAL:")
            print(f"  • Valeurs manquantes: {total_missing_final}")
            print(f"  • Réduction: {total_missing - total_missing_final} valeurs")
            print("=" * 80)
        
        return self
    
    # =========================================================================
    # ÉTAPE 3: TRAITEMENT DE L'ASYMÉTRIE ET OUTLIERS
    # =========================================================================
    def ____3_Traitement_de_Asymetrie_et_Outliers():
            
    def transformation_logarithmique(
        self,
        colonnes: Optional[List[str]] = None,
        auto_detect: bool = True,
        skew_threshold: float = 1.0,
        suffix: str = '_log'
    ) -> 'DataCleaner':
        """
        ÉTAPE 3.1: Transformation Logarithmique.
        
        Applique log(x + 1) aux variables asymétriques pour:
        - Réduire l'asymétrie (skewness)
        - Normaliser les distributions
        - Stabiliser la variance
        - Gérer les outliers
        
        IMPORTANT: Ajoute 1 avant log pour gérer les valeurs nulles.
        
        Args:
            colonnes: Liste de colonnes à transformer (si None, auto-détection)
            auto_detect: Détecter automatiquement les colonnes asymétriques
            skew_threshold: Seuil de skewness pour auto-détection (1.0 par défaut)
            suffix: Suffixe pour les nouvelles colonnes
            
        Returns:
            Self pour chaînage des méthodes
        """
        # Identification des colonnes numériques
        types_cols = self._identifier_colonnes_par_type()
        num_cols   = types_cols['numeriques']
        
        # Exclure les colonnes indicateurs
        num_cols = [c for c in num_cols if '_Manquant' not in c]
        
        if not num_cols:
            if self.verbose:
                print("\n⚠️  Aucune colonne numérique disponible")
            return self
        
        # Déterminer les colonnes à transformer
        if auto_detect and colonnes is None:
            # Auto-détection basée sur le skewness
            cols_a_transformer = []
            skewness_info      = []
            
            for col in num_cols:
                skew_val = self.df[col].skew()
                
                if abs(skew_val) > skew_threshold:
                    cols_a_transformer.append(col)
                    skewness_info.append({
                        'colonne'  : col,
                        'skewness' : skew_val
                    })
        
        elif colonnes is not None:
            # Utiliser les colonnes spécifiées
            cols_a_transformer = [c for c in colonnes if c in num_cols]
            skewness_info      = [
                {'colonne': c, 'skewness': self.df[c].skew()} 
                for c in cols_a_transformer
            ]
        
        else:
            if self.verbose:
                print("\n⚠️  Aucune colonne à transformer")
            return self
        
        if not cols_a_transformer:
            if self.verbose:
                print(f"\n⚠️  Aucune colonne avec |skewness| > {skew_threshold}")
            return self
        
        # Application de la transformation
        transformations = []
        
        for col in cols_a_transformer:
            # Statistiques AVANT transformation
            skew_avant = self.df[col].skew()
            min_avant  = self.df[col].min()
            max_avant  = self.df[col].max()
            
            # Vérifier si la colonne contient des valeurs négatives
            has_negative = (self.df[col] < 0).any()
            
            if has_negative:
                # Décalage pour rendre toutes les valeurs positives
                decalage           = abs(self.df[col].min()) + 1
                nom_col_transformed = f"{col}{suffix}"
                
                self.df[nom_col_transformed] = np.log1p(self.df[col] + decalage)
                
                transformations.append({
                    'colonne'       : col,
                    'colonne_log'   : nom_col_transformed,
                    'skew_avant'    : skew_avant,
                    'skew_apres'    : self.df[nom_col_transformed].skew(),
                    'decalage'      : decalage,
                    'has_negative'  : True,
                    'min_avant'     : min_avant,
                    'max_avant'     : max_avant
                })
            else:
                # Transformation directe avec log1p
                nom_col_transformed = f"{col}{suffix}"
                
                self.df[nom_col_transformed] = np.log1p(self.df[col])
                
                transformations.append({
                    'colonne'       : col,
                    'colonne_log'   : nom_col_transformed,
                    'skew_avant'    : skew_avant,
                    'skew_apres'    : self.df[nom_col_transformed].skew(),
                    'decalage'      : 0,
                    'has_negative'  : False,
                    'min_avant'     : min_avant,
                    'max_avant'     : max_avant
                })
        
        # Enregistrement dans l'historique
        self.history.append({
            'operation'            : 'transformation_logarithmique',
            'auto_detect'          : auto_detect,
            'skew_threshold'       : skew_threshold,
            'nb_transformations'   : len(transformations),
            'colonnes_originales'  : cols_a_transformer,
            'details'              : transformations
        })
        
        if self.verbose:
            print("\n============================================================================")
            print("ÉTAPE 3.1 : TRANSFORMATION LOGARITHMIQUE")
            print("============================================================================")
            print(f"Méthode..................: {'Auto-détection' if auto_detect else 'Manuel'}")
            if auto_detect:
                print(f"Seuil de skewness........: |skew| > {skew_threshold}")
            print(f"Colonnes transformées....: {len(transformations)}")
            print()
            
            # Affichage détaillé
            for trans in transformations:
                print(f"  • {trans['colonne']:<40} → {trans['colonne_log']}")
                print(f"    Skewness avant.......: {trans['skew_avant']:>8.3f}")
                print(f"    Skewness après.......: {trans['skew_apres']:>8.3f}")
                print(f"    Amélioration.........: {abs(trans['skew_avant']) - abs(trans['skew_apres']):>8.3f}")
                
                if trans['has_negative']:
                    print(f"    ⚠️ Valeurs négatives → Décalage: +{trans['decalage']:.2f}")
                
                print()
            
            print("============================================================================")
        
        return self
    
    def winsorisation(
        self,
        colonnes: Optional[List[str]] = None,
        percentile_bas: float = 0.01,
        percentile_haut: float = 0.99,
        inplace: bool = False,
        suffix: str = '_wins'
    ) -> 'DataCleaner':
        """
        ÉTAPE 3.2: Winsorisation/Écrêtage des Outliers.
        
        Cappe les valeurs extrêmes aux percentiles spécifiés pour:
        - Réduire l'impact des outliers
        - Préserver la structure des données (vs suppression)
        - Améliorer la robustesse des modèles
        
        Args:
            colonnes: Liste de colonnes à traiter (si None, toutes les numériques)
            percentile_bas: Percentile inférieur (0.01 = 1%)
            percentile_haut: Percentile supérieur (0.99 = 99%)
            inplace: Modifier les colonnes originales (True) ou créer nouvelles (False)
            suffix: Suffixe pour les nouvelles colonnes (si inplace=False)
            
        Returns:
            Self pour chaînage des méthodes
        """
        # Identification des colonnes numériques
        types_cols = self._identifier_colonnes_par_type()
        num_cols   = types_cols['numeriques']
        
        # Exclure les indicateurs et colonnes log
        num_cols = [
            c for c in num_cols 
            if '_Manquant' not in c and '_log' not in c
        ]
        
        # Déterminer les colonnes à traiter
        if colonnes is not None:
            cols_a_traiter = [c for c in colonnes if c in num_cols]
        else:
            cols_a_traiter = num_cols
        
        if not cols_a_traiter:
            if self.verbose:
                print("\n⚠️  Aucune colonne à winsoriser")
            return self
        
        # Application de la winsorisation
        winsorisations = []
        
        for col in cols_a_traiter:
            # Calcul des bornes
            q_bas  = self.df[col].quantile(percentile_bas)
            q_haut = self.df[col].quantile(percentile_haut)
            
            # Comptage des valeurs cappées
            nb_bas   = (self.df[col] < q_bas).sum()
            nb_haut  = (self.df[col] > q_haut).sum()
            nb_total = nb_bas + nb_haut
            pct_cappe = (nb_total / len(self.df)) * 100
            
            # Statistiques AVANT
            min_avant  = self.df[col].min()
            max_avant  = self.df[col].max()
            mean_avant = self.df[col].mean()
            std_avant  = self.df[col].std()
            
            # Application du capping
            if inplace:
                # Modification directe
                self.df[col] = self.df[col].clip(lower=q_bas, upper=q_haut)
                col_finale   = col
            else:
                # Création d'une nouvelle colonne
                col_finale            = f"{col}{suffix}"
                self.df[col_finale]   = self.df[col].clip(lower=q_bas, upper=q_haut)
            
            # Statistiques APRÈS
            min_apres  = self.df[col_finale].min()
            max_apres  = self.df[col_finale].max()
            mean_apres = self.df[col_finale].mean()
            std_apres  = self.df[col_finale].std()
            
            winsorisations.append({
                'colonne'        : col,
                'colonne_finale' : col_finale,
                'borne_inf'      : q_bas,
                'borne_sup'      : q_haut,
                'nb_cappe_bas'   : nb_bas,
                'nb_cappe_haut'  : nb_haut,
                'nb_total_cappe' : nb_total,
                'pct_cappe'      : pct_cappe,
                'stats_avant'    : {
                    'min'  : min_avant,
                    'max'  : max_avant,
                    'mean' : mean_avant,
                    'std'  : std_avant
                },
                'stats_apres'    : {
                    'min'  : min_apres,
                    'max'  : max_apres,
                    'mean' : mean_apres,
                    'std'  : std_apres
                }
            })
        
        # Enregistrement dans l'historique
        self.history.append({
            'operation'           : 'winsorisation',
            'percentile_bas'      : percentile_bas,
            'percentile_haut'     : percentile_haut,
            'inplace'             : inplace,
            'nb_colonnes_traitees': len(winsorisations),
            'colonnes'            : cols_a_traiter,
            'details'             : winsorisations
        })
        
        if self.verbose:
            print("\n============================================================================")
            print("ÉTAPE 3.2 : WINSORISATION/ÉCRÊTAGE")
            print("============================================================================")
            print(f"Percentiles..............: {percentile_bas*100:.0f}% - {percentile_haut*100:.0f}%")
            print(f"Mode.....................: {'Inplace' if inplace else 'Nouvelles colonnes'}")
            print(f"Colonnes traitées........: {len(winsorisations)}")
            print()
            
            # Affichage détaillé
            for wins in winsorisations:
                col_affichage = wins['colonne']
                if not inplace:
                    col_affichage = f"{wins['colonne']} → {wins['colonne_finale']}"
                
                print(f"  • {col_affichage}")
                print(f"    Bornes...............: [{wins['borne_inf']:.2f}, {wins['borne_sup']:.2f}]")
                print(f"    Valeurs cappées......: {wins['nb_total_cappe']} ({wins['pct_cappe']:.2f}%)")
                print(f"      - Bas (< {wins['borne_inf']:.2f})...: {wins['nb_cappe_bas']}")
                print(f"      - Haut (> {wins['borne_sup']:.2f}).: {wins['nb_cappe_haut']}")
                print()
            
            print("============================================================================")
        
        return self
    
    def traitement_asymetrie_complet(
        self,
        log_colonnes: Optional[List[str]] = None,
        log_auto_detect: bool = True,
        log_skew_threshold: float = 1.0,
        wins_colonnes: Optional[List[str]] = None,
        wins_percentiles: tuple = (0.01, 0.99),
        wins_inplace: bool = False
    ) -> 'DataCleaner':
        """
        Exécute la séquence complète de traitement de l'asymétrie (Étape 3).
        
        PIPELINE COMPLET:
        3.1 → Transformation logarithmique
        3.2 → Winsorisation
        
        Args:
            log_colonnes: Colonnes pour transformation log (None = auto)
            log_auto_detect: Auto-détection des colonnes asymétriques
            log_skew_threshold: Seuil de skewness
            wins_colonnes: Colonnes pour winsorisation (None = toutes)
            wins_percentiles: Tuple (percentile_bas, percentile_haut)
            wins_inplace: Modifier colonnes originales ou créer nouvelles
            
        Returns:
            Self pour chaînage des méthodes
        """
        if self.verbose:
            print("\n============================================================================")
            print("ÉTAPE 3: TRAITEMENT COMPLET DE L'ASYMÉTRIE ET OUTLIERS")
            print("============================================================================")
        
        # 3.1 Transformation logarithmique
        self.transformation_logarithmique(
            colonnes=log_colonnes,
            auto_detect=log_auto_detect,
            skew_threshold=log_skew_threshold
        )
        
        # 3.2 Winsorisation
        self.winsorisation(
            colonnes=wins_colonnes,
            percentile_bas=wins_percentiles[0],
            percentile_haut=wins_percentiles[1],
            inplace=wins_inplace
        )
        
        if self.verbose:
            print("\n============================================================================")
            print("BILAN ÉTAPE 3 COMPLÈTE")
            print("============================================================================")
            print(f"Shape finale.............: {self.df.shape}")
            
            # Compter les nouvelles colonnes
            nb_cols_log  = len([c for c in self.df.columns if '_log' in c])
            nb_cols_wins = len([c for c in self.df.columns if '_wins' in c])
            
            print(f"Colonnes log créées......: {nb_cols_log}")
            print(f"Colonnes wins créées.....: {nb_cols_wins}")
            print("============================================================================")
        
        return self
    
    
    # =========================================================================
    # GENERER RAPPORT
    # =========================================================================
    def ____GENERER_RAPPORT(): 

    def generer_rapport_etape2(self) -> pd.DataFrame:
        """
        Génère un rapport consolidé de l'Étape 2 (Gestion des Valeurs Manquantes).
        
        Returns:
            DataFrame avec le résumé des opérations
        """
        etape2_ops = [
            op for op in self.history 
            if op['operation'] in [
                'creation_indicateurs_missing', 
                'imputation_categoriques',
                'imputation_numeriques'
            ]
        ]
        
        if not etape2_ops:
            print("⚠️  Aucune opération d'Étape 2 trouvée dans l'historique")
            return pd.DataFrame()
        
        rapport_data = []
        
        for op in etape2_ops:
            if op['operation'] == 'creation_indicateurs_missing':
                rapport_data.append({
                    'Étape': '2.1 - Indicateurs',
                    'Type': 'Numériques',
                    'Nb_Actions': op['nb_indicateurs_num'],
                    'Details': ', '.join(op['indicateurs_numeriques']) 
                               if op['indicateurs_numeriques'] else 'Aucun'
                })
                rapport_data.append({
                    'Étape': '2.1 - Indicateurs',
                    'Type': 'Catégorielles',
                    'Nb_Actions': op['nb_indicateurs_cat'],
                    'Details': ', '.join(op['indicateurs_categoriques']) 
                               if op['indicateurs_categoriques'] else 'Aucun'
                })
            
            elif op['operation'] == 'imputation_categoriques':
                rapport_data.append({
                    'Étape': '2.2 - Imputation Cat.',
                    'Type': f"Valeur: '{op['valeur_defaut']}'",
                    'Nb_Actions': op['nb_colonnes_imputees'],
                    'Details': ', '.join(op['colonnes'])
                })
            
            elif op['operation'] == 'imputation_numeriques':
                total_imputations = sum(
                    detail['nb_values_imputees'] 
                    for detail in op['details']
                )
                rapport_data.append({
                    'Étape': '2.3 - Imputation Num.',
                    'Type': f"Stratégie: {op['strategie']}",
                    'Nb_Actions': f"{op['nb_colonnes_imputees']} cols, {total_imputations} valeurs",
                    'Details': ', '.join(op['colonnes'])
                })
            
            elif op['operation'] == 'imputation_numeriques':
                total_imputations = sum(
                    detail['nb_values_imputees'] 
                    for detail in op['details']
                )
                rapport_data.append({
                    'Étape': '2.3 - Imputation Num.',
                    'Type': f"Stratégie: {op['strategie']}",
                    'Nb_Actions': f"{op['nb_colonnes_imputees']} cols, {total_imputations} valeurs",
                    'Details': ', '.join(op['colonnes'])
                })
        
        return pd.DataFrame(rapport_data)