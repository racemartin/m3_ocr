# ============================================================
# CLASE MODELER: Pipeline ML Orientado a Objetos
# ============================================================


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer, mean_absolute_error, 
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import time
import warnings

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DEFAULT
# ============================================================

DEFAULT_METRICS = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'mape': make_scorer(
        lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        greater_is_better=False
    )
}

# ============================================================
# 🎯 CLASE PRINCIPAL: Modeler
# ============================================================

class Modeler:
    """
    Pipeline completo de Machine Learning con seguimiento de experimentos.
    
    Características:
    ----------------
    - Entrenamiento con validación cruzada
    - Evaluación automática en train y test
    - Detección de overfitting
    - Comparación de múltiples modelos
    - Visualizaciones automáticas
    - Historial de experimentos
    - Selección del mejor modelo
    
    Ejemplo de uso:
    ---------------
    >>> modeler = Modeler(X_train, y_train, X_test, y_test)
    >>> modeler.entrenar_modelo(LinearRegression(), 'Linear')
    >>> modeler.entrenar_modelo(RandomForestRegressor(), 'RF')
    >>> modeler.comparar_modelos()
    >>> modeler.visualizar_mejor_modelo()
    >>> best_model = modeler.get_mejor_modelo()
    """
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: Optional[Dict] = None
    ):
        """
        Inicializa el Modeler.
        
        Parámetros:
        -----------
        X_train, y_train : datos de entrenamiento
        X_test, y_test : datos de test
        config : dict opcional con configuración
            - RANDOM_STATE: semilla aleatoria
            - CV_FOLDS: número de folds para CV
            - METRICS: diccionario de métricas sklearn
        """
        # Datos
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Configuración
        default_config = {
            'RANDOM_STATE': 42,
            'CV_FOLDS': 5,
            'METRICS': DEFAULT_METRICS
        }
        self.config = {**default_config, **(config or {})}
        
        # Fijar semilla
        np.random.seed(self.config['RANDOM_STATE'])
        
        # Historial de experimentos
        self.results_history = {}
        self.experiment_counter = 0
        
        # Mejor modelo
        self._best_model_name = None
        
        # Mostrar info de inicialización
        self._print_initialization_info()
    
    def _print_initialization_info(self):
        """Imprime información de inicialización."""
        print(f"\n{'='*70}")
        print("🚀 MODELER INICIALIZADO")
        print(f"{'='*70}")
        print(f"📊 Datos de entrenamiento : {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"📊 Datos de test          : {self.X_test.shape[0]} samples")
        print(f"🎲 Random State           : {self.config['RANDOM_STATE']}")
        print(f"🔄 CV Folds               : {self.config['CV_FOLDS']}")
        print(f"📏 Métricas               : {list(self.config['METRICS'].keys())}")
        print(f"{'='*70}\n")





    
    def backwards_elimination(
        self,
        X: pd.DataFrame, 
        y: pd.Series, 
        seuil_pvalue: float = 0.05,
        verbose: bool = True
    ) -> dict:
        """
        Automatise l'élimination backwards pour la régression linéaire.
        
        Paramètres:
        -----------
        X : pd.DataFrame
            Features (variables indépendantes)
        y : pd.Series
            Target (variable dépendante)
        seuil_pvalue : float (default=0.05)
            Seuil pour éliminer une variable
        verbose : bool (default=True)
            Afficher les étapes détaillées
        
        Retourne:
        ---------
        dict contenant:
            - 'model': Modèle final OLS ajusté
            - 'features': Liste des features retenues
            - 'history': Historique des éliminations
            - 'vif': VIF du modèle final
        """
        
        X_working = X.copy()
        features_restantes = list(X_working.columns)
        historique = []
        iteration = 0
        
        if verbose:
            print("="*80)
            print("🚀 DÉMARRAGE DE L'ÉLIMINATION BACKWARDS")
            print("="*80)
            print(f"Nombre initial de features : {len(features_restantes)}")
            print(f"Features : {features_restantes}")
            print(f"Seuil p-value : {seuil_pvalue}")
            print(f"Taille dataset : {len(X)} observations\n")
        
        while True:
            iteration += 1
            
            # Ajustement du modèle
            model = sm.OLS(endog=y, exog=X_working).fit()
            
            # Récupération des p-values
            pvalues = model.pvalues
            max_pvalue = pvalues.max()
            feature_max_pvalue = pvalues.idxmax()
            
            if verbose:
                print(f"\n{'─'*80}")
                print(f"ITÉRATION {iteration}")
                print(f"{'─'*80}")
                print(f"Features actuelles : {len(features_restantes)}")
                print(f"R² ajusté         : {model.rsquared_adj:.4f}")
                print(f"AIC               : {model.aic:.2f}")
                print(f"Condition Number  : {model.condition_number:.2e}")
                print(f"\n📊 P-values des features :")
                for feat, pval in pvalues.items():
                    statut = "✅" if pval < seuil_pvalue else "❌"
                    print(f"  {statut} {feat:30s} : p-value = {pval:.4f}")
            
            # Critère d'arrêt : toutes les p-values < seuil
            if max_pvalue < seuil_pvalue:
                if verbose:
                    print(f"\n{'='*80}")
                    print("✅ CONVERGENCE : Toutes les features sont significatives !")
                    print(f"{'='*80}")
                break
            
            # Élimination de la feature avec la plus grande p-value
            if verbose:
                print(f"\n🗑️  ÉLIMINATION : '{feature_max_pvalue}' (p-value = {max_pvalue:.4f})")
            
            historique.append({
                'iteration': iteration,
                'feature_eliminee': feature_max_pvalue,
                'pvalue': max_pvalue,
                'r2_adj': model.rsquared_adj,
                'aic': model.aic,
                'features_restantes': features_restantes.copy()
            })
            
            # Retirer la feature
            X_working = X_working.drop(columns=[feature_max_pvalue])
            features_restantes.remove(feature_max_pvalue)
            
            # Sécurité : arrêt si plus de features
            if len(features_restantes) == 0:
                print("\n⚠️  ATTENTION : Plus aucune feature restante !")
                break
        
        # Modèle final
        model_final = sm.OLS(endog=y, exog=X_working).fit()
        
        # Calcul des VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_working.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_working.values, i) 
            for i in range(X_working.shape[1])
        ]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        if verbose:
            print(f"\n{'='*80}")
            print("🏆 MODÈLE FINAL")
            print(f"{'='*80}")
            print(f"Features retenues   : {features_restantes}")
            print(f"Nombre de features  : {len(features_restantes)}")
            print(f"R² ajusté           : {model_final.rsquared_adj:.4f}")
            print(f"AIC                 : {model_final.aic:.2f}")
            print(f"Condition Number    : {model_final.condition_number:.2e}")
            
            print(f"\n{'─'*80}")
            print("📊 COEFFICIENTS FINAUX")
            print(f"{'─'*80}")
            for feat, coef in model_final.params.items():
                pval = model_final.pvalues[feat]
                print(f"  {feat:30s} : {coef:>12.2f}  (p-value = {pval:.4f})")
            
            print(f"\n{'─'*80}")
            print("📊 VIF - VÉRIFICATION MULTICOLINÉARITÉ")
            print(f"{'─'*80}")
            for _, row in vif_data.iterrows():
                alerte = "⚠️" if row['VIF'] > 10 else ("⚡" if row['VIF'] > 5 else "✅")
                print(f"  {alerte} {row['Feature']:30s} : VIF = {row['VIF']:.2f}")
            
            if (vif_data['VIF'] > 10).any():
                print("\n⚠️  ATTENTION : Multicolinéarité forte détectée (VIF > 10)")
            
            print(f"\n{'─'*80}")
            print("📜 HISTORIQUE DES ÉLIMINATIONS")
            print(f"{'─'*80}")
            for h in historique:
                print(f"  Iter {h['iteration']} : {h['feature_eliminee']:30s} "
                      f"(p={h['pvalue']:.4f}, R²={h['r2_adj']:.4f})")
        
        return {
            'model': model_final,
            'features': features_restantes,
            'history': historique,
            'vif': vif_data,
            'summary': model_final.summary()
        }


    def interpreter_modele(self, resultats: dict, y_name: str = "prix"):
        """
        Interprète les coefficients du modèle final de manière pédagogique.
        
        Paramètres:
        -----------
        resultats : dict
            Résultat de la fonction backwards_elimination()
        y_name : str
            Nom de la variable cible
        """
        model = resultats['model']
        
        print("\n" + "="*80)
        print("🎓 INTERPRÉTATION DU MODÈLE (Toutes choses égales par ailleurs)")
        print("="*80)
        
        # Équation mathématique
        equation = f"{y_name} = "
        termes = []
        for feat, coef in model.params.items():
            signe = "+" if coef >= 0 else "-"
            termes.append(f"{signe} {abs(coef):.2f} × {feat}")
        equation += " ".join(termes)
        
        print(f"\n📐 ÉQUATION DU MODÈLE :")
        print(f"   {equation}")
        
        print(f"\n{'─'*80}")
        print("💡 INTERPRÉTATION PRATIQUE :")
        print(f"{'─'*80}")
        
        for feat, coef in model.params.items():
            
            # Interprétations contextualisées
            if "surface" in feat.lower():
                unite = "m²"
                exemple = 10
                impact = coef * exemple
                print(f"\n  📏 {feat} :")
                print(f"     • +1 {unite} → {y_name} change de {coef:+,.2f}€")
                print(f"     • +{exemple} {unite} → {y_name} change de {impact:+,.2f}€")
                
            elif "piece" in feat.lower() or "room" in feat.lower():
                print(f"\n  🚪 {feat} :")
                print(f"     • +1 pièce → {y_name} change de {coef:+,.2f}€")
                if coef < 0:
                    print(f"     ⚠️  Coefficient négatif : possiblement dû à la multicolinéarité")
                    print(f"         avec la surface (+ de pièces = - de m² par pièce)")
                
            elif "annee" in feat.lower() or "year" in feat.lower():
                print(f"\n  📅 {feat} :")
                print(f"     • +1 an → {y_name} change de {coef:+,.2f}€")
                if abs(coef) < 100:
                    print(f"     💡 Impact très faible : l'année a peu d'influence")
                
            else:
                print(f"\n  🔹 {feat} :")
                print(f"     • +1 unité → {y_name} change de {coef:+,.2f}€")
        
        print(f"\n{'='*80}")
        print("⚠️  RAPPEL : Ces interprétations sont valables 'toutes choses égales par ailleurs'")
        print("    On ne peut raisonner que sur UNE feature à la fois !")
        print("="*80 + "\n")
    
    # ================================================================
    # MÉTODOS DE ENTRENAMIENTO
    # ================================================================
    
    def entrenar_modelo(
        self,
        model,
        model_name: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena un modelo y guarda resultados en el historial.
        
        Parámetros:
        -----------
        model : estimador sklearn
        model_name : str, nombre identificador
        verbose : bool, mostrar logs
        
        Returns:
        --------
        dict : resultados completos del experimento
        """
        self.experiment_counter += 1
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 EXPERIMENTO #{self.experiment_counter}: {model_name}")
            print(f"{'='*70}")
        
        # 1. Validación Cruzada
        cv_results = self._cross_validate(model, verbose)
        
        # 2. Entrenar en todo el train
        trained_model, train_time = self._fit_model(model, verbose)
        
        # 3. Predecir
        y_train_pred = self._predict(trained_model, self.X_train, "Train", verbose)
        y_test_pred = self._predict(trained_model, self.X_test, "Test", verbose)
        
        # 4. Evaluar
        train_scores = self._evaluate(self.y_train, y_train_pred, "Train", verbose)
        test_scores = self._evaluate(self.y_test, y_test_pred, "Test", verbose)
        
        # 5. Detectar overfitting
        overfitting, diagnostics = self._detect_overfitting(
            train_scores, test_scores, verbose
        )
        
        # 6. Resumen
        if verbose:
            self._print_summary(train_scores, test_scores)
        
        # 7. Guardar resultados
        results = {
            'experiment_id': self.experiment_counter,
            'model_name': model_name,
            'model': trained_model,
            'cv_scores': cv_results,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_time': train_time,
            'overfitting': overfitting,
            'diagnostics': diagnostics,
            'predictions': {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            },
            'timestamp': pd.Timestamp.now()
        }
        
        self.results_history[model_name] = results
        
        # Actualizar mejor modelo
        self._update_best_model()
        
        if verbose:
            print(f"{'='*70}\n")
        
        return results
    
    def _cross_validate(self, model, verbose: bool) -> Dict[str, Dict]:
        """Ejecuta validación cruzada."""
        if verbose:
            print(f"\n🔄 Validación Cruzada ({self.config['CV_FOLDS']} folds)...")
        
        cv_results = cross_validate(
            estimator=model,
            X=self.X_train,
            y=self.y_train,
            cv=self.config['CV_FOLDS'],
            scoring=self.config['METRICS'],
            return_train_score=True,
            n_jobs=-1
        )
        
        # Procesar scores
        cv_scores = {}
        for metric_name in self.config['METRICS'].keys():
            train_scores = cv_results[f'train_{metric_name}']
            test_scores = cv_results[f'test_{metric_name}']
            
            # Invertir negativos
            if metric_name in ['mae', 'rmse', 'mape']:
                train_scores = -train_scores
                test_scores = -test_scores
            
            cv_scores[metric_name] = {
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'cv_mean': test_scores.mean(),
                'cv_std': test_scores.std()
            }
        
        if verbose:
            print(f"   Resultados:")
            for metric, scores in cv_scores.items():
                print(f"   {metric.upper():6s} → "
                      f"Train: {scores['train_mean']:.4f} (±{scores['train_std']:.4f}) | "
                      f"CV: {scores['cv_mean']:.4f} (±{scores['cv_std']:.4f})")
        
        return cv_scores
    
    def _fit_model(self, model, verbose: bool) -> Tuple[Any, float]:
        """Entrena el modelo en todo el train set."""
        if verbose:
            print(f"\n🏋️  Entrenando en {self.X_train.shape[0]} samples...")
        
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        if verbose:
            print(f"   ✅ Completado en {train_time:.2f}s")
        
        return model, train_time
    
    def _predict(
        self, 
        model, 
        X: pd.DataFrame, 
        dataset_name: str, 
        verbose: bool
    ) -> np.ndarray:
        """Realiza predicciones."""
        if verbose:
            print(f"\n🔮 Predicción ({dataset_name})...")
        
        predictions = model.predict(X)
        
        if verbose:
            print(f"   Media: {predictions.mean():.2f} | "
                  f"Std: {predictions.std():.2f} | "
                  f"Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        return predictions
    
    def _evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        dataset_name: str,
        verbose: bool
    ) -> Dict[str, float]:
        """Calcula métricas."""
        scores = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return scores
    
    def _detect_overfitting(
        self,
        train_scores: Dict,
        test_scores: Dict,
        verbose: bool
    ) -> Tuple[bool, Dict]:
        """Detecta overfitting."""
        overfitting = False
        diagnostics = {}
        
        # R² gap
        r2_gap = train_scores['r2'] - test_scores['r2']
        diagnostics['r2_gap'] = r2_gap
        if r2_gap > 0.1:
            overfitting = True
        
        # MAE ratio
        mae_ratio = test_scores['mae'] / train_scores['mae'] if train_scores['mae'] > 0 else 1
        diagnostics['mae_ratio'] = mae_ratio
        if mae_ratio > 1.5:
            overfitting = True
        
        # RMSE ratio
        rmse_ratio = test_scores['rmse'] / train_scores['rmse'] if train_scores['rmse'] > 0 else 1
        diagnostics['rmse_ratio'] = rmse_ratio
        if rmse_ratio > 1.5:
            overfitting = True
        
        diagnostics['overfitting'] = overfitting
        
        if verbose:
            if overfitting:
                print(f"\n⚠️  OVERFITTING DETECTADO:")
                if r2_gap > 0.1:
                    print(f"   • R² gap: {r2_gap:.3f}")
                if mae_ratio > 1.5:
                    print(f"   • MAE ratio: {mae_ratio:.2f}x")
                if rmse_ratio > 1.5:
                    print(f"   • RMSE ratio: {rmse_ratio:.2f}x")
            else:
                print(f"\n✅ Sin overfitting significativo")
        
        return overfitting, diagnostics
    
    def _print_summary(self, train_scores: Dict, test_scores: Dict):
        """Imprime resumen comparativo."""
        print(f"\n{'─'*60}")
        print(f"📊 RESUMEN")
        print(f"{'─'*60}")
        print(f"{'Métrica':10s} | {'Train':>10s} | {'Test':>10s} | {'Δ':>10s}")
        print(f"{'-'*50}")
        
        for metric in ['r2', 'mae', 'rmse', 'mape']:
            train_val = train_scores[metric]
            test_val = test_scores[metric]
            diff = train_val - test_val
            print(f"{metric.upper():10s} | {train_val:10.4f} | {test_val:10.4f} | {diff:+10.4f}")
    
    def _update_best_model(self):
        """Actualiza el mejor modelo basado en R² test."""
        if not self.results_history:
            return
        
        best = max(
            self.results_history.items(),
            key=lambda x: x[1]['test_scores']['r2']
        )
        self._best_model_name = best[0]
    
    # ================================================================
    # MÉTODOS DE COMPARACIÓN Y ANÁLISIS
    # ================================================================
    
    def comparar_modelos(self, sort_by: str = 'r2') -> pd.DataFrame:
        """
        Genera tabla comparativa de todos los modelos entrenados.
        
        Parámetros:
        -----------
        sort_by : str, métrica para ordenar ('r2', 'mae', 'rmse', 'mape')
        
        Returns:
        --------
        pd.DataFrame : tabla comparativa
        """
        if not self.results_history:
            print("⚠️  No hay modelos entrenados todavía.")
            return pd.DataFrame()
        
        rows = []
        for model_name, results in self.results_history.items():
            row = {
                'Modelo': model_name,
                'R² (CV)': results['cv_scores']['r2']['cv_mean'],
                'R² (Test)': results['test_scores']['r2'],
                'MAE (Test)': results['test_scores']['mae'],
                'RMSE (Test)': results['test_scores']['rmse'],
                'MAPE (%)': results['test_scores']['mape'],
                'Tiempo (s)': results['train_time'],
                'Overfitting': '⚠️' if results['overfitting'] else '✅',
                'Experimento': results['experiment_id']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ordenar
        ascending = True if sort_by in ['mae', 'rmse', 'mape'] else False
        sort_col = {
            'r2': 'R² (Test)',
            'mae': 'MAE (Test)',
            'rmse': 'RMSE (Test)',
            'mape': 'MAPE (%)'
        }.get(sort_by, 'R² (Test)')
        
        df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        
        # Marcar el mejor
        df['Mejor'] = ''
        df.loc[0, 'Mejor'] = '🏆'
        
        print(f"\n{'='*80}")
        print(f"📊 COMPARACIÓN DE MODELOS (ordenado por {sort_by.upper()})")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*80}")
        print(f"🏆 Mejor modelo: {df.loc[0, 'Modelo']} "
              f"(R² = {df.loc[0, 'R² (Test)']:.4f})")
        print(f"{'='*80}\n")
        
        return df
    
    def get_mejor_modelo(self) -> Tuple[str, Any, Dict]:
        """
        Obtiene el mejor modelo entrenado.
        
        Returns:
        --------
        tuple : (nombre, modelo, resultados)
        """
        if not self._best_model_name:
            raise ValueError("No hay modelos entrenados.")
        
        results = self.results_history[self._best_model_name]
        return (
            self._best_model_name,
            results['model'],
            results
        )
    
    def get_modelo(self, model_name: str) -> Tuple[Any, Dict]:
        """
        Obtiene un modelo específico por nombre.
        
        Returns:
        --------
        tuple : (modelo, resultados)
        """
        if model_name not in self.results_history:
            raise KeyError(f"Modelo '{model_name}' no encontrado. "
                          f"Disponibles: {list(self.results_history.keys())}")
        
        results = self.results_history[model_name]
        return results['model'], results
    
    # ================================================================
    # MÉTODOS DE VISUALIZACIÓN
    # ================================================================
    
    def visualizar_modelo(
        self, 
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Visualiza predicciones y residuos de un modelo.
        
        Parámetros:
        -----------
        model_name : str, opcional. Si None, usa el mejor modelo
        figsize : tuple
        """
        if model_name is None:
            model_name = self._best_model_name
        
        if model_name not in self.results_history:
            raise KeyError(f"Modelo '{model_name}' no encontrado.")
        
        results = self.results_history[model_name]
        y_true = self.y_test
        y_pred = results['predictions']['y_test_pred']
        scores = results['test_scores']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Scatter: Predicciones vs Reales
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Predicción perfecta')
        
        ax.set_xlabel('Valores Reales', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicciones', fontsize=12, fontweight='bold')
        ax.set_title(f'Predicciones vs Reales\n{model_name}', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        
        metrics_text = (f"R² = {scores['r2']:.4f}\n"
                       f"MAE = {scores['mae']:.2f}\n"
                       f"RMSE = {scores['rmse']:.2f}")
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes,
                fontsize=10, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Histograma de Residuos
        ax = axes[1]
        residuos = y_true - y_pred
        
        ax.hist(residuos, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                   label='Residuo = 0', alpha=0.8)
        
        ax.set_xlabel('Residuos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribución de Residuos\n{model_name}', 
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        mean_res = residuos.mean()
        std_res = residuos.std()
        stats_text = f'μ = {mean_res:.2f}\nσ = {std_res:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_mejor_modelo(self):
        """Visualiza el mejor modelo."""
        self.visualizar_modelo(model_name=None)
    
    def visualizar_comparacion(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Visualiza comparación de todos los modelos.
        
        Parámetros:
        -----------
        figsize : tuple
        """
        if len(self.results_history) < 2:
            print("⚠️  Se necesitan al menos 2 modelos para comparar.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        model_names = list(self.results_history.keys())
        n_models = len(model_names)
        
        # 1. R² CV vs Test
        ax = axes[0, 0]
        x = np.arange(n_models)
        width = 0.35
        
        r2_cv = [self.results_history[m]['cv_scores']['r2']['cv_mean'] for m in model_names]
        r2_test = [self.results_history[m]['test_scores']['r2'] for m in model_names]
        
        ax.bar(x - width/2, r2_cv, width, label='R² CV', alpha=0.8, 
               color='skyblue', edgecolor='black')
        ax.bar(x + width/2, r2_test, width, label='R² Test', alpha=0.8, 
               color='coral', edgecolor='black')
        
        ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('R²: CV vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. MAE Comparison
        ax = axes[0, 1]
        mae_test = [self.results_history[m]['test_scores']['mae'] for m in model_names]
        colors = ['green' if mae == min(mae_test) else 'steelblue' for mae in mae_test]
        
        bars = ax.barh(model_names, mae_test, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('MAE (Test)', fontsize=12, fontweight='bold')
        ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, val) in enumerate(zip(bars, mae_test)):
            ax.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        # 3. Tiempo de entrenamiento
        ax = axes[1, 0]
        train_times = [self.results_history[m]['train_time'] for m in model_names]
        
        bars = ax.bar(model_names, train_times, alpha=0.8, 
                      color='mediumpurple', edgecolor='black')
        ax.set_ylabel('Tiempo (s)', fontsize=12, fontweight='bold')
        ax.set_title('Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, train_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
        
        # 4. Overfitting indicators
        ax = axes[1, 1]
        ax.axis('off')
        
        best_name = self._best_model_name
        best_r2 = self.results_history[best_name]['test_scores']['r2']
        best_mae = self.results_history[best_name]['test_scores']['mae']
        
        summary_text = f"""
        🏆 MEJOR MODELO
        
        Nombre: {best_name}
        R²: {best_r2:.4f}
        MAE: {best_mae:.2f}
        
        Total modelos: {n_models}
        Experimentos: {self.experiment_counter}
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_cv_curves(
        self, 
        metric: str = 'r2',
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Visualiza curvas de validación cruzada.
        
        Parámetros:
        -----------
        metric : str, métrica a visualizar
        figsize : tuple
        """
        if not self.results_history:
            print("⚠️  No hay modelos entrenados.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        model_names = list(self.results_history.keys())
        x = np.arange(len(model_names))
        
        train_means = []
        train_stds = []
        cv_means = []
        cv_stds = []
        
        for name in model_names:
            cv_scores = self.results_history[name]['cv_scores'][metric]
            train_means.append(cv_scores['train_mean'])
            train_stds.append(cv_scores['train_std'])
            cv_means.append(cv_scores['cv_mean'])
            cv_stds.append(cv_scores['cv_std'])
        
        ax.errorbar(x - 0.2, train_means, yerr=train_stds, fmt='o-', 
                    label='Train', linewidth=2, markersize=8, capsize=5)
        ax.errorbar(x + 0.2, cv_means, yerr=cv_stds, fmt='s-', 
                    label='CV', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Validación Cruzada: {metric.upper()}', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()
    
    # ================================================================
    # MÉTODOS UTILITARIOS
    # ================================================================
    
    def resumen(self):
        """Imprime resumen del estado actual."""
        print(f"\n{'='*70}")
        print("📋 RESUMEN DEL MODELER")
        print(f"{'='*70}")
        print(f"Modelos entrenados    : {len(self.results_history)}")
        print(f"Experimentos totales  : {self.experiment_counter}")
        
        if self._best_model_name:
            best_r2 = self.results_history[self._best_model_name]['test_scores']['r2']
            print(f"Mejor modelo          : {self._best_model_name} (R² = {best_r2:.4f})")
        
        print(f"{'='*70}\n")
    
    def limpiar_historial(self):
        """Limpia el historial de experimentos."""
        self.results_history = {}
        self.experiment_counter = 0
        self._best_model_name = None
        print("✅ Historial limpiado.")
    
    def exportar_resultados(self, filepath: str = 'model_results.csv'):
        """
        Exporta tabla comparativa a CSV.
        
        Parámetros:
        -----------
        filepath : str, ruta del archivo
        """
        df = self.comparar_modelos()
        df.to_csv(filepath, index=False)
        print(f"✅ Resultados exportados a: {filepath}")
    
    def __repr__(self):
        return (f"Modeler(modelos={len(self.results_history)}, "
                f"experimentos={self.experiment_counter}, "
                f"mejor={self._best_model_name})")


# ============================================================
# 🎯 EJEMPLO DE USO
# ============================================================

## Inicializar
#modeler = Modeler(X_train_global, y_train_global, 


