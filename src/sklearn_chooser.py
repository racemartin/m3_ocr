"""
Scikit-learn Algorithm Chooser
Basé sur le flowchart officiel de scikit-learn
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Types de tâches ML"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


@dataclass
class ModelRecommendation:
    """Recommandation de modèle"""
    name: str
    sklearn_class: str
    description: str
    parameters: Dict[str, str]
    notes: List[str]


class ScikitLearnChooser:
    """Choisisseur d'algorithme scikit-learn selon le flowchart"""
    
    def __init__(self):
        self.history = []
    
    def choose_algorithm(
        self,
        n_samples: int,
        task: str,
        has_labeled_data: bool = True,
        predicting_category: bool = True,
        predicting_quantity: bool = False,
        n_features: int = None,
        text_data: bool = False,
        need_structure: bool = False,
        tough_luck: bool = False
    ) -> ModelRecommendation:
        """
        Choisit l'algorithme approprié selon le flowchart scikit-learn.
        
        Args:
            n_samples: Nombre d'échantillons
            task: Type de tâche ('classification', 'regression', 'clustering', 'dimensionality_reduction')
            has_labeled_data: Les données sont-elles étiquetées?
            predicting_category: Prédire une catégorie?
            predicting_quantity: Prédire une quantité?
            n_features: Nombre de features
            text_data: Données textuelles?
            need_structure: Besoin de structure?
            tough_luck: Problème difficile?
        
        Returns:
            ModelRecommendation: Recommandation de modèle
        """
        
        # START: Obtenir les données étiquetées
        if not has_labeled_data:
            return self._choose_unsupervised(n_samples, n_features, need_structure)
        
        # Données étiquetées
        if n_samples < 50:
            return ModelRecommendation(
                name="Get more data",
                sklearn_class="N/A",
                description="Vous avez besoin de plus de données (< 50 échantillons)",
                parameters={},
                notes=["Collectez plus d'échantillons avant d'entraîner un modèle"]
            )
        
        # Classification ou Régression?
        if predicting_category:
            return self._choose_classification(n_samples, text_data, tough_luck)
        
        if predicting_quantity:
            return self._choose_regression(n_samples, n_features)
        
        # Par défaut, demander plus d'informations
        return ModelRecommendation(
            name="Need more information",
            sklearn_class="N/A",
            description="Précisez si vous prédisez une catégorie ou une quantité",
            parameters={},
            notes=["Définissez clairement votre problème"]
        )
    
    def _choose_classification(
        self, 
        n_samples: int, 
        text_data: bool,
        tough_luck: bool
    ) -> ModelRecommendation:
        """Choisit un algorithme de classification"""
        
        if n_samples < 100_000:
            # < 100K samples
            if text_data:
                return ModelRecommendation(
                    name="Naive Bayes",
                    sklearn_class="sklearn.naive_bayes.MultinomialNB",
                    description="Idéal pour la classification de texte",
                    parameters={
                        "alpha": "1.0 (smoothing parameter)"
                    },
                    notes=[
                        "Très rapide",
                        "Fonctionne bien avec peu de données",
                        "Bon pour le text mining"
                    ]
                )
            
            # Essayer Linear SVC
            return ModelRecommendation(
                name="Linear SVC",
                sklearn_class="sklearn.svm.LinearSVC",
                description="Support Vector Classification avec noyau linéaire",
                parameters={
                    "C": "1.0 (regularization)",
                    "max_iter": "1000"
                },
                notes=[
                    "Efficace pour données linéairement séparables",
                    "Rapide sur datasets moyens",
                    "Si ça ne marche pas, essayez KNeighborsClassifier ou SVC"
                ]
            )
        
        else:
            # >= 100K samples
            return ModelRecommendation(
                name="SGD Classifier",
                sklearn_class="sklearn.linear_model.SGDClassifier",
                description="Classificateur avec descente de gradient stochastique",
                parameters={
                    "loss": "'hinge' or 'log_loss'",
                    "alpha": "0.0001 (regularization)",
                    "max_iter": "1000"
                },
                notes=[
                    "Très efficace sur grands datasets",
                    "Scalable",
                    "Supporte l'apprentissage incrémental"
                ]
            )
    
    def _choose_regression(
        self, 
        n_samples: int,
        n_features: int = None
    ) -> ModelRecommendation:
        """Choisit un algorithme de régression"""
        
        if n_samples < 100_000:
            # < 100K samples
            if n_features and n_features > 100:
                # Few features should be important
                return ModelRecommendation(
                    name="Lasso (L1) / ElasticNet",
                    sklearn_class="sklearn.linear_model.Lasso",
                    description="Régression linéaire avec régularisation L1",
                    parameters={
                        "alpha": "1.0 (regularization strength)"
                    },
                    notes=[
                        "Effectue une sélection de features automatique",
                        "Met certains coefficients à zéro",
                        "ElasticNet combine L1 et L2"
                    ]
                )
            else:
                # Regular regression
                return ModelRecommendation(
                    name="Ridge Regression (L2)",
                    sklearn_class="sklearn.linear_model.Ridge",
                    description="Régression linéaire avec régularisation L2",
                    parameters={
                        "alpha": "1.0 (regularization strength)"
                    },
                    notes=[
                        "Bonne pour la plupart des problèmes",
                        "Réduit l'overfitting",
                        "Plus stable que Lasso"
                    ]
                )
        else:
            # >= 100K samples
            return ModelRecommendation(
                name="SGD Regressor",
                sklearn_class="sklearn.linear_model.SGDRegressor",
                description="Régression avec descente de gradient stochastique",
                parameters={
                    "loss": "'squared_error'",
                    "alpha": "0.0001",
                    "max_iter": "1000"
                },
                notes=[
                    "Très rapide sur gros datasets",
                    "Scalable",
                    "Supporte l'apprentissage en ligne"
                ]
            )
    
    def _choose_unsupervised(
        self,
        n_samples: int,
        n_features: int = None,
        need_structure: bool = False
    ) -> ModelRecommendation:
        """Choisit un algorithme non supervisé"""
        
        # Clustering ou Dimensionality Reduction?
        if need_structure:
            # Dimensionality Reduction
            if n_samples < 10_000:
                return ModelRecommendation(
                    name="IsoMap",
                    sklearn_class="sklearn.manifold.Isomap",
                    description="Isometric Mapping pour réduction de dimensionnalité",
                    parameters={
                        "n_components": "2 or 3",
                        "n_neighbors": "5"
                    },
                    notes=[
                        "Préserve les distances géodésiques",
                        "Bon pour la visualisation",
                        "Peut être lent sur gros datasets"
                    ]
                )
            else:
                return ModelRecommendation(
                    name="Kernel Approximation + LLE",
                    sklearn_class="sklearn.decomposition.KernelPCA",
                    description="Approximation de noyau avec Locally Linear Embedding",
                    parameters={
                        "n_components": "depends on use case",
                        "kernel": "'rbf' or 'poly'"
                    },
                    notes=[
                        "Scalable",
                        "Préserve la structure locale",
                        "Spectral Embedding ou LLE sont aussi possibles"
                    ]
                )
        
        else:
            # Clustering
            if n_samples < 10_000:
                # Small dataset
                return ModelRecommendation(
                    name="KMeans",
                    sklearn_class="sklearn.cluster.KMeans",
                    description="Clustering par K-moyennes",
                    parameters={
                        "n_clusters": "must be specified",
                        "init": "'k-means++'",
                        "n_init": "10"
                    },
                    notes=[
                        "Simple et efficace",
                        "Nécessite de connaître K à l'avance",
                        "Sensible aux outliers"
                    ]
                )
            else:
                # Large dataset
                return ModelRecommendation(
                    name="MiniBatch KMeans",
                    sklearn_class="sklearn.cluster.MiniBatchKMeans",
                    description="KMeans avec mini-batches pour grands datasets",
                    parameters={
                        "n_clusters": "must be specified",
                        "batch_size": "100"
                    },
                    notes=[
                        "Plus rapide que KMeans classique",
                        "Scalable",
                        "Léger compromis sur la qualité"
                    ]
                )
    
    def interactive_chooser(self):
        """Mode interactif pour choisir un algorithme"""
        print("=" * 60)
        print("🤖 SCIKIT-LEARN ALGORITHM CHOOSER")
        print("=" * 60)
        print()
        
        # Question 1: Nombre d'échantillons
        while True:
            try:
                n_samples = int(input("📊 Combien d'échantillons avez-vous? "))
                if n_samples < 0:
                    print("❌ Le nombre doit être positif!")
                    continue
                break
            except ValueError:
                print("❌ Entrez un nombre valide!")
        
        if n_samples < 50:
            print("\n⚠️  Vous avez besoin de plus de données (< 50 échantillons)")
            print("💡 Collectez plus d'échantillons avant d'entraîner un modèle")
            return
        
        print()
        
        # Question 2: Données étiquetées?
        labeled = input("🏷️  Avez-vous des données étiquetées? (oui/non): ").lower().strip()
        has_labeled_data = labeled in ['oui', 'o', 'yes', 'y']
        
        if not has_labeled_data:
            print()
            structure = input("🔍 Cherchez-vous une structure dans les données? (oui/non): ").lower().strip()
            need_structure = structure in ['oui', 'o', 'yes', 'y']
            
            recommendation = self._choose_unsupervised(n_samples, need_structure=need_structure)
            self._display_recommendation(recommendation)
            return
        
        print()
        
        # Question 3: Classification ou Régression?
        print("🎯 Que voulez-vous prédire?")
        print("  1. Une catégorie (classification)")
        print("  2. Une quantité (régression)")
        
        choice = input("Votre choix (1/2): ").strip()
        
        if choice == "1":
            # Classification
            print()
            text = input("📝 Travaillez-vous avec des données textuelles? (oui/non): ").lower().strip()
            text_data = text in ['oui', 'o', 'yes', 'y']
            
            recommendation = self._choose_classification(n_samples, text_data, False)
            
        elif choice == "2":
            # Régression
            print()
            try:
                n_features_str = input("📈 Combien de features? (appuyez sur Entrée si inconnu): ").strip()
                n_features = int(n_features_str) if n_features_str else None
            except ValueError:
                n_features = None
            
            recommendation = self._choose_regression(n_samples, n_features)
        
        else:
            print("❌ Choix invalide!")
            return
        
        self._display_recommendation(recommendation)
    
    def _display_recommendation(self, rec: ModelRecommendation):
        """Affiche la recommandation de manière lisible"""
        print()
        print("=" * 60)
        print("✅ RECOMMANDATION")
        print("=" * 60)
        print()
        print(f"🎯 Algorithme recommandé: {rec.name}")
        print(f"📦 Classe scikit-learn: {rec.sklearn_class}")
        print()
        print(f"📝 Description:")
        print(f"   {rec.description}")
        print()
        
        if rec.parameters:
            print("⚙️  Paramètres principaux:")
            for param, value in rec.parameters.items():
                print(f"   • {param}: {value}")
            print()
        
        if rec.notes:
            print("💡 Notes importantes:")
            for note in rec.notes:
                print(f"   • {note}")
        
        print()
        print("=" * 60)
        
        # Code exemple
        if rec.sklearn_class != "N/A":
            print()
            print("📋 Exemple de code:")
            print("-" * 60)
            print(f"from {rec.sklearn_class.rsplit('.', 1)[0]} import {rec.name.replace(' ', '')}")
            print()
            print(f"# Créer le modèle")
            print(f"model = {rec.name.replace(' ', '')}()")
            print()
            print(f"# Entraîner")
            print(f"model.fit(X_train, y_train)")
            print()
            print(f"# Prédire")
            print(f"predictions = model.predict(X_test)")
            print("-" * 60)
        
        print()


def main():
    """Fonction principale"""
    chooser = ScikitLearnChooser()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║      SCIKIT-LEARN ALGORITHM CHOOSER                       ║
║      Basé sur le flowchart officiel                       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\nOptions:")
        print("  1. Mode interactif (recommandé)")
        print("  2. Exemples prédéfinis")
        print("  3. Quitter")
        print()
        
        choice = input("Votre choix (1-3): ").strip()
        
        if choice == "1":
            chooser.interactive_chooser()
            
        elif choice == "2":
            # Exemples
            print("\n" + "=" * 60)
            print("EXEMPLES")
            print("=" * 60)
            
            examples = [
                {
                    "name": "Classification de texte (spam)",
                    "params": {
                        "n_samples": 5000,
                        "task": "classification",
                        "has_labeled_data": True,
                        "predicting_category": True,
                        "text_data": True
                    }
                },
                {
                    "name": "Prédiction de prix immobiliers",
                    "params": {
                        "n_samples": 1500,
                        "task": "regression",
                        "has_labeled_data": True,
                        "predicting_quantity": True,
                        "n_features": 15
                    }
                },
                {
                    "name": "Segmentation de clients (clustering)",
                    "params": {
                        "n_samples": 8000,
                        "task": "clustering",
                        "has_labeled_data": False,
                        "need_structure": False
                    }
                },
                {
                    "name": "Classification d'images (grand dataset)",
                    "params": {
                        "n_samples": 150000,
                        "task": "classification",
                        "has_labeled_data": True,
                        "predicting_category": True,
                        "text_data": False
                    }
                }
            ]
            
            for i, example in enumerate(examples, 1):
                print(f"\n{i}. {example['name']}")
                rec = chooser.choose_algorithm(**example['params'])
                print(f"   → {rec.name} ({rec.sklearn_class})")
            
            print()
            
        elif choice == "3":
            print("\n👋 Au revoir!")
            break
            
        else:
            print("❌ Choix invalide!")


if __name__ == "__main__":
    main()