# Poetry M3 - Prédisez la consommation d'énergie des bâtiments dans la ville de Seattle.

<div align="left">
  <img src="https://user.oc-static.com/upload/2024/09/11/17260684381511_Capture%20d%E2%80%99e%CC%81cran%202024-09-11%20a%CC%80%2017.22.25.png" width="200px">
</div>


## **Objetif**: Prédire les **émissions de CO2** et la **consommation totale d’énergie** de **bâtiments non destinés à l’habitation**

## 🛠️ Technologies

- Python 3.12+
- Poetry (gestion des dépendances)
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Lab

## 📦 Installation

https://github.com/racemartin/m3-ocr

```bash
# Cloner le dépôt
git clone https://github.com/racemartin/m3-ocr.git
cd m3-ocr


# Installer les dépendances avec Poetry
poetry install

# verifier que toutes les dependences sont presents
poetry run python .\check_deps.py

# Activer l'environnement virtuel
poetry shell
```

## 🚀 Utilisation

```bash
# Exécuter Jupyter Lab
poetry run jupyter lab

# Exécuter des scripts de test
poetry run python .\tests\test_python-dotenv.py  # WIN
poetry run python test/test_python-dotenv.py     # OSX/LINUX

```

### Notebooks Importants (Par ordre de implementation)

```bash                          
                                   1.Problem. 2.Anal.Exp 3.Feat.Enge. 4.Modelisation   5.Interpretation
                                   ---------- ---------- ------------ --------------   ----------------
STEP_1_Problematique               Prj.Info.  -          -            -                -
STEP_2_Analyse_Exploratoire_EDA    -          Use RAW    -            -                -

STEP_3_FEA_ENG_SET1 (Many Methods) -          -          RAW > SetV1° -                -
STEP_3_FEA_ENG_SET2 (Fit & Trans)  -          _          RAW > SetV2  -                -

STEP_4_MOD_SET1_CUSTOM             -          -          -            SetV1° > Model1  Model1 > Eval.& Opt.
STEP_4_MOD_SET2_CUSTOM             -          -          -            SetV2  > Model2  Model2 > Eval.& Opt.
STEP_4_MOD_SET2_PIPELINE           -          -          -            SetV2  > Model3  -

FULL_Sklearn_Pipeline              -          -          -            RAW    > Model4  -

Note: SetV1° contiens des Features Categorielles sans traiter.
```

# Exécuter des scripts
```bash
poetry run python src/main.py   # No pertinent. Code place dans le Notebooks.
```

## 📝 Structure du projet

```bash
.
$ tree -a -L1
.
├── .DS_Store
├── .env
├── .git
├── .gitignore
├── .idea
├── .ipynb_checkpoints
├── .venv
├── README.md
├── check_deps.py
├── data
├── doc
├── notebooks
├── poetry.lock
├── pyproject.toml
├── src
└── tests

```

## 📚 Dépendances principales

### Production
```bash
🔹 Socle Scientifique & Données
python (>=3.12)           : Interpréteur de base.
pandas (^2.3.3)           : Manipulation de données (DataFrames).
numpy (2.2.2)             : Calcul matriciel et numérique.
scipy (^1.16.3)           : Algorithmes mathématiques avancés.

🔹 Intelligence Artificielle & Statistiques
scikit-learn (^1.7.2)     : Cœur de votre pipeline (Modèles & Pipelines).
category-encoders         : Indispensable pour votre TargetEncoder (À ajouter).
statsmodels (^0.14.5)     : Analyse statistique.

🔹 Visualisation (Optionnel en pur déploiement)
matplotlib (<=3.10.0)     : Graphiques de base.
seaborn (^0.13.2)         : Graphiques statistiques.

🔹 Deep Learning & Vision (Spécifique Windows)
transformers              : Modèles NLP (Hugging Face).
huggingface-hub           : Accès aux modèles pré-entraînés.
datasets                  : Gestion des jeux de données complexes.
opencv-python-headless    : Traitement d image optimisé pour serveurs.

🔹 Utilitaires & Configuration
python-dotenv             : Chargement des clés et secrets.
tqdm                      : Suivi visuel des processus longs.
joblib                    : Pour le chargement (load) de votre fichier .pkl.
```
### Développement
```bash
🔹 Analyse Exploratoire & Data Profiling
ydata-profiling           : Génération automatique de rapports EDA.
polars                    : Alternative ultra-rapide à Pandas.
tabulate                  : Formatage élégant des tables dans la console.

🔹 Écosystème Jupyter (Interface & Interactivité)
jupyter / jupyterlab      : Votre environnement de travail interactif.
ipykernel                 : Le moteur d exécution Python pour Jupyter.
ipywidgets                : Menus et curseurs interactifs.
j-contrib-nbextensions    : Améliorations de productivité pour notebooks.

🔹 Qualité de Code & Standardisation
pre-commit                : Automatisation des vérifications Git.
black                     : Formateur de code strict (Style LeCun).
flake8                    : Analyseur de style et détection d erreurs.

🔹 Utilitaires d Expérimentation
requests                  : Requêtes HTTP pour APIs ou téléchargements.
pillow (PIL)              : Manipulation d images pour tests Vision.
```

## 👤 Auteur

**Rafael Cerezo Martín**
- Email: rafael.cerezo.martin@icloud.com
- GitHub: [@racemartin](https://github.com/racemartin)

## 📄 Licence

MIT License - voir le fichier [LICENSE](LICENSE) pour plus de détails.
