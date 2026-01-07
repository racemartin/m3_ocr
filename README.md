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

poetry run python .\tests\test_token_validity.py # WIN
poetry run python test/test_token_validity.py    # OSX/LINUX
```

# Exécuter des scripts
```bash
poetry run python src/main.py   # No pertinent. Code place dans le Notebooks.
```

## 📝 Structure du projet

```bash
.
$ tree -a -L3
.

```

## 📚 Dépendances principales

### Production
```bash

```
### Développement
```bash

```

## 👤 Auteur

**Rafael Cerezo Martín**
- Email: rafael.cerezo.martin@icloud.com
- GitHub: [@racemartin](https://github.com/racemartin)

## 📄 Licence

MIT License - voir le fichier [LICENSE](LICENSE) pour plus de détails.
