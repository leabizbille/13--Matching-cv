# 📄 Matching CV - Projet de comparaison de CV et fiches de poste

Ce projet Python permet de comparer des **CV** et des **fiches de poste** afin de calculer différentes mesures de similarité (**cosinus, distance euclidienne, similarité de Jaccard**) entre les textes extraits. Il offre une solution complète d’analyse textuelle pour l’aide au recrutement. Le projet propose l’extraction de texte à partir de fichiers **PDF**, le chargement de fichiers texte encodés en **UTF-8** ou **ISO-8859-1**, le calcul des similarités via **TF-IDF, CountVectorizer, HashingVectorizer et Jaccard**, des tests unitaires automatisés pour garantir la fiabilité, et une intégration continue (CI) avec **GitHub Actions** qui exécute les tests à chaque push.

---

## 🚀 Fonctionnalités principales

* Extraction de texte depuis **PDF** et **TXT**
* Vectorisation des textes avec **TF-IDF**, **CountVectorizer**, **HashingVectorizer**
* Calcul de similarités : **Cosine Similarity**, **Euclidean Distance**, **Jaccard Similarity**
* Interface interactive via **Streamlit**
* Utilisation des **stopwords français (NLTK)**
* Tests unitaires automatisés avec **pytest**
* Intégration Continue (CI) avec GitHub Actions

---

## 📦 Installation

Cloner le dépôt :

```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
```

Créer et activer un environnement virtuel :
**Windows PowerShell** :

```bash
python -m venv myenv
.\myenv\Scripts\activate
```

**Linux / macOS** :

```bash
python -m venv myenv
source myenv/bin/activate
```

Installer les dépendances :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Télécharger les stopwords de NLTK :

```python
import nltk
nltk.download('stopwords')
```

Vérifier l’environnement :

```bash
python --version
# Windows
where python
# Linux / macOS
which python
```

---

## 🧪 Tests unitaires

Lancer les tests avec :

```bash
python -m pytest
```

---

## ⚙️ Intégration Continue (CI)

Un workflow GitHub Actions est inclus. À chaque **push sur `main`** :

* Installation de Python et des dépendances
* Exécution des tests unitaires
* Validation automatique de la qualité du code

---

## 🎨 Interface Streamlit

L’application permet :

* Dépôt de plusieurs **CV (PDF ou TXT)**
* Dépôt d’une **fiche de poste**
* Choix du **vectorizer** : TF-IDF, Count, Hashing
* Choix de la **méthode de comparaison** : Cosinus, Euclidienne, Jaccard
* Résultats affichés dans un tableau récapitulatif

Lancer l’application avec :

```bash
streamlit run app.py
```

---

## 📊 Exemple d’interprétation

* **Cosine Similarity** → proche de 1 = documents très similaires
* **Euclidean Distance** → plus la valeur est faible, plus les documents sont proches
* **Jaccard Similarity** → valeur entre 0 (aucune similarité) et 1 (identiques)

---

## 📚 Dépendances principales

* [Streamlit](https://streamlit.io/) – Interface utilisateur
* [PyPDF2](https://pypi.org/project/pypdf2/) – Extraction de texte PDF
* [scikit-learn](https://scikit-learn.org/stable/) – Vectorisation & mesures de similarité
* [Pandas](https://pandas.pydata.org/) – Manipulation de données
* [NLTK](https://www.nltk.org/) – Stopwords français
* [Pytest](https://docs.pytest.org/) – Tests unitaires

---

## 👩‍💻 LB

✨ N’hésitez pas à contribuer, suggérer des idées ou ouvrir des issues !

