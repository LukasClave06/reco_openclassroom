# 📰 My Content – Système de recommandation d'articles

## 🎯 Objectif

Prototyper un MVP d'application de recommandation d'articles à destination des particuliers, en utilisant une approche **Content-Based Filtering**.


---

## 🧠 Fonctionnalités principales

> ✅ "En tant qu’utilisateur, je vais recevoir une sélection de cinq articles."

- Recommandations personnalisées via une Azure Function
- Interface web Flask pour tester les recommandations utilisateur
- Architecture adaptable pour accueillir de nouveaux utilisateurs et articles

---

## 🧱 Architecture du projet

systeme_reco/
│
├── app_flask/ # Interface web utilisateur (Flask)
│ ├── app.py
│ ├── templates/
│ └── static/
│
├── azure_function/ # Azure Function (serverless)
│ ├── function_app.py
│ ├── requirements.txt
│ ├── local.settings.json (local uniquement)
│ └── host.json
│
├── data/ # Données statiques (non versionnées)
│ ├── articles_metadata.csv
│ ├── clicks_sample.csv
│ └── articles_embeddings.pickle
│
├── notebooks/ # Analyses et prototypage
│ └── analyse_explo_et_modeles.ipynb
│
├── tests/ # À compléter si besoin
├── reco/ # Dossier technique si besoin d’extensions
├── requirements.txt # Dépendances principales
└── README.md


---

## 🛠 Lancer le projet en local

### 🔹 Prérequis
- Python 3.10
- Azure Functions Core Tools
- Environnement virtuel (recommandé)

### 🔹 1. Installer les dépendances
```bash
python -m venv reco
source reco/bin/activate  # ou .\reco\Scripts\activate sous Windows
pip install -r requirements.txt

2. Lancer l’API Flask

cd app_flask
python app.py

3. Lancer l’Azure Function (en local)

cd azure_function
func start

Test via URL : http://localhost:7071/api/recommendation?user_id=1

Exemple de réponse de l’Azure Function

[
  {"article_id": 38208, "similarity": 0.8156},
  {"article_id": 118369, "similarity": 0.8132},
  {"article_id": 118495, "similarity": 0.8115},
  {"article_id": 115695, "similarity": 0.8075},
  {"article_id": 107779, "similarity": 0.7997}
]

Modèle utilisé

Approche Content-Based Filtering avec :

- Embeddings d’articles (.pickle)
- Similarité cosinus
- Moyenne des embeddings des articles lus

Dépendances principales :

azure-functions
pandas
numpy
scikit-learn
flask

Architecture cible :

- Déploiement de l’Azure Function sur Azure (plan serverless gratuit)
- Prise en compte de nouveaux utilisateurs/articles avec mise à jour automatique de la matrice

Auteur :

Projet réalisé par Lukas Clave – CTO fictif de My Content