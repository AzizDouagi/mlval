# NeuralFinance SaaS ML Platform

Application Streamlit de démonstration ML pour la finance SaaS :
- entraînement de modèles (classification, segmentation, anomalies, régression, NLP),
- visualisation interactive multi-pages,
- persistance des modèles dans `models/`.

## Lancement local

### 1) Installer les dépendances
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Lancer l'application
```bash
streamlit run streamlit_app/app.py
```

## Déploiement Streamlit Community Cloud

### Paramètres de déploiement
- Repository: ce dépôt GitHub
- Branch: `main` (ou votre branche cible)
- Main file path: `streamlit_app/app.py`
- Python version: 3.10+ recommandé

### Fichiers requis
- `requirements.txt` à la racine (présent)
- Données CSV dans `data/invoices_ml (1).csv`
- Modèles pré-entraînés dans `models/` (optionnel: sinon entraînement via l'UI)

## Structure utile
- Application: `streamlit_app/app.py`
- Pages: `streamlit_app/pages/`
- Utilitaires: `streamlit_app/utils/`
- Modèles: `models/`
- Données: `data/invoices_ml (1).csv`
