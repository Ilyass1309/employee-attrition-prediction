# Analyse Prédictive de l'Attrition des Employés — HumanForYou

Projet d'intelligence artificielle réalisé dans le cadre de la formation CESI, visant à analyser et prédire le taux de rotation du personnel (~15 %/an) de l'entreprise pharmaceutique **HumanForYou**, à partir de données RH de 2015.

## Objectifs

- Identifier les facteurs clés influençant le départ des employés
- Construire un modèle prédictif fiable malgré le déséquilibre des classes
- Fournir des recommandations actionnables pour réduire l'attrition
- Garantir une démarche éthique conforme aux 7 exigences de la Commission Européenne pour une IA digne de confiance

## Données

5 fichiers CSV situés dans `data/raw/` :

| Fichier | Description |
|---------|-------------|
| `general_data.csv` | Données démographiques et professionnelles (âge, salaire, poste, ancienneté...) |
| `employee_survey_data.csv` | Enquête employés (satisfaction environnement, job, équilibre vie pro/perso) |
| `manager_survey_data.csv` | Évaluations managers (performance, implication) |
| `in_time.csv` | Horaires d'arrivée quotidiens |
| `out_time.csv` | Horaires de départ quotidiens |

## Pipeline d'analyse

Le notebook principal [`attrition_analysis.ipynb`](attrition_analysis.ipynb) suit les étapes suivantes :

1. **Chargement et fusion** des 5 sources de données
2. **Nettoyage** — traitement des valeurs manquantes (imputation adaptée au type : enquête vs données factuelles), suppression des colonnes constantes (`Over18`, `EmployeeCount`, `StandardHours`), détection des outliers via IQR
3. **Feature Engineering** — calcul des heures travaillées à partir de `in_time`/`out_time`, analyse temporelle mensuelle
4. **Encodage** — label encoding (binaires), ordinal encoding (variables ordonnées), one-hot encoding (catégorielles nominales)
5. **EDA** — distribution de la cible, matrice de corrélation, top corrélations avec l'attrition, distributions par classe
6. **Modélisation** — 4 modèles comparés :
   - Random Forest (`class_weight='balanced'`)
   - Régression Logistique
   - XGBoost
   - Random Forest + SMOTE
7. **Optimisation** — RandomizedSearchCV (XGBoost) et GridSearchCV (Random Forest), optimisation sur le F1-Score
8. **Interprétabilité** — SHAP values pour expliquer les prédictions
9. **Analyse du seuil de décision** — ajustement du seuil selon le contexte métier (coût d'un faux négatif vs faux positif)
10. **Recommandations** — pistes concrètes et chiffrées pour la direction RH

### Choix méthodologiques clés

- **Standardisation après le split** train/test pour éviter toute fuite de données
- **Stratification** du split pour conserver les proportions de classes
- **Métriques adaptées** au déséquilibre : F1-score, AUC-ROC, rappel, précision (pas l'accuracy)
- **Comparaison SMOTE vs class_weight** pour le traitement du déséquilibre

## Livrables

| Livrable | Description |
|----------|-------------|
| [`attrition_analysis.ipynb`](attrition_analysis.ipynb) | Notebook complet (analyse, modélisation, recommandations) |
| [`livrables/livrable_ethique_humanforyou.docx`](livrables/livrable_ethique_humanforyou.docx) | Démarche éthique — respect des 7 exigences CE (autonomie humaine, robustesse, confidentialité, transparence, non-discrimination, bien-être sociétal, responsabilité) |

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### Dépendances principales

- `pandas`, `numpy`, `scipy` — manipulation de données
- `scikit-learn` — modélisation et évaluation
- `imbalanced-learn` — SMOTE
- `xgboost` — gradient boosting
- `shap` — interprétabilité des modèles
- `matplotlib`, `seaborn` — visualisation
- `missingno` — visualisation des données manquantes

> Python 3.13+ recommandé (compatibilité des dépendances).

## Structure du projet

```
employee-attrition-prediction/
├── data/
│   └── raw/                  # Données brutes CSV
├── notebooks/                # Notebooks exploratoires
├── livrables/                # Documents livrables (éthique)
├── attrition_analysis.ipynb  # Notebook principal
├── requirements.txt
└── README.md
```
