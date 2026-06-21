# Projet Fouille de Données

Projet académique réalisé dans le cadre du **Master GLSI — Semestre 2**, couvrant les principales techniques de fouille de données et d'apprentissage automatique.

---

## Contenu du projet

| Partie | Sujet | Notebook |
|---|---|---|
| **1** | Ensemble Learning & BoostinVg (AdaBoost) | `partie1_ensemble_boosting/partie1_ensemble_boosting.ipynb` |
| **2** | Identification et traitement des valeurs manquantes | `partie2_valeurs_manquantes/` |
| **3** | Validation appliquée au clustering hiérarchique | `partie3_clustering_validation/` |
| **4** | Tuning des hyperparamètres + validation croisée | `partie4_tuning_cv/partie4_tuning_cv.ipynb` |

---

## Dataset utilisé

**Breast Cancer Wisconsin** (disponible via `sklearn.datasets.load_breast_cancer`)
- 569 échantillons, 30 features numériques
- Tâche : classification binaire (tumeur maligne / bénigne)
- Utilisé dans les parties 1 et 4

---

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter scipy
```

## Lancer un notebook

```bash
jupyter notebook partie1_ensemble_boosting/partie1_ensemble_boosting.ipynb
```

---

## Partie 1 — Ensemble Learning & Boosting

**Théorie :** comparaison des trois familles d'Ensemble Learning (Bagging, Boosting, Stacking) et analyse du compromis biais-variance.

**Implémentation :**
- AdaBoost **from scratch** (DecisionStump + mise à jour des poids)
- Comparaison avec `sklearn.ensemble.AdaBoostClassifier`
- Visualisations : évolution de l'erreur, poids α des stumps, comparaison des méthodes

---

## Partie 4 — Tuning des hyperparamètres & Validation croisée

**Théorie :** distinction paramètres / hyperparamètres, problème de contamination du jeu de test, k-fold cross-validation, GridSearch vs RandomizedSearch.

**Implémentation :**
- K-fold cross-validation **from scratch**
- `GridSearchCV` sur un arbre de décision + heatmap des résultats
- `RandomizedSearchCV` sur Random Forest
- Validation curve et Learning curve pour diagnostiquer biais/variance
- Comparaison finale CV score vs Test score
