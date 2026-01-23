# Airfoil CFD Simulator : GNN & PINN Hybrid approach

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![OpenFOAM](https://img.shields.io/badge/CFD-OpenFOAM-005a9c.svg)

Ce projet présente un simulateur aérodynamique intelligent capable de prédire les champs de pression et de vitesse autour de profils d'ailes NACA 4-chiffres. Il utilise une architecture hybride combinant les **Graph Neural Networks (GNN)** et les principes des **Physics-Informed Neural Networks (PINN)**.

## 🚀 Points Clés
- **Vitesse de prédiction** : ~15ms (contre ~120s pour OpenFOAM, soit une accélération de x8000).
- **Architecture** : Graph Network basé sur MeshGraphNet (4 couches, 128 unités cachées).
- **Innovation** : 
    - **Smart Density Sampling** : Échantillonnage haute densité dans la couche limite (près du mur) pour capturer les gradients critiques.
    - **Hybrid Loss (PINN)** : La fonction de perte impose des contraintes physiques aux frontières (Inlet, No-Slip sur le mur).
- **Précision** : Erreur de vitesse moyenne < 0.4 m/s sur les cas de test.

## 📁 Structure du Projet
- `src/` : Code source complet (Génération de données, entraînement, validation).
- `airfoil_gnn_best.pt` : Poids du modèle entraîné (Version V5 ).
- `normalizer_stats.pt` : Statistiques de normalisation pour l'inférence.
- `journal/` : Historique du développement et de l'optimisation (V1 à V5).

## 🛠️ Installation & Utilisation
1. **Pré-requis** : PyTorch, PyTorch Geometric, PyVista, Scikit-Learn.
2. **Entraînement** : `python src/train.py`
3. **Validation** : `python src/validate_gnn_vs_openfoam.py`

## 📊 Résultats
Le modèle a été validé par rapport à des simulations OpenFOAM (SimpleFoam) et des données théoriques XFOIL.

### Courbe de Convergence
![Convergence](learning_curve_v5_perso.png)

### Comparaison des Champs de Vitesse
![perso_v5_val_sim_0001_naca_1316](perso_v5_val_sim_0001_naca_1316.png)

## 🧠 Méthodologie
1. **Génération** : Création automatique de maillages Gmsh et exécution de simulations OpenFOAM en parallèle.
2. **Graph Construction** : Conversion des maillages en graphes KNN (K-Nearest Neighbors) avec attributs géométriques relatifs.
3. **Entraînement** : Utilisation du scheduler `OneCycleLR` pour une convergence rapide et stable.

---
*Projet Personel réalisé dans le cadre d'une recherche sur l'accélération de la conception aéronautique par l'IA.*
