# Airfoil & Rocket Nozzle SciML Simulator (GNN-PINN Hybrid)

Ce projet présente un simulateur aérodynamique intelligent capable de prédire les champs physiques complexes avec une précision quasi-CFD. Initialement focalisé sur l'aéronautique (ailes NACA), le projet évolue vers des applications spatiales critiques (tuyères supersoniques) en intégrant des principes de **Scientific Machine Learning (SciML)**.

Il utilise une architecture hybride combinant les **Graph Neural Networks (GNN)** et les principes des **Physics-Informed Neural Networks (PINN)** pour servir de modèle de substitution (*Surrogate Model*) temps réel sur matériel contraint (Edge AI).

## 🚀 Points Clés & Performance
- **Accélération Massive** : Prédiction en **~15ms** (vs ~120s pour OpenFOAM), soit un gain de vitesse de **x8000**.
- **Dataset Robuste** : Entraîné et validé sur **500 simulations RANS** générées automatiquement, couvrant une large variété de géométries NACA.
- **Haute Précision** : Erreur moyenne relative **< 2%** sur les champs de vitesse et de pression par rapport au solveur physique de référence.
- **Architecture Avancée** :
    - Graph Network basé sur **MeshGraphNet** (4 couches de message passing, 128 unités cachées).
    - **Smart Density Sampling** : Échantillonnage adaptatif (100% des points en couche limite, 10% en champ lointain) pour capturer la physique critique sans compromis.
    - **Hybrid Loss (PINN)** : La fonction de coût intègre des contraintes physiques (Conditions aux limites, No-Slip sur le mur, Équations de conservation).

## 📁 Structure du Projet
- `src/airfoil2D/` : Pipeline original pour les profils d'ailes NACA.
- `src/lavalNozzle/` : Nouveau pipeline pour les tuyères supersoniques (En cours).
- `airfoil_gnn_best.pt` : Poids du modèle entraîné.
- `normalizer_stats.pt` : Statistiques de normalisation pour l'inférence.

## 🛰️ Roadmap Spatiale (Objectifs 2026)
Le projet pivote actuellement vers des cas d'usage à haute valeur ajoutée pour l'ingénierie spatiale :

1.  **Pivot Supersonique** : Transition du régime incompressible (aile) vers le régime **compressible** (tuyère de Laval).
2.  **Capturation de Chocs (Shock Capturing)** : Entraînement sur solvers `rhoCentralFoam` pour prédire les diamants de Mach et les ondes de choc.
3.  **Physique Augmentée** : Intégration de la conservation de la masse et de l'énergie comme contraintes fortes dans le GNN.
4.  **Optimisation Multiobjectif** : Utilisation du modèle comme moteur pour l'optimisation de forme temps réel (Shape Optimization).

## 🛠️ Installation & Utilisation
1. **Pré-requis** : PyTorch, PyTorch Geometric, PyVista, Scikit-Learn, OpenFOAM.
2. **Génération Dataset** : `python src/airfoil2D/generate_dataset.py`
3. **Création des Graphes** : `python src/airfoil2D/extract_to_graphs.py`
4. **Entraînement** : `python src/airfoil2D/train.py`
5. **Validation** : `python src/airfoil2D/validate_gnn_vs_openfoam.py`

## 📊 Résultats (Version Aile 2D)
Le modèle a été validé par rapport à des simulations OpenFOAM (SimpleFoam) avec un écart < 2%.

![perso_v5_val_sim_0001_naca_1316](data/perso_v5_val_sim_0001_naca_1316.png)

---
*Projet réalisé pour démontrer la puissance du SciML appliqué à l'ingénierie spatiale sous contraintes matérielles.*
