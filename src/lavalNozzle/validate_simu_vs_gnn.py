import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
from model import NozzleGNN
from generate_nozzle import generate_nozzle
from train import Normalizer
from dataset import NozzleDataset
from torch_geometric.loader import DataLoader
import pandas as pd
import re
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def validate(resolution=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # --- 1. CHARGEMENT DU MODÈLE ---
    model = NozzleGNN(input_dim=19, hidden_dim=64, output_dim_local=6, output_dim_global=4, num_layers=5).to(device)
    model.load_state_dict(torch.load("nozzle_gnn_best_v26.pt", map_location=device))
    model.eval()

    # --- 2. RESTAURATION DES NORMALISEURS ---
    # On charge les statistiques sauvegardées pendant l'entraînement
    if not os.path.exists("normalizer_stats_nozzle.pt"):
        print("Error: 'normalizer_stats_nozzle.pt' not found. You must run training first to generate it.")
        return
        
    stats = torch.load("normalizer_stats_nozzle.pt", map_location=device)
    
    # Restauration utilisant les moyennes/stds sauvegardées
    normalizer_x = Normalizer(mean=stats['x_mean'], std=stats['x_std'], device=device)
    normalizer_y = Normalizer(mean=stats['y_mean'], std=stats['y_std'], device=device)
    normalizer_edges = Normalizer(mean=stats['edge_mean'], std=stats['edge_std'], device=device)
    normalizer_bc = Normalizer(mean=stats['bc_mean'], std=stats['bc_std'], device=device)
    normalizer_case = Normalizer(mean=stats['case_mean'], std=stats['case_std'], device=device)
    normalizer_global = Normalizer(mean=stats['global_mean'], std=stats['global_std'], device=device)

    
    # --- 3. CHOIX DES CAS DE TEST ---

    test_cases = [
        {"path": "data/graphs/nozzle/sim_0001_T143_E453_I225.pt"},
        {"path": "data/graphs/nozzle/sim_0002_T139_E282_I207.pt"}
    ]
    
    for case in test_cases:
        graph_path = case["path"]
        if not os.path.exists(graph_path): continue
            
        data = torch.load(graph_path, weights_only=False)
        data.y = torch.cat([data.y_p, data.y_u, data.y_T, data.y_rho, data.y_mach], dim=1)
        data = data.to(device)

        # --- Correction du batch manquant ---
        # torch.load() d'un graphe seul ne crée pas data.batch (contrairement au DataLoader).
        # Sans batch, global_vars[None] ajoute une dimension parasite → erreur 2D vs 3D.
        # On crée manuellement le vecteur batch : tous les nœuds appartiennent au graphe 0.
        if data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        # --- A. Préparation des données ---
        pos_real = data.pos.cpu().numpy()[:, :2]

        # Récupérer les données de la ligne correspondante dans le dataframe global (passé en argument)
        df_global = pd.read_csv("dataset_nozzle.csv")
        sim_id = int(re.search(r'sim_(\d+)', graph_path).group(1))
        case_row = df_global[df_global['id'] == sim_id].iloc[0]
        # --- Extraction des paramètres Nozzle ---
        R_exit = case_row["R_exit"]
        L_conv = case_row["L_convergent"]
        L_div = case_row["L_divergent"]
        # --- Calcul des limites de la géométrie ---
        x_exit = 2*(L_conv + L_div)
        r_outlet = 2*R_exit
        
        # 1. Normalise les données
        data.x = normalizer_x.encode(data.x)
        data.edge_attr = normalizer_edges.encode(data.edge_attr)
        data.bc_params = normalizer_bc.encode(data.bc_params)
        data.case_params = normalizer_case.encode(data.case_params)
        global_params = torch.stack([data.thrust, data.isp, data.p_ratio, data.m_dot], dim=1)
        data.global_params = normalizer_global.encode(global_params)
        
        # 2. Prediction IA
        with torch.no_grad():
            prediction_local, prediction_global = model(data)

        # 3. Retour aux unités réelles (dénormalisation)
        pred_real = normalizer_y.decode(prediction_local).cpu().numpy()
        p_pred = pred_real[:, 0]
        u_pred = pred_real[:, 1:3]
        T_pred = pred_real[:, 3]
        rho_pred = pred_real[:, 4]
        mach_pred = pred_real[:, 5]
        
        # 4. Données de vérité (OpenFOAM)
        y_true = data.y.cpu().numpy()
        p_true = y_true[:, 0]
        u_true = y_true[:, 1:3]
        T_true = y_true[:, 3]   
        rho_true = y_true[:, 4]
        mach_true = y_true[:, 5]
        # pos_real a été défini plus haut depuis data.pos (ou reconstruit en fallback)
        
        # --- C. Calculs Physiques ---
        
        vel_mag_pred = np.sqrt(u_pred[:, 0]**2 + u_pred[:, 1]**2)
        vel_mag_true = np.sqrt(u_true[:, 0]**2 + u_true[:, 1]**2)
        error_vel = np.abs(vel_mag_pred - vel_mag_true)
        
        print(f"\n--- Analyse PINN : {os.path.basename(graph_path)} ---")
        print(f"Erreur Vitesse Moyenne : {np.mean(error_vel):.2f} m/s")
        print(f"Erreur Vitesse Max     : {np.max(error_vel):.2f} m/s")

        error_p = np.abs(p_pred - p_true)
        error_T = np.abs(T_pred - T_true)
        error_rho = np.abs(rho_pred - rho_true)
        error_mach = np.abs(mach_pred - mach_true)

        print(f"Erreur Pression Moyenne : {np.mean(error_p):.2f} Pa")
        print(f"Erreur Pression Max     : {np.max(error_p):.2f} Pa")
        print(f"Erreur Temperature Moyenne : {np.mean(error_T):.2f} K")
        print(f"Erreur Temperature Max     : {np.max(error_T):.2f} K")
        print(f"Erreur Masse Volumique Moyenne : {np.mean(error_rho):.2f} kg/m^3")
        print(f"Erreur Masse Volumique Max     : {np.max(error_rho):.2f} kg/m^3")
        print(f"Erreur Mach Moyenne : {np.mean(error_mach):.2f}")
        print(f"Erreur Mach Max     : {np.max(error_mach):.2f}")
        
        # --- E. Tracer la comparaison des champs de Mach ---
        fig2, (ax_mt, ax_mp, ax_me) = plt.subplots(3, 1, figsize=(12, 12))
        
        vmax_mach = 4.0
        vmax_err_mach = 0.5 

        # --- Interpolation sur une grille régulière ---
        # 1. Création de la grille cible
        y_max = r_outlet * 1.1
        grid_x, grid_y = np.mgrid[-0.05:(x_exit+0.1):500j, -y_max:y_max:200j]
        
        # 2. Préparation des données miroirs pour l'interpolation
        pos_sym = pos_real.copy()
        pos_sym[:, 1] = -pos_real[:, 1]
        
        # Concaténation
        points_all = np.concatenate((pos_real, pos_sym), axis=0)
        mach_true_all = np.concatenate((mach_true, mach_true), axis=0)
        mach_pred_all = np.concatenate((mach_pred, mach_pred), axis=0)
        
        # 3. Interpolation (Linear pour la précision des chocs)
        grid_mach_true = griddata(points_all, mach_true_all, (grid_x, grid_y), method='linear', fill_value=0)
        grid_mach_pred = griddata(points_all, mach_pred_all, (grid_x, grid_y), method='linear', fill_value=0)
        
        # Calcul de l'erreur sur la grille interpolée
        grid_error = np.abs(grid_mach_true - grid_mach_pred)

        # 4. Créer un masque basé sur la distance aux points réels
        # On utilise un KDTree pour trouver la distance grille -> points réels

        tree = cKDTree(points_all)
        # On aplatit la grille pour le calcul
        grid_coords = np.c_[grid_x.ravel(), grid_y.ravel()]
        dists, _ = tree.query(grid_coords)
        
        # Seuil de masquage (environ 1.5x la distance max entre points, soit ~0.05 - 0.1)
        mask_threshold = 0.05
        mask_void = dists > mask_threshold
        
        # Appliquer le masque (NaN pour que imshow ne dessine rien)
        grid_mach_true.ravel()[mask_void] = np.nan
        grid_mach_pred.ravel()[mask_void] = np.nan
        grid_error.ravel()[mask_void] = np.nan
        
        # --- Affichage (Pcolormesh) ---
        extent = (-0.05, x_exit+0.1, -y_max, y_max)
        
        # Champ Vérité
        im1 = ax_mt.imshow(grid_mach_true.T, extent=extent, origin='lower', cmap='jet', vmin=0, vmax=vmax_mach)
        ax_mt.set_title(f"Mach Number - OpenFOAM (Truth)")
        ax_mt.set_aspect('equal')
        plt.colorbar(im1, ax=ax_mt, label="Mach")
        
        # Champ GNN
        im2 = ax_mp.imshow(grid_mach_pred.T, extent=extent, origin='lower', cmap='jet', vmin=0, vmax=vmax_mach)
        ax_mp.set_title(f"Mach Number - GNN Prediction")
        ax_mp.set_aspect('equal')
        plt.colorbar(im2, ax=ax_mp, label="Mach")
        
        # Champ Erreur
        im3 = ax_me.imshow(grid_error.T, extent=extent, origin='lower', cmap='hot', vmin=0, vmax=vmax_err_mach)
        ax_me.set_title(f"Absolute Error Map (Interpolated)")
        ax_me.set_aspect('equal')
        plt.colorbar(im3, ax=ax_me, label="Mach Error")
        
        field_output = f"data/nozzle/figures/nozzle_mach_comp_v26_{os.path.basename(graph_path).replace('.pt', '.png')}"
        plt.tight_layout()
        fig2.savefig(field_output, dpi=200)
        print(f"Detailed field comparison saved : {field_output}")
        plt.close(fig2)

        # --- F. Profils 1D (Radiaux et Axiaux) ---
        fig3, (ax_long, ax_rad1, ax_rad2) = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. Profil Longitudinal (Axe de symétrie Y=0)
        # On cherche l'indice de la grille y qui est le plus proche de 0
        idx_y0 = np.argmin(np.abs(grid_y[0, :]))
        x_line = grid_x[:, idx_y0]
        mach_true_long = grid_mach_true[:, idx_y0]
        mach_pred_long = grid_mach_pred[:, idx_y0]
        
        ax_long.plot(x_line, mach_true_long, 'k-', linewidth=2, label='OpenFOAM (Centerline)')
        ax_long.plot(x_line, mach_pred_long, 'r--', linewidth=2, label='GNN (Centerline)')
        ax_long.set_title(f"Profil de Mach sur l'axe de symétrie (Y=0) - {os.path.basename(graph_path)}")
        ax_long.set_xlabel("X (m)")
        ax_long.set_ylabel("Mach")
        ax_long.legend()
        ax_long.grid(True, alpha=0.3)
        
        # 2. Profil Radial au Col (X = L_conv)
        idx_x_col = np.argmin(np.abs(grid_x[:, 0] - L_conv))
        y_line = grid_y[idx_x_col, :]
        mach_true_rad_col = grid_mach_true[idx_x_col, :]
        mach_pred_rad_col = grid_mach_pred[idx_x_col, :]
        
        ax_rad1.plot(y_line, mach_true_rad_col, 'k-', linewidth=2, label='OpenFOAM (Throat)')
        ax_rad1.plot(y_line, mach_pred_rad_col, 'r--', linewidth=2, label='GNN (Throat)')
        ax_rad1.set_title(f"Profil Radial au Col (X = {L_conv:.2f}m)")
        ax_rad1.set_xlabel("Y (m)")
        ax_rad1.set_ylabel("Mach")
        ax_rad1.legend()
        ax_rad1.grid(True, alpha=0.3)
        
        # 3. Profil Radial à la Sortie (X = L_conv + L_div)
        idx_x_exit = np.argmin(np.abs(grid_x[:, 0] - (L_conv + L_div)))
        mach_true_rad_exit = grid_mach_true[idx_x_exit, :]
        mach_pred_rad_exit = grid_mach_pred[idx_x_exit, :]
        
        ax_rad2.plot(y_line, mach_true_rad_exit, 'k-', linewidth=2, label='OpenFOAM (Exit)')
        ax_rad2.plot(y_line, mach_pred_rad_exit, 'r--', linewidth=2, label='GNN (Exit)')
        ax_rad2.set_title(f"Profil Radial à la Sortie (X = {(L_conv + L_div):.2f}m)")
        ax_rad2.set_xlabel("Y (m)")
        ax_rad2.set_ylabel("Mach")
        ax_rad2.legend()
        ax_rad2.grid(True, alpha=0.3)
        
        profile_output = f"data/nozzle/figures/nozzle_profiles_v26_{os.path.basename(graph_path).replace('.pt', '.png')}"
        plt.tight_layout()
        fig3.savefig(profile_output, dpi=200)
        print(f"1D profiles saved : {profile_output}")
        plt.close(fig3)

if __name__ == "__main__":
    validate()
