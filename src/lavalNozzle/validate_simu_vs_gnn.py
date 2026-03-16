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
    model = NozzleGNN(input_dim=21, hidden_dim=64, output_dim_local=6, output_dim_global=4, num_layers=5).to(device)
    model.load_state_dict(torch.load("nozzle_gnn_best_v33.pt", map_location=device))
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
        
        # --- E. Tracer la comparaison des champs de Mach (Scatter Plot — points réels, sans interpolation) ---
        fig2, axes = plt.subplots(3, 1, figsize=(14, 13))
        ax_mt, ax_mp, ax_me = axes

        vmax_mach = 4.0
        vmax_err_mach = 0.5
        error_mach_scatter = np.abs(mach_pred - mach_true)

        # Données miroir (symétrie axiale) pour afficher le demi-domaine bas
        pos_sym = pos_real.copy()
        pos_sym[:, 1] = -pos_real[:, 1]
        pos_full = np.concatenate((pos_real, pos_sym), axis=0)
        mach_true_full = np.concatenate((mach_true, mach_true), axis=0)
        mach_pred_full = np.concatenate((mach_pred, mach_pred), axis=0)
        error_full = np.concatenate((error_mach_scatter, error_mach_scatter), axis=0)

        # Taille des marqueurs : petite pour un rendu dense, sans recouvrement excessif
        s = 1.5

        sc1 = ax_mt.scatter(pos_full[:, 0], pos_full[:, 1], c=mach_true_full,
                            cmap='jet', vmin=0, vmax=vmax_mach, s=s, linewidths=0)
        ax_mt.set_title("Mach Number — OpenFOAM (Truth) · Points réels")
        ax_mt.set_xlabel("X (m)")
        ax_mt.set_ylabel("Y (m)")
        ax_mt.set_aspect('equal')
        plt.colorbar(sc1, ax=ax_mt, label="Mach")

        sc2 = ax_mp.scatter(pos_full[:, 0], pos_full[:, 1], c=mach_pred_full,
                            cmap='jet', vmin=0, vmax=vmax_mach, s=s, linewidths=0)
        ax_mp.set_title("Mach Number — GNN Prediction · Points réels")
        ax_mp.set_xlabel("X (m)")
        ax_mp.set_ylabel("Y (m)")
        ax_mp.set_aspect('equal')
        plt.colorbar(sc2, ax=ax_mp, label="Mach")

        sc3 = ax_me.scatter(pos_full[:, 0], pos_full[:, 1], c=error_full,
                            cmap='hot', vmin=0, vmax=vmax_err_mach, s=s, linewidths=0)
        ax_me.set_title("Absolute Error |GNN − CFD| · Points réels")
        ax_me.set_xlabel("X (m)")
        ax_me.set_ylabel("Y (m)")
        ax_me.set_aspect('equal')
        plt.colorbar(sc3, ax=ax_me, label="|ΔMach|")

        field_output = f"data/nozzle/figures/nozzle_mach_comp_v33_{os.path.basename(graph_path).replace('.pt', '.png')}"
        plt.tight_layout()
        fig2.savefig(field_output, dpi=200)
        print(f"Detailed field comparison saved : {field_output}")
        plt.close(fig2)

        # --- F. Profils 1D — extraction KDTree + binage (lissage sans interpolation) ---
        fig3, (ax_long, ax_rad1, ax_rad2) = plt.subplots(3, 1, figsize=(10, 15))
        tree_real = cKDTree(pos_real)

        def extract_profile_binned(coord_vals, field_true, field_pred, n_bins=120):
            """Bine les points par coord, moyenne dans chaque bin → courbe lisse sans zigzag."""
            sort_idx = np.argsort(coord_vals)
            cv = coord_vals[sort_idx]; ft = field_true[sort_idx]; fp = field_pred[sort_idx]
            bins = np.linspace(cv.min(), cv.max(), n_bins + 1)
            bin_idx = np.clip(np.digitize(cv, bins) - 1, 0, n_bins - 1)
            coord_b, true_b, pred_b = [], [], []
            for b in range(n_bins):
                mb = bin_idx == b
                if mb.sum() > 0:
                    coord_b.append(cv[mb].mean())
                    true_b.append(ft[mb].mean())
                    pred_b.append(fp[mb].mean())
            return np.array(coord_b), np.array(true_b), np.array(pred_b)

        # ── 1. Profil longitudinal (axe Y = 0) ──
        x_query = np.linspace(pos_real[:, 0].min(), pos_real[:, 0].max(), 600)
        query_axis = np.column_stack([x_query, np.zeros(600)])
        dists_a, idx_a = tree_real.query(query_axis)
        res_a = (pos_real[:, 0].max() - pos_real[:, 0].min()) / 600
        valid_a = dists_a < 6 * res_a
        if valid_a.sum() > 5:
            x_b, mt_b, mp_b = extract_profile_binned(
                pos_real[idx_a[valid_a], 0], mach_true[idx_a[valid_a]], mach_pred[idx_a[valid_a]], n_bins=200)
            ax_long.plot(x_b, mt_b, 'k-', linewidth=2, label='OpenFOAM (Centerline)')
            ax_long.plot(x_b, mp_b, 'r--', linewidth=2, label='GNN (Centerline)')
        else:
            ax_long.text(0.5, 0.5, "Pas assez de points sur l'axe", ha='center', transform=ax_long.transAxes)
        ax_long.set_title(f"Profil de Mach sur l'axe (Y≈0) — {os.path.basename(graph_path)}")
        ax_long.set_xlabel("X (m)"); ax_long.set_ylabel("Mach")
        ax_long.legend(); ax_long.grid(True, alpha=0.3)

        # ── Fonction profil radial réutilisable ──
        def radial_profile(ax, x_target, label_foam, label_gnn, title):
            y_max_loc = pos_real[:, 1].max()
            y_query = np.linspace(0, y_max_loc, 400)
            query_rad = np.column_stack([np.full(400, x_target), y_query])
            dists_r, idx_r = tree_real.query(query_rad)
            res_r = y_max_loc / 400
            valid_r = dists_r < 6 * res_r
            if valid_r.sum() > 5:
                y_b, mt_b, mp_b = extract_profile_binned(
                    pos_real[idx_r[valid_r], 1], mach_true[idx_r[valid_r]], mach_pred[idx_r[valid_r]], n_bins=100)
                ax.plot(y_b, mt_b, 'k-', linewidth=2, label=label_foam)
                ax.plot(y_b, mp_b, 'r--', linewidth=2, label=label_gnn)
            else:
                ax.text(0.5, 0.5, "Pas assez de points", ha='center', transform=ax.transAxes)
            ax.set_title(title); ax.set_xlabel("Y (m)"); ax.set_ylabel("Mach")
            ax.legend(); ax.grid(True, alpha=0.3)

        # ── 2. Col ──
        radial_profile(ax_rad1, L_conv,
                       'OpenFOAM (Throat)', 'GNN (Throat)',
                       f"Profil Radial au Col (X = {L_conv:.2f}m)")
        # ── 3. Sortie ──
        radial_profile(ax_rad2, L_conv + L_div,
                       'OpenFOAM (Exit)', 'GNN (Exit)',
                       f"Profil Radial à la Sortie (X = {L_conv + L_div:.2f}m)")

        profile_output = f"data/nozzle/figures/nozzle_profiles_v33_{os.path.basename(graph_path).replace('.pt', '.png')}"
        plt.tight_layout()
        fig3.savefig(profile_output, dpi=200)
        print(f"1D profiles saved : {profile_output}")
        plt.close(fig3)

if __name__ == "__main__":
    validate()
