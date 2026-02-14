import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
from model import AirfoilGNN
from generate_naca import generate_naca4, save_airfoil
from train import Normalizer
from dataset import AirfoilDataset
from torch_geometric.loader import DataLoader

def run_xfoil_headless(m, p, t):
    """Extraction Cp via XFOIL."""
    naca_code = f"{int(m*100)}{int(p*10)}{int(t*100):02d}"
    cp_file = f"cp_xf_{naca_code}.txt"
    if os.path.exists(cp_file): os.remove(cp_file)
    
    xfoil_input = f"""NACA {naca_code}
OPER
VISC 1.6e6
ITER 200
ALFA 0
CPWR {cp_file}
"""
    try:
        process = subprocess.Popen(
            ['xfoil'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=xfoil_input, timeout=15)
        
        if os.path.exists(cp_file):
            # Essayer de charger avec différents en-têtes potentiels
            for skip in [1, 2, 3]:
                try:
                    data = np.loadtxt(cp_file, skiprows=skip)
                    if data.ndim == 2 and data.shape[1] >= 2:
                        os.remove(cp_file)
                        return data
                except:
                    continue
    except Exception as e:
        print(f"XFOIL error: {e}")
    return None

def validate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # --- 1. CHARGEMENT DU MODÈLE ---
    model = AirfoilGNN(input_dim=10, hidden_dim=128, output_dim=3, num_layers=4).to(device)
    model.load_state_dict(torch.load("airfoil_gnn_best.pt", map_location=device))
    model.eval()

    # --- 2. RESTAURATION DES NORMALISEURS ---
    # On charge les statistiques sauvegardées pendant l'entraînement
    if not os.path.exists("normalizer_stats.pt"):
        print("Error: 'normalizer_stats.pt' not found. You must run training first to generate it.")
        return
        
    stats = torch.load("normalizer_stats.pt", map_location=device)
    
    # Restauration utilisant les moyennes/stds sauvegardées
    normalizer_x = Normalizer(mean=stats['x_mean'], std=stats['x_std'], device=device)
    normalizer_y = Normalizer(mean=stats['y_mean'], std=stats['y_std'], device=device)
    normalizer_edges = Normalizer(mean=stats['edge_mean'], std=stats['edge_std'], device=device)

    
    # --- 3. CHOIX DES CAS DE TEST ---

    test_cases = [
        {"path": "data/graphs/sim_0001_naca_1316.pt", "naca": (0.01, 0.3, 0.16)},
        {"path": "data/graphs/sim_0002_naca_1616.pt", "naca": (0.01, 0.6, 0.16)}
    ]
    
    for case in test_cases:
        graph_path = case["path"]
        if not os.path.exists(graph_path): continue
            
        data = torch.load(graph_path, weights_only=False)
        data.y = torch.cat([data.y_p, data.y_u], dim=1)
        data = data.to(device)
        m, p, t = case["naca"]

        # --- A. Préparation des données ---
        pos_real = data.x.clone().cpu().numpy()[:, :2]
        
        # 1. Normalise les données
        data.x = normalizer_x.encode(data.x)
        data.edge_attr = normalizer_edges.encode(data.edge_attr)
        
        # 2. Prediction IA
        with torch.no_grad():
            prediction = model(data)

        # 3. Retour aux unités réelles (dénormalisation)
        pred_real = normalizer_y.decode(prediction).cpu().numpy()
        p_pred = pred_real[:, 0]
        u_pred = pred_real[:, 1:]
        
        # 4. Données de vérité (OpenFOAM)
        y_true = data.y.cpu().numpy()
        p_true = y_true[:, 0]
        u_true = y_true[:, 1:]
        pos_np = data.x.cpu().numpy()

        pos_real = normalizer_x.decode(data.x).cpu().numpy()[:, :2]
            
        # --- C. XFOIL Reference ---
        xf_data = run_xfoil_headless(m, p, t)
        
        # --- D. Calculs Physiques ---

        U_inf = 25      
        p_inf_kin = 0.0
        
        cp_pred = (p_pred - p_inf_kin) / (0.5 * U_inf**2)
        cp_true = (p_true - p_inf_kin) / (0.5 * U_inf**2)
        
        vel_mag_pred = np.sqrt(u_pred[:, 0]**2 + u_pred[:, 1]**2)
        vel_mag_true = np.sqrt(u_true[:, 0]**2 + u_true[:, 1]**2)
        error_vel = np.abs(vel_mag_pred - vel_mag_true)
        
        
        print(f"\n--- Analyse PINN : {os.path.basename(graph_path)} ---")
        print(f"Erreur Vitesse Moyenne : {np.mean(error_vel):.2f} m/s")
        print(f"Erreur Vitesse Max     : {np.max(error_vel):.2f} m/s")
        
        # --- E. Tracer la comparaison des champs de vitesse ---
        fig2, (ax_t, ax_p, ax_e) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Échelle de couleurs cohérente pour la vitesse et l'erreur
        vmax_vel = 35 # m/s
        vmax_err = 10 # m/s (Fixe pour comparer les essais)
        
        # Champ Vérité
        sc1 = ax_t.scatter(pos_real[:, 0], pos_real[:, 1], c=vel_mag_true, cmap='viridis', s=1, vmin=0, vmax=vmax_vel)
        ax_t.set_title(f"Velocity Magnitude - OpenFOAM (Truth)")
        ax_t.set_xlim(-0.1, 1.1)
        ax_t.set_ylim(-0.2, 0.2)
        ax_t.set_aspect('equal')
        plt.colorbar(sc1, ax=ax_t, label="m/s")
        
        # Champ GNN
        sc2 = ax_p.scatter(pos_real[:, 0], pos_real[:, 1], c=vel_mag_pred, cmap='viridis', s=1, vmin=0, vmax=vmax_vel)
        ax_p.set_title(f"Velocity Magnitude - GNN + PINN")
        ax_p.set_xlim(-0.1, 1.1)
        ax_p.set_ylim(-0.2, 0.2)
        ax_p.set_aspect('equal')
        plt.colorbar(sc2, ax=ax_p, label="m/s")
        
        # Champ Erreur
        sc3 = ax_e.scatter(pos_real[:, 0], pos_real[:, 1], c=error_vel, cmap='hot', s=1, vmin=0, vmax=vmax_err)
        ax_e.set_title(f"Absolute Error Map (Max: {np.max(error_vel):.1f} m/s)")
        ax_e.set_xlim(-0.1, 1.1)
        ax_e.set_ylim(-0.2, 0.2)
        ax_e.set_aspect('equal')
        plt.colorbar(sc3, ax=ax_e, label="m/s")
        
        field_output = f"data/perso_field_comp_v5_{os.path.basename(graph_path).replace('.pt', '.png')}"
        plt.tight_layout()
        fig2.savefig(field_output, dpi=200)
        print(f"Detailed field comparison saved : {field_output}")
        
        # Garder aussi la logique de tracé Cp existante
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Tracé de comparaison Cp
        mask = (pos_real[:, 0] >= -0.01) & (pos_real[:, 0] <= 1.01) & (np.abs(pos_real[:, 1]) <= 0.04)
        x_surf = pos_real[mask, 0]
        ax1.scatter(x_surf, cp_true[mask], label="OpenFOAM (Truth)", color="#3498db", s=12, alpha=0.4)
        ax1.scatter(x_surf, cp_pred[mask], label="GNN + PINN (Pred)", color="#e74c3c", s=12, alpha=0.4)
        if xf_data is not None:
            x_xf = xf_data[:, 0]
            cp_xf = xf_data[:, 1] if xf_data.shape[1] == 2 else xf_data[:, 2]
            ax1.plot(x_xf, cp_xf, label="XFOIL (Ref)", color="black", linewidth=1.2, linestyle="-")
        ax1.invert_yaxis()
        ax1.set_xlabel("Chord (x/c)")
        ax1.set_ylabel("Cp")
        ax1.set_title(f"Validation: Cp Distribution (NACA {int(m*100)}{int(p*10)}{int(t*100)})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        sc = ax2.scatter(pos_real[:, 0], pos_real[:, 1], c=error_vel, cmap='hot', s=1, alpha=0.6, vmin=0, vmax=vmax_err)
        plt.colorbar(sc, ax=ax2, label="Abs Error (m/s)")
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.15, 0.15)
        ax2.set_aspect('equal')
        ax2.set_title("PINN v5: Velocity Magnitude Error Map")
        output_name = f"data/perso_v5_val_{os.path.basename(graph_path).replace('.pt', '.png')}"
        plt.tight_layout()
        plt.savefig(output_name, dpi=150)
        plt.close('all') # Nettoyage

if __name__ == "__main__":
    validate()
