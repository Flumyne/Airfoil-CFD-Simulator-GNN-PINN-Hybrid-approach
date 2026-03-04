import os
import pyvista as pv
import torch
from torch_geometric.data import Data
import numpy as np
import glob
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import re
import pandas as pd

def get_latest_time(case_dir):
    """Trouve le répertoire temporel le plus récent dans un cas OpenFOAM."""
    time_dirs = []
    for d in os.listdir(case_dir):
        if (d.isdigit() or (d.replace('.', '', 1).isdigit())) and float(d) > 0:
            time_dirs.append(d)
    
    if not time_dirs:
        return None
    
    # Trier par valeur flottante
    time_dirs.sort(key=float)
    return time_dirs[-1]

def process_case_to_graph(case_dir, output_file, df_global):
    """
    Lit un cas OpenFOAM et le convertit en objet PyTorch Geometric Data.
    """
    # 1. Créer un fichier .foam factice pour PyVista
    case_name = os.path.basename(case_dir.rstrip('/'))
    foam_file = os.path.join(case_dir, f"{case_name}.foam")
    if not os.path.exists(foam_file):
        open(foam_file, 'w').close()

    try:
        # 2. Lire le cas avec PyVista
        reader = pv.OpenFOAMReader(foam_file)
        
        # Selectionne le dossier le plus récent de la simulation
        reader.set_active_time_value(float(get_latest_time(case_dir)))
        
        # Récupérer les données de la ligne correspondante dans le dataframe global (passé en argument)
        sim_id = int(re.search(r'sim_(\d+)', case_name).group(1))
        case_row = df_global[df_global['id'] == sim_id].iloc[0]
        
        # Lire le maillage
        mesh = reader.read()
        internal_mesh = mesh["internalMesh"]

        # 3. Extraire les coordonnées des nœuds
        mesh_points = internal_mesh.cell_data_to_point_data()

        node_type_full = torch.zeros(mesh_points.n_points, dtype=torch.long)

        # Constantes pour la lisibilité 
        TYPE_FLUID =0
        TYPE_INLET =1
        TYPE_WALL =2
        TYPE_OUTLET =3
        TYPE_SYMMETRY =4

        wall_points_all = []
        symmetry_points_all = []

        # Initialiser KDTree pour tous les points
        internal_tree = KDTree(internal_mesh.points)
        
        # Trouver les points du patch de l'aile
        if 'boundary' in mesh.keys():
            boundary_block = mesh['boundary']
            for patch_name in boundary_block.keys():
                patch = boundary_block[patch_name]
                if patch.n_points == 0 : continue
                current_type = -1
                if 'nozzle' in patch_name.lower():
                    current_type = TYPE_WALL
                    wall_points_all.append(patch.points) 
                elif 'inlet' in patch_name.lower():
                    current_type = TYPE_INLET
                elif 'outlet' in patch_name.lower():
                    current_type = TYPE_OUTLET
                elif 'symmetry' in patch_name.lower():
                    current_type = TYPE_SYMMETRY    
                    symmetry_points_all.append(patch.points)
                else:
                    current_type = TYPE_FLUID
                
                if current_type != -1:
                    _, found_ids = internal_tree.query(patch.points)
                    node_type_full[found_ids] = current_type


        # --- ÉCHANTILLONNAGE INTELLIGENT (BASÉ SUR LA DENSITÉ) ---
        dist_to_wall_full = np.full(mesh_points.n_points, 100.0)
        dist_to_symmetry_full = np.full(mesh_points.n_points, 100.0)
        
        if wall_points_all:
            all_wall_pts = np.concatenate(wall_points_all, axis=0)
            wall_tree = KDTree(all_wall_pts)
            dists, _ = wall_tree.query(mesh_points.points)
            dist_to_wall_full = dists
        
        if symmetry_points_all:
            all_symmetry_pts = np.concatenate(symmetry_points_all, axis=0)
            symmetry_tree = KDTree(all_symmetry_pts)
            dists, _ = symmetry_tree.query(mesh_points.points)
            dist_to_symmetry_full = dists

        # --- Extraction des paramètres Nozzle ---
        R_inlet = case_row["R_inlet"]
        R_throat = case_row["R_throat"]
        R_exit = case_row["R_exit"]
        L_conv = case_row["L_convergent"]
        L_div = case_row["L_divergent"]
         
        # --- ECHANTILLONNAGE BASÉ SUR LE CHOC (GRADIENT DE PRESSION) ---
        # 1. Calcul du gradient approximatif (basique mais suffisant)
        pressures = mesh_points.point_data["p"]
        # On utilise les voisins KDTree pour estimer la variance locale de P
        # Si la variance est forte -> Choc -> On garde
        dists_k, ids_k = internal_tree.query(mesh_points.points, k=5)
        local_p_std = np.std(pressures[ids_k], axis=1)
        
        # Seuil adaptatif : On garde les 15% des points avec le plus fort gradient
        threshold_shock = np.percentile(local_p_std, 80)
        
        # Échantillonnage hybride : 
        # - 100% au mur/col (Physique de couche limite)
        # - 100% si Choc (Gradient fort)
        # - 10% ailleurs (Zone calme)
        
        indices_to_keep = []
        for i, (d, s, p_var) in enumerate(zip(dist_to_wall_full, dist_to_symmetry_full, local_p_std)):
            x_val = internal_mesh.points[i, 0]
            
            is_shock = (p_var > threshold_shock) and (x_val > L_conv) # Choc uniquement dans le divergent/jet
            is_wall_symmetry = (d < R_throat/2) or (abs(x_val) < L_conv) or (s < 0.05)
            
            if is_wall_symmetry or is_shock:
                indices_to_keep.append(i)
            else:                              # Champ lointain
                if i % 4 == 0:                # Encore plus agressif loin (1/4)
                    indices_to_keep.append(i)
        
        indices_to_keep = np.array(indices_to_keep)

        points_kept = mesh_points.points[indices_to_keep]

        # Supprimer la coordonnée Z
        pos = torch.tensor(points_kept[:, :2], dtype=torch.float) 

        p = torch.tensor(mesh_points.point_data["p"][indices_to_keep], dtype=torch.float).view(-1, 1)
        u = torch.tensor(mesh_points.point_data["U"][indices_to_keep, :2], dtype=torch.float)
        T = torch.tensor(mesh_points.point_data["T"][indices_to_keep], dtype=torch.float).view(-1, 1)
        rho = torch.tensor(mesh_points.point_data["rho"][indices_to_keep], dtype=torch.float).view(-1, 1)
        
        # Calcul du Mach
        U_mag = np.sqrt(np.sum(mesh_points.point_data["U"][indices_to_keep, :2]**2, axis=1))
        sound_speed = np.sqrt(1.4 * 287 * mesh_points.point_data["T"][indices_to_keep])
        mach_np = U_mag / sound_speed
        Mach = torch.tensor(mach_np, dtype=torch.float).view(-1, 1)

        node_type = node_type_full[indices_to_keep]

        # --- CORRECTION PHYSIQUE : Forcer Uy=0 sur l'axe de symétrie ---
        mask_sym = (node_type == 4)
        u[mask_sym, 1] = 0.0  # On force la composante Y à zéro
        
        # Distance au mur pour les points conservés
        dist_to_wall = torch.tensor(dist_to_wall_full[indices_to_keep], dtype=torch.float).view(-1, 1)
        dist_to_symmetry = torch.tensor(dist_to_symmetry_full[indices_to_keep], dtype=torch.float).view(-1, 1)

        # 4. RECONSTRUCTION DE LA CONNECTIVITÉ 
        pos_np = pos.numpy()
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(pos_np)
        _, indices = nbrs.kneighbors(pos_np)
        indices = indices[:, 1:] # Supprimer les auto-connexions
        sources_nodes = np.repeat(np.arange(len(pos)), 5)
        target_nodes = indices.flatten()
        edge_index = torch.tensor(np.array([sources_nodes, target_nodes]), dtype=torch.long)
        
        # --- Calcul des Edge Features (Vecteur relatif et distance) ---
        
        # 1. On récupère les indices des points des arêtes
        src, dst = edge_index
        
        # 2. On calcule le vecteur du déplacement (u_ij = pos_i - pos_j)
        # 3. On calcule la distance entre les points (d_ij = ||pos_i - pos_j||)
        d_ij = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)

        # Filtre les arêtes avec une distance inférieure à 1e-8
        # 3. Différences de positions (Vecteur relatif non normalisé)
        rel_pos = pos[src] - pos[dst]
        
        # 4. Distance euclidienne
        d_ij = torch.norm(rel_pos, dim=1, keepdim=True)
        
        # 5. Vecteur unitaire (Direction pure)
        u_ij = rel_pos / (d_ij + 1e-8)

        # 6. Attributs d'arêtes : [dx, dy, distance, unit_dx, unit_dy] -> 5 features
        edge_attr = torch.cat([rel_pos, d_ij, u_ij], dim=1)
        
        type_one_hot = torch.nn.functional.one_hot(node_type, num_classes=5).float()

        # Étendre les paramètres Nozzle à tous les nœuds
        nozzle_features = torch.tensor([[R_inlet, R_throat, R_exit, L_conv, L_div]], dtype=torch.float).expand(pos.shape[0], -1)

        # --- Calcul de la poussée (Thrust) sur le patch de sortie ---
        thrust = 0.0
        p_amb = 35000.0 # Valeur de setup_openfoam.py
        
        outlet_mask = (node_type == TYPE_OUTLET)
        if outlet_mask.any():
            # On récupère les données de sortie
            p_exit = p[outlet_mask].numpy()
            u_exit = u[outlet_mask, 0].numpy() # Ux
            rho_exit = rho[outlet_mask].numpy()
            
            # Approximation de l'intégrale (Thrust = sum[ (rho*u^2 + p - p_amb) * Delta_y ])
            term_mom = np.mean(rho_exit * u_exit**2)
            term_press = np.mean(p_exit - p_amb)
            thrust = (term_mom + term_press) * R_exit
            
            # Débit massique (m_dot = sum[ rho * u * Delta_y ])
            m_dot = np.mean(rho_exit * u_exit) * R_exit
            
            # Isp (Impulsion spécifique) = F / (m_dot * g0)
            isp = thrust / (m_dot * 9.81)
            
            # Rapport de pression moyen en sortie
            p_ratio_val = float(np.mean(p_exit) / p_amb)
        else:
            p_ratio_val = 1.0

        
        # --- Zone One-hot-encoding 0 = convergent, 1 = divergent, 2 = atmosphere---
        # pos[:, 0] est un tensor [N] → opérations vectorisées (pas de if scalaire)
        pos_x = pos[:, 0]  # tensor [N]

        mask_conv = pos_x <= L_conv
        mask_div  = (pos_x > L_conv) & (pos_x <= L_conv + L_div)
        mask_atm  = pos_x > (L_conv + L_div)

        '''
        zone_labels = torch.zeros(pos_x.shape[0], dtype=torch.long)
        zone_labels[mask_conv] = 0
        zone_labels[mask_div]  = 1
        zone_labels[mask_atm]  = 2

        zone_type_one_hot = torch.nn.functional.one_hot(zone_labels, num_classes=3).float()  # [N, 3]
        '''

        # --- X_normalisé par zone (convergent: -1→0, divergent: 0→1, atmo: 1→...) ---
        x_norm = torch.zeros(pos_x.shape[0], dtype=torch.float)
        x_norm[mask_conv] = (pos_x[mask_conv] - L_conv) / L_conv
        x_norm[mask_div]  = (pos_x[mask_div]  - L_conv) / L_div
        x_norm[mask_atm]  = 1.0 + (pos_x[mask_atm] - (L_conv + L_div)) / (L_conv + L_div)

        x_norm = x_norm.view(-1, 1)  # [N, 1]

        # Caractéristiques finales des nœuds (input_dim=8)
        x_features = torch.cat([x_norm, type_one_hot, dist_to_wall, dist_to_symmetry], dim=1)

        # Récupération des conditions limites
        reader.set_active_time_value(0.0)
        mesh = reader.read()
        T_inlet = mesh['boundary']['inlet']['T'][0]
        p_inlet = mesh['boundary']['inlet']['p'][0]
        p_outlet = mesh['boundary']['outlet']['p'][0]
        p_ratio_imposed = p_outlet / p_inlet

        # 6. Construire l'objet Data
        data = Data(
            x=x_features,
            pos=pos,          # Coordonnées réelles [N, 2] pour la visualisation
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y_p=p, y_u=u, y_T=T, y_rho=rho, y_mach=Mach,
            node_type=node_type
        )
        data.thrust = torch.tensor([thrust], dtype=torch.float)
        data.m_dot = torch.tensor([m_dot], dtype=torch.float)
        data.isp = torch.tensor([isp], dtype=torch.float)
        data.p_ratio = torch.tensor([p_ratio_val], dtype=torch.float)
        data.bc_params = torch.tensor([[T_inlet, p_inlet, p_ratio_imposed]], dtype=torch.float)
        data.case_params = torch.tensor([[R_inlet, R_throat, R_exit, L_conv, L_div]], dtype=torch.float)
        data.case_name = case_name

        # 7. Sauvegarder
        torch.save(data, output_file)
        return True

    except Exception as e:
        print(f"Error processing {case_dir}: {e}")
        return False
    finally:
        if os.path.exists(foam_file):
            os.remove(foam_file)



def main():
    sim_root = "simulations/nozzle"
    output_dir = "data/graphs/nozzle"
    csv_global_path = "dataset_nozzle.csv"
    
    if not os.path.exists(csv_global_path):
        print("Error: dataset_nozzle.csv not found!")
        return
        
    df_global = pd.read_csv(csv_global_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cases = glob.glob(os.path.join(sim_root, "sim_*"))
    print(f"Found {len(cases)} cases to process.")
    
    success_count = 0
    for case in cases:
        case_name = os.path.basename(case)
        out_path = os.path.join(output_dir, f"{case_name}.pt")
            
        print(f"Processing {case_name}...")
        if process_case_to_graph(case, out_path, df_global):
            success_count += 1
            
    print(f"Successfully processed {success_count}/{len(cases)} cases.")

if __name__ == "__main__":
    main()
