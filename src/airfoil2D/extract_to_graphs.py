import os
import pyvista as pv
import torch
from torch_geometric.data import Data
import numpy as np
import glob
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import re

def get_latest_time(case_dir):
    """Trouve le répertoire temporel le plus récent dans un cas OpenFOAM."""
    time_dirs = []
    for d in os.listdir(case_dir):
        if (d.isdigit() or (d.replace('.', '', 1).isdigit())) and float(d) > 0:
            time_dirs.append(d)
    
    if not time_dirs:
        return None
    
    # Trier par valeur flottante pour gérer 10, 100, etc. correctement
    time_dirs.sort(key=float)
    return time_dirs[-1]

def process_case_to_graph(case_dir, output_file):
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

        wall_points_all = []

        # Initialiser KDTree pour tous les points
        internal_tree = KDTree(internal_mesh.points)
        
        # Trouver les points du patch de l'aile
        if 'boundary' in mesh.keys():
            boundary_block = mesh['boundary']
            for patch_name in boundary_block.keys():
                patch = boundary_block[patch_name]
                if patch.n_points == 0 : continue
                current_type = -1
                if 'airfoil' in patch_name.lower():
                    current_type = TYPE_WALL
                    wall_points_all.append(patch.points) 
                elif 'inlet' in patch_name.lower():
                    current_type = TYPE_INLET
                elif 'outlet' in patch_name.lower():
                    current_type = TYPE_OUTLET
                else:
                    current_type = TYPE_FLUID
                
                if current_type != -1:
                    _, found_ids = internal_tree.query(patch.points)
                    node_type_full[found_ids] = current_type


        # --- ÉCHANTILLONNAGE INTELLIGENT (BASÉ SUR LA DENSITÉ) ---
        dist_to_wall_full = np.full(mesh_points.n_points, 100.0)
        
        if wall_points_all:
            all_wall_pts = np.concatenate(wall_points_all, axis=0)
            wall_tree = KDTree(all_wall_pts)
            dists, _ = wall_tree.query(mesh_points.points)
            dist_to_wall_full = dists

        # Définir la probabilité d'échantillonnage basée sur la distance
        # Près du mur (< 0.05) : Garder 100% (1/1) -> Couche limite haute fidélité
        # Champ moyen (< 0.5) : Garder 33% (1/3)
        # Champ lointain (> 0.5) : Garder 10% (1/10)
        
        indices_to_keep = []
        for i, d in enumerate(dist_to_wall_full):
            if d < 0.05:
                # Tout garder
                indices_to_keep.append(i)
            elif d < 0.5:
                if i % 3 == 0: 
                    indices_to_keep.append(i)
            else:
                if i % 10 == 0:
                    indices_to_keep.append(i)
        
        indices_to_keep = np.array(indices_to_keep)

        points_kept = mesh_points.points[indices_to_keep]

        # Supprimer la coordonnée Z
        pos = torch.tensor(points_kept[:, :2], dtype=torch.float) 

        p = torch.tensor(mesh_points.point_data["p"][indices_to_keep], dtype=torch.float).view(-1, 1)
        u = torch.tensor(mesh_points.point_data["U"][indices_to_keep, :2], dtype=torch.float)

        node_type = node_type_full[indices_to_keep]
        
        # Distance au mur pour les points conservés (déjà calculée)
        dist_to_wall = torch.tensor(dist_to_wall_full[indices_to_keep], dtype=torch.float).view(-1, 1)

        # 4. RECONSTRUCTION DE LA CONNECTIVITÉ 
        pos_np = pos.numpy()
        nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(pos_np)
        _, indices = nbrs.kneighbors(pos_np)
        indices = indices[:, 1:] # Supprimer les auto-connexions
        sources_nodes = np.repeat(np.arange(len(pos)), 6)
        target_nodes = indices.flatten()
        edge_index = torch.tensor(np.array([sources_nodes, target_nodes]), dtype=torch.long)
        
        # --- Calcul des Edge Features (Vecteur relatif et distance) ---
        
        # 1. On récupère les indices des points des arêtes
        src, dst = edge_index
        
        # 2. On calcule le vecteur du déplacement (u_ij = pos_i - pos_j)
        u_ij = pos[src] - pos[dst]
        
        # 3. On calcule la distance entre les points (d_ij = ||pos_i - pos_j||)
        d_ij = torch.norm(u_ij, dim=1, keepdim=True)

        # Filtre les arêtes avec une distance inférieure à 1e-8
        mask = (d_ij > 1e-8).view(-1)
        edge_index = edge_index[:, mask]
        u_ij = u_ij[mask]
        d_ij = d_ij[mask]
        
        # 4. On normalise le vecteur du déplacement (u_ij = u_ij / d_ij)
        u_ij = u_ij / d_ij

        edge_attr = torch.cat([u_ij, d_ij.view(-1, 1)], dim=1)
        
        type_one_hot = torch.nn.functional.one_hot(node_type, num_classes=4).float()

        # --- Extraction des paramètres NACA pour le contexte global ---
        # Parser le code NACA : NACA MP TT -> m=M/100, p=P/10, t=TT/100
        m_val, p_val, t_val = 0.0, 0.0, 0.0
        match = re.search(r'naca_(\d{4})', case_name)
        if match:
            code = match.group(1)
            m_val = float(code[0]) / 100.0
            p_val = float(code[1]) / 10.0
            t_val = float(code[2:]) / 100.0
        
        # Étendre les paramètres NACA à tous les nœuds
        naca_features = torch.tensor([[m_val, p_val, t_val]], dtype=torch.float).expand(pos.shape[0], -1)
        
        # Caractéristiques finales des nœuds
        x_features = torch.cat([pos, naca_features, type_one_hot, dist_to_wall], dim=1)

        # 6. Construire l'objet Data
        data = Data(x=x_features, edge_index=edge_index, edge_attr=edge_attr, y_p=p, y_u=u, node_type=node_type)
        data.naca_code = case_name # Stocker pour référence

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
    sim_root = "simulations"
    output_dir = "data/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    cases = glob.glob(os.path.join(sim_root, "sim_*"))
    print(f"Found {len(cases)} cases to process.")
    
    success_count = 0
    for case in cases:
        case_name = os.path.basename(case)
        out_path = os.path.join(output_dir, f"{case_name}.pt")
            
        print(f"Processing {case_name}...")
        if process_case_to_graph(case, out_path):
            success_count += 1
            
    print(f"Successfully processed {success_count}/{len(cases)} cases.")

if __name__ == "__main__":
    main()
