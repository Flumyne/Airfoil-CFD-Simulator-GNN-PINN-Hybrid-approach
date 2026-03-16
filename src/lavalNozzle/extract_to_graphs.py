import os
import pyvista as pv
import torch
from torch_geometric.data import Data
import numpy as np
import glob
#from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from torch_geometric.nn import radius_graph
from torch_geometric.utils import k_hop_subgraph
import re
import pandas as pd
import scipy.optimize as opt
import itertools


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
        

        points_kept = mesh_points.points

        # Supprimer la coordonnée Z
        pos = torch.tensor(points_kept[:, :2], dtype=torch.float) 

        p = torch.tensor(mesh_points.point_data["p"], dtype=torch.float).view(-1, 1)
        u = torch.tensor(mesh_points.point_data["U"][:, :2], dtype=torch.float)
        T = torch.tensor(mesh_points.point_data["T"][:], dtype=torch.float).view(-1, 1)
        rho = torch.tensor(mesh_points.point_data["rho"][:], dtype=torch.float).view(-1, 1)
        
        # Calcul du Mach
        U_mag = np.sqrt(np.sum(mesh_points.point_data["U"][:, :2]**2, axis=1))
        sound_speed = np.sqrt(1.4 * 287 * mesh_points.point_data["T"])
        mach_np = U_mag / sound_speed
        Mach = torch.tensor(mach_np, dtype=torch.float).view(-1, 1)

        node_type = node_type_full

        # --- CORRECTION PHYSIQUE : Forcer Uy=0 sur l'axe de symétrie ---
        mask_sym = (node_type == 4)
        u[mask_sym, 1] = 0.0  # On force la composante Y à zéro
        
        # Distance au mur pour les points conservés
        dist_to_wall = torch.tensor(dist_to_wall_full, dtype=torch.float).view(-1, 1)
        dist_to_symmetry = torch.tensor(dist_to_symmetry_full, dtype=torch.float).view(-1, 1)

        # 4. RECONSTRUCTION DE LA CONNECTIVITÉ (Radius)
        edge_index = radius_graph(pos, r = 0.2, loop = False, max_num_neighbors=16)
        
        # --- Calcul des Edge Features (Vecteur relatif et distance) ---
        
        # 1. On récupère les indices des points des arêtes
        src, dst = edge_index
        
        # 2. On calcule le vecteur du déplacement (u_ij = pos_i - pos_j)
        # 3. On calcule la distance entre les points (d_ij = ||pos_i - pos_j||)
        d_ij = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)

        # 3. Différences de positions (Vecteur relatif non normalisé)
        rel_pos = pos[src] - pos[dst]
        
        # 4. Distance euclidienne
        d_ij = torch.norm(rel_pos, dim=1, keepdim=True)
        
        # 5. Vecteur unitaire (Direction pure)
        u_ij = rel_pos / (d_ij + 1e-8)

        # ── SHOCK DETECTOR (vectorisé, Li & Sun 2024) ──────────────────────────────
        # Source : https://watermark02.silverchair.com/046123_1_5.0200168.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAABaEwggWdBgkqhkiG9w0BBwagggWOMIIFigIBADCCBYMGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMYr446JJdX21sOHWCAgEQgIIFVKEIMECwkpZ05nNN7qipq8OUfrFPPVhYzsWzLK3nQwy7gLFF4Wpov7Ya9BUILTJt5vi54roMCYPG2waBV4KhO8RKUQvmCFaUtUUf-WHFSoXmQNVUWGi0N1DQdFNOJi9bxa3U3TqF9gxlMEUqTyB8cd_9_3eYPhEOE2tu5Bggf2hI5xLQxqkLChPZLsJGMBaeMSQmaxysqrB4Ff8Uh3y4KbWsxDvFcnUpxePkl33Z9qkg7pYe6c4D56XUQ_bhthgXg5qaN42sfjfeetYFYqt_eQdBN1qnouXgQY53CHqMDqu5FaromXcz2pqlhZbYkh8n-r8hidMSrc6J4lCzCYrbmMI9HCnFjPpPfaOcGLzlx6Px0ggo-14ofQfKIvy8ycDdDXDkInc_6UV5OjZawcABTlyWgJZajd2VQz1xDV4B-S1_OfiF_jHSNBNmM_xWoB4h1YGRcZLZa1OaNpYNVTnjNP_2FYCeidMrxG9lN5i6u2DipvLaINDOsLL3-UZyPzqvMPlR5A0H0C7Ubp1wG_O4yokybSi9TPD8Xo4cz_0hz823sz61slUwIzE9e98X-jchJCCdkknfe4WZ057f0QaLuxG1X50QlrQ3fGdxsfUXKLzta3BpTgy-5b8tog2LjGs7dehZdhRRTpTiSVL8WkbbL_8DmACSevF_yC3qbm9iwFkQgsCq2N-bFH_7tj91RxoKPqG8NEs3LJAhZ1juZfcBZtTOYfoJqv6r44lfvuYBjmFug1qdowTqXSaOOHWMuUIIMJJwNOdEvCmouJkGvzeWjEddA80UAcP__UGXUP0WCheS-zPt4D9ENlITJQiN3CqG1i-LHh61k1rxerbuqs3ildD6REIf_usXE_CjDoqZUDuvVIv6qv02QaZKwx14z7ReYqlTYJLSsVe4xvUaW2wCBV7BWwlT_HQPbalKaYIL6gNnhBmGla324e6qFbY-275begc_vHStN16YLPvbImAJ_5nZCgvsQ8x8zDGhI9v2GCgMMCYb790xImWmCapTzuan9Kh2Ff_7mE3bYQ3RX9YmIwTYb2HE7NlICBv8IjXrnKT6jHcxqYHOYgZnhL9UbcEDb2NzTnei9EjjZSe2tdL4BCwqXT0zhISGyhRyV4iyY6fiBMLizlWCu8eCPtTFXeLGbefgK1kKnLhpaGK6bFpbChQE-ucrinNedmBG3k4JMNqV5UA-h-GLZCHJlNIc7iIe22eWgJPM_5U2rpPN8_vujUk8dLrMg1yOE_EEkeAoHHJV34X24z-54x3_YY3CBm_3SRFey2q9a0w5qoUIyLnJk3JXNkeGiGl3GoA-FT8gy7wyNv3WnCIZFUqmYta6M7WoDueJ425X39HhUU7h_JP3UsQoo3zAYEKzrI8sikfdZ16S16Cqd2vE48k2tyinMS7Escg8vPB_JoZt6h26wshD8HtJmt2PzjVm8RJqEiuUsPSFxKs-VYS5TfT5ecCwXy8E2HjepgX_6nRgQnMzKRvW9PwBHRQu4v5QV21hfQisxiBXj-rcubG3l6BE7b2OkXAaeeJIvAnRl0WMaZwsO6iuOEHHQCfK-Y4KTivNvY2X6xHUj2slhxVO-FYemp_HlR2UmQgRGFwFoylTdfyHUQdyC4GMnp7QAop6BjHIvWR7RkXOi3RXwxkXkaRNgN98QCh7paGDRRz7yhpxHLUXwaosb-rVcbO4cETO0Im7M56_bvVot2Te0Mf0U6daZfkG4HFRaidkzh8WIHrLvLVyf9i0yAopEqL-z4P_nsq4CxmPKkUOxiHDHkZZlGoqr4YOScdGuNqMbSjuB4_LdLJsROi2QUDpd_vy

        # Normalisation par l'écart-type local 
        p_std   = p.std().clamp(min=1e-8)
        rho_std = rho.std().clamp(min=1e-8)
        T_std   = T.std().clamp(min=1e-8)
        Ux_std  = u[:, 0].std().clamp(min=1e-8)

        # Concaténation des 4 variables normalisées → shape [N, 4]
        h = torch.cat([
            p   / p_std,
            rho / rho_std,
            T   / T_std,
            (u[:, 0] / Ux_std).view(-1, 1)
        ], dim=1)

        # delta_h_k : [E, 4] — gradient normalisé par arête 
        delta_h_k = (h[src] - h[dst]) / (d_ij + 1e-8)

        epsilon = 1e-8
        num_nodes = pos.shape[0]

        # Étape 1 — Gradient moyen arrivant à chaque nœud r  →  [N, 4]
        dh_sum = torch.zeros(num_nodes, 4)
        count  = torch.zeros(num_nodes, 1)
        # scatter_add_ accumule les gradients entrants par nœud destination
        dh_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 4), delta_h_k)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1))
        dh_mean = dh_sum / (count + epsilon)            # [N, 4]

        # Étape 2 — Pour chaque arête k=(s→r), comparer Δh_k vs gradient moyen du nœud r
        dh_ref  = dh_mean[dst]                          # [E, 4]  (référence au nœud dest)

        # Étape 3 — κ_mn (Eq. 15 du papier) pour chaque arête
        dot     = (delta_h_k * dh_ref).sum(dim=1)       # [E]  produit scalaire
        sq_k    = (delta_h_k ** 2).sum(dim=1)           # [E]  norme² de l'arête
        sq_ref  = (dh_ref ** 2).sum(dim=1)              # [E]  norme² de la référence
        kappa_edge = (2 * dot.abs() + epsilon) / (sq_k + sq_ref + epsilon)  # [E]  ∈ [0,1]

        # Étape 4 — κ_s[r] = min sur toutes les arêtes entrantes de r  →  [N]
        kappa_node = torch.ones(num_nodes)
        # scatter_reduce 'amin' = minimum par nœud destination
        kappa_node.scatter_reduce_(0, dst, kappa_edge, reduce='amin', include_self=True)

        # Étape 5 — r_s[r] = min(1, κ_s[r] / 0.4)  →  [N]
        r_s = (kappa_node / 0.4).clamp(max=1.0)

        # Étape 6 — Signal binaire par nœud : 1 si choc, 0 sinon  →  [N]
        shock_node = (r_s < 1.0).float()

        # Étape 7 — Projection nœud → arête via nœud destination  →  [E, 1]
        shock_detector = shock_node[dst].view(-1, 1)

        # 6. Attributs d'arêtes : [dx, dy, distance, unit_dx, unit_dy, choc_detecteur] -> 6 features
        edge_attr = torch.cat([rel_pos, d_ij, u_ij, shock_detector], dim=1)
        
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

        
        # --- X_normalisé par zone (convergent: -1→0, divergent: 0→1, atmo: 1→...) ---
        pos_x = pos[:, 0]  # tensor [N]

        mask_conv = pos_x <= L_conv
        mask_div  = (pos_x > L_conv) & (pos_x <= L_conv + L_div)
        mask_atm  = pos_x > (L_conv + L_div)

        x_norm = torch.zeros(pos_x.shape[0], dtype=torch.float)
        x_norm[mask_conv] = (pos_x[mask_conv] - L_conv) / L_conv
        x_norm[mask_div]  = (pos_x[mask_div]  - L_conv) / L_div
        x_norm[mask_atm]  = 1.0 + (pos_x[mask_atm] - (L_conv + L_div)) / (L_conv + L_div)

        x_norm = x_norm.view(-1, 1)  # [N, 1]

        # Récupération des conditions limites
        reader.set_active_time_value(0.0)
        mesh = reader.read()
        T_inlet = mesh['boundary']['inlet']['T'][0]
        p_inlet = mesh['boundary']['inlet']['p'][0]
        p_outlet = mesh['boundary']['outlet']['p'][0]
        p_ratio_imposed = p_outlet / p_inlet

        # --- Detection de choc par la théorie --- 

        A_ratio = (R_exit/R_throat)**2
        gamma = 1.4

        def f(M):
            '''
            Relation section-Mach isentropique : A/A* = f(M)
            Formule exacte : (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M²)] ^ ((gamma+1)/(2(gamma-1)))
            '''
            return (1/M)*((2/(gamma+1))*(1 + (gamma -1)/2*M**2))**((gamma+1)/(2*(gamma-1))) - A_ratio

        Mach_1D_sup = opt.brentq(f, 1.001, 15.0)   # Solution supersonique (M > 1)
        Mach_1D_sub = opt.brentq(f, 0.01,  0.999)  # Solution subsonique  (M < 1)

        p_III = p_inlet * (1 + (gamma-1)/(2) * Mach_1D_sup**2)**(-gamma/(gamma-1)) # adapté supersonique

        p_II = p_inlet * (1 + (gamma-1)/(2) * Mach_1D_sub**2)**(-gamma/(gamma-1)) # col sonique et divergent subsonique

        p_shock_exit = p_III * (1 + (2*gamma)/(gamma+1) * (Mach_1D_sup**2 - 1))

        # 0 = subsonique, 1 = choc_divergent, 2 = adapté, 3 = choc_jet
        if p_outlet > p_II:
            regime = 0
        elif p_outlet > p_shock_exit and p_outlet < p_II:           # p_III < p_post_choc < p_outlet < p_II → choc dans divergent
            regime = 1
        elif abs(p_outlet - p_III) / (p_III + 1e-8) < 1e-3:  # ≈ p_III → adapté, tolérance 0.1%
            regime = 2
        else:                                  # p_outlet < p_III → sous-détendu (disques de Mach)
            regime = 3


        regime_one_hot = torch.nn.functional.one_hot(torch.tensor([int(regime)]), num_classes=4).float() 

        # --- Calcul de la position de shock dans le divergent --- #
        # Source : https://www.youtube.com/watch?v=b0wvwkKqoVw&t

        def pressure_ratio_from_given_area_ratio(A_ratio_given):
            # P_ratio (pe/p0) = (pe/p02)*(p02/p2)*(p2/p1)*(p1/p01=p0)

                def f2(M):
                    '''
                    Relation section-Mach isentropique : A/A* = f(M)
                    Formule exacte : (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M²)] ^ ((gamma+1)/(2(gamma-1)))
                    '''
                    return (1/M)*((2/(gamma+1))*(1 + (gamma -1)/2*M**2))**((gamma+1)/(2*(gamma-1))) - (A_ratio_given)
                M1 = opt.brentq(f2, 1.0, 15.0)
                p1_p0 = (1 + (gamma - 1)/2 * M1**2)**((-gamma)/(gamma-1))
                p2_p1 = 1 + (2*gamma)/(gamma + 1)*(M1**2 -1)

                M2 = np.sqrt((1 + (gamma -1)/2 * M1**2)/(gamma*M1**2 - (gamma -1)/2))
                p02_p2 = (1 + (gamma - 1)/2 * M2**2)**(gamma/(gamma-1)) 

                def f3(A_ratio):
                    '''
                    Relation section-Mach isentropique : A/A* = f(M)
                    Formule exacte : (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M²)] ^ ((gamma+1)/(2(gamma-1)))
                    '''
                    return (1/M2)*((2/(gamma+1))*(1 + (gamma -1)/2*M2**2))**((gamma+1)/(2*(gamma-1))) - A_ratio
                A_ratio_post_shock = opt.brentq(f3, 1.0, A_ratio)
                Ae_A2 = A_ratio / A_ratio_given * A_ratio_post_shock

                def f4(M):
                    '''
                    Relation section-Mach isentropique : A/A* = f(M)
                    Formule exacte : (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M²)] ^ ((gamma+1)/(2(gamma-1)))
                    '''
                    return (1/M)*((2/(gamma+1))*(1 + (gamma -1)/2*M**2))**((gamma+1)/(2*(gamma-1))) - Ae_A2

                Me = opt.brentq(f4, 0.001, 0.9999)
                pe_p0e = (1 + (gamma -1)/2 * Me**2)**(-gamma/(gamma-1))
                pe_p0 = pe_p0e*p02_p2*p2_p1*p1_p0
                return pe_p0

        if (regime == 0) :
            x_shock_norm = -1
        elif regime == 2 : 
            x_shock_norm = 1   
        elif regime == 3 : 
            # Formule empirique Ashkenas & Sherman 1966 
            # x_choc/D_e = 0.67* sqrt(p_e/p_inf)
            D_e = 2*R_exit
            p_inf = p_outlet
            p_e = p_III

            x_shock_norm = 1 + D_e*0.67*np.sqrt(p_e/p_inf)
 
        elif regime == 1 :

            A_ratio_shock = opt.brentq(lambda A: pressure_ratio_from_given_area_ratio(A) - p_ratio_imposed, 1.0001, A_ratio-0.0001, xtol=1e-3)

            R_target = R_throat *  np.sqrt(A_ratio_shock)   

            def eq(x): 
                return spline(x) - R_target

            x_shock = opt.brentq(eq,L_conv, L_conv+L_div)
            x_shock_norm = (x_shock - L_conv)/(L_div)

        print(f"  [{case_name}] A_ratio={A_ratio:.3f} | M_sup={Mach_1D_sup:.3f} | M_sub={Mach_1D_sub:.3f} | "
        f"p_II={p_II:.0f} | p_III={p_III:.0f} | p_outlet={p_outlet:.0f} | regime={regime} | pos_shock={x_shock_norm}") 


        # Caractéristiques finales des nœuds (input_dim=8)
        x_features = torch.cat([x_norm, type_one_hot, dist_to_wall, dist_to_symmetry], dim=1)

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
        data.regime = regime_one_hot
        data.shock_pos = torch.tensor([[x_shock_norm]], dtype=torch.float)
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
