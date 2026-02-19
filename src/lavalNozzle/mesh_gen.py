import os
import subprocess

def create_geo_file(nozzle_file, output_geo, l_convergent, lc_tuyere=None, lc_col=None, lc_domain=None, l_domain=2.0, R_exit=4.0):
    """
    Crée un fichier Gmsh .geo avec raffinement adaptatif à la géométrie.
    Les paramètres de maillage sont calculés automatiquement si non fournis.
    """
    points = []
    with open(nozzle_file, 'r') as f:
        for line in f:
            points.append(list(map(float, line.split())))
    
    # Extraire le rayon du col (minimum de y dans le profil)
    R_throat = min(pt[1] for pt in points)
    R_exit_geom = max(pt[1] for pt in points)
    L_total = max(pt[0] for pt in points)
    
    # --- PARAMÈTRES ADAPTATIFS ---
    # Taille de maille sur la paroi : ~R_throat / 12
    if lc_tuyere is None:
        lc_tuyere = max(0.003, R_throat / 12.0)
    # Taille au col : plus fin (~R_throat / 20)
    if lc_col is None:
        lc_col = max(0.002, R_throat / 20.0)
    # Taille dans le domaine lointain
    if lc_domain is None:
        lc_domain = max(0.05, L_total / 10.0)
    
    # BoundaryLayer adaptatif (plus conservateur pour éviter volumes négatifs)
    bl_first_cell = max(0.0005, R_throat / 80.0)
    bl_thickness = max(0.004, R_throat / 12.0)
    bl_ratio = 1.15 # Ratio plus doux
    
    # Distances de raffinement
    dist_min_wall = max(0.05, R_throat * 1.5)
    dist_max_wall = max(0.1, R_throat * 3.0)
    dist_min_throat = max(0.05, R_throat * 2.0)
    dist_max_throat = max(0.15, R_throat * 5.0)
    
    print(f"  [Mesh] R_throat={R_throat:.4f} lc_tuyere={lc_tuyere:.4f} lc_col={lc_col:.4f} BL_size={bl_first_cell:.5f} BL_thick={bl_thickness:.4f}")
    
    with open(output_geo, 'w') as f:
        f.write(f"lc_tuyere = {lc_tuyere};\n")
        f.write(f"lc_domain = {lc_domain};\n\n")
        f.write(f"l_domain = {l_domain};\n")
        l_outlet = 2*l_domain
        f.write(f"l_outlet = {l_outlet};\n")
        f.write(f"r_exit = {R_exit};\n\n")
        
        # Points de la tuyère 
        for i, (x, y) in enumerate(points):
            f.write(f"Point({i+1}) = {{{x}, {y}, 0, lc_tuyere}};\n")
        
        # Spline de la tuyère Convergente
        point_ids = ", ".join([str(i+1) for i in range(len(points))])
        f.write(f"Spline(1) = {{{point_ids}}};\n\n")
        
        # Points du domaine (Champ lointain et ligne de symétrie)

        f.write(f"Point(1001) = {{0.0, 0.0, 0.0, lc_domain}};\n")
        f.write(f"Point(1002) = {{l_domain, 0.0, 0.0, lc_tuyere}};\n")
        f.write(f"Point(1003) = {{l_outlet, 0.0, 0.0, lc_domain}};\n")    
        f.write(f"Point(1005) = {{l_outlet, r_exit, 0.0, lc_domain}};\n\n")

        
        # Lignes du domaine
        f.write("Line(2) = {1, 1001};\n")
        f.write("Line(3) = {1001, 1002};\n")
        f.write("Line(4) = {1002, 1003};\n")
        f.write("Line(5) = {1003, 1005};\n")
        f.write(f"Line(6) = {{1005, {len(points)}}};\n\n")

        # Surface Domaine
        f.write("Curve Loop(1) = {3, 4, 5, 6, -1, 2};\n")
        f.write("Plane Surface(1) = {1};\n\n")

        # Identifier l'index du col (le point le plus proche de l_convergent)
        throat_idx = 1
        min_diff = float('inf')
        for i, pt in enumerate(points):
            diff = abs(pt[0] - l_convergent)
            if diff < min_diff:
                min_diff = diff
                throat_idx = i + 1 

        f.write(f"throat_idx = {throat_idx};\n")

        # --- STRATÉGIE DE RAFFINEMENT (Dist + Threshold) ---
        # 1. On calcule la distance à la tuyère
        f.write("Field[1] = Distance;\n")
        f.write("Field[1].CurvesList = {1};\n")
        f.write("Field[1].Sampling = 100;\n")

        # 2. Zone de maillage fin proche de la paroi
        f.write("Field[2] = Threshold;\n")
        f.write("Field[2].IField = 1;\n")
        f.write(f"Field[2].LcMin = {lc_tuyere};\n")
        f.write(f"Field[2].LcMax = {lc_domain};\n")
        f.write(f"Field[2].DistMin = {dist_min_wall};\n")
        f.write(f"Field[2].DistMax = {dist_max_wall};\n")

        # 3. Raffinement spécifique au COL (Throat)
        f.write("Field[3] = Distance;\n")
        f.write(f"Field[3].NodesList = {{throat_idx}};\n") 
        f.write("Field[4] = Threshold;\n")
        f.write("Field[4].IField = 3;\n")
        f.write(f"Field[4].LcMin = {lc_col};\n")
        f.write(f"Field[4].LcMax = {lc_domain};\n")
        f.write(f"Field[4].DistMin = {dist_min_throat};\n")
        f.write(f"Field[4].DistMax = {dist_max_throat};\n")

        # Raffinement le long du jet
        jet_height = max(0.2, R_exit_geom * 1.5)
        f.write("Field[6] = Box;\n")
        f.write(f"Field[6].VIn = {max(0.005, lc_tuyere * 1.5)};\n")
        f.write(f"Field[6].VOut = {lc_domain * 0.5};\n")
        f.write(f"Field[6].XMin = {l_domain};\n")
        f.write(f"Field[6].XMax = {l_outlet};\n")
        f.write(f"Field[6].YMin = 0;\n")
        f.write(f"Field[6].YMax = {jet_height};\n")

        # Couches limites (Désactivé pour Laminaire/Robustesse)
        # f.write("Field[7] = BoundaryLayer;\n")
        # f.write("Field[7].CurvesList = {1};\n") 
        # ... (Boundary layer lines commented out)
        
        # Minimum de tous les raffinements (sans BL)
        f.write("Field[5] = Min;\n")
        f.write("Field[5].FieldsList = {2, 4, 6};\n") # On retire 7 de la liste
        f.write("Background Field = 5;\n\n")

        f.write("Mesh.Algorithm = 6;\n") # Frontal-Delaunay

        # Extruder pour OpenFOAM
        f.write("out[] = Extrude {0, 0, 0.1} {\n")
        f.write("  Surface{1}; Layers{1}; Recombine;\n")
        f.write("};\n\n")
        
        # Groupes physiques
        f.write("Physical Surface(\"front\") = {1};\n")
        f.write("Physical Surface(\"back\") = {out[0]};\n")
        f.write("Physical Surface(\"inlet\") = {out[7]};\n")
        f.write("Physical Surface(\"outlet\") = {out[4],out[5]};\n")
        f.write("Physical Surface(\"nozzle\") = {out[6]};\n")
        f.write("Physical Surface(\"symmetry\") = {out[2],out[3]};\n")
        f.write("Physical Volume(\"internal\") = {out[1]};\n")
        
def run_gmsh(geo_file, msh_file):
    cmd = ["gmsh", "-3", geo_file, "-o", msh_file, "-format", "msh2"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    nozzle_dat = "data/nozzle/nozzle_010302103.dat"
    geo_file = "data/mesh.geo"
    msh_file = "data/mesh.msh"
    create_geo_file(nozzle_dat, geo_file, l_convergent=0.3, l_domain=1.3, R_exit=1)
    run_gmsh(geo_file, msh_file)