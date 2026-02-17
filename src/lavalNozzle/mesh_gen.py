import os
import subprocess

def create_geo_file(nozzle_file, output_geo,l_convergent, lc_tuyere=0.008, lc_col=0.005, lc_domain=0.15, l_domain=2.0, r_exit=4.0):
    """
    Crée un fichier Gmsh .geo avec raffinement local au bord d'attaque et bord de fuite.
    """
    points = []
    with open(nozzle_file, 'r') as f:
        for line in f:
            points.append(list(map(float, line.split())))
    
    with open(output_geo, 'w') as f:
        f.write(f"lc_tuyere = {lc_tuyere};\n")
        f.write(f"lc_domain = {lc_domain};\n\n")
        f.write(f"l_domain = {l_domain};\n")
        f.write(f"r_exit = {r_exit};\n\n")
        l_outlet = 2*l_domain
        f.write(f"l_outlet = {l_outlet};\n")
        
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

        # 2. Identifier l'index du col (le point le plus proche de l_convergent)
        throat_idx = 1 # Par défaut point 1
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

        # 2. On définit une zone de maillage fin (lc_tuyere) proche de la paroi
        f.write("Field[2] = Threshold;\n")
        f.write("Field[2].IField = 1;\n")
        f.write(f"Field[2].LcMin = {lc_tuyere};\n")      # Taille au contact
        f.write(f"Field[2].LcMax = {lc_domain};\n")      # Taille loin
        f.write("Field[2].DistMin = 0.2;\n")            # Fin jusqu'à 15cm
        f.write("Field[2].DistMax = 0.3;\n")             # Transition vers le gros maillage


        # 3. On raffine spécifiquement le COL (Throat) pour capturer Mach 1
        # On utilise le point du col (milieu du tableau points)
        f.write("Field[3] = Distance;\n")
        f.write(f"Field[3].NodesList = {{throat_idx}};\n") 
        f.write("Field[4] = Threshold;\n")
        f.write("Field[4].IField = 3;\n")
        f.write(f"Field[4].LcMin = {lc_col};\n") # Deux fois plus fin au col
        f.write(f"Field[4].LcMax = {lc_domain};\n")
        f.write("Field[4].DistMin = 0.2;\n")
        f.write("Field[4].DistMax = 0.5;\n")

        # Raffinement le long du jet
        f.write("Field[6] = Box;\n")
        f.write("Field[6].VIn = 0.01;\n")  # Fin dans le jet (optimisé)
        f.write("Field[6].VOut = 0.05;\n")
        f.write(f"Field[6].XMin = {l_domain};\n")   # Commence après le col
        f.write(f"Field[6].XMax = {l_outlet};\n")   # Jusqu'au bout du domaine
        f.write(f"Field[6].YMin = 0;\n")
        f.write(f"Field[6].YMax = 0.4;\n")

        # Champ de Boundary Layer (Couches limites)
        f.write("Field[7] = BoundaryLayer;\n")
        f.write("Field[7].CurvesList = {1};\n") 
        f.write(f"Field[7].PointsList = {{1,{len(points)}}};\n")
        f.write(f"Field[7].FanPointsList = {{1,{len(points)}}};\n")
        f.write("Field[7].Size = 0.0005;\n")    
        f.write("Field[7].Ratio = 1.2;\n")     
        f.write("Field[7].IntersectMetrics = 1;\n")
        f.write("Field[7].Quads = 1;\n")       
        f.write("Field[7].Thickness = 0.01;\n") 

        f.write("BoundaryLayer Field = 7;\n\n")

        # 4. On prend le minimum de tous les raffinements
        f.write("Field[5] = Min;\n")
        f.write("Field[5].FieldsList = {2, 4, 6};\n")
        f.write("Background Field = 5;\n\n")


        f.write("Mesh.Algorithm = 6;\n") # Frontal-Delaunay (très bon pour GNN)

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
    create_geo_file(nozzle_dat, geo_file,l_convergent=0.3, lc_tuyere=0.008, lc_col=0.005, lc_domain=0.15, l_domain=1.3, r_exit=1)
    run_gmsh(geo_file, msh_file)
