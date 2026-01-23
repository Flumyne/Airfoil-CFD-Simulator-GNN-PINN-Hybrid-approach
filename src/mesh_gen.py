import os
import subprocess

def create_geo_file(airfoil_file, output_geo, lc_airfoil=0.015, lc_domain=0.5):
    """
    Crée un fichier Gmsh .geo avec raffinement local au bord d'attaque et bord de fuite.
    """
    points = []
    with open(airfoil_file, 'r') as f:
        for line in f:
            points.append(list(map(float, line.split())))
    
    with open(output_geo, 'w') as f:
        f.write(f"lc_airfoil = {lc_airfoil};\n")
        f.write(f"lc_domain = {lc_domain};\n\n")
        
        # Points de l'aile
        # Le point 100 est typiquement le Bord d'Attaque pour NACA 200 pts
        for i, (x, y) in enumerate(points):
            f.write(f"Point({i+1}) = {{{x}, {y}, 0, lc_airfoil}};\n")
        
        # Spline de l'aile
        point_ids = ", ".join([str(i+1) for i in range(len(points))])
        f.write(f"Spline(1) = {{{point_ids}, 1}};\n\n")
        
        # Points du domaine (Champ lointain)
        f.write("Point(1000) = {-10, -10, 0, lc_domain};\n")
        f.write("Point(1001) = {20, -10, 0, lc_domain};\n")
        f.write("Point(1002) = {20, 10, 0, lc_domain};\n")
        f.write("Point(1003) = {-10, 10, 0, lc_domain};\n\n")
        
        # Lignes du domaine
        f.write("Line(2) = {1000, 1001};\n")
        f.write("Line(3) = {1001, 1002};\n")
        f.write("Line(4) = {1002, 1003};\n")
        f.write("Line(5) = {1003, 1000};\n\n")
        
        # Surfaces
        f.write("Curve Loop(1) = {2, 3, 4, 5};\n")
        f.write("Curve Loop(2) = {1};\n")
        f.write("Plane Surface(1) = {1, 2};\n\n")

        # --- CHAMPS DE RAFFINEMENT ---
        # Raffiner le Bord d'Attaque (Point 100) et le Bord de Fuite (Points 1, 200)
        f.write("Field[1] = Distance;\n")
        f.write("Field[1].NodesList = {1, 100, 199};\n") # BF et BA
        f.write("Field[2] = Threshold;\n")
        f.write("Field[2].IField = 1;\n")
        f.write(f"Field[2].LcMin = {lc_airfoil / 4};\n")
        f.write(f"Field[2].LcMax = {lc_domain};\n")
        f.write("Field[2].DistMin = 0.1;\n")
        f.write("Field[2].DistMax = 1.0;\n")
        f.write("Background Field = 2;\n\n")
        
        # Extruder pour OpenFOAM
        f.write("out[] = Extrude {0, 0, 0.1} {\n")
        f.write("  Surface{1}; Layers{1}; Recombine;\n")
        f.write("};\n\n")
        
        # Groupes physiques
        f.write("Physical Surface(\"front\") = {1};\n")
        f.write("Physical Surface(\"back\") = {out[0]};\n")
        f.write("Physical Surface(\"inlet\") = {out[5], out[4], out[2]};\n")
        f.write("Physical Surface(\"outlet\") = {out[3]};\n")
        f.write("Physical Surface(\"airfoil\") = {out[6]};\n")
        f.write("Physical Volume(\"internal\") = {out[1]};\n")

def run_gmsh(geo_file, msh_file):
    cmd = ["gmsh", "-3", geo_file, "-o", msh_file, "-format", "msh2"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    airfoil_dat = "data/airfoils/naca2412.dat"
    geo_file = "data/mesh.geo"
    msh_file = "data/mesh.msh"
    create_geo_file(airfoil_dat, geo_file)
    run_gmsh(geo_file, msh_file)
