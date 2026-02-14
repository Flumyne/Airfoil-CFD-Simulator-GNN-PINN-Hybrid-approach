import os
import random
import time
import subprocess
import shutil
import csv
import re

# Importer nos outils de configuration
from generate_naca import generate_naca
from mesh_gen import create_geo_file, run_gmsh
from setup_openfoam import setup_case

def clean_simulation_dir(sim_dir):
    """Nettoie en profondeur un répertoire de simulation."""
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(os.path.join(sim_dir, "constant"), exist_ok=True)
    os.makedirs(os.path.join(sim_dir, "system"), exist_ok=True)

def parse_force_coeffs(case_dir):
    """
    Analyse la dernière ligne du fichier coefficient.dat pour obtenir Cl et Cd.
    """
    # Trouver le répertoire postProcessing
    pp_dir = os.path.join(case_dir, "postProcessing/forceCoeffs1/0")
    coeff_file = os.path.join(pp_dir, "coefficient.dat")
    
    if not os.path.exists(coeff_file):
        return None, None
        
    try:
        with open(coeff_file, 'r') as f:
            lines = f.readlines()
            # Obtenir la dernière ligne non vide
            last_line = ""
            for line in reversed(lines):
                if line.strip() and not line.strip().startswith('#'):
                    last_line = line
                    break
            
            if not last_line:
                return None, None
                
            # Parser les colonnes (Temps, Cd, Cd(f), Cd(r), Cl, ...)
            parts = last_line.split()
            # Index 1 est Cd, Index 4 est Cl
            cd = float(parts[1])
            cl = float(parts[4])
            return cl, cd
    except Exception as e:
        print(f"Error parsing coefficients: {e}")
        return None, None

def generate_transport_properties(case_dir):
    """
    Génère le fichier constant/transportProperties.
    """
    tp_path = os.path.join(case_dir, "constant/transportProperties")
    with open(tp_path, 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 1.5e-05;
""")

def run_simulation(m, p, t, sim_id):
    """
    Orchestre le pipeline complet pour une aile.
    """
    start_time = time.time()
    
    # 1. Nommage et Chemins
    naca_code = f"{int(m*100)}{int(p*10)}{int(t*100):02d}" 
    case_name = f"sim_{sim_id:04d}_naca_{naca_code}"
    
    base_dir = os.getcwd()
    sim_dir = os.path.join(base_dir, "simulations", case_name)
    data_dir = os.path.join(base_dir, "data")
    airfoil_file = os.path.join(data_dir, "airfoils", f"{case_name}.dat")
    geo_file = os.path.join(data_dir, "geo", f"{case_name}.geo")
    msh_file = os.path.join(data_dir, "mesh", f"{case_name}.msh")
    
    # S'assurer que les répertoires existent
    os.makedirs(os.path.join(data_dir, "airfoils"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "geo"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "mesh"), exist_ok=True)
    
    print(f"--- Starting Case {sim_id}: NACA {naca_code} (m={m}, p={p}, t={t}) ---")
    
    try:
        # 2. Générer la Géométrie
        generate_naca(m, p, t, 200, airfoil_file)
        
        # 3. Générer le Maillage
        create_geo_file(airfoil_file, geo_file)
        run_gmsh(geo_file, msh_file)
        
        # 4. Configurer la Structure du Cas OpenFOAM
        clean_simulation_dir(sim_dir)
        
        # Configurer la Physique
        setup_case(sim_dir)
        generate_transport_properties(sim_dir) 
        
        # Commande source environnement OpenFOAM
        of_source = "source /usr/lib/openfoam/openfoam2512/etc/bashrc"
        
        # Convertir le Maillage (Exécuter dans le répertoire sim)
        # supprimer tout polyMesh existant au cas où
        if os.path.exists(os.path.join(sim_dir, "constant/polyMesh")):
            shutil.rmtree(os.path.join(sim_dir, "constant/polyMesh"))
            
        cmd_mesh = f"{of_source} && cd {sim_dir} && gmshToFoam {msh_file}"
        subprocess.run(cmd_mesh, shell=True, executable='/bin/bash', check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Réappliquer setup_case pour corriger les patchs
        setup_case(sim_dir)
        generate_transport_properties(sim_dir) 
        
        # 5. Lancer la Simulation (Parallèle)
        cmd_run = f"{of_source} && cd {sim_dir} && decomposePar > log.decompose && mpirun -np 4 simpleFoam -parallel > log.simpleFoam && reconstructPar -latestTime > log.reconstructPar"
        subprocess.run(cmd_run, shell=True, executable='/bin/bash', check=True)
        
        # 6. Extraire les Résultats
        cl, cd = parse_force_coeffs(sim_dir)
        
        duration = time.time() - start_time
        print(f"   -> Done in {duration:.1f}s. Cl={cl}, Cd={cd}")
        
        return {
            "id": sim_id,
            "m": m, "p": p, "t": t,
            "cl": cl, "cd": cd,
            "status": "success",
            "duration": duration,
            "error": ""
        }
        
    except Exception as e:
        print(f"   -> FAILED: {e}")
        return {
            "id": sim_id,
            "m": m, "p": p, "t": t,
            "cl": None, "cd": None,
            "status": "failed",
            "duration": time.time() - start_time,
            "error": str(e)
        }

def main():
    # Cible TOTALE d'échantillons réussis/total
    TARGET_TOTAL = 500
    
    # Fichier Dataset
    csv_file = "dataset.csv"
    
    existing_ids = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_ids.append(int(row['id']))
                except:
                    pass
    
    current_count = len(existing_ids)
    last_id = max(existing_ids) if existing_ids else 0
    
    print(f"Current dataset has {current_count} entries. Last ID: {last_id}")
    
    if current_count >= TARGET_TOTAL:
        print(f"Target of {TARGET_TOTAL} already reached. Nothing to do.")
        return

    remaining = TARGET_TOTAL - current_count
    print(f"Starting generation of {remaining} more samples to reach {TARGET_TOTAL}...")
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "m", "p", "t", "cl", "cd", "status", "duration", "error"])
        if current_count == 0:
            writer.writeheader()
            
        for i in range(remaining):
            current_id = last_id + i + 1
            
            # Générer des paramètres NACA aléatoires
            m = round(random.uniform(0.0, 0.08), 3)
            p = round(random.uniform(0.2, 0.6), 1)
            t = round(random.uniform(0.08, 0.18), 2)
            
            # Exécuter
            result = run_simulation(m, p, t, current_id)
            
            # Sauvegarder la ligne
            writer.writerow(result)
            f.flush()

if __name__ == "__main__":
    main()
