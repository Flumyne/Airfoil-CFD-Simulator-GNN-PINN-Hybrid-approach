import os
import random
import time
import subprocess
import shutil
import csv
import re

# Importer nos outils de configuration
from generate_nozzle import generate_nozzle
from mesh_gen import create_geo_file, run_gmsh
from setup_openfoam import setup_case

def clean_simulation_dir(sim_dir):
    """Nettoie en profondeur un répertoire de simulation."""
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(os.path.join(sim_dir, "constant"), exist_ok=True)
    os.makedirs(os.path.join(sim_dir, "system"), exist_ok=True)

def extract_mass_flow(sim_dir):
    """
    Extrait le débit massique final depuis log.simu.
    """
    log_path = os.path.join(sim_dir, "log.simu")
    if not os.path.exists(log_path):
        return None, None
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        inlet_flow = None
        outlet_flow = None
        for line in reversed(lines):
            if "sum(inlet) of phi =" in line:
                inlet_flow = abs(float(line.split('=')[1]))
            if "sum(outlet) of phi =" in line:
                outlet_flow = abs(float(line.split('=')[1]))
            if inlet_flow and outlet_flow:
                break
        return inlet_flow, outlet_flow
    except:
        return None, None

def run_simulation(R_throat, R_exit, R_inlet, L_divergent, L_convergent, sim_id):
    """
    Orchestre le pipeline complet pour une tuyère.
    """
    start_time = time.time()
    
    # Code compact pour le nom du dossier
    nozzle_code = f"T{int(R_throat*1000)}_E{int(R_exit*1000)}_I{int(R_inlet*1000)}"
    case_name = f"sim_{sim_id:04d}_{nozzle_code}"
    
    base_dir = os.getcwd()
    sim_dir = os.path.join(base_dir, "simulations", "nozzle", case_name)
    data_dir = os.path.join(base_dir, "data", "nozzle")
    
    nozzle_file = os.path.join(data_dir, "profiles", f"{case_name}.dat")
    geo_file = os.path.join(data_dir, "geo", f"{case_name}.geo")
    msh_file = os.path.join(data_dir, "mesh", f"{case_name}.msh")
    
    # S'assurer que les répertoires existent
    os.makedirs(os.path.dirname(nozzle_file), exist_ok=True)
    os.makedirs(os.path.dirname(geo_file), exist_ok=True)
    os.makedirs(os.path.dirname(msh_file), exist_ok=True)
    clean_simulation_dir(sim_dir)
    
    print(f"--- Starting Case {sim_id}: {nozzle_code} ---")
    
    try:
        # 1. Générer la géométrie (x, y)
        generate_nozzle(R_throat, R_exit, R_inlet, L_divergent, L_convergent, nozzle_file)
        
        # 2. Maillage avec r_exit adaptatif
        r_domain = R_exit * 3.0
        l_domain = L_convergent + L_divergent
        create_geo_file(nozzle_file, geo_file, L_convergent, l_domain=l_domain, R_exit=r_domain)
        run_gmsh(geo_file, msh_file) 
        
        # 3. Setup OpenFOAM Initial
        setup_case(sim_dir)
        
        # 4. Conversion Maillage et Correction Patchs
        of_source = "source /usr/lib/openfoam/openfoam2512/etc/bashrc"
        cmd_mesh = f"{of_source} && cd {sim_dir} && gmshToFoam {msh_file} > log.gmshToFoam && checkMesh > log.checkMesh"
        subprocess.run(cmd_mesh, shell=True, executable='/bin/bash', check=True)
        
        # Ré-appliquer setup_case pour corriger le fichier boundary écrasé par gmshToFoam
        setup_case(sim_dir)
        
        # 5. Lancer la Simulation (Directement car pas de code dynamique)
        cmd_run = (
            f"{of_source} && cd {sim_dir} && "
            f"decomposePar -force > log.decompose && "
            f"mpirun -np 4 rhoCentralFoam -parallel > log.simu && "
            f"reconstructPar -latestTime >> log.simu"
        )
        subprocess.run(cmd_run, shell=True, executable='/bin/bash', check=True)
        
        # 6. Extraction Résultats
        m_in, m_out = extract_mass_flow(sim_dir)
        
        # 7. Ménage (très important pour le disque !)
        subprocess.run(f"rm -rf {sim_dir}/processor*", shell=True)
        
        duration = time.time() - start_time
        status = "success"
        err_msg = ""
        print(f"   -> Done in {duration:.1f}s. MassFlow Err: {abs(m_in-m_out)/max(m_in,1e-6)*100:.2f}%")
        
    except Exception as e:
        duration = time.time() - start_time
        status = "failed"
        err_msg = str(e)
        m_in, m_out = 0, 0
        print(f"   -> Failed: {e}")

    return {
        "id": sim_id,
        "R_throat": R_throat, "R_exit": R_exit, "R_inlet": R_inlet,
        "L_divergent": L_divergent, "L_convergent": L_convergent,
        "m_in": m_in, "m_out": m_out,
        "status": status, "duration": duration, "error": err_msg
    }

def main():
    # Cible TOTALE d'échantillons réussis/total
    TARGET_TOTAL = 500
    
    # Fichier Dataset
    csv_file = "dataset_nozzle.csv"
    
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
        writer = csv.DictWriter(f, fieldnames=["id", "R_throat", "R_exit", "R_inlet", "L_divergent", "L_convergent", "m_in", "m_out", "status", "duration", "error"])
        if current_count == 0:
            writer.writeheader()
            
        for i in range(remaining):
            current_id = last_id + i + 1
            
            # Paramètres cohérents : col entre 5cm et 15cm
            R_throat = round(random.uniform(0.05, 0.15), 3)
            # Sortie entre 1.5x et 4x le col
            R_exit = round(R_throat * random.uniform(1.5, 4.0), 3)
            # Entrée entre 1.2x le col et R_exit (jamais plus grand que la sortie)
            R_inlet = round(R_throat * random.uniform(1.2, min(2.5, R_exit / R_throat)), 3)
            
            L_convergent = round(random.uniform(0.2, 0.5), 2)
            L_divergent = round(random.uniform(0.6, 1.5), 2)
            
            # Exécuter
            result = run_simulation(R_throat, R_exit, R_inlet, L_divergent, L_convergent, current_id)
            
            # Sauvegarder la ligne
            writer.writerow(result)
            f.flush()

if __name__ == "__main__":
    main()
