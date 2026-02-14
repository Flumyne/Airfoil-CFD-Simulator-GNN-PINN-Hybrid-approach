import numpy as np
import matplotlib.pyplot as plt
import os


def generate_nozzle(R_throat, R_exit,R_inlet, L_divergent, L_convergent):
    """
    Génère une tuyère de Laval avec les paramètres suivants :
    - R_inlet : Rayon de l'entrée
    - R_throat : Rayon du col
    - R_exit : Rayon de sortie
    - L_divergent : Longueur du divergent
    - L_convergent : Longueur du convergent
    """
    
    x_convergent = np.linspace(0, L_convergent, 100)
    x_divergent = np.linspace(L_convergent, L_convergent + L_divergent, 100)

    coef_convergent = (R_throat - R_inlet) / L_convergent
    coef_divergent = (R_exit - R_throat) / L_divergent

    y_convergent = R_inlet + coef_convergent * x_convergent
    y_divergent = R_throat + coef_divergent * (x_divergent - L_convergent)

    x_pos = np.concatenate([x_convergent, x_divergent])
    y_pos = np.concatenate([y_convergent, y_divergent])

    y_neg = -y_pos
    x_neg = x_pos

    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    
    return x, y

def save_nozzle(x, y, filename):
    with open(filename, 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")


if __name__ == "__main__":
    x,y = generate_nozzle(1, 4, 2, 1, 1)
    os.makedirs("data/nozzle", exist_ok=True)
    save_nozzle(x, y, "data/nozzle/nozzle_14211.dat")    
    print("Nozzle generated and saved to data/nozzle/nozzle_14211.dat")        
