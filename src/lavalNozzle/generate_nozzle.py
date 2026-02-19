import numpy as np
import os
from scipy.interpolate import PchipInterpolator


def generate_nozzle(R_throat, R_exit,R_inlet, L_divergent, L_convergent,filename):
    """
    Génère une demi-tuyère de Laval avec les paramètres suivants :
    - R_inlet : Rayon de l'entrée
    - R_throat : Rayon du col
    - R_exit : Rayon de sortie
    - L_divergent : Longueur du divergent
    - L_convergent : Longueur du convergent
    """
    
    # Points de contrôle de la courbe
    x_control = np.array([0, L_convergent, L_convergent + L_divergent])
    y_control = np.array([R_inlet, R_throat, R_exit])

    # Interpolation avec spline cubique
    spline = PchipInterpolator(x_control, y_control)

    x = np.linspace(0, L_convergent + L_divergent, 100)
    y = spline(x)
    
    with open(filename, 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    return x, y


if __name__ == "__main__":
    os.makedirs("data/nozzle", exist_ok=True)
    x,y = generate_nozzle(0.1, 0.3, 0.2, 1, 0.3, "data/nozzle/nozzle_010302103.dat")
    print("Nozzle generated and saved to data/nozzle/nozzle_010302103.dat")        
