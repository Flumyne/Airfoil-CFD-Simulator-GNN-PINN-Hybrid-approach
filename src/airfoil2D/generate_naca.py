import numpy as np
import matplotlib.pyplot as plt
import os

def generate_naca4(m, p, t, n_points=100):
    """
    Générer les coordonnées pour une aile NACA 4 chiffres.
    m : cambrure max (0 à 0.09)
    p : position cambrure max (0 à 0.9)
    t : épaisseur max (0 à 0.3)
    """
    x = np.linspace(0, 1, n_points)
    
    # Moitié de l'épaisseur du profil
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Ligne de cambrure et gradient
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if m > 0:
        mask1 = (x >= 0) & (x < p)
        mask2 = (x >= p) & (x <= 1)
        
        yc[mask1] = (m / p**2) * (2 * p * x[mask1] - x[mask1]**2)
        dyc_dx[mask1] = (2 * m / p**2) * (p - x[mask1])
        
        yc[mask2] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[mask2] - x[mask2]**2)
        dyc_dx[mask2] = (2 * m / (1 - p)**2) * (p - x[mask2])
    
    theta = np.arctan(dyc_dx)
    
    # Surfaces supérieure et inférieure
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combiner en un seul ensemble de points (sens horaire depuis le bord de fuite)
    # Surface supérieure : BF -> BA
    # Surface inférieure : BA -> BF
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    return x_coords, y_coords

def save_airfoil(x, y, filename):
    with open(filename, 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

def generate_naca(m, p, t, n_points, filename):
    """Fonction pour appels externes"""
    x, y = generate_naca4(m, p, t, n_points)
    save_airfoil(x, y, filename)

if __name__ == "__main__":
    # Exemple : NACA 2412
    m, p, t = 0.02, 0.4, 0.12
    x, y = generate_naca4(m, p, t)
    
    os.makedirs("data/airfoils", exist_ok=True)
    save_airfoil(x, y, "data/airfoils/naca2412.dat")
    
    plt.plot(x, y)
    plt.axis('equal')
    plt.title(f"NACA {int(m*100)}{int(p*10)}{int(t*100):02d}")
    plt.savefig("data/airfoils/naca2412.png")
    print("Airfoil generated and saved to data/airfoils/naca2412.dat")
