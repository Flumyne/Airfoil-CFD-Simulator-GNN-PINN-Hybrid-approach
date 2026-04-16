import sys
import os
from pathlib import Path

# --- PATHS ---
APP_DIR = Path(__file__).parent.absolute()
ROOT_DIR = APP_DIR.parent.parent.absolute()
AIRFOIL_DIR = ROOT_DIR / "src" / "airfoil2D"

# Ensure imports from airfoil2D work
if str(AIRFOIL_DIR) not in sys.path:
    sys.path.append(str(AIRFOIL_DIR))

# --- CONSTANTS ---
MODEL_PATH = ROOT_DIR / "pinn_airfoil_model_V2.pth"

# Physics/Domain
X_MIN, X_MAX = -1.0, 2.0
Y_MIN, Y_MAX = -1.0, 1.0
RHO = 1.0
MU = 0.01

# Default Airfoil (NACA 2412)
DEFAULT_M = 2
DEFAULT_P = 4
DEFAULT_T = 12

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))