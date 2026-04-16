import streamlit as st 
import numpy as np
from core.pinn_model import predict_field
from src.airfoil2D.PINN_Airfoil import calc_force
from config import MU, RHO

@st.cache_data(ttl=3600)
def compute_field_cached(_model, alpha_deg: float, m: int, p: int, t: int, grid_size: int, device: str):
    """
    Cached function to compute flow fields. 
    """
    return predict_field(_model, alpha_deg, m, p, t, grid_size, device)

def compute_aerodynamics_coeffs(model, alpha_deg: float, m: int, p: int, t: int, device: str): 
    """
    Computes Lift (Cl) and Drag (Cd) coefficients using the PINN's physics-informed force calculation.
    """
    alpha_rad = alpha_deg * np.pi / 180
    cl, cd = calc_force(m, p, t, MU, RHO, alpha_rad, model, device)
    return cl, cd
