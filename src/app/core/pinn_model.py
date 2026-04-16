import streamlit as st
import torch 
import numpy as np
from shapely.geometry import Point
import sys
import os

# Import from config
from config import X_MIN, X_MAX, Y_MIN, Y_MAX

# Import Model architecture and utils from airfoil2D
from src.airfoil2D.PINN_Airfoil import PINN, generate_airfoil

@st.cache_resource
def load_pinn_model(model_path: str, device: str):
    """
    Loads the PINN model weights and sets to eval mode.
    Cached by streamlit to avoid reloading on every interaction.
    """
    model = PINN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_field(model, alpha_deg, m, p, t, grid_size, device): 
    """
    Inference function for the PINN model.
    Returns the predicted fields (u, v, p) on a grid, masked by the airfoil.
    """
    # Grid for global visualization
    x_grid = np.linspace(X_MIN, X_MAX, grid_size)
    y_grid = np.linspace(Y_MIN, Y_MAX, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    alpha_rad = alpha_deg * np.pi / 180

    # Convert to tensors for prediction
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    y_test = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)

    x_test_np = x_test.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    # Generate airfoil geometry for masking
    x_airfoil, y_airfoil, surface = generate_airfoil(m, p, t, device)

    # Prediction
    model.eval()
    with torch.no_grad():
        # Predict fields
        out = model(x_test, y_test, torch.full_like(x_test, alpha_rad)).cpu().numpy()
        u_pred = out[:, 0:1].reshape(grid_size, grid_size)
        v_pred = out[:, 1:2].reshape(grid_size, grid_size)
        p_pred = out[:, 2:3].reshape(grid_size, grid_size)

        # Masking logic
        # Optimize masking: only check points near the airfoil
        mask_grid = np.array([surface.contains(Point(val_x, val_y)) for val_x, val_y in zip(x_test_np.flatten(), y_test_np.flatten())])
        mask_2d = mask_grid.reshape(grid_size, grid_size)
        
        u_pred_masked = u_pred.copy()
        u_pred_masked[mask_2d] = np.nan

        v_pred_masked = v_pred.copy()
        v_pred_masked[mask_2d] = np.nan

        p_pred_masked = p_pred.copy()
        p_pred_masked[mask_2d] = np.nan

    # Convert tensors back to numpy for visualization
    x_airfoil_np = x_airfoil.cpu().detach().numpy()
    y_airfoil_np = y_airfoil.cpu().detach().numpy()

    return {
        "u": u_pred_masked,
        "v": v_pred_masked,
        "p": p_pred_masked,
        "X": X,
        "Y": Y,
        "x_airfoil": x_airfoil_np,
        "y_airfoil": y_airfoil_np
    }