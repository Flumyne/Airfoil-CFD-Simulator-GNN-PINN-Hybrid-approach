import streamlit as st 
from core.field_viz import compute_field_cached, compute_aerodynamics_coeffs
from core.pinn_model import load_pinn_model
from config import DEFAULT_M, DEFAULT_P, DEFAULT_T, MODEL_PATH, X_MIN, X_MAX, Y_MIN, Y_MAX 
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="Airfoil Model", layout="wide"
)
st.markdown("# 2D PINN Airfoil Model NACA 2412")
st.sidebar.header("2D Airfoil Model")

tab1,tab2 = st.tabs(
    ["Input", "Flow Fields"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_pinn_model(MODEL_PATH, device)

with st.expander("🛠️ Debug : Statistiques du Modèle"):
    st.write("Ces valeurs ne doivent pas être 0 et 1 :")
    st.write(f"**Mu (moyennes) :** {model.mu.cpu().numpy()}")
    st.write(f"**Sigma (std) :** {model.sigma.cpu().numpy()}")

with tab1 :
    input_alpha = st.slider("Choose the angle of attack", -10.0, 15.0, 0.1)
    input_grid_size = st.select_slider("Choose the grid resolution", [256, 512, 1024, 2048])
    if st.button("Run Simulation", width="stretch"):
        start_time = time.time()
        field = compute_field_cached(model, input_alpha, DEFAULT_M, DEFAULT_P, DEFAULT_T, input_grid_size, device)
        st.session_state.u = field["u"]
        st.session_state.v = field["v"]
        st.session_state.p = field["p"]
        st.session_state.x = field["X"]
        st.session_state.y = field["Y"]
        st.session_state.x_a = field["x_airfoil"]
        st.session_state.y_a = field["y_airfoil"]
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.markdown(f"PINN Simulation Time : {elapsed_time} secondes")
        st.markdown("### Aerodynamics Coefficients")
        cl, cd = compute_aerodynamics_coeffs(model, input_alpha, DEFAULT_M, DEFAULT_P, DEFAULT_T, device)
        st.markdown(f"##### CL value = {cl}")
        st.markdown(f"##### CD value = {cd}")
with tab2: 
    st.markdown("### Flow Field Visualisation")
    start_visu_time = time.time()
    if "u" in st.session_state:

        u = st.session_state.u
        v = st.session_state.v
        p = st.session_state.p
        x = st.session_state.x
        y = st.session_state.y
        x_airfoil = st.session_state.x_a
        y_airfoil = st.session_state.y_a

        u_optimum = u[~np.isnan(u)]
        v_optimum = v[~np.isnan(v)]
        p_optimum = p[~np.isnan(p)]

        u_min = u_optimum.min()
        u_max = u_optimum.max()
        v_min = v_optimum.min()
        v_max = v_optimum.max()
        p_min = p_optimum.min()
        p_max = p_optimum.max()

        # --- Plotting ---

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        # 1. PINN Prediction u 
        u_plot = axes[0].scatter(x,y, c=u,alpha=0.5, edgecolors='none', cmap="jet", marker='o', s=2, vmin=u_min, vmax=u_max)
        # Overlay the sparse data points used for training
        axes[0].plot(x_airfoil, y_airfoil, color='black', linewidth=2 )
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlim([X_MIN, X_MAX])
        axes[0].set_ylim([Y_MIN, Y_MAX])
        axes[0].set_title(f"PINN Simulation for u field")
        fig.colorbar(u_plot, ax=axes[0], fraction=0.046, pad=0.04)

        # 2. PINN Prediction v 
        v_plot = axes[1].scatter(x, y, c=v, alpha=0.5, edgecolors='none', cmap="jet", marker='o', s=2, vmin=v_min, vmax=v_max)
        # Overlay the sparse data points used for training
        axes[1].plot(x_airfoil, y_airfoil, color='black', linewidth=2 )
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_xlim([X_MIN, X_MAX])
        axes[1].set_ylim([Y_MIN, Y_MAX])
        axes[1].set_title(f"PINN Simulation for v field")
        fig.colorbar(v_plot, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. PINN Prediction p 
        p_plot = axes[2].scatter(x,y, c=p,alpha=0.5, edgecolors='none', cmap="jet", marker='o', s=2, vmin=p_min, vmax=p_max)
        # Overlay the sparse data points used for training
        axes[2].plot(x_airfoil, y_airfoil, color='black', linewidth=2 )
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_xlim([X_MIN, X_MAX])
        axes[2].set_ylim([Y_MIN, Y_MAX])
        axes[2].set_title(f"PINN Simulation for p field")
        fig.colorbar(p_plot, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)
        end_visu_time = time.time()
        elapsed_visu_time = end_visu_time - start_visu_time
        st.markdown(f"Plot Display Time : {elapsed_visu_time} secondes")
    else : 
        st.markdown("#### Press the button Run Simulation in the first tab ")    