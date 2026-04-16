import streamlit as st 

st.set_page_config(
    page_title="Surrogate Model for CFD",
)

st.title("Surrogate Model for CFD by Florian Royon-Chalendard")
st.markdown(" Welcome to my Surrogate Model for CFD Simulation where you can test 2D simulation for naca 2412 airfoil")
st.markdown("### PINN Parametric(New ✨)")
st.markdown(" - **Zero supervised data**: The model learns only from the Navier-Stokes equations and boundary conditions, without any CFD dataset.")
st.markdown(" - **Parametric**: A single model covers a continuous range of angles of attack (AOA ∈ [-10°, 15°]).")
st.markdown(" - **Simulation Time** : 380s for OpenFoam vs 5s for PINN.")
st.markdown(" - **Quantitative validation**: Comparison with OpenFOAM (simpleFoam) for the $(u, v, p)$ fields and the aerodynamic coefficients ($C_l$, $C_d$).")


col1, col2, col3 = st.columns(3)
with col1:
    st.image("Comparison_CFD_PINN_V2_AOA_-4.0.png", caption="Comparison CFD vs PINN for AOA = -4°")
with col2:    
    st.image("Comparison_CFD_PINN_V2_AOA_0.0.png", caption="Comparison CFD vs PINN for AOA = 0°")
with col3:    
    st.image("Comparison_CFD_PINN_V2_AOA_8.0.png", caption="Comparison CFD vs PINN for AOA = 8°")

st.page_link("pages/🛩️_PINN_Airfoil.py", label="Simulator", icon="🛩️")

st.sidebar.success("Select a mode above")

