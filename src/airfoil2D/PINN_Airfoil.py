import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.airfoil2D.generate_naca import generate_naca4
from shapely.geometry import Point, Polygon
import argparse
import pyvista as pv
import os 
import random

# Inspired by "Physics-informed deep learning for simultaneous surrogate modeling and PDE-constrained optimization of an airfoil geometry"
# Yubiao Sun, Ushnish Sengupta, Matthew Juniper

class Normalizer:
    def __init__(self, tensor=None, mean=None, std=None, device='cpu'):
        super(Normalizer,self).__init__()
        # Si on a une tenseur, on calcule les stats
        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).detach().to(device)
            self.std = torch.std(tensor, dim=0).detach().to(device)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = mean.to(device)
            self.std = std.to(device)
        
    def encode(self, x):
        return (x - self.mean) / self.std    
    
    def decode(self, x):
        return x * self.std + self.mean    

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self    

# --- 2. PINN Architecture (MLP with SiLU) ---
hidden_layer = 128 

class FourierEmbedding(nn.Module):
    def __init__(self, n_freq=6):
        super().__init__()
        freqs = 2**torch.linspace(0, n_freq-1, n_freq)
        self.register_buffer('freqs',freqs)

    def forward(self,x):
        x_proj = x.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1).flatten(1)    

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.register_buffer('mu', torch.zeros(3))
        self.register_buffer('sigma', torch.ones(3))
        self.fourier_xy = FourierEmbedding(n_freq=6)
        self.local_net = nn.Sequential(
            nn.Linear(2*2*6, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
        )
        self.global_net = nn.Sequential(
            nn.Linear(1, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, 3)
        )

    def forward(self, x, y, alpha):
        # Concatenate x and y for the network input
        inputs_local = torch.cat([x, y], dim=1)
        alpha_norm = (alpha - self.mu[2:3]) / self.sigma[2:3]

        spatial_fourier = self.fourier_xy(inputs_local)
        spatial = self.local_net(spatial_fourier)
        param = self.global_net(alpha_norm)

        combined = spatial*param

        return self.output_net(combined)

def calc_loss(x_col, y_col, x_bc, y_bc, x_airfoil, y_airfoil, u_bc, v_bc, p_bc, rho, mu, alpha, mask_side, model):
    
    # --- 1. Loss PDE (Physics) ---
    out = model(x_col, y_col, torch.full_like(x_col,alpha))
    u_pred_col = out[:,0:1]
    v_pred_col = out[:,1:2]
    p_pred_col = out[:,2:3]
    
    # Calculate first derivatives (du/dx, du/dy) using autograd
    du_dx = torch.autograd.grad(u_pred_col, x_col, torch.ones_like(u_pred_col), create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u_pred_col, y_col, torch.ones_like(u_pred_col), create_graph=True, retain_graph=True)[0]
    
    # Calculate second derivatives (d2u/dx2, d2u/dy2)
    d2u_dx2 = torch.autograd.grad(du_dx, x_col, torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y_col, torch.ones_like(du_dy), create_graph=True, retain_graph=True)[0]

    # Calculate first derivatives (dv/dx, dv/dy) using autograd
    dv_dx = torch.autograd.grad(v_pred_col, x_col, torch.ones_like(v_pred_col), create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v_pred_col, y_col, torch.ones_like(v_pred_col), create_graph=True, retain_graph=True)[0]
    
    # Calculate second derivatives (d2v/dx2, d2v/dy2)
    d2v_dx2 = torch.autograd.grad(dv_dx, x_col, torch.ones_like(dv_dx), create_graph=True, retain_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy, y_col, torch.ones_like(dv_dy), create_graph=True, retain_graph=True)[0]

    # Calculate first derivatives (dp/dx, dp/dy) using autograd
    dp_dx = torch.autograd.grad(p_pred_col, x_col, torch.ones_like(p_pred_col), create_graph=True, retain_graph=True)[0]
    dp_dy = torch.autograd.grad(p_pred_col, y_col, torch.ones_like(p_pred_col), create_graph=True, retain_graph=True)[0]
    
    # Navier-Stokes incompressible equation 
    # rho u . grad(u) = - grad(p) + mu*grad**2(u)
    # grad(u) = 0
    ns_x = rho*(u_pred_col*du_dx + v_pred_col*du_dy) + dp_dx - mu*(d2u_dx2 + d2u_dy2)
    ns_y = rho*(u_pred_col*dv_dx + v_pred_col*dv_dy) + dp_dy - mu*(d2v_dx2 + d2v_dy2)
    loss_ns = torch.mean(ns_x ** 2) + torch.mean(ns_y ** 2)
    loss_continuity = torch.mean((du_dx + dv_dy) ** 2)
    loss_pde = loss_ns + loss_continuity
    
    # --- 2. Loss BC (Boundary Conditions) ---
    out_bc = model(x_bc, y_bc, torch.full_like(x_bc,alpha))
    u_pred_bc = out_bc[:,0:1]
    v_pred_bc = out_bc[:,1:2]
    p_pred_bc = out_bc[:,2:3]

    # Calculate first derivatives (du/dx, du/dy) using autograd
    du_dx_bc = torch.autograd.grad(u_pred_bc, x_bc, torch.ones_like(u_pred_bc), create_graph=True, retain_graph=True)[0]
    dv_dx_bc = torch.autograd.grad(v_pred_bc, x_bc, torch.ones_like(v_pred_bc), create_graph=True, retain_graph=True)[0]

    loss_inlet_u = torch.mean((u_pred_bc[mask_side == 0] - u_bc[mask_side == 0]) ** 2)
    loss_inlet_v = torch.mean((v_pred_bc[mask_side == 0] - v_bc[mask_side == 0]) ** 2)
    loss_inlet = loss_inlet_u + loss_inlet_v

    loss_top_u = torch.mean((u_pred_bc[mask_side == 2] - u_bc[mask_side == 2]) ** 2)
    loss_top_v = torch.mean((v_pred_bc[mask_side == 2] - v_bc[mask_side == 2]) ** 2)
    loss_top = loss_top_u + loss_top_v

    loss_bot_u = torch.mean((u_pred_bc[mask_side == 3] - u_bc[mask_side == 3]) ** 2)
    loss_bot_v = torch.mean((v_pred_bc[mask_side == 3] - v_bc[mask_side == 3]) ** 2)
    loss_bot = loss_bot_u + loss_bot_v

    nu = mu/rho
    loss_outlet_x = torch.mean((-p_pred_bc[mask_side == 1] + nu*(du_dx_bc[mask_side == 1]))** 2)
    loss_outlet_y = torch.mean((nu*(dv_dx_bc[mask_side == 1]))** 2)
    loss_outlet = loss_outlet_x + loss_outlet_y
    #loss_outlet = torch.mean((p_pred_bc[mask_side == 1]) ** 2)

    out_wall = model(x_airfoil, y_airfoil, torch.full_like(x_airfoil,alpha))
    u_pred_wall = out_wall[:,0:1]
    v_pred_wall = out_wall[:,1:2]
    loss_wall_airfoil_u = torch.mean((u_pred_wall) ** 2)
    loss_wall_airfoil_v = torch.mean((v_pred_wall) ** 2)
    loss_wall_airfoil = loss_wall_airfoil_u + loss_wall_airfoil_v


    # --- Total Loss --
    loss = 15*loss_pde + 4*loss_top + 4*loss_bot + 5*loss_inlet + 1*loss_outlet + 10*loss_wall_airfoil
    return loss, loss_pde, loss_inlet, loss_outlet, loss_top, loss_bot, loss_wall_airfoil

def calc_force(m, p, t, mu, rho, alpha, model, device):

    x_s, y_s = generate_naca4(m, p, t, 1000)

    vec_dx = []
    vec_dy = []
    dl = []
    vec_n = []
    x_mid = []
    y_mid = []
    for i in range(len(x_s)-1):
        vec_dx.append(x_s[i+1] - x_s[i]) 
        vec_dy.append(y_s[i+1] - y_s[i])
        dl.append(np.sqrt(vec_dx[i]**2 + vec_dy[i]**2))
        vec_n.append((vec_dy[i]/dl[i], -vec_dx[i]/dl[i]))
        x_mid.append((x_s[i] + x_s[i+1])/2)
        y_mid.append((y_s[i] + y_s[i+1])/2)
       
    vec_dx = torch.tensor(vec_dx).to(device).view(-1,1)
    vec_dy = torch.tensor(vec_dy).to(device).view(-1,1)
    dl = torch.tensor(dl).to(device).view(-1,1)
    vec_n = torch.tensor(vec_n).to(device)
    x_mid = torch.tensor(x_mid, requires_grad=True, dtype=torch.float32).to(device).view(-1,1)
    y_mid = torch.tensor(y_mid, requires_grad=True, dtype=torch.float32).to(device).view(-1,1)  

    # Prediction
    model.eval()
    out = model(x_mid, y_mid, torch.full_like(x_mid,alpha))
    u_pred = out[:,0:1]
    v_pred = out[:,1:2]
    p_pred = out[:,2:3]

    du_dx = torch.autograd.grad(u_pred, x_mid, torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u_pred, y_mid, torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v_pred, x_mid, torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v_pred, y_mid, torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]

    F_v_x = torch.sum(mu*(du_dx*vec_n[:,0:1] + du_dy*vec_n[:,1:2])* dl)
    F_v_y = torch.sum(mu*(dv_dx*vec_n[:,0:1] + dv_dy*vec_n[:,1:2])* dl)
    Fv = torch.stack([F_v_x.detach().view(1), F_v_y.detach().view(1)]).to(device)
    Fp = torch.sum(-p_pred * vec_n * dl, dim=0)

    F_tot = Fp + Fv
    vec_drag = torch.tensor((np.cos(alpha),np.sin(alpha))).to(device)
    vec_lift = torch.tensor((-np.sin(alpha), np.cos(alpha))).to(device)
    Drag = torch.sum(F_tot * vec_drag)
    Lift = torch.sum(F_tot * vec_lift)

    V_inf = 0.3
    Cl = Lift/(0.5*rho*V_inf**2)
    Cd = Drag/(0.5*rho*V_inf**2)

    return Cl,Cd

# --- 1. Data Generation (Collocation, Boundary, and Sparse Data) ---
def generate_airfoil(m,p,t, device) :
    x_airfoil,y_airfoil = generate_naca4(m,p,t,2000)
    surface = Polygon(np.column_stack((x_airfoil, y_airfoil)))

    x_airfoil = torch.tensor(x_airfoil, dtype=torch.float32).to(device).view(-1,1)
    y_airfoil = torch.tensor(y_airfoil, dtype=torch.float32).to(device).view(-1,1)

    return x_airfoil, y_airfoil, surface

def data_generation(x_airfoil, y_airfoil, surface, batch_size, device, fixed_alpha=None):

    x_max_local = x_airfoil.max() + 0.1
    x_min_local = x_airfoil.min() - 0.1
    y_max_local = y_airfoil.max() + 0.1
    y_min_local = y_airfoil.min() - 0.1

    if fixed_alpha is not None:
        alpha_rad = fixed_alpha
    else:
        alpha = round(random.uniform(-10, 15), 2)
        alpha_rad = alpha * np.pi / 180

    # 1. Collocation Points (Physics Loss): Random points inside the domain
    num_collocation_global = batch_size
    num_collocation_local = int(batch_size/3)

    x_col_global = (x_max - x_min)*torch.rand(num_collocation_global, 1, requires_grad=True).to(device) + x_min
    y_col_global = (y_max - y_min)*torch.rand(num_collocation_global, 1, requires_grad=True).to(device) + y_min

    x_col_local = (x_max_local - x_min_local)*torch.rand(num_collocation_local, 1, requires_grad=True).to(device) + x_min_local
    y_col_local = (y_max_local - y_min_local)*torch.rand(num_collocation_local, 1, requires_grad=True).to(device) + y_min_local

    x_col = torch.cat([x_col_global, x_col_local])
    y_col = torch.cat([y_col_global,y_col_local])

    x_np =  x_col.detach().cpu().numpy()
    y_np =  y_col.detach().cpu().numpy()

    mask = [not surface.contains(Point(x,y)) for x,y in zip(x_np,y_np)]
    x_col = x_col[mask].detach().requires_grad_(True)
    y_col = y_col[mask].detach().requires_grad_(True)


    # 2. Boundary Points (Boundary Condition Loss): 
    num_bc = int(batch_size/4)
    x_bc = (x_max - x_min)*torch.rand(num_bc, 1).to(device) + x_min
    y_bc = (y_max - y_min)*torch.rand(num_bc, 1).to(device) + y_min
    u_bc = torch.ones(num_bc, 1).to(device)
    v_bc = torch.zeros(num_bc, 1).to(device)
    p_bc = torch.zeros(num_bc, 1).to(device)

    # Enforce boundaries
    mask_side = torch.randint(0, 4, (num_bc, 1))
    x_bc[mask_side == 0] = x_min # Left boundary
    x_bc[mask_side == 1] = x_max # Right boundary
    y_bc[mask_side == 2] = y_min # Bottom boundary
    y_bc[mask_side == 3] = y_max # Top boundary

    u_max = 0.3
    H = y_max - y_min
    y_inlet = y_bc[mask_side == 0]
    #profile = 4 * u_max * (y_inlet-y_min) * (H - (y_inlet-y_min)) / (H ** 2) # Left boundary
    u_bc[mask_side == 0] = u_max*np.cos(alpha_rad)
    u_bc[mask_side == 2] = u_max*np.cos(alpha_rad) # Bottom boundary
    u_bc[mask_side == 3] = u_max*np.cos(alpha_rad) # Top boundary

    v_bc[mask_side == 0] = u_max*np.sin(alpha_rad) # Left boundary
    v_bc[mask_side == 2] = u_max*np.sin(alpha_rad) # Bottom boundary
    v_bc[mask_side == 3] = u_max*np.sin(alpha_rad) # Top boundary

    x_bc = x_bc.requires_grad_(True)
    y_bc = y_bc.requires_grad_(True)

    return x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, alpha_rad, mask_side

def train(x_airfoil, y_airfoil, surface, rho, mu, batch_size, device):

    # Compute normalizer stats: use data for x,y and full alpha range for alpha
    x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, alpha, mask_side = data_generation(x_airfoil, y_airfoil, surface, batch_size, device)
    alpha_range = torch.linspace(-10*np.pi/180, 15*np.pi/180, x_col.shape[0]).unsqueeze(1).to(device)
    x_cat = torch.cat([x_col, y_col, alpha_range], dim=1).to(device)
    x_norm = Normalizer(x_cat, device=device)
    model = PINN().to(device)

    with torch.no_grad(): 
        model.mu.copy_(x_norm.mean)
        model.sigma.copy_(x_norm.std)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min = 1e-5)

    # --- 4. Training Loop and Loss Function ---

    epochs = 1500
    loss_history = []
    loss_stop = []
    loss_pde_history = []
    loss_inlet_history = []
    loss_outlet_history = []
    loss_airfoil_history = []
    loss_top_bottom_history = []


    print("Starting training...")

    for epoch in range(epochs):
        optimizer.zero_grad()

        x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, alpha, mask_side = data_generation(x_airfoil, y_airfoil, surface, batch_size, device)
        
        loss, loss_pde, loss_inlet, loss_outlet, loss_top, loss_bot, loss_wall_airfoil = calc_loss(x_col, y_col, x_bc, y_bc, x_airfoil, y_airfoil, u_bc, v_bc, p_bc, rho, mu, alpha, mask_side, model)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        loss_pde_history.append(loss_pde.item())
        loss_airfoil_history.append(loss_wall_airfoil.item())
        loss_inlet_history.append(loss_inlet.item())
        loss_outlet_history.append(loss_outlet.item())
        loss_top_bottom_history.append(loss_top.item() + loss_bot.item())
        
        if epoch % 100 == 0:
            loss_stop.append(loss.item())
            print(f"Epoch {epoch}/{epochs} | Loss_Total: {loss.item():.5f} (PDE: {loss_pde:.5f}, Inlet: {loss_inlet:.5f}, Outlet: {loss_outlet:.5f}, Top: {loss_top:.5f}, Bot: {loss_bot:.5f}, Airfoil: {loss_wall_airfoil:.5f})")
            if len(loss_stop) > 3:
                if abs(loss_stop[-1] - loss_stop[-2]) < 1e-6 and abs(loss_stop[-2] - loss_stop[-3]) < 1e-6:
                    print("Loss stabilize, training finished !")
                    break      

    print("Phase 1 (AdamW) finished.")

    # --- Phase 2: L-BFGS with FIXED multi-angle batch ---
    # All angles in ONE batch → data never changes → L-BFGS is stable
    del optimizer
    torch.cuda.empty_cache()

    angles_deg = np.linspace(-10, 15, 20)
    batch_per_angle = 1200  

    # Generate and concatenate data for ALL angles at once
    all_x_col, all_y_col = [], []
    all_x_bc, all_y_bc, all_u_bc, all_v_bc, all_p_bc = [], [], [], [], []
    all_alpha_col, all_alpha_bc = [], []
    all_mask_side = []
    bc_offset = 0

    for angle_deg in angles_deg:
        angle_rad = angle_deg * np.pi / 180
        x_c, y_c, x_b, y_b, u_b, v_b, p_b, alpha_val, m_side = data_generation(
            x_airfoil, y_airfoil, surface, batch_per_angle, device, fixed_alpha=angle_rad)
        
        all_x_col.append(x_c)
        all_y_col.append(y_c)
        all_x_bc.append(x_b)
        all_y_bc.append(y_b)
        all_u_bc.append(u_b)
        all_v_bc.append(v_b)
        all_p_bc.append(p_b)
        all_alpha_col.append(torch.full_like(x_c, alpha_val))
        all_alpha_bc.append(torch.full_like(x_b, alpha_val))
        all_mask_side.append(m_side)

    # Concatenate everything into fixed tensors
    fix_x_col = torch.cat(all_x_col).detach().requires_grad_(True)
    fix_y_col = torch.cat(all_y_col).detach().requires_grad_(True)
    fix_x_bc = torch.cat(all_x_bc).detach().requires_grad_(True)
    fix_y_bc = torch.cat(all_y_bc).detach().requires_grad_(True)
    fix_u_bc = torch.cat(all_u_bc)
    fix_v_bc = torch.cat(all_v_bc)
    fix_p_bc = torch.cat(all_p_bc)
    fix_mask_side = torch.cat(all_mask_side)

    
    optimizer2 = torch.optim.LBFGS(model.parameters(), max_iter=40, history_size=50, line_search_fn='strong_wolfe')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss = float('inf')

    lbfgs_epochs = 250
    for epoch2 in range(lbfgs_epochs):
        def closure():
            optimizer2.zero_grad()
            total_loss = 0
            for i, angle_deg in enumerate(angles_deg):
                angle_rad = angle_deg * np.pi / 180
                loss_i, *_ = calc_loss(
                    all_x_col[i], all_y_col[i],
                    all_x_bc[i], all_y_bc[i],
                    x_airfoil, y_airfoil,
                    all_u_bc[i], all_v_bc[i], all_p_bc[i],
                    rho, mu, angle_rad, all_mask_side[i], model)
                total_loss = total_loss + loss_i
            total_loss = total_loss / len(angles_deg)
            total_loss.backward()
            return total_loss

        optimizer2.step(closure)

        # Evaluate and log
        total_loss = 0
        total_pde = 0
        total_inlet = 0
        total_outlet = 0
        total_top = 0
        total_bot = 0
        total_airfoil = 0
        for i, angle_deg in enumerate(angles_deg):
            angle_rad = angle_deg * np.pi / 180
            l, l_pde, l_inlet, l_outlet, l_top, l_bot, l_wall_airfoil= calc_loss(
                all_x_col[i], all_y_col[i],
                all_x_bc[i], all_y_bc[i],
                x_airfoil, y_airfoil,
                all_u_bc[i], all_v_bc[i], all_p_bc[i],
                rho, mu, angle_rad, all_mask_side[i], model)
            total_loss += l.item()
            total_pde += l_pde.item()
            total_inlet += l_inlet.item()
            total_outlet += l_outlet.item()
            total_top += l_top.item()
            total_bot += l_bot.item()
            total_airfoil += l_wall_airfoil.item()

        
        avg_loss = total_loss / len(angles_deg)
        avg_pde = total_pde / len(angles_deg)
        avg_inlet = total_inlet / len(angles_deg)
        avg_outlet = total_outlet / len(angles_deg)
        avg_bot = total_bot / len(angles_deg)
        avg_top = total_top / len(angles_deg)
        avg_airfoil = total_airfoil / len(angles_deg)

        # NaN detection
        if np.isnan(avg_loss):
            print(f"  L-BFGS epoch {epoch2}: NaN detected, rolling back.")
            model.load_state_dict(best_state)
            break

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        loss_history.append(avg_loss)
        loss_pde_history.append(avg_pde)
        loss_inlet_history.append(avg_inlet)
        loss_outlet_history.append(avg_outlet)
        loss_airfoil_history.append(avg_airfoil)
        loss_top_bottom_history.append(avg_bot + avg_top)

        if epoch2 % 10 == 0:
            print(f"  L-BFGS epoch {epoch2}/{lbfgs_epochs} | Avg Loss: {avg_loss:.6f} (PDE: {avg_pde:.5f}, Inlet: {avg_inlet:.5f}, Outlet: {avg_outlet:.5f}, Top: {avg_top:.5f}, Bot: {avg_bot:.5f}, Airfoil: {avg_airfoil:.5f})")

    # Restore best model from Phase 2
    model.load_state_dict(best_state)
    print(f"Phase 2 (L-BFGS) finished. Best loss: {best_loss:.6f}")

    torch.save(model.state_dict(), "pinn_airfoil_model_V2.pth")
    print("Modèle enregistré avec succès !")

    visualize_loss(loss_history, loss_pde_history, loss_inlet_history, loss_outlet_history, loss_top_bottom_history, loss_airfoil_history)

# --- 5. Results and Visualization (Super-Resolution Proof) ---
def visualize_loss(loss_history, loss_pde_history, loss_inlet_history, loss_outlet_history, loss_top_bottom_history, loss_airfoil_history):

    fig_res, axes_res = plt.subplots(3, 2, figsize=(12, 8))

    axes_res[0,0].plot(loss_history)
    axes_res[0,0].set_title(f"Total Loss")
    axes_res[0,0].set_yscale('log')

    axes_res[1,0].plot(loss_pde_history)
    axes_res[1,0].set_title(f"PDE")
    axes_res[1,0].set_yscale('log')

    axes_res[2,0].plot(loss_inlet_history)
    axes_res[2,0].set_title(f"Inlet")
    axes_res[2,0].set_yscale('log')

    axes_res[0,1].plot(loss_outlet_history)
    axes_res[0,1].set_title(f"Outlet")
    axes_res[0,1].set_yscale('log')

    axes_res[1,1].plot(loss_top_bottom_history)
    axes_res[1,1].set_title(f"Top and Bottom Wall")
    axes_res[1,1].set_yscale('log')

    axes_res[2,1].plot(loss_airfoil_history)
    axes_res[2,1].set_title(f"Airfoil Surface")
    axes_res[2,1].set_yscale('log')

    plt.tight_layout()
    plt.savefig("Residual_PINN_V2.png", dpi=150, bbox_inches='tight')

def visualise_field(model, x_min, x_max, y_min, y_max, x_airfoil, y_airfoil, alpha, grid_size, device):
    # Grid for global visualization
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Convert to tensors for prediction
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    y_test = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)

    x_test_np =  x_test.detach().cpu().numpy()
    y_test_np =  y_test.detach().cpu().numpy()

    surface = Polygon(np.column_stack((x_airfoil.cpu().numpy(), y_airfoil.cpu().numpy())))

    # Prediction
    model.eval()
    with torch.no_grad():
        out = model(x_test, y_test, torch.full_like(x_test,alpha)).cpu().numpy()
        u_pred = out[:,0:1].reshape(grid_size,grid_size)
        v_pred = out[:,1:2].reshape(grid_size,grid_size)
        p_pred = out[:,2:3].reshape(grid_size,grid_size)

        # Logique de masquage
        mask_grid = np.array([surface.contains(Point(x, y)) for x, y in zip(x_test_np, y_test_np)])
        u_pred_masked = u_pred.copy()
        u_pred_masked[mask_grid.reshape(grid_size, grid_size)] = np.nan

        v_pred_masked = v_pred.copy()
        v_pred_masked[mask_grid.reshape(grid_size, grid_size)] = np.nan

        p_pred_masked = p_pred.copy()
        p_pred_masked[mask_grid.reshape(grid_size, grid_size)] = np.nan

        u_optimum = u_pred_masked[~np.isnan(u_pred_masked)]
        v_optimum = v_pred_masked[~np.isnan(v_pred_masked)]
        p_optimum = p_pred_masked[~np.isnan(p_pred_masked)]

        u_min = u_optimum.min()
        u_max = u_optimum.max()
        v_min = v_optimum.min()
        v_max = v_optimum.max()
        p_min = p_optimum.min()
        p_max = p_optimum.max()

    # --- Plotting ---

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # 1. PINN Prediction u 
    u_plot = axes[0].scatter(X,Y, c=u_pred_masked,alpha=0.5, edgecolors='none', cmap="jet", marker='o', s=2, vmin=u_min, vmax=u_max)
    # Overlay the sparse data points used for training
    axes[0].plot(x_airfoil.cpu(), y_airfoil.cpu(), color='black', linewidth=2 )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlim([x_min, x_max])
    axes[0].set_ylim([y_min, y_max])
    axes[0].set_title(f"PINN Simulation for u field")
    fig.colorbar(u_plot, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. PINN Prediction v 
    v_plot = axes[1].scatter(X, Y, c=v_pred_masked, alpha=0.5, edgecolors='none', cmap="jet", marker='o', s=2, vmin=v_min, vmax=v_max)
    # Overlay the sparse data points used for training
    axes[1].plot(x_airfoil.cpu(), y_airfoil.cpu(), color='black', linewidth=2 )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlim([x_min, x_max])
    axes[1].set_ylim([y_min, y_max])
    axes[1].set_title(f"PINN Simulation for v field")
    fig.colorbar(v_plot, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. PINN Prediction p 
    p_plot = axes[2].scatter(X,Y, c=p_pred_masked,alpha=0.5, edgecolors='none', cmap="jet", marker='o', s=2, vmin=p_min, vmax=p_max)
    # Overlay the sparse data points used for training
    axes[2].plot(x_airfoil.cpu(), y_airfoil.cpu(), color='black', linewidth=2 )
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xlim([x_min, x_max])
    axes[2].set_ylim([y_min, y_max])
    axes[2].set_title(f"PINN Simulation for p field")
    fig.colorbar(p_plot, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("Field_Global_PINN_V2.png", dpi=150, bbox_inches='tight')

def comparison_cfd(m, p, t, rho, mu, model, alpha, of, device, cfd_file):

    if not os.path.exists(cfd_file):
        with open(cfd_file, 'w') as f:
            pass
        
    reader = pv.OpenFOAMReader(cfd_file)

    # AOA = 0°
    reader.set_active_time_value(of)

    # Lire le maillage
    mesh = reader.read()
    internal_mesh = mesh["internalMesh"]    

    coords = torch.tensor(internal_mesh.points, dtype=torch.float).to(device)
    velocity_cfd = internal_mesh.point_data["U"]
    u_cfd = velocity_cfd[:,0:1]
    v_cfd = velocity_cfd[:,1:2]
    p_cfd = internal_mesh.point_data["p"]

    alpha_deg = alpha*180/np.pi

    data = np.loadtxt(f"Validation_PINN_OF/airFoil_PINN_Validation/postProcessing/forces/0/coefficient_{alpha_deg:.0f}.dat", comments='#')
    cl_cfd = data[-1,4]
    cd_cfd = data[-1,1]

    Cl_pred, Cd_pred = calc_force(m, p, t, mu, rho, alpha, model, device)

    print(f'cl_pred = {Cl_pred:.3f}')
    print(f'cl_cfd = {cl_cfd:.3f}')
    print(f'cd_pred = {Cd_pred:.3f}')
    print(f'cd_cfd = {cd_cfd:.3f}')

    # Prediction
    model.eval()
    with torch.no_grad():
        out = model(coords[:,0:1], coords[:,1:2], torch.full_like(coords[:,0:1],alpha)).cpu().numpy()
        u_pred = out[:,0:1]
        v_pred = out[:,1:2]
        p_pred = out[:,2:3]
   

    u_min = u_cfd.min()
    u_max = u_cfd.max()
    v_min = v_cfd.min()
    v_max = v_cfd.max()
    p_min = p_cfd.min()
    p_max = p_cfd.max()

    diff_u = np.linalg.norm(u_cfd.flatten() - u_pred.flatten()) / (np.linalg.norm(u_cfd.flatten()) + 1e-8)
    map_u = (u_cfd.flatten() - u_pred.flatten())/(u_max.flatten())
    diff_v = np.linalg.norm(v_cfd.flatten() - v_pred.flatten()) / (np.linalg.norm(v_cfd.flatten()) + 1e-8)  
    map_v = (v_cfd.flatten() - v_pred.flatten())/(v_max.flatten())
    diff_p = np.linalg.norm(p_cfd.flatten() - p_pred.flatten()) / (np.linalg.norm(p_cfd.flatten()) + 1e-8)  
    map_p = (p_cfd.flatten() - p_pred.flatten())/(p_max.flatten())
    internal_mesh.point_data["error_U"] = map_u
    internal_mesh.point_data["u_cfd"] = u_cfd.flatten()
    internal_mesh.point_data["u_PINN"] = u_pred.flatten()
    internal_mesh.point_data["error_V"] = map_v
    internal_mesh.point_data["v_cfd"] = v_cfd.flatten()
    internal_mesh.point_data["v_PINN"] = v_pred.flatten()
    internal_mesh.point_data["error_p"] = map_p
    internal_mesh.point_data["p_cfd"] = p_cfd.flatten()
    internal_mesh.point_data["p_PINN"] = p_pred.flatten()

    err_cl = abs(Cl_pred - cl_cfd) / abs(cl_cfd) * 100 if cl_cfd != 0 else 0
    err_cd = abs(Cd_pred - cd_cfd) / abs(cd_cfd) * 100 if cd_cfd != 0 else 0

    title_text = (f"Cl CFD: {cl_cfd:.3f} vs PINN: {float(Cl_pred):.3f} (Err: {err_cl:.1f}%) | "
                  f"Cd CFD: {cd_cfd:.3f} vs PINN: {float(Cd_pred):.3f} (Err: {err_cd:.1f}%)")

    plotter = pv.Plotter(off_screen=True, shape=(3,3), window_size=[1920, 1080])
    
    # Place le titre dans le subplot du milieu (centré par défaut sur la case)
    plotter.subplot(0, 1)
    plotter.add_text(title_text, position='upper_edge', font_size=7, color="black", shadow=True)
    
    # Rebascule sur le premier subplot pour les graphiques
    plotter.subplot(0, 0)
    plotter.add_mesh(internal_mesh.copy(), scalars="error_U", cmap="coolwarm", scalar_bar_args={"title": "Error U "})
    plotter.add_text(f"Relative L2 Error for u: {diff_u*100:.2f}%")
    plotter.view_xy()
    plotter.subplot(0,1)
    plotter.add_mesh(internal_mesh.copy(), scalars="u_cfd", cmap="jet", scalar_bar_args={"title": " u_cfd (m/s)"}, clim=[u_min, u_max])
    plotter.view_xy()
    plotter.subplot(0,2)
    plotter.add_mesh(internal_mesh.copy(), scalars="u_PINN", cmap="jet", scalar_bar_args={"title": " u_pinn (m/s)"}, clim=[u_min, u_max])
    plotter.view_xy()

    plotter.subplot(1,0)
    plotter.add_mesh(internal_mesh.copy(), scalars="error_V", cmap="coolwarm", scalar_bar_args={"title": "Error V "})
    plotter.add_text(f"Relative L2 Error for v: {diff_v*100:.2f}%")
    plotter.view_xy()
    plotter.subplot(1,1)
    plotter.add_mesh(internal_mesh.copy(), scalars="v_cfd", cmap="jet", scalar_bar_args={"title": " v_cfd (m/s)"}, clim=[v_min, v_max])
    plotter.view_xy()
    plotter.subplot(1,2)
    plotter.add_mesh(internal_mesh.copy(), scalars="v_PINN", cmap="jet", scalar_bar_args={"title": " v_pinn (m/s)"}, clim=[v_min, v_max])
    plotter.view_xy()

    plotter.subplot(2,0)
    plotter.add_mesh(internal_mesh.copy(), scalars="error_p", cmap="coolwarm", scalar_bar_args={"title": "Error p "})
    plotter.add_text(f"Relative L2 Error for p: {diff_p*100:.2f}%")
    plotter.view_xy()
    plotter.subplot(2,1)
    plotter.add_mesh(internal_mesh.copy(), scalars="p_cfd", cmap="jet", scalar_bar_args={"title": " p_cfd (m/s)"}, clim=[p_min, p_max])
    plotter.view_xy()
    plotter.subplot(2,2)
    plotter.add_mesh(internal_mesh.copy(), scalars="p_PINN", cmap="jet", scalar_bar_args={"title": " p_pinn (m/s)"}, clim=[p_min, p_max])
    plotter.view_xy()

    
    plotter.screenshot(f"Comparison_CFD_PINN_V2_AOA_{alpha_deg:.1f}.png")
    plotter.close()

        
if __name__ == "__main__":

    # --- Argument Definition ---
    parser = argparse.ArgumentParser(description= "PINN Airfoil Simulator")
    parser.add_argument('-t', action='store_true', help='Launch Training')
    parser.add_argument('-v', type=str, help='Path to the model .pth')
    parser.add_argument('-c', type=str, help='Path to the model .pth')
    args = parser.parse_args()

    # Configuration for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    x_max = 2.0
    x_min = -1
    y_max = 1
    y_min = -1

    m = 2 
    p = 4 
    t = 12

    rho = 1.0
    mu = 0.01
    
    x_airfoil, y_airfoil, surface = generate_airfoil(m, p, t, device)

    # Entrainement 
    if args.t:
        print("Training Mode Starting...")
        batch_size = 18000
        train(x_airfoil, y_airfoil, surface, rho, mu, batch_size, device)


    # Visulisation 
    elif args.v:
        print("Visualization Mode launch ")
        alpha = 5.0
        alpha_rad = alpha*np.pi/180
        x_airfoil = x_airfoil.cpu()
        y_airfoil = y_airfoil.cpu()
        model = PINN().to(device)
        model.load_state_dict(torch.load(args.v))
        visualise_field(model, x_min, x_max, y_min, y_max, x_airfoil, y_airfoil, alpha_rad, grid_size=1024, device=device )

    # Comparaison 
    elif args.c:
        print("Comparison Mode launch ")
        model = PINN().to(device)
        model.load_state_dict(torch.load(args.c))

        # AOA = 0
        alpha = 0.0
        alpha_rad = alpha*np.pi/180
        comparison_cfd(m, p, t, rho, mu, model, alpha_rad, 2752, device, "Validation_PINN_OF/airFoil_PINN_Validation/airfoil_PINN_Validation.foam")
       
        # AOA = -4
        alpha = -4.0
        alpha_rad = alpha*np.pi/180
        comparison_cfd(m, p, t, rho, mu, model, alpha_rad, 2775, device, "Validation_PINN_OF/airFoil_PINN_Validation/airfoil_PINN_Validation.foam"  )

        # AOA = 8
        alpha = 8.0
        alpha_rad = alpha*np.pi/180
        comparison_cfd(m, p, t, rho, mu, model, alpha_rad, 2978, device, "Validation_PINN_OF/airFoil_PINN_Validation/airfoil_PINN_Validation.foam"  )
      