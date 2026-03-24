import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generate_naca import generate_naca4
from shapely.geometry import Point, Polygon

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


def calc_loss(x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side):
    
    # --- 1. Loss PDE (Physics) ---
    out = model(x_col, y_col)
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
    out_bc = model(x_bc, y_bc)
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
    loss_wall_top = loss_top_u + loss_top_v

    loss_bot_u = torch.mean((u_pred_bc[mask_side == 3] - u_bc[mask_side == 3]) ** 2)
    loss_bot_v = torch.mean((v_pred_bc[mask_side == 3] - v_bc[mask_side == 3]) ** 2)
    loss_wall_bot = loss_bot_u + loss_bot_v

    nu = mu/rho
    loss_outlet_x = torch.mean((-p_pred_bc[mask_side == 1] + nu*(du_dx_bc[mask_side == 1]))** 2)
    loss_outlet_y = torch.mean((nu*(dv_dx_bc[mask_side == 1]))** 2)
    loss_outlet = loss_outlet_x + loss_outlet_y
    #loss_outlet = torch.mean((p_pred_bc[mask_side == 1]) ** 2)

    out_wall = model(x_airfoil, y_airfoil)
    u_pred_wall = out_wall[:,0:1]
    v_pred_wall = out_wall[:,1:2]
    loss_wall_airfoil_u = torch.mean((u_pred_wall) ** 2)
    loss_wall_airfoil_v = torch.mean((v_pred_wall) ** 2)
    loss_wall_airfoil = loss_wall_airfoil_u + loss_wall_airfoil_v


    # --- Total Loss --
    loss = 10*loss_pde + 5*loss_wall_top + 5*loss_wall_bot + 10*loss_inlet + 2*loss_outlet + 20*loss_wall_airfoil
    return loss, loss_pde, loss_inlet, loss_outlet, loss_wall_top, loss_wall_bot, loss_wall_airfoil


# Configuration for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. Data Generation (Collocation, Boundary, and Sparse Data) ---
x_brut,y_brut = generate_naca4(2,4,12,2000)
x_airfoil,y_airfoil = x_brut, y_brut
surface = Polygon(np.column_stack((x_airfoil, y_airfoil)))

x_airfoil = torch.tensor(x_airfoil, dtype=torch.float32).to(device).view(-1,1)
y_airfoil = torch.tensor(y_airfoil, dtype=torch.float32).to(device).view(-1,1)

x_max_local = 1.1
x_min_local = -0.1
y_max_local = 0.2
y_min_local = -0.2

x_max = 2.0
x_min = -1
y_max = 1
y_min = -1

def data_generation(batch_size = 10000):

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
    u_bc[mask_side == 0] = 4 * u_max * (y_inlet-y_min) * (H - (y_inlet-y_min)) / (H ** 2) # Left boundary
    u_bc[mask_side == 2] = 0 # Bottom boundary
    u_bc[mask_side == 3] = 0 # Top boundary

    v_bc[mask_side == 0] = 0 # Left boundary
    v_bc[mask_side == 2] = 0 # Bottom boundary
    v_bc[mask_side == 3] = 0 # Top boundary

    x_bc = x_bc.requires_grad_(True)
    y_bc = y_bc.requires_grad_(True)

    return x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side

x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side = data_generation(18000)
x_cat = torch.cat([x_col,y_col], dim=1).to(device)
x_norm = Normalizer(x_cat, device=device)

# --- 2. PINN Architecture (MLP with SiLU) ---
hidden_layer = 128 

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.register_buffer('mu', x_norm.mean)
        self.register_buffer('sigma', x_norm.std)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_layer),
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
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, 3)
        )

    def forward(self, x, y):
        # Concatenate x and y for the network input
        inputs = torch.cat([x, y], dim=1)
        inputs_norm = (inputs - self.mu) / self.sigma
        return self.net(inputs_norm)

model = PINN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
optimizer2 = torch.optim.LBFGS(model.parameters(), max_iter=20, history_size=20)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=100)

# --- 4. Training Loop and Loss Function ---

epochs = 3001
loss_history = []
loss_stop = []
loss_pde_history = []
loss_inlet_history = []
loss_outlet_history = []
loss_airfoil_history = []
loss_top_bottom_history = []
rho = 1.0
mu = 0.01

print("Starting training...")

for epoch in range(epochs):
    optimizer.zero_grad()

    x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side = data_generation(18000)
    
    loss, loss_pde, loss_inlet, loss_outlet, loss_wall_top, loss_wall_bot, loss_wall_airfoil = calc_loss(x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side)
    
    loss.backward()
    optimizer.step()
    scheduler.step(loss.detach())
    
    loss_history.append(loss.item())
    loss_pde_history.append(loss_pde.item())
    loss_airfoil_history.append(loss_wall_airfoil.item())
    loss_inlet_history.append(loss_inlet.item())
    loss_outlet_history.append(loss_outlet.item())
    loss_top_bottom_history.append(loss_wall_top.item() + loss_wall_bot.item())
    
    if epoch % 250 == 0:
        loss_stop.append(loss.item())
        print(f"Epoch {epoch}/{epochs} | Loss_Total: {loss.item():.5f} (PDE: {loss_pde:.5f}, Inlet: {loss_inlet:.5f}, Outlet: {loss_outlet:.5f}, Top: {loss_wall_top:.5f}, Bot: {loss_wall_bot:.5f}, Airfoil: {loss_wall_airfoil:.5f})")
        if len(loss_stop) > 1:
            if abs(loss_stop[-1] - loss_stop[-2]) < 1e-5:
                print("Loss stabilize, training finished !")
                break      

print("1St Training finished.")

del optimizer
torch.cuda.empty_cache()
x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side = data_generation(10000)


for epoch2 in range(201):
    def closure() :
        optimizer2.zero_grad()
        loss, loss_pde, loss_inlet, loss_outlet, loss_wall_top, loss_wall_bot, loss_wall_airfoil = calc_loss(x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side)
        loss.backward()
        return loss

    optimizer2.step(closure)

    loss, loss_pde, loss_inlet, loss_outlet, loss_wall_top, loss_wall_bot, loss_wall_airfoil = calc_loss(x_col, y_col, x_bc, y_bc, u_bc, v_bc, p_bc, mask_side)
    loss_history.append(loss.item())
    loss_pde_history.append(loss_pde.item())
    loss_airfoil_history.append(loss_wall_airfoil.item())
    loss_inlet_history.append(loss_inlet.item())
    loss_outlet_history.append(loss_outlet.item())
    loss_top_bottom_history.append(loss_wall_top.item() + loss_wall_bot.item())
    
    if epoch2 % 50 == 0:
        loss_stop.append(loss.item())
        print(f"Epoch {epoch2}/{200} | Loss_Total: {loss.item():.5f} (PDE: {loss_pde:.5f}, Inlet: {loss_inlet:.5f}, Outlet: {loss_outlet:.5f}, Top: {loss_wall_top:.5f}, Bot: {loss_wall_bot:.5f}, Airfoil: {loss_wall_airfoil:.5f})")
        if len(loss_stop) > 1:
            if abs(loss_stop[-1] - loss_stop[-2]) < 1e-5:
                print("Loss stabilize, training finished !")
                break      

print("2nd Training finished.")


torch.save(model.state_dict(), "pinn_airfoil_model.pth")
print("Modèle enregistré avec succès !")


# --- 5. Results and Visualization (Super-Resolution Proof) ---

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
plt.savefig("Residual_PINN.png", dpi=150, bbox_inches='tight')

# Grid for global visualization
grid_size = 512
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)

# Convert to tensors for prediction
x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
y_test = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)

x_test_np =  x_test.detach().cpu().numpy()
y_test_np =  y_test.detach().cpu().numpy()

# Prediction
model.eval()
with torch.no_grad():
    out = model(x_test, y_test).cpu().numpy()
    u_pred = out[:,0:1].reshape(grid_size,grid_size)
    v_pred = out[:,1:2].reshape(grid_size,grid_size)
    p_pred = out[:,2:3].reshape(grid_size,grid_size)
    u_min = u_pred.min()
    u_max = u_pred.max()
    v_min = v_pred.min()
    v_max = v_pred.max()
    p_min = p_pred.min()
    p_max = p_pred.max()
    # Logique de masquage
    mask_grid = np.array([surface.contains(Point(x, y)) for x, y in zip(x_test_np, y_test_np)])
    u_pred_masked = u_pred.copy()
    u_pred_masked[mask_grid.reshape(grid_size, grid_size)] = np.nan

    v_pred_masked = v_pred.copy()
    v_pred_masked[mask_grid.reshape(grid_size, grid_size)] = np.nan

    p_pred_masked = p_pred.copy()
    p_pred_masked[mask_grid.reshape(grid_size, grid_size)] = np.nan

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
plt.savefig("Field_Global_PINN.png", dpi=150, bbox_inches='tight')
