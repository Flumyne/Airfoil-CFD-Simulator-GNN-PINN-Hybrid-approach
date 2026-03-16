import torch
from torch_geometric.loader import DataLoader
from dataset import NozzleDataset
from model import NozzleGNN
import torch.nn.functional
from torch_geometric.utils import scatter
import time
import os
from torch_geometric.data import Data
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt


class Normalizer:
    def __init__(self, tensor=None, mean=None, std=None, device='cpu'):
        super(Normalizer,self).__init__()
        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
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

def compute_hybrid_loss(pred_norm_local, pred_norm_global, batch, normalizer_y):
    '''
    pred : [N,6] -> (P, Ux, Uy, T, rho, Mach)
    batch : Simu OpenFOAM
    '''
    # FIX 1 : device inféré depuis les tenseurs (pas hardcodé 'cpu')
    device = pred_norm_local.device

    # --- A. PERTE DONNÉES ---
    mask_internal = (batch.node_type == 0)
    loss_data = (torch.nn.functional.l1_loss(pred_norm_local[mask_internal], batch.y[mask_internal]))

    # --- B. PERTE FRONTIÈRE ---
    # 1. INLET (TYPE 1) - Condition de Dirchlet Pression et Temperature
    mask_inlet = (batch.node_type == 1)
    if mask_inlet.any():
        loss_phy_inlet = torch.nn.functional.l1_loss(pred_norm_local[mask_inlet, 0], batch.y[mask_inlet,0]) + torch.nn.functional.l1_loss(pred_norm_local[mask_inlet, 3], batch.y[mask_inlet,3])
    else:
        loss_phy_inlet = torch.tensor(0.0, device=device)

    # 2. WALL (TYPE 2) - U slip, V slip = 0
    mask_wall = (batch.node_type == 2)
    if mask_wall.any():
        loss_wall = torch.nn.functional.l1_loss(pred_norm_local[mask_wall, 1], batch.y[mask_wall, 1]) + torch.nn.functional.l1_loss(pred_norm_local[mask_wall, 2], batch.y[mask_wall, 2])
    else:
        loss_wall = torch.tensor(0.0, device=device)

    # 3. SYMMETRY U_y = 0
    mask_symmetry = (batch.node_type == 4)
    if mask_symmetry.any():
        # On calcule la cible normalisée qui correspond à Uy_physique = 0
        uy_target_norm = (0.0 - normalizer_y.mean[2]) / normalizer_y.std[2]
        # On utilise L1_loss pour rester à la même échelle que loss_data
        loss_symmetry = torch.nn.functional.l1_loss(
            pred_norm_local[mask_symmetry, 2], 
            torch.full_like(pred_norm_local[mask_symmetry, 2], uy_target_norm)
        )
    else:
        loss_symmetry = torch.tensor(0.0, device=device)

    # 4. PARAMS (TYPE 3) - Poussée, ISP, p_ratio, m_dot (espace normalisé global)

    loss_thrust  = torch.nn.functional.l1_loss(pred_norm_global[:, 0], batch.global_params[:, 0])
    loss_isp     = torch.nn.functional.l1_loss(pred_norm_global[:, 1], batch.global_params[:, 1])
    loss_p_ratio = torch.nn.functional.l1_loss(pred_norm_global[:, 2], batch.global_params[:, 2])
    loss_mdot    = torch.nn.functional.l1_loss(pred_norm_global[:, 3], batch.global_params[:, 3])
    loss_params = 0.25*loss_mdot + 0.25*loss_p_ratio + 0.25*loss_thrust + 0.25*loss_isp

    # 5. OUTLET
    mask_outlet = (batch.node_type == 3)
    if mask_outlet.any():
        loss_outlet = torch.nn.functional.l1_loss(pred_norm_local[mask_outlet], batch.y[mask_outlet])
    else:
        loss_outlet = torch.tensor(0.0, device=device)

    # --- C. PERTE DE GRADIENT MACH ---
    pred_phys = normalizer_y.decode(pred_norm_local)
    batch_phys = normalizer_y.decode(batch.y)
    
    mach_pred = pred_phys[:, 5]
    mach_true = batch_phys[:, 5]
    
    src, dst = batch.edge_index
 
    grad_pred = mach_pred[dst] - mach_pred[src]
    grad_true = mach_true[dst] - mach_true[src] 

    shock_mask_low = torch.abs(grad_true) < 0.1
    shock_mask_medium = (torch.abs(grad_true) > 0.1) & (torch.abs(grad_true) < 0.5)

    if shock_mask_medium.any():
        loss_grad_shock_medium = torch.nn.functional.l1_loss(grad_pred[shock_mask_medium], grad_true[shock_mask_medium])
    else:
        loss_grad_shock_medium = torch.tensor(0.0, device=device)

    if shock_mask_low.any():    
        loss_grad_shock_low = torch.nn.functional.l1_loss(grad_pred[shock_mask_low], grad_true[shock_mask_low])
    else:
        loss_grad_shock_low = torch.tensor(0.0, device=device)

    loss_grad = 2*loss_grad_shock_low + 1*loss_grad_shock_medium 
    

    # --- D. PERTE TOTAL ---
    loss = (2.5 * loss_data) + (0.5 * loss_phy_inlet) + (1.5* loss_wall) + (1.0 * loss_symmetry) + (0.5 * loss_outlet) + (1.0 * loss_grad) + (0.1*loss_params)
    return loss, loss_data, loss_phy_inlet, loss_wall, loss_symmetry, loss_outlet, loss_grad, loss_params


def train():

    # 1. Configuration du périphérique
    device = torch.device('cpu')
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 6:
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)} (Compute {major}.{minor})")
        else:
            print(f"GPU {torch.cuda.get_device_name(0)} (Compute {major}.{minor}) is too old. Falling back to CPU.")
    
    print(f"Final device selection: {device}")

    scaler = GradScaler('cuda') if device.index is not None or device.type == 'cuda' else None

    # 2. Charger le Dataset
    #dataset = NozzleDataset("data/graphs/nozzle")
    # Pour google Colab
    #dataset = NozzleDataset(root_dir="/content/data/graphs/nozzle")
    # Pour Kaggle Notebook
    dataset = NozzleDataset(root_dir="/kaggle/input/datasets/florianroyon/graphs-v16/data/graphs/nozzle")
    # Séparer en train/validation (75/25) 
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.75)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
   
    # Sauvegarde/Chargement des stats pour la validation
    stats_path = "normalizer_stats_nozzle.pt"
    if os.path.exists(stats_path):
        print(f"Loading existing normalizer stats from {stats_path}...")
        stats = torch.load(stats_path, map_location=device)
        normalizer_x = Normalizer(mean=stats['x_mean'], std=stats['x_std'], device=device)
        normalizer_y = Normalizer(mean=stats['y_mean'], std=stats['y_std'], device=device)
        normalizer_edges = Normalizer(mean=stats['edge_mean'], std=stats['edge_std'], device=device)
        normalizer_bc = Normalizer(mean=stats['bc_mean'], std=stats['bc_std'], device=device)
        normalizer_case = Normalizer(mean=stats['case_mean'], std=stats['case_std'], device=device)
        normalizer_global = Normalizer(mean=stats['global_mean'], std=stats['global_std'], device=device)
    else:
        print("Concatenating training data and computing stats...")
        all_x = torch.cat([data.x for data in train_dataset], dim=0)
        all_y = torch.cat([data.y for data in train_dataset], dim=0)
        all_edges = torch.cat([data.edge_attr for data in train_dataset], dim=0)
        all_bc = torch.cat([data.bc_params for data in train_dataset], dim=0)
        all_case = torch.cat([data.case_params for data in train_dataset], dim=0)
        all_global = torch.stack([torch.tensor([data.thrust, data.isp, data.p_ratio, data.m_dot], device=device) for data in train_dataset])        
        normalizer_x = Normalizer(tensor=all_x, device=device)
        normalizer_y = Normalizer(tensor=all_y, device=device)
        normalizer_edges = Normalizer(tensor=all_edges, device=device)
        normalizer_bc = Normalizer(tensor=all_bc, device=device)
        normalizer_case = Normalizer(tensor=all_case, device=device)
        normalizer_global = Normalizer(tensor=all_global, device=device)

        torch.save({
            'x_mean': normalizer_x.mean.cpu(),
            'x_std': normalizer_x.std.cpu(),
            'y_mean': normalizer_y.mean.cpu(),
            'y_std': normalizer_y.std.cpu(),
            'edge_mean': normalizer_edges.mean.cpu(),
            'edge_std': normalizer_edges.std.cpu(),
            'bc_mean': normalizer_bc.mean.cpu(),
            'bc_std': normalizer_bc.std.cpu(),
            'case_mean': normalizer_case.mean.cpu(),
            'case_std': normalizer_case.std.cpu(),
            'global_mean': normalizer_global.mean.cpu(),
            'global_std': normalizer_global.std.cpu()
        }, stats_path)
        print(f"Normalizer statistics saved to {stats_path}")
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = NozzleGNN(input_dim=21, hidden_dim=64, output_dim_local=6, output_dim_global=4, num_layers=5).to(device)  
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  
    epochs = 150
    # Scheduler CosineAnnealingWarmRestarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=0.00005)
    
    # 4. Boucle d'Entraînement
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Système de reprise (Resume)
    start_epoch = 1
    checkpoint_path = "nozzle_gnn_last_v33.pt"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Restarting at epoch {start_epoch} with Best Val Loss: {best_val_loss:.6f}")

    print(f"Starting training from epoch {start_epoch} to {epochs}...")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)

            # Ajout du bruit (Uniquement colonnes 0,9, 10 : X_norm et distances)
            noise_std = 0.006
            cols_idx = [0,6, 7]
            
            x_noisy = batch.x.clone()
            noise = torch.randn(x_noisy.size(0), len(cols_idx), device=device) * noise_std
            x_noisy[:, cols_idx] = x_noisy[:, cols_idx] + noise
            batch.x = x_noisy

            # Normalise les données
            batch.x = normalizer_x.encode(batch.x)
            batch.edge_attr = normalizer_edges.encode(batch.edge_attr)
            batch.y = normalizer_y.encode(batch.y)
            batch.bc_params = normalizer_bc.encode(batch.bc_params)
            batch.case_params = normalizer_case.encode(batch.case_params)
            # torch.stack -> [batch_size, 4] : ordre [thrust, isp, p_ratio, m_dot]
            batch.global_params = torch.stack([batch.thrust, batch.isp, batch.p_ratio, batch.m_dot], dim=1)
            batch.global_params = normalizer_global.encode(batch.global_params)

            # Prédiction
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                pred_norm_local, pred_norm_global = model(batch)
                loss, loss_data, loss_phy_inlet, loss_wall, loss_symmetry, loss_outlet, loss_grad, loss_params = compute_hybrid_loss(pred_norm_local, pred_norm_global, batch, normalizer_y)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Gradient Clipping (Priorité Haute) pour éviter les pics de loss
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                step_result = scaler.step(optimizer)
                scaler.update()
                
                # Avancer le scheduler seulement si l'optimiseur a avancé
                if step_result is not None:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.6f} | Data Loss: {loss_data.item():.6f} | Wall Loss: {loss_wall.item():.6f} | Outlet Loss: {loss_outlet.item():.6f} | Inlet Loss: {loss_phy_inlet.item():.6f} | Sym Loss: {loss_symmetry.item():.6f} | Grad Loss: {loss_grad.item():.6f} | Params Loss: {loss_params.item():.6f} ")
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Normaliser la cible
                batch.x = normalizer_x.encode(batch.x)
                batch.edge_attr = normalizer_edges.encode(batch.edge_attr)
                batch.y = normalizer_y.encode(batch.y)
                batch.bc_params = normalizer_bc.encode(batch.bc_params)
                batch.case_params = normalizer_case.encode(batch.case_params)
                # torch.stack -> [batch_size, 4] : ordre [thrust, isp, p_ratio, m_dot]
                batch.global_params = torch.stack([batch.thrust, batch.isp, batch.p_ratio, batch.m_dot], dim=1)
                batch.global_params = normalizer_global.encode(batch.global_params)
                
                with autocast(device_type=device.type):
                    out_local, out_global = model(batch)
                    loss, loss_data, loss_phy_inlet, loss_wall, loss_symmetry, loss_outlet, loss_grad, loss_params = compute_hybrid_loss(out_local, out_global, batch, normalizer_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {duration:.1f}s")
        
        # Sauvegarder le meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "nozzle_gnn_best_v33.pt")
            print(f"  --> Best model saved with Val Loss: {best_val_loss:.6f}")
        
        # Sauvegarder systématiquement le dernier état (checkpoint)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }, "nozzle_gnn_last_v33.pt")
            
    print("Training finished!")
    # Après la fin de la boucle for epoch:
    print("Génération du graphique de convergence...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Nozzle Training Convergence (v33)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    plt.savefig('learning_curve_nozzle_v33.png') 
    plt.show() 

if __name__ == "__main__":
    train()
