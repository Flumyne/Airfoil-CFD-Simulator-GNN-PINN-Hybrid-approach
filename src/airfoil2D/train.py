import torch
from torch_geometric.loader import DataLoader
from dataset import AirfoilDataset
from model import AirfoilGNN
import torch.nn.functional as F
from torch_geometric.utils import scatter
import time
import os
from torch_geometric.data import Data
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt


class Normalizer:
    def __init__(self, tensor=None, mean=None, std=None, device='cpu'):
        super(Normalizer,self).__init__()
        # Si on a une tenseur, on calcule les stats
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

def compute_hybrid_loss(pred, batch, lambda_data=1.0):
    '''
    pred : [N,3] -> (P, Ux, Uy)
    batch : Objet Data contenant x,y(pression, vitesse),node_type,edge_index
    '''

    # --- A. PERTE DONNÉES ---
    mask_internal = (batch.node_type == 0) | (batch.node_type == 3)
    loss_data = torch.nn.functional.mse_loss(pred[mask_internal], batch.y[mask_internal])

    # --- B. PERTE FRONTIÈRE ---
    # 1. INLET (TYPE 1) - Condition de Dirchlet
    mask_inlet = (batch.node_type == 1)
    if mask_inlet.any():
        loss_phy_inlet = torch.nn.functional.mse_loss(pred[mask_inlet], batch.y[mask_inlet])
    else:
        loss_phy_inlet = torch.tensor(0.0, device=device)

    # 2. WALL (TYPE 2) - U = 0
    mask_wall = (batch.node_type == 2)
    if mask_wall.any():
        val_pred_wall = pred[mask_wall, 1:]
        loss_wall = torch.nn.functional.mse_loss(val_pred_wall, batch.y[mask_wall, 1:])
    else:
        loss_wall = torch.tensor(0.0, device=device)

    # --- C. PERTE TOTAL ---
    loss = (lambda_data * loss_data) +  (1.0 * loss_phy_inlet) + (5.0 * loss_wall)
    return loss, loss_data, loss_phy_inlet, loss_wall


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

    '''
    # Données de tests
    dataset = []
    for _ in range(10): # 10 graphes bidons
        pos = torch.randn(100, 2)
        x = torch.cat([pos, torch.randn(100, 8)], dim=1) # 10 caractéristiques
        edge_index = torch.randint(0, 100, (2, 300))
        edge_attr = torch.randn(300, 3) # dx, dy, dist
        y = torch.randn(100, 3) # P, Ux, Uy
        node_type = torch.randint(0, 4, (100,))
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, node_type=node_type) 
        dataset.append(data)

    train_dataset = dataset[:8]
    val_dataset = dataset[8:]
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    '''
    # 2. Charger le Dataset
    dataset = AirfoilDataset("data/graphs")
    
    # Séparer en train/validation (80/20)
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,num_workers=2,pin_memory=True)
   
    # Concatener les données d'entrainement pour avoir mu et sigma
    print("Concatenating training data...")
    all_x = torch.cat([data.x for data in train_dataset], dim=0)
    all_y = torch.cat([data.y for data in train_dataset], dim=0)
    all_edges = torch.cat([data.edge_attr for data in train_dataset], dim=0)
    
    # Crée les normaliseurs
    normalizer_x = Normalizer(tensor = all_x, device=device)
    normalizer_y = Normalizer(tensor = all_y, device=device)
    normalizer_edges = Normalizer(tensor = all_edges, device=device)

    # Sauvegarde des stats pour la validation
    torch.save({
        'x_mean': normalizer_x.mean.cpu(),
        'x_std': normalizer_x.std.cpu(),
        'y_mean': normalizer_y.mean.cpu(),
        'y_std': normalizer_y.std.cpu(),
        'edge_mean': normalizer_edges.mean.cpu(),
        'edge_std': normalizer_edges.std.cpu()
    }, "normalizer_stats.pt")
    print("Normalizer statistics saved to normalizer_stats.pt")
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Modèle, Optimiseur, Perte
    model = AirfoilGNN(input_dim=10, hidden_dim=128, output_dim=3, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # LR initial (sera écrasé/géré par le scheduler)
    epochs = 50
    # Scheduler OneCycle : Super-convergence
    # Max LR = 0.005 pour éviter l'instabilité, augmente puis diminue
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, 
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=epochs,
                                                    pct_start=0.3)
    
    # 4. Boucle d'Entraînement
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)

            # Ajout du bruit
            noise_std = 0.003
            noise = torch.randn_like(batch.x)*noise_std
            batch.x = batch.x + noise

            # Normalise les données
            batch.x = normalizer_x.encode(batch.x)
            batch.edge_attr = normalizer_edges.encode(batch.edge_attr)
            batch.y = normalizer_y.encode(batch.y)

            # Prédiction
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                pred_norm = model(batch)
                loss, loss_data, loss_phy_inlet, loss_wall = compute_hybrid_loss(pred_norm, batch, lambda_data=1.0)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                step_result = scaler.step(optimizer)
                scaler.update()
                
                # Avancer le scheduler seulement si l'optimiseur a avancé
                if step_result is not None:
                    scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.6f}")
            
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
                
                with autocast(device_type=device.type):
                    out = model(batch)
                    loss, loss_data, loss_phy_inlet, loss_wall = compute_hybrid_loss(out, batch, lambda_data=1.0)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {duration:.1f}s")
        
        # Sauvegarder le meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "airfoil_gnn_best.pt")
            print(f"  --> Best model saved with Val Loss: {best_val_loss:.6f}")
            
        # Sauvegarder un checkpoint du modèle
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_checkpoint_e{epoch}.pt")

    print("Training finished!")
    torch.save(model.state_dict(), "airfoil_gnn_final.pt")
    # Après la fin de la boucle for epoch:
    print("Génération du graphique de convergence...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Convergence du GNN (Hybrid Loss)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    plt.savefig('learning_curve_v5_antigravity.png') 
    plt.show() 

if __name__ == "__main__":
    if not os.path.exists("data/graphs"):
        print("Error: No data in data/graphs. Please run src/extract_to_graphs.py first.")
    else:
        train()
