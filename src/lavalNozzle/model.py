import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, SiLU, LayerNorm
from torch_geometric.nn import global_mean_pool

class GNSBlock(MessagePassing):
    """
    Cette couche implémente la logique "MeshGraphNet" : 
    Elle utilise les features du noeud ET les vecteurs relatifs (edge_attr).
    """
    def __init__(self, hidden_dim):
        super(GNSBlock, self).__init__(aggr='add') # Aggrégation par somme
        
        # MLP pour traiter le message (v_i + v_j + edge_attr)
        self.edge_mlp = Seq(Linear(2 * hidden_dim + 5, hidden_dim), # +5 car edge_attr a 5 composantes (dx, dy, dist, n_x, n_y)
                            SiLU(),
                            Linear(hidden_dim, hidden_dim),
                            LayerNorm(hidden_dim))
        
        # MLP pour la mise à jour du noeud
        self.node_mlp = Seq(Linear(2 * hidden_dim, hidden_dim),
                            SiLU(),
                            Linear(hidden_dim, hidden_dim),
                            LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        # On lance la propagation de messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Le message dépend du voisin (x_j), du noeud (x_i) et de la géométrie (edge_attr)
        # Concaténation : [x_i, x_j, edge_attr]
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1) 
        return self.edge_mlp(tmp)

    def update(self, aggr_out, x):
        # Mise à jour du noeud avec la somme des messages reçus
        tmp = torch.cat([x, aggr_out], dim=1)
        return x + self.node_mlp(tmp) # Connexion résiduelle (x + ...)

class NozzleGNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim_local=6, output_dim_global=4, num_layers=5): # input_dim = 8 variable + 8 constantes pas dans x
        super(NozzleGNN, self).__init__()

        # 1. Encoder : Features physiques -> Espace Latent
        self.encoder = Seq(
            Linear(input_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim), 
            LayerNorm(hidden_dim)
        )
        
        # 2. Processor : Boucle de Message Passing
        self.processor = nn.ModuleList([
            GNSBlock(hidden_dim) for _ in range(num_layers)
        ])

        # 3. Decoder : Espace Latent -> Physique
        self.decoder_local = Seq(
            Linear(hidden_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, output_dim_local)
        )

        self.decoder_global = Seq(
            Linear(hidden_dim, hidden_dim//2),
            SiLU(),
            Linear(hidden_dim//2, output_dim_global)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        global_vars = torch.cat([data.case_params, data.bc_params], dim=1)

        x = torch.cat([x, global_vars[batch]], dim=1)

        # Encodage initial
        x = self.encoder(x)
        
        # Passage dans les blocs (Chaque bloc contient sa propre résiduelle interne)
        for block in self.processor:
            x = block(x, edge_index, edge_attr)
        
        out_local = self.decoder_local(x)
        
        pooling = global_mean_pool(x, batch)

        out_global = self.decoder_global(pooling)

        return out_local, out_global

if __name__ == "__main__":
    from torch_geometric.data import Data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Instanciation
    # input_dim=13 correspond bien à notre calcul (x, y, R_inlet, R_throat, R_exit, L_conv, L_div, dist, 5 types)
    model = NozzleGNN(input_dim=12, hidden_dim=128, output_dim=6, num_layers=5).to(device)
    model.load_state_dict(torch.load("nozzle_gnn_best_v14.pt", map_location=device))
    print("Modèle chargé.")
    
    # 2. Données fictives
    # 4 nœuds, 13 features par nœud
    x = torch.randn(4, 13)
    
    # 4 arêtes (connectivité)
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
    
    # ATTENTION : edge_attr doit avoir une dimension de 3 (dx, dy, distance)
    edge_attr = torch.randn(4, 3) 
    
    # Création de l'objet Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 3. Passage dans le modèle
    output = model(data)
    
    # Vérification
    print("Output shape:", output.shape) 
