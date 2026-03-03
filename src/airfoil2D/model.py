import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm

class GNSBlock(MessagePassing):
    """
    Cette couche implémente la logique "MeshGraphNet" : 
    Elle utilise les features du noeud ET les vecteurs relatifs (edge_attr).
    """
    def __init__(self, hidden_dim):
        super(GNSBlock, self).__init__(aggr='add') # Aggrégation par somme
        
        # MLP pour traiter le message (v_j + edge_attr)
        self.edge_mlp = Seq(Linear(2 * hidden_dim + 3, hidden_dim), # +3 car edge_attr a 3 composantes (dx, dy, dist)
                            ReLU(),
                            Linear(hidden_dim, hidden_dim),
                            LayerNorm(hidden_dim))
        
        # MLP pour la mise à jour du noeud
        self.node_mlp = Seq(Linear(2 * hidden_dim, hidden_dim),
                            ReLU(),
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

class AirfoilGNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=3, num_layers=4):
        super(AirfoilGNN, self).__init__()

        # 1. Encoder : Features physiques -> Espace Latent
        self.encoder = Seq(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )
        
        # 2. Processor : Boucle de Message Passing
        self.processor = nn.ModuleList([
            GNSBlock(hidden_dim) for _ in range(num_layers)
        ])

        # 3. Decoder : Espace Latent -> Physique (P, Ux, Uy)
        self.decoder = Seq(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encoder
        h = self.encoder(x)

        # Traiter (M passes de passage de message)
        for layer in self.processor:
            h = layer(h, edge_index, edge_attr)

        # Décoder
        out = self.decoder(h)
        return out

if __name__ == "__main__":
    from torch_geometric.data import Data
    
    # 1. Instanciation
    # input_dim=10 correspond bien à notre calcul (x, y, m, p, t, dist, 4 types)
    model = AirfoilGNN(input_dim=10, hidden_dim=128, output_dim=3, num_layers=3)
    print("Modèle chargé.")
    
    # 2. Données fictives
    # 4 nœuds, 10 features par nœud
    x = torch.randn(4, 10)
    
    # 4 arêtes (connectivité)
    edge_index = torch.tensor([[1-3]], dtype=torch.long)
    
    # ATTENTION : edge_attr doit avoir une dimension de 3 (dx, dy, distance)
    edge_attr = torch.randn(4, 3) 
    
    # Création de l'objet Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 3. Passage dans le modèle
    output = model(data)
    
    # Vérification
    print("Output shape:", output.shape) 
    # Doit afficher : torch.Size([3, 4]) -> 4 nœuds, 3 valeurs prédites (P, Ux, Uy)
