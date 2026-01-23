import torch
import time
import os
from model import AirfoilGNN

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Périphérique utilisé : {device}")

    # 1. Charger le modèle
    model = AirfoilGNN(input_dim=10, hidden_dim=128, output_dim=3, num_layers=4).to(device)
    if os.path.exists("airfoil_gnn_best.pt"):
        model.load_state_dict(torch.load("airfoil_gnn_best.pt", map_location=device))
    model.eval()

    # 2. Création d'un graphe fictif représentatif (env. 2000-3000 nœuds comme les simulations)
    # [N, 10] features, [2, E] edges, [E, 3] edge_attr
    num_nodes = 2500
    num_edges = num_nodes * 6 # Car k=6 dans NearestNeighbors
    
    dummy_x = torch.randn(num_nodes, 10).to(device)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
    dummy_edge_attr = torch.randn(num_edges, 3).to(device)
    
    from torch_geometric.data import Data
    data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr)

    print(f"Benchmarking sur un graphe de {num_nodes} nœuds...")

    # Warp-up (nécessaire pour CUDA/GPU)
    for _ in range(10):
        _ = model(data)

    # 3. Mesure précise
    start_time = time.time()
    iterations = 100
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize() # Attendre que le GPU finisse pour mesurer le temps réel

    end_time = time.time()
    avg_time = (end_time - start_time) / iterations

    print("-" * 30)
    print(f"Temps de calcul moyen : {avg_time*1000:.2f} ms")
    print(f"Fréquence : {1/avg_time:.1f} prédictions par seconde")
    print("-" * 30)
    print("Comparaison OpenFOAM : ~120 secondes")
    print(f"Accélération estimée : x{int(120/avg_time):,}")

if __name__ == "__main__":
    benchmark()
