import os
import torch
from torch_geometric.data import Dataset
import glob

class AirfoilDataset(Dataset):
    """
    Classe pour charger les graphes d'ailes traités (fichiers .pt).
    """
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super(AirfoilDataset, self).__init__(root_dir, transform, pre_transform)
        self.file_list = glob.glob(os.path.join(root_dir, "*.pt"))
        
    def len(self):
        return len(self.file_list)

    def get(self, idx):
        data = torch.load(self.file_list[idx], weights_only=False)
        
        # Combiner y_p et y_u en un seul tenseur cible [N, 3]
        # data.y_p est [N, 1], data.y_u est [N, 2]
        data.y = torch.cat([data.y_p, data.y_u], dim=1)
        
        return data

if __name__ == "__main__":
    dataset = AirfoilDataset("data/graphs")
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample node features shape: {sample.x.shape}")
        print(f"Sample edges shape: {sample.edge_index.shape}")
        print(f"Sample target shape: {sample.y.shape}")
