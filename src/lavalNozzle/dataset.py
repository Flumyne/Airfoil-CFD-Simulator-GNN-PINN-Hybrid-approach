import os
import torch
from torch_geometric.data import Dataset
import glob

class NozzleDataset(Dataset):
    """
    Classe pour charger les graphes d'ailes traités (fichiers .pt).
    """
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super(NozzleDataset, self).__init__(root_dir, transform, pre_transform)
        self.file_list = glob.glob(os.path.join(root_dir, "*.pt"))
        
    def len(self):
        return len(self.file_list)

    def get(self, idx):
        data = torch.load(self.file_list[idx], weights_only=False)
        data.y = torch.cat([data.y_p, data.y_u, data.y_T, data.y_rho, data.y_mach], dim=1)
        return data

if __name__ == "__main__":
    dataset = NozzleDataset("data/graphs/nozzle")
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample node features shape: {sample.x.shape}")
        print(f"Sample edges shape: {sample.edge_index.shape}")
        print(f"Sample target shape: {sample.y.shape}")
