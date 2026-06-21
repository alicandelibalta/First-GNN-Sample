import torch
import torch_geometric
from torch_geometric.data import Data

import networkx as netx
import matplotlib.pyplot as plt  
from torch_geometric.utils.convert import to_networkx

print("PyTorch Başarıyla Yüklendi:", torch.__version__)
print("Grafik İşlemci (GPU) Aktif mi?:", torch.cuda.is_available())
print("PyG Başarıyla Yüklendi:", torch_geometric.__version__)

# 4 node'lu bir graph
edge_list = torch.tensor(
    [[0, 0, 0, 1, 1, 2, 2, 3],  # source nodes  
     [0, 1, 2, 1, 2, 2, 3, 0]], # target nodes
    dtype=torch.long
)

# Feature'ları torch.float yaptık çünkü ondalıklı sayılar (8.5, 0.5) var.

node_features = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0],                # node 0
    [1.0, 1.0, 1.0, 1.0, 1.0],                # node 1
    [-1.0, 258.0, 252844962.0, -528948268.0, 0.0], # node 2
    [0.0, 8.5, 0.5, -0.5, 1.0]                # node 3
], dtype=torch.float)

# Edge weightler, yukarıda edge_list'te yer alan tüm çiftler için 1 er tane feature.
edge_weight = torch.tensor([1., 22., -5., 0., -0., 5555555555555., -747474774747447., 0.5], dtype=torch.float)

# PyG Data Objesi
data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_weight)

# NetworkX formatına dönüştürme, ot_indirected ayarı ile verdiğimiz yönleri koru dedik
plot = to_networkx(data, to_undirected=False)

# Görselleştirme Ayarları
plt.figure(figsize=(6, 6))
netx.draw_networkx(
    plot, 
    with_labels=True, 
    node_color='lightblue', 
    node_size=700, 
    font_weight='bold', 
    arrowsize=20
)

plt.show()