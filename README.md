# Micro Graph Neural Networks

Welcome to my first GNN try

This repository contains the foundational modules for processing complex structural relationships, CAD/STEP topologies, and engineering metadata using **Graph Neural Networks (GNNs)** with PyTorch Geometric (PyG).

---

## 🚀 Quick Start: Constructing and Visualizing Directed Engineering Graphs

Engineering structures (such as CAD assemblies, STEP topologies, or pipeline systems) are naturally represented as **Directed Graphs**. Below is a core implementation demonstrating how to build a 4-node graph with multidimensional node features, extreme edge weights (simulating structural boundaries, forces, or connection states), and render it cleanly using NetworkX.

### Code Example (`visualize_graph.py`)

```python
import torch
import torch_geometric
from torch_geometric.data import Data

import networkx as netx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

print("PyTorch Version:", torch.__version__)
print("CUDA Available?:", torch.cuda.is_available())
print("PyG Version:", torch_geometric.__version__)

# 1. Define Topology (Directed Edges: Source -> Target)
edge_list = torch.tensor(
    [[0, 0, 0, 1, 1, 2, 2, 3],  # Source nodes  
     [0, 1, 2, 1, 2, 2, 3, 0]], # Target nodes
    dtype=torch.long
)

# 2. Define Node Features (e.g., volume, surface area, bounding box vectors)
node_features = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0],                # Node 0
    [1.0, 1.0, 1.0, 1.0, 1.0],                # Node 1
    [-1.0, 258.0, 252844962.0, -528948268.0, 0.0], # Node 2 (Extreme Feature Values)
    [0.0, 8.5, 0.5, -0.5, 1.0]                # Node 3
], dtype=torch.float)

# 3. Define Edge Weights (e.g., mechanical stress, clearance, alignment tolerances)
edge_weight = torch.tensor([
    1., 22., -5., 0., -0., 5555555555555., -747474774747447., 0.5
], dtype=torch.float)

# 4. Pack into PyG Data Object
data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_weight)

# 5. Convert to NetworkX & Retain Edge Direction (to_undirected=False)
plot = to_networkx(data, to_undirected=False)

# 6. Render the Topology
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