# AHD-SLE

A PyTorch implementation for the paper below: AHD-SLE: Anomalous Hyperedge Detection on Hypergraph Symmetric Line Expansion

## 1. Code Structure
- ```data/``` : under this folder, we provide several hypergraph datasets
    - dblp, iAF1260b, iJO1366, reverb, uspto
    - *.edges.neg : negative hyperedge dataset, format: node_id, hyperedge_id
    - *.edges.pos : positive hyperedge dataset, format: node_id, hyperedge_id
    - *.npz : node feature, it a sparse matrix
    
- ```layers.py```: standard GCN layers
- ```logutil.py```: logging function
- ```main.py```: the running script
- ```models.py```: AHD-SLE model implementations
- ```SLE.py```: the SLE transformation script
- ```utils.py```: auxiliary functions

## 2. Running AHD-SLE
```python
# select a dataset from dblp, iAF1260b, iJO1366, reverb, uspto
python main.py --dataset [DATASET]
```