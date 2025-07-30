# GIGN RNA–Ligand Binding Affinity Prediction

This repository provides scripts and pipelines to **retrain and fine-tune the GIGN (Geometric Interaction Graph Neural Network, GIGN) model** for *RNA–ligand binding affinity prediction*.  
All scripts and data dependencies are self-contained within this folder.

---

## What is GIGN?

**GIGN** is an advanced geometric graph neural network that:

- **Incorporates 3D structure and physical interactions:**  
  Unlike most ML approaches, GIGN explicitly models the spatial and physicochemical interactions essential for molecular binding.

- **Heterogeneous interaction layer:**  
  Unifies covalent and noncovalent interactions in message passing, enabling more accurate and biologically relevant node representations.

> For more details, see:  
> [Graph Interaction Graph Network for Protein–Ligand Binding Affinity Prediction (J. Phys. Chem. Lett. 2023)](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03906)  
> [Official GIGN GitHub](https://github.com/guaguabujianle/GIGN)

---

## Installation

1. **Recommended Python version:** 3.8 or newer  
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    > Key dependencies: `torch`, `torch_geometric`, `dgl`, `rdkit`, `pandas`, `numpy`, `wandb` (optional: for logging)

---

## Usage

### 1. Data Preprocessing

Prepare graphs and features:
```
python preprocessing.py
```

### 2. GIGN Retraining

Train GIGN on preprocessed data:
```
python train_pdbbind.py
```

### 3. GIGN Fine-tuning
Refine a pre-trained model:
```
python finetuning_pdbbind.py
```

### 4. Prediction
Predict RNA–ligand binding affinities:
```
test_gign_model.py
```




