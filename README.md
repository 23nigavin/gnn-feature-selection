# GNN Feature Selection

This repository explores how graph neural networks behave when node features are flooded with irrelevant or misleading dimensions. The main experiments use the Cora citation graph and compare a baseline GCN against preprocessing approaches such as graph-aware feature selection, PCA, autoencoder embeddings, and a simplified learned feature mask.

The project also includes a binary synthetic graph generator for testing specific assumptions, such as weak feature-label correlation, homophily strength, sparse useful features, and train-only spurious features.

## Project Structure

```text
.
├── README.md
├── requirements.txt
└── src
    ├── main.py                    # Main Cora experiment script
    ├── synthetic_experiments.py   # Synthetic scenario experiment script
    ├── experiments.py             # Experiment runners for GCN, feature selection, and autoencoder baselines
    ├── gnn.py                     # Two-layer GCN model plus train/test helpers
    ├── preprocessing_selection.py # Feature-selection methods
    ├── autoencoder.py             # Plain autoencoder model and training helpers
    ├── pca.py                     # Transductive unsupervised PCA helper
    ├── masked_gnn.py              # Simplified top-k learned mask baseline
    ├── noise.py                   # Junk-feature corruption helper
    ├── util.py                    # Graph feature aggregation helper
    ├── plotting.py                # Plotting functions for accuracy curves
    └── synthetic_data.py          # Binary synthetic graph dataset generator
```

Generated outputs such as `.png` plots and `.csv` result files may also appear under `src/` after experiments are run.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Main Experiment

From the repository root:

```bash
python src/main.py
```

This loads the Cora dataset through PyTorch Geometric and runs noise sweeps. The most important experiment is the **dense junk** setting, because it most directly tests the original hypothesis: whether graph-aware feature selection is robust when many irrelevant feature dimensions are appended.

The main script compares:

- GCN on noisy features with no feature selection
- raw L1 feature selection
- graph-aware L1 feature selection
- raw mutual information feature selection
- graph-aware mutual information feature selection
- PCA embeddings
- autoencoder embeddings
- simplified learned top-k feature mask

The script produces accuracy curves for:

- accuracy vs. noise level for dense junk, feature dropout, and bit flip
- accuracy vs. number of selected features `k`

The dense-junk graph should be emphasized in writeups. Feature dropout and bit flip are useful extensions/controls.

Results saved by `src/main.py` are reported as mean +- standard deviation over 3 random seeds.

## Core Components

### GCN Baseline

`src/gnn.py` defines a simple two-layer GCN:

```text
GCNConv -> ReLU -> Dropout -> GCNConv
```

It also provides `train(...)` and `test(...)` helpers used by the experiment runners.

### Junk Features

`src/noise.py` adds random binary junk dimensions to node features:

```python
add_junk_features(x, num_junk_features)
```

These dimensions are intended to be irrelevant, so accuracy should drop if the model overfits or becomes distracted by them.

### Graph-Aware Feature Selection

`src/preprocessing_selection.py` contains feature-selection methods:

- `select_top_k_features_l1`
- `select_features_mutual_info`

The main Cora experiment compares raw and graph-aware versions of these selectors. Graph-aware selection first aggregates node features over the graph, then scores features using only the training labels. Aggregation happens in `src/util.py`:

```python
aggregate_features(x, edge_index)
```

This means feature selection can score features using local graph context, not just each node's raw feature vector.

The main noise sweeps use `k=256` for feature-selection methods and for the dimensionality of PCA/autoencoder baselines.

### PCA and Autoencoder Baselines

PCA and autoencoder are unsupervised transductive baselines. They are fit using **all nodes**, but they do not use labels.

`src/pca.py` fits PCA on the full noisy feature matrix.

`src/autoencoder.py` defines `BetterDenoisingAutoencoder`, a vanilla MLP autoencoder. The class name is historical; in the current experiments the autoencoder is **not denoising**. It reconstructs the noisy/junk-augmented feature matrix and then passes the latent representation to the downstream GCN.

The intended pipeline is:

```text
node features
    -> apply noise/corruption
    -> train autoencoder to reconstruct the same noisy/corrupted features
    -> encode features into latent z
    -> train GCN on z
```

The autoencoder supports:

- MSE reconstruction loss for continuous features
- BCE-with-logits reconstruction loss for binary features

### Learned Mask Baseline

`src/masked_gnn.py` defines a simplified learned mask baseline. It is not a direct implementation of a specific paper. It learns one mask logit per feature, keeps the top-k entries, and trains a GCN on the masked feature matrix.

## Synthetic Data

`src/synthetic_data.py` creates binary synthetic graph datasets for controlled tests.

The generator controls:

- number of classes and nodes per class
- graph homophily through `p_in` and `p_out`
- number and strength of truly useful binary signal features
- number of random junk features
- train-only spurious features
- bit-flip and masking corruption

Predefined scenarios include:

```python
easy_homophily
weak_features_strong_graph
weak_graph_strong_features
few_signal_features
partial_class_signal
train_only_spurious
anti_spurious_test
```

To run the predefined synthetic scenario sweep:

```bash
python src/synthetic_experiments.py
```

This evaluates a small set of methods on several synthetic assumptions:

- no feature selection
- raw L1 feature selection
- graph-aware L1 feature selection
- raw mutual information feature selection
- graph-aware mutual information feature selection
- autoencoder embeddings

The script writes a CSV named `synthetic_results.csv`.
The synthetic script also records feature-recovery metrics for feature-selection methods, such as how many selected columns came from true signal, junk, or spurious feature groups.

Example:

```python
from synthetic_data import make_scenario
from experiments import run_preprocessing_selection_experiment_avg

dataset = make_scenario("weak_features_strong_graph", seed=0)

mean_acc, std_acc = run_preprocessing_selection_experiment_avg(
    dataset,
    noise_ratio=0.0,
    noise_type="dense_junk",
    k=256,
    selection_method="l1",
    graph_aware=True,
    seeds=[0, 1, 2],
)

print(mean_acc, std_acc)
```

When importing from an interactive session, make sure `src/` is on your Python path, or run from inside `src/`.

## Notes

- The main script uses Cora from `torch_geometric.datasets.Planetoid`.
- PyTorch Geometric downloads dataset files into `data/Planetoid`.
- Supervised feature-selection methods use only training labels.
- PCA and autoencoder are unsupervised transductive methods fit on all nodes.
- The learned mask baseline is a simplified top-k mask baseline, not a paper reproduction.
- The synthetic dataset stores metadata such as `data.feature_groups`, `data.useful_signal_by_class`, and `data.assumptions` to help evaluate whether feature selection is choosing true signal or junk/spurious dimensions.
