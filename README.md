# GNN Feature Selection

This repository explores how graph neural networks behave when node features are flooded with irrelevant or misleading dimensions. The main experiments use the Cora citation graph and compare a baseline GCN against preprocessing approaches such as graph-aware feature selection and denoising autoencoder embeddings.

The project also includes a binary synthetic graph generator for testing specific assumptions, such as weak feature-label correlation, homophily strength, sparse useful features, and train-only spurious features.

## Project Structure

```text
.
├── README.md
├── requirements.txt
└── src
    ├── main.py                    # Main Cora experiment script
    ├── experiments.py             # Experiment runners for GCN, feature selection, and autoencoder baselines
    ├── gnn.py                     # Two-layer GCN model plus train/test helpers
    ├── preprocessing_selection.py # Feature-selection methods
    ├── autoencoder.py             # Denoising autoencoder model and training helpers
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

This loads the Cora dataset through PyTorch Geometric, adds different amounts of binary junk features, and compares:

- GCN on clean features
- GCN on noisy/junk-augmented features
- GCN after L1-based graph-aware feature selection

The script produces accuracy curves for:

- accuracy vs. junk-feature noise level
- accuracy vs. number of selected features `k`

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

`src/preprocessing_selection.py` contains several feature-selection methods:

- `select_top_k_features_l1`
- `select_features_permutation`
- `select_features_correlation`
- `select_features_mutual_info`

The main Cora experiment currently uses L1 logistic regression after graph feature aggregation. Aggregation happens in `src/util.py`:

```python
aggregate_features(x, edge_index)
```

This means feature selection can score features using local graph context, not just each node's raw feature vector.

### Autoencoder Baseline

`src/autoencoder.py` defines `BetterDenoisingAutoencoder`, a vanilla MLP denoising autoencoder.

The intended pipeline is:

```text
clean node features
    -> append junk features
    -> train autoencoder to reconstruct clean features
    -> encode noisy features into latent z
    -> train GCN on z
```

The autoencoder supports:

- MSE reconstruction loss for continuous features
- BCE-with-logits reconstruction loss for binary features

Use BCE for binary synthetic features:

```python
run_autoencoder_experiment_avg(dataset, reconstruction_loss_type="bce")
```

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

Example:

```python
from synthetic_data import make_scenario
from experiments import run_l1_selection_experiment_avg

dataset = make_scenario("weak_features_strong_graph", seed=0)

acc = run_l1_selection_experiment_avg(
    dataset,
    noise_ratio=0.0,
    use_feature_selection=True,
    k=dataset[0].x.shape[1],
    seeds=[0, 1, 2, 3, 4],
)

print(acc)
```

When importing from an interactive session, make sure `src/` is on your Python path, or run from inside `src/`.

## Experiment Runners

`src/experiments.py` provides reusable experiment functions:

- `run_l1_selection_experiment`
- `run_l1_selection_experiment_avg`
- `run_autoencoder_experiment`
- `run_autoencoder_experiment_avg`
- `run_graph_autoencoder_experiment`
- `run_graph_autoencoder_experiment_avg`

The `_avg` variants run multiple seeds and return mean test accuracy.

## Notes

- The main script uses Cora from `torch_geometric.datasets.Planetoid`.
- PyTorch Geometric downloads dataset files into `data/Planetoid`.
- The current feature-selection experiment uses graph-aggregated features for selection, then trains the GCN on the selected original feature columns.
- The synthetic dataset stores metadata such as `data.feature_groups`, `data.useful_signal_by_class`, and `data.assumptions` to help evaluate whether feature selection is choosing true signal or junk/spurious dimensions.
