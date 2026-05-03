from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class SyntheticGraphConfig:
    """
    Configuration for creating a synthetic graph dataset.

    The purpose of this dataset is to test when graph-aware feature selection
    is useful. We control the graph structure, the true useful features,
    random junk features, and misleading train-only features.
    """

    # Basic dataset size
    num_classes: int = 3
    nodes_per_class: int = 100

    # Graph structure:
    # p_in is the probability of connecting two nodes from the same class.
    # p_out is the probability of connecting two nodes from different classes.
    # If p_in > p_out, then the graph has homophily.
    p_in: float = 0.08
    p_out: float = 0.01

    # True signal features:
    # These are binary features that are actually related to the class label.
    # signal_delta controls how strong the relationship is.
    # Small delta = weak feature-label correlation.
    num_signal_features: int = 9
    signal_base_prob: float = 0.5
    signal_delta: float = 0.15

    # If signal_classes is None, all classes get useful signal features.
    # If signal_classes = (0,), then only class 0 has useful signal features.
    signal_classes: Optional[Tuple[int, ...]] = None

    # Junk features:
    # These are random binary features that should not help classification.
    num_junk_features: int = 100
    junk_prob: float = 0.5

    # Spurious features:
    # These features are correlated with the labels only on the training nodes.
    # This tests whether the model is learning a real signal or a shortcut.
    num_spurious_features: int = 0
    spurious_train_delta: float = 0.45
    spurious_test_mode: str = "random"  # options: "random" or "anti"

    # Optional noise added after the features are created.
    bit_flip_prob: float = 0.0
    mask_prob: float = 0.0

    # Train/validation split per class.
    train_per_class: int = 20
    val_per_class: int = 30

    # Random seed for reproducibility.
    seed: int = 42


# These are the main synthetic settings we use in experiments.
# Each scenario tests a different assumption about when graph-aware feature
# selection should help.
SCENARIOS: Dict[str, SyntheticGraphConfig] = {
    # Easy case: graph is very homophilous and features are strong.
    "easy_homophily": SyntheticGraphConfig(
        p_in=0.10,
        p_out=0.005,
        num_signal_features=12,
        signal_delta=0.30,
        num_junk_features=60,
    ),

    # Useful features are weak, but the graph structure is strong.
    # A graph-aware method should be especially helpful here.
    "weak_features_strong_graph": SyntheticGraphConfig(
        p_in=0.10,
        p_out=0.005,
        num_signal_features=9,
        signal_delta=0.08,
        num_junk_features=150,
    ),

    # Features are strong, but the graph has weak homophily.
    # Here, the graph structure may not help as much.
    "weak_graph_strong_features": SyntheticGraphConfig(
        p_in=0.04,
        p_out=0.03,
        num_signal_features=9,
        signal_delta=0.30,
        num_junk_features=150,
    ),

    # Only a few features are truly useful.
    # This tests whether feature selection can find sparse signal.
    "few_signal_features": SyntheticGraphConfig(
        p_in=0.08,
        p_out=0.01,
        num_signal_features=3,
        signal_delta=0.20,
        num_junk_features=250,
    ),

    # Only one class has useful signal features.
    # This tests a partial-signal setting.
    "partial_class_signal": SyntheticGraphConfig(
        p_in=0.10,
        p_out=0.005,
        num_signal_features=6,
        signal_delta=0.25,
        signal_classes=(0,),
        num_junk_features=150,
    ),

    # Spurious features work on train nodes but become random on non-train nodes.
    "train_only_spurious": SyntheticGraphConfig(
        p_in=0.08,
        p_out=0.01,
        num_signal_features=6,
        signal_delta=0.12,
        num_junk_features=100,
        num_spurious_features=30,
        spurious_train_delta=0.48,
        spurious_test_mode="random",
    ),

    # Spurious features work on train nodes but become misleading on test nodes.
    "anti_spurious_test": SyntheticGraphConfig(
        p_in=0.08,
        p_out=0.01,
        num_signal_features=6,
        signal_delta=0.12,
        num_junk_features=100,
        num_spurious_features=30,
        spurious_train_delta=0.48,
        spurious_test_mode="anti",
    ),
}


class SyntheticDataset:
    """
    Small wrapper so the synthetic graph acts like a PyTorch Geometric dataset.

    This dataset only contains one graph, so dataset[0] returns the graph.
    """

    def __init__(self, data: Data, num_classes: int, name: str = "synthetic"):
        self.data = data
        self.num_classes = num_classes
        self.name = name

    def __getitem__(self, index: int) -> Data:
        if index != 0:
            raise IndexError("SyntheticDataset only contains one graph.")
        return self.data

    def __len__(self) -> int:
        return 1


def make_binary_class_features(
    labels: np.ndarray,
    num_features: int,
    num_classes: int,
    base_prob: float,
    delta: float,
    active_classes: Optional[Tuple[int, ...]],
    rng: np.random.Generator,
):
    """
    Create binary features that are weakly or strongly correlated with labels.

    The idea:
    - Each feature is assigned to one class.
    - Nodes from that class are more likely to have the feature equal to 1.
    - Nodes from other classes are less likely to have the feature equal to 1.

    Example:
    If base_prob = 0.5 and delta = 0.15:
        owning class:     P(feature = 1) = 0.65
        non-owning class: P(feature = 1) = 0.35

    This keeps the features binary instead of Gaussian.
    """

    if num_features <= 0:
        return np.empty((len(labels), 0), dtype=np.float32), {}

    if active_classes is None:
        active_classes = tuple(range(num_classes))

    # Start all probabilities at the lower probability.
    probs = np.full(
        (len(labels), num_features),
        base_prob - delta,
        dtype=np.float32,
    )

    # Keep track of which features are actually useful for each class.
    useful_by_class = {c: [] for c in range(num_classes)}

    for feature_idx in range(num_features):
        # Assign features evenly across classes.
        # Example with 3 classes:
        # feature 0 -> class 0
        # feature 1 -> class 1
        # feature 2 -> class 2
        # feature 3 -> class 0
        owner_class = feature_idx % num_classes

        # If this class is not supposed to have signal, make the feature random.
        if owner_class not in active_classes:
            probs[:, feature_idx] = base_prob
            continue

        # Nodes from the owning class get a higher probability of feature = 1.
        probs[labels == owner_class, feature_idx] = base_prob + delta
        useful_by_class[owner_class].append(feature_idx)

    # Keep probabilities valid and sample binary 0/1 values.
    probs = np.clip(probs, 0.01, 0.99)
    x = rng.binomial(1, probs).astype(np.float32)

    return x, useful_by_class


def make_synthetic_graph_data(
    config: SyntheticGraphConfig = SyntheticGraphConfig(),
) -> Data:
    """
    Create one synthetic PyTorch Geometric graph.

    The final graph has:
    - node features: data.x
    - clean features before corruption: data.x_clean
    - graph edges: data.edge_index
    - node labels: data.y
    - train/val/test masks
    """

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    # Create node labels
    # Labels are grouped by class.
    # Example for 3 classes and 100 nodes per class:
    # [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2]
    labels = np.repeat(np.arange(config.num_classes), config.nodes_per_class)
    num_nodes = len(labels)

    # Create train, validation, and test masks
    # We sample the same number of train/val nodes from each class.
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for c in range(config.num_classes):
        class_nodes = np.where(labels == c)[0]
        rng.shuffle(class_nodes)

        train_nodes = class_nodes[:config.train_per_class]

        val_start = config.train_per_class
        val_end = config.train_per_class + config.val_per_class
        val_nodes = class_nodes[val_start:val_end]

        test_nodes = class_nodes[val_end:]

        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

    train_mask_t = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool)

    # Create graph edges
    # This is a simple homophily-based graph.
    # Same-label node pairs connect with probability p_in.
    # Different-label node pairs connect with probability p_out.
    src = []
    dst = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if labels[i] == labels[j]:
                edge_prob = config.p_in
            else:
                edge_prob = config.p_out

            if rng.random() < edge_prob:
                # Add both directions so the graph is undirected.
                src.append(i)
                dst.append(j)

                src.append(j)
                dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Create true signal features
    # These are the features we want a good feature-selection method to find.
    x_signal, useful_signal_by_class = make_binary_class_features(
        labels=labels,
        num_features=config.num_signal_features,
        num_classes=config.num_classes,
        base_prob=config.signal_base_prob,
        delta=config.signal_delta,
        active_classes=config.signal_classes,
        rng=rng,
    )

    # Create junk features
    # These features are completely random and should not be useful.
    x_junk = rng.binomial(
        1,
        config.junk_prob,
        size=(num_nodes, config.num_junk_features),
    ).astype(np.float32)

    # Create spurious features
    # These features look useful on the training nodes.
    # However, they do not generalize to validation/test nodes.
    #
    # This is meant to test whether the model learns real graph-aware signal
    # or just memorizes shortcuts from the training set.
    x_spurious, spurious_by_class = make_binary_class_features(
        labels=labels,
        num_features=config.num_spurious_features,
        num_classes=config.num_classes,
        base_prob=0.5,
        delta=config.spurious_train_delta,
        active_classes=tuple(range(config.num_classes)),
        rng=rng,
    )

    if config.num_spurious_features > 0:
        non_train = ~train_mask

        if config.spurious_test_mode == "random":
            # Outside training, the spurious features become random.
            x_spurious[non_train] = rng.binomial(
                1,
                0.5,
                size=x_spurious[non_train].shape,
            )

        elif config.spurious_test_mode == "anti":
            # Outside training, the spurious features become misleading.
            # A 1 becomes 0 and a 0 becomes 1.
            x_spurious[non_train] = 1.0 - x_spurious[non_train]

        else:
            raise ValueError("spurious_test_mode must be 'random' or 'anti'.")

    # Combine all feature groups
    # Feature order:
    # [signal features | junk features | spurious features]
    x_clean = np.concatenate([x_signal, x_junk, x_spurious], axis=1)

    # Make a copy that we can optionally corrupt.
    x = x_clean.copy()

    # Optional feature corruption
    # Bit flipping changes some 0s to 1s and some 1s to 0s.
    if config.bit_flip_prob > 0:
        flip = rng.random(x.shape) < config.bit_flip_prob
        x[flip] = 1.0 - x[flip]

    # Masking sets some feature values to 0.
    if config.mask_prob > 0:
        keep = rng.random(x.shape) > config.mask_prob
        x *= keep

    # Convert everything into a PyTorch Geometric Data object
    signal_end = config.num_signal_features
    junk_end = signal_end + config.num_junk_features

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        x_clean=torch.tensor(x_clean, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=train_mask_t,
        val_mask=val_mask_t,
        test_mask=test_mask_t,
    )

    # Store which columns belong to each feature type.
    # This is useful for checking whether feature selection picked the right features.
    data.feature_groups = {
        "signal": list(range(0, signal_end)),
        "junk": list(range(signal_end, junk_end)),
        "spurious": list(range(junk_end, junk_end + config.num_spurious_features)),
    }

    # Store the ground-truth useful features.
    data.useful_signal_by_class = useful_signal_by_class
    data.spurious_by_class = spurious_by_class

    # Store a summary of what assumptions this synthetic graph is testing.
    data.assumptions = {
        "binary_features": True,
        "homophily": config.p_in > config.p_out,
        "p_in": config.p_in,
        "p_out": config.p_out,
        "weak_label_signal": config.signal_delta <= 0.12,
        "few_signal_features": config.num_signal_features <= config.num_classes * 2,
        "partial_class_signal": config.signal_classes is not None,
        "train_only_spurious_features": config.num_spurious_features > 0,
    }

    data.synthetic_config = config

    return data


def make_synthetic_dataset(
    config: SyntheticGraphConfig = SyntheticGraphConfig(),
    name: str = "synthetic",
) -> SyntheticDataset:
    """
    Create a one-graph synthetic dataset.
    """
    data = make_synthetic_graph_data(config)
    return SyntheticDataset(data, config.num_classes, name)


def make_scenario(name: str, seed: Optional[int] = None) -> SyntheticDataset:
    """
    Create one of the predefined synthetic scenarios.

    Example:
        dataset = make_scenario("weak_features_strong_graph")
        data = dataset[0]
    """

    if name not in SCENARIOS:
        valid_options = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{name}'. Valid options: {valid_options}")

    config = SCENARIOS[name]

    # Optionally override the seed so we can run the same scenario multiple times.
    if seed is not None:
        config = SyntheticGraphConfig(**{**config.__dict__, "seed": seed})

    return make_synthetic_dataset(config, name)