import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class BetterDenoisingAutoencoder(nn.Module):
    """Fully connected autoencoder for node-feature reconstruction.

    The model itself is a vanilla MLP autoencoder. It becomes "denoising" when
    training passes noisy/junk-augmented features as input and clean features as
    the reconstruction target.

    Args:
        input_dim: Number of input feature columns seen by the encoder.
        latent_dim: Size of the compressed representation used by the downstream GCN.
        target_dim: Number of feature columns produced by the decoder.
        hidden_dim: Width of the hidden layer in both encoder and decoder.
        dropout: Dropout probability applied in the encoder and decoder.
    """

    def __init__(self, input_dim: int, latent_dim: int, target_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_noisy: torch.Tensor):
        """Encode input features and reconstruct the configured target shape."""
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return z, x_hat


def reconstruction_loss(model, x_input, x_target, mask, loss_type="mse"):
    """Compute reconstruction loss on the nodes selected by a boolean mask.

    Use MSE for continuous features and BCE-with-logits for binary features.
    The decoder returns raw values/logits; no sigmoid is applied inside the model.
    """

    _, x_hat = model(x_input[mask])

    if loss_type == "mse":
        return F.mse_loss(x_hat, x_target[mask])

    if loss_type == "bce":
        target = x_target[mask]

        pos = target.sum(dim=0)
        neg = target.size(0) - pos

        pos_weight = torch.where(
            pos > 0,
            neg / pos.clamp(min=1.0),
            torch.ones_like(pos),
        ).clamp(max=50)

        return F.binary_cross_entropy_with_logits(
            x_hat,
            target,
            pos_weight=pos_weight,
        )

    raise ValueError("loss_type must be 'mse' or 'bce'.")


def train_autoencoder(
    model,
    x_input,
    x_target,
    train_mask,
    val_mask,
    lr,
    weight_decay,
    epochs,
    patience,
    verbose=True,
    reconstruction_loss_type="mse",
):
    """Train an autoencoder with validation-based early stopping.

    Args:
        model: Autoencoder model to train.
        x_input: Input feature matrix. This may be clean, noisy, or junk-augmented.
        x_target: Reconstruction target. Use x_input for a plain autoencoder and
            clean features for a denoising autoencoder.
        train_mask: Boolean mask selecting nodes used for gradient updates.
        val_mask: Boolean mask selecting nodes used for early stopping.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        epochs: Maximum number of training epochs.
        patience: Stop after this many epochs without validation improvement.
        verbose: Whether to print periodic training progress.
        reconstruction_loss_type: "mse" for continuous targets or "bce" for binary targets.

    Returns:
        A tuple of the best validation model and a history dictionary containing
        train and validation reconstruction losses.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_loss = reconstruction_loss(
            model,
            x_input,
            x_target,
            train_mask,
            loss_type=reconstruction_loss_type,
        )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = reconstruction_loss(
                model,
                x_input,
                x_target,
                val_mask,
                loss_type=reconstruction_loss_type,
            )

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val_loss - 1e-5:
            best_val_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(
                f"[AE] Epoch {epoch:03d} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"[AE] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def encode_features(model, x_input):
    """Return normalized latent embeddings from the trained encoder.

    The downstream GCN uses these embeddings as its node features.
    """

    model.eval()
    with torch.no_grad():
        z = model.encoder(x_input)
        z = F.normalize(z, p=2, dim=1)
    return z
