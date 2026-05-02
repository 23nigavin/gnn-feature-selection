# Baseline autoencoder model
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class BetterDenoisingAutoencoder(nn.Module):
    """
    Encoder sees noisy augmented input of size input_dim.
    Decoder reconstructs only original clean feature dimension of size target_dim.
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
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return z, x_hat

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
    verbose=True
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        _, x_hat = model(x_input[train_mask])
        train_loss = F.mse_loss(x_hat, x_target[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, x_hat_val = model(x_input[val_mask])
            val_loss = F.mse_loss(x_hat_val, x_target[val_mask])

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
    model.eval()
    with torch.no_grad():
        z = model.encoder(x_input)
        z = F.normalize(z, p=2, dim=1)   # helps stabilize downstream GCN
    return z
