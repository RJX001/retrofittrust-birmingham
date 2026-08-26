"""Shallow PyOD AutoEncoder for EPC data-quality screening.

Architecture (CURSOR_BUILD_SPEC §4):
    input → encoder hidden → bottleneck → decoder hidden → linear output

- Bottleneck is sized comfortably below half the (one-hot-expanded) input dim.
- Modest L2 + dropout on the two hidden layers only, not the bottleneck or output.
- Early stopping on a validation reconstruction-loss split.
- Optional light denoising (small Gaussian input corruption at train time).

This is a preprocessing detector, not the ranking model. The core ranking model
remains LightGBM / XGBoost (see Program 1). SEED = 42 throughout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

from retrofittrust.config import SEED

try:
    import torch
    from torch import nn
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.utils.stat_models import pairwise_distances_no_broadcast
    from pyod.utils.torch_utility import LinearBlock, TorchDataset
except ImportError as exc:  # pragma: no cover - import-time environment check
    raise ImportError(
        "retrofittrust.quality.autoencoder requires pyod and torch. "
        "Install pinned pyod from requirements.txt plus PyTorch "
        "(PyOD 2.0.2 AutoEncoder is PyTorch-backed, not Keras)."
    ) from exc

def infer_hidden_dims(
    n_features: int,
    hidden_dim: Optional[int] = None,
    bottleneck_dim: Optional[int] = None,
) -> tuple[int, int]:
    """Choose a shallow encoder/decoder width and a small bottleneck.

    The bottleneck is forced below half the input dimensionality so the
    autoencoder must compress rather than copy. Hidden width sits between
    bottleneck and input.
    """
    if n_features < 2:
        raise ValueError(f"Need at least 2 features, got {n_features}.")

    max_bottleneck = max(1, (n_features // 2) - 1)
    if bottleneck_dim is None:
        bottleneck_dim = max(2, n_features // 4) if n_features >= 8 else max(1, n_features // 3)
    bottleneck_dim = int(max(1, min(int(bottleneck_dim), max_bottleneck)))

    if hidden_dim is None:
        hidden_dim = max(bottleneck_dim + 1, n_features // 2)
    hidden_dim = int(max(bottleneck_dim + 1, hidden_dim))
    return hidden_dim, bottleneck_dim


class ShallowAutoEncoderModel(nn.Module):
    """One encoder hidden, bottleneck (no dropout), one decoder hidden, linear out."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout_rate: float,
        hidden_activation_name: str = "relu",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Dropout on hidden layers only — bottleneck and output stay deterministic.
        self.encoder_hidden = LinearBlock(
            n_features,
            hidden_dim,
            activation_name=hidden_activation_name,
            batch_norm=False,
            dropout_rate=dropout_rate,
        )
        self.bottleneck = LinearBlock(
            hidden_dim,
            bottleneck_dim,
            activation_name=hidden_activation_name,
            batch_norm=False,
            dropout_rate=0.0,
        )
        self.decoder_hidden = LinearBlock(
            bottleneck_dim,
            hidden_dim,
            activation_name=hidden_activation_name,
            batch_norm=False,
            dropout_rate=dropout_rate,
        )
        # Linear output: required to reconstruct standardised numeric values.
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_hidden(x)
        x = self.bottleneck(x)
        x = self.decoder_hidden(x)
        return self.output(x)


class ShallowAutoEncoder(AutoEncoder):
    """PyOD AutoEncoder with the Program 2 architecture constraints.

    Parameters
    ----------
    hidden_dim, bottleneck_dim
        If omitted, sized from ``n_features`` at ``fit`` time via
        :func:`infer_hidden_dims`. Bottleneck is always < n_features / 2.
    dropout_rate
        Applied to encoder/decoder hidden layers only.
    l2
        Weight decay applied to hidden-layer weights only (not bottleneck/output).
    denoise_std
        Std of Gaussian noise added to *inputs* during training (target stays
        clean). Set 0 to disable. Light denoising is preferred but not required.
    patience, min_delta, validation_fraction
        Early stopping on held-out reconstruction loss.
    preprocessing
        PyOD z-score of columns inside the detector. Safe to leave True even
        if Program 1 already standardised (near no-op on mean-0 unit-variance
        data). Pass False to skip a second pass.
    """

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        dropout_rate: float = 0.2,
        l2: float = 1e-4,
        denoise_std: float = 0.05,
        patience: int = 8,
        min_delta: float = 1e-4,
        validation_fraction: float = 0.15,
        epoch_num: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        contamination: float = 0.1,
        preprocessing: bool = True,
        random_state: int = SEED,
        verbose: int = 0,
        hidden_activation_name: str = "relu",
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            contamination=contamination,
            preprocessing=preprocessing,
            lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name="adam",
            device=device,
            random_state=random_state,
            verbose=verbose,
            # L2 is applied via param groups in training_prepare, not globally.
            optimizer_params={"weight_decay": 0.0},
            hidden_neuron_list=[32, 16],
            hidden_activation_name=hidden_activation_name,
            batch_norm=False,
            dropout_rate=dropout_rate,
        )
        self.requested_hidden_dim = hidden_dim
        self.requested_bottleneck_dim = bottleneck_dim
        self.l2 = float(l2)
        self.denoise_std = float(denoise_std)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.validation_fraction = float(validation_fraction)
        self.hidden_dim_: Optional[int] = None
        self.bottleneck_dim_: Optional[int] = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []
        self.best_epoch_: Optional[int] = None
        self.stopped_early_: bool = False

    def build_model(self) -> None:
        hidden_dim, bottleneck_dim = infer_hidden_dims(
            self.feature_size,
            hidden_dim=self.requested_hidden_dim,
            bottleneck_dim=self.requested_bottleneck_dim,
        )
        if bottleneck_dim >= self.feature_size / 2:
            raise ValueError(
                f"Bottleneck {bottleneck_dim} must be < half input dim "
                f"({self.feature_size / 2:.1f})."
            )
        self.hidden_dim_ = hidden_dim
        self.bottleneck_dim_ = bottleneck_dim
        self.hidden_neuron_list = [hidden_dim, bottleneck_dim]
        self.model = ShallowAutoEncoderModel(
            n_features=self.feature_size,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout_rate=self.dropout_rate,
            hidden_activation_name=self.hidden_activation_name,
        )

    def training_prepare(self) -> None:
        """Adam with L2 on hidden Linear weights only; bottleneck/output are L2-free."""
        self.model = self.model.to(self.device)
        hidden_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder_hidden") or name.startswith("decoder_hidden"):
                hidden_params.append(param)
            else:
                other_params.append(param)
        self.optimizer = torch.optim.Adam(
            [
                {"params": hidden_params, "weight_decay": self.l2},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        self.model.train()

    def training_forward(self, batch_data):
        x = batch_data.to(self.device)
        target = x
        if self.denoise_std > 0:
            x = x + torch.randn_like(x) * self.denoise_std
        self.optimizer.zero_grad()
        reconstructed = self.model(x)
        loss = self.criterion(reconstructed, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluating_forward(self, batch_data):
        x = batch_data
        x_gpu = x.to(self.device)
        reconstructed = self.model(x_gpu)
        return pairwise_distances_no_broadcast(x.numpy(), reconstructed.cpu().numpy())

    def _make_loader(self, X: np.ndarray, shuffle: bool, drop_last: bool):
        if self.preprocessing:
            dataset = TorchDataset(X=X, y=None, mean=self.X_mean, std=self.X_std)
        else:
            dataset = TorchDataset(X=X, y=None)
        batch_size = min(self.batch_size, max(1, len(X)))
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last and len(X) >= batch_size * 2,
        )

    def _reconstruction_loss(self, loader) -> float:
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device)
                reconstructed = self.model(x)
                losses.append(float(self.criterion(reconstructed, x).item()))
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def fit(self, X, y=None):
        """Fit with a validation split and early stopping on reconstruction loss."""
        X = check_array(X, dtype=np.float64)
        self._set_n_classes(y)
        self.data_num, self.feature_size = X.shape
        self._set_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        n = X.shape[0]
        use_val = n >= 50 and 0.0 < self.validation_fraction < 0.5
        if use_val:
            X_train, X_val = train_test_split(
                X,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=True,
            )
        else:
            X_train, X_val = X, None

        if self.preprocessing:
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
            self.X_std = np.where(self.X_std < 1e-8, 1.0, self.X_std)
        else:
            self.X_mean = None
            self.X_std = None

        self.build_model()
        self.training_prepare()

        train_loader = self._make_loader(X_train, shuffle=True, drop_last=True)
        val_loader = (
            self._make_loader(X_val, shuffle=False, drop_last=False)
            if X_val is not None
            else None
        )
        self._train_with_early_stopping(train_loader, val_loader)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def _train_with_early_stopping(self, train_loader, val_loader) -> None:
        self.train_loss_history_ = []
        self.val_loss_history_ = []
        best_val = float("inf")
        best_state = None
        wait = 0
        self.stopped_early_ = False
        self.best_epoch_ = self.epoch_num

        iterator = range(self.epoch_num)
        if self.verbose == 1:
            try:
                import tqdm

                iterator = tqdm.trange(self.epoch_num, desc="AE training")
            except ImportError:
                pass

        for epoch in iterator:
            epoch_losses: list[float] = []
            self.model.train()
            for batch in train_loader:
                epoch_losses.append(float(self.training_forward(batch)))
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            self.train_loss_history_.append(train_loss)

            if val_loader is not None:
                val_loss = self._reconstruction_loss(val_loader)
            else:
                val_loss = train_loss
            self.val_loss_history_.append(val_loss)

            if self.verbose == 2:
                print(
                    f"Epoch {epoch + 1}/{self.epoch_num} "
                    f"train={train_loss:.4f} val={val_loss:.4f}"
                )

            if val_loss < best_val - self.min_delta:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
                self.best_epoch_ = epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    self.stopped_early_ = True
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()

    def _transformed(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.preprocessing:
            return (X - self.X_mean) / (self.X_std + 1e-8)
        return X

    def reconstruct(self, X) -> np.ndarray:
        """Return reconstructions in the space the network sees (standardised if enabled)."""
        X = check_array(X, dtype=np.float64)
        xp = self._transformed(X)
        self.model.eval()
        reconstructed_chunks: list[np.ndarray] = []
        batch_size = min(self.batch_size, max(1, len(xp)))
        with torch.no_grad():
            for start in range(0, len(xp), batch_size):
                batch = torch.as_tensor(
                    xp[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                reconstructed_chunks.append(self.model(batch).cpu().numpy())
        return np.concatenate(reconstructed_chunks, axis=0)

    def per_feature_reconstruction_error(self, X) -> np.ndarray:
        """Squared residual per feature in the (standardised) model space.

        Used to explain *which* EPC field looks implausible, not just that a
        record is anomalous. Shape ``(n_samples, n_features)``.
        """
        X = check_array(X, dtype=np.float64)
        xp = self._transformed(X)
        reconstructed = self.reconstruct(X)
        return (xp - reconstructed) ** 2
