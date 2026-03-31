from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    hidden_layer_sizes: tuple[int, int] = (256, 64),
    max_iter: int = 80,
) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def mlp_predict_proba(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def mlp_entropy_uncertainty(model: MLPClassifier, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(x)[:, 1]
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return probs, entropy
