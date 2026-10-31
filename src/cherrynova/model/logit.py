"""Binary logistic regression trained by minimizing the cross-entropy loss
with mini-batch stochastic gradient descent (SGD), implemented in PyTorch."""

import numpy as np
import torch


class LogisticRegression:
    """Binary logistic regression: h(x) = sigmoid(w^T x + b).

    Member variables
    ----------------
    learning_rate, epochs, batch_size, seed : training settings
    coef_       : fitted weights w (for standardized features)
    intercept_  : fitted bias b
    mean_, std_ : feature means and standard deviations used for scaling
    loss_history: training cross-entropy after each epoch
    """

    def __init__(self, learning_rate=0.1, epochs=50, batch_size=64, seed=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.coef_ = None
        self.intercept_ = None
        self.mean_ = None
        self.std_ = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + torch.exp(-z))

    @staticmethod
    def cross_entropy(p, y, eps=1e-12):
        """Mean binary cross-entropy (negative log-likelihood)."""
        p = torch.clamp(p, eps, 1 - eps)
        return -torch.mean(y * torch.log(p) + (1 - y) * torch.log(1 - p))

    def _scale(self, X):
        X = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float64)
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        X = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float64)
        y = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.float64)
        n, d = X.shape

        # Standardize features so SGD behaves well
        self.mean_ = X.mean(dim=0)
        self.std_ = X.std(dim=0, unbiased=False)
        Z = (X - self.mean_) / self.std_

        w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([w, b], lr=self.learning_rate)
        gen = torch.Generator().manual_seed(self.seed)

        self.loss_history = []
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)       # shuffle each epoch
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                p = self.sigmoid(Z[idx] @ w + b)
                loss = self.cross_entropy(p, y[idx])
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                full_loss = self.cross_entropy(self.sigmoid(Z @ w + b), y)
            self.loss_history.append(full_loss.item())

        self.coef_ = w.detach()
        self.intercept_ = b.detach()
        return self

    def predict_proba(self, X):
        """Probability that each sample belongs to class 1."""
        with torch.no_grad():
            return self.sigmoid(self._scale(X) @ self.coef_ + self.intercept_)

    def predict(self, X, threshold=0.5):
        """Class labels (0 or 1)."""
        return (self.predict_proba(X) >= threshold).long()

    def loss(self, X, y):
        """Cross-entropy of the fitted model on data (X, y)."""
        y = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.float64)
        return self.cross_entropy(self.predict_proba(X), y).item()
