"""Linear regression y = w0 + w1 * x trained with gradient descent in PyTorch."""

import matplotlib.pyplot as plt
import numpy as np
import torch


class LinearRegression:
    """Simple linear regression model trained with SGD."""

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Parameters w0 (intercept) and w1 (slope); requires_grad=True lets
        # autograd compute d(loss)/dw0 and d(loss)/dw1 for us.
        self.w0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self.w1 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

        # SGD optimizer and mean squared error loss
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        # Lists that store w0, w1, and the loss as training progresses
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, x):
        """Linear model: compute and return the response y = w0 + w1 * x."""
        return self.w0 + self.w1 * x

    def fit(self, X_train, y_train, X_test, y_test):
        """Train with gradient descent, then print R^2 on the test data."""
        # Convert inputs to 1D float64 tensors
        X_train = torch.as_tensor(np.asarray(X_train, dtype=np.float64)).reshape(-1)
        y_train = torch.as_tensor(np.asarray(y_train, dtype=np.float64)).reshape(-1)
        X_test = torch.as_tensor(np.asarray(X_test, dtype=np.float64)).reshape(-1)
        y_test = torch.as_tensor(np.asarray(y_test, dtype=np.float64)).reshape(-1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()                  # clear old gradients
            y_pred = self.forward(X_train)              # forward pass
            loss = self.loss_fn(y_pred, y_train)        # mean squared error
            loss.backward()                             # backward pass (autograd)
            self.optimizer.step()                       # w <- w - lr * gradient

            # record progress
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        # R^2 on the test data: 1 - SS_res / SS_tot
        with torch.no_grad():
            y_test_pred = self.forward(X_test)
            ss_res = torch.sum((y_test - y_test_pred) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            self.r2 = (1 - ss_res / ss_tot).item()
        print(f"R^2 on test data: {self.r2:.4f}")

    def predict(self, x_new):
        """Predict the response (e.g. AnnualProduction) for new input data (e.g. BCR).

        x_new can be a single number, a list, or an array.
        """
        x_new = torch.as_tensor(np.asarray(x_new, dtype=np.float64))
        with torch.no_grad():
            y_new = self.forward(x_new)
        if y_new.dim() == 0:
            return y_new.item()
        return y_new.numpy()

    def analysis_plot(self, X, y, xlabel="x", ylabel="y"):
        """Plot the data with the fitted line, and w0, w1, and loss versus epoch."""
        X = np.asarray(X, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        epochs = range(1, len(self.loss_history) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # 1. Original data and fitted regression line
        ax = axes[0, 0]
        ax.scatter(X, y, alpha=0.6, label="data")
        x_line = np.linspace(X.min(), X.max(), 100)
        ax.plot(x_line, self.w0.item() + self.w1.item() * x_line,
                color="red", label="fitted line")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Data and fitted regression line")
        ax.legend()

        # 2. w0 as training progressed
        axes[0, 1].plot(epochs, self.w0_history)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("w0")
        axes[0, 1].set_title("w0 versus epoch")

        # 3. w1 as training progressed
        axes[1, 0].plot(epochs, self.w1_history)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("w1")
        axes[1, 0].set_title("w1 versus epoch")

        # 4. Loss as training progressed
        axes[1, 1].plot(epochs, self.loss_history)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss (MSE)")
        axes[1, 1].set_title("Loss versus epoch")

        fig.tight_layout()
        plt.show()

import numpy as np


class CauchyRegression:
    """Multiple linear regression trained with the Cauchy loss in PyTorch.

    Loss: l(y_hat, y) = (c^2 / 2) * log(1 + ((y - y_hat) / c)^2)
    """

    def __init__(self, c=1.0, learning_rate=0.5, epochs=2000):
        self.c = c
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None          # [w0, w1, ..., wp] in original units
        self.loss_history = []

    def cauchy_loss(self, y_hat, y):
        r = (y - y_hat) / self.c
        return torch.mean(self.c**2 / 2 * torch.log1p(r**2))

    def fit(self, X, y):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
        self.X_, self.y_ = X, y

        # Standardize features for stable training
        mu = X.mean(dim=0)
        sd = X.std(dim=0, unbiased=False)
        Z = (X - mu) / sd

        b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([b0, b], lr=self.learning_rate)

        self.loss_history = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self.cauchy_loss(b0 + Z @ b, y)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

        # Convert weights back to the original feature scale
        with torch.no_grad():
            w = b / sd
            w0 = b0 - torch.sum(b * mu / sd)
            self.coef_ = torch.cat([w0, w])
        return self

    def predict(self, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        return self.coef_[0] + X @ self.coef_[1:]

    def final_loss(self):
        return self.cauchy_loss(self.predict(self.X_), self.y_).item()

    def confidence_interval(self, alpha=0.05):
        """Wald confidence intervals from the inverse Hessian of the
        Cauchy negative log-likelihood (scale c) at the fitted coefficients."""
        Xd = torch.cat([torch.ones(len(self.y_), 1, dtype=torch.float64), self.X_], dim=1)

        def nll(w):
            r = (self.y_ - Xd @ w) / self.c
            return torch.sum(torch.log1p(r**2))

        H = torch.autograd.functional.hessian(nll, self.coef_)
        se = torch.sqrt(torch.diag(torch.linalg.inv(H)))
        z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1 - alpha / 2)).item()
        return self.coef_ - z * se, self.coef_ + z * se, se
