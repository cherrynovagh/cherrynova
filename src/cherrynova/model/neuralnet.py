"""Neural network models for the cherrynova package.

Contains TorchNet, a small fully connected PyTorch classifier with
built-in standardization, training, loss plotting, ONNX export and
ONNX-based prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class TorchNet(nn.Module):
    """Fully connected neural network for classification.

    Architecture: input -> 4 neurons (ReLU) -> 3 neurons (ReLU) -> output.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of outputs. Use 1 for binary classification
        (BCEWithLogitsLoss) or the number of classes for multi-class
        classification (CrossEntropyLoss).
    hidden1 : int
        Neurons in the first hidden layer (default 4).
    hidden2 : int
        Neurons in the second hidden layer (default 3).
    lr : float
        Learning rate for the Adam optimizer (default 0.01).
    """

    def __init__(self, input_dim, output_dim=1, hidden1=4, hidden2=3, lr=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

        # Binary vs. multi-class loss
        if output_dim == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.scaler = StandardScaler()
        self.losses = []

    # 1. Data standardization
    def standardize(self, X, fit=True):
        """Standardize features with StandardScaler.

        Use fit=True on the training data and fit=False on new data so the
        same mean and standard deviation are reused.
        """
        X = np.asarray(X, dtype=np.float32)
        if fit:
            return self.scaler.fit_transform(X).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    # 2. Forward pass
    def forward(self, x):
        return self.network(x)

    # 3. Training
    def train_model(self, X, y, epochs=1000, verbose=True):
        """Train with the Adam optimizer on already standardized X."""
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        if self.output_dim == 1:
            y_t = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        else:
            y_t = torch.tensor(np.asarray(y), dtype=torch.long)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.losses = []
        self.train()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = self.loss_fn(self(X_t), y_t)
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())

            if verbose and (epoch == 1 or epoch % 100 == 0):
                print(f"Epoch {epoch:4d}/{epochs}  Loss: {loss.item():.4f}")

        return self.losses

    # 4. Loss plot
    def plot_loss(self):
        """Plot training loss against epoch number."""
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(self.losses) + 1), self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs. Epochs")
        plt.grid(True, alpha=0.3)
        plt.show()

    # 5. Save in ONNX format
    def save_onnx(self, path="torchnet.onnx"):
        """Export the trained model to an ONNX file."""
        self.eval()
        dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)
        torch.onnx.export(
            self,
            (dummy_input,),
            path,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"Model saved to {path}")
        return path

    # 6. Load the ONNX model and predict
    def predict_onnx(self, X_new, path="torchnet.onnx"):
        """Load the saved ONNX model and predict class labels for X_new.

        X_new is raw (unstandardized) data; it is scaled with the scaler
        fitted during standardize().
        """
        import onnxruntime as ort

        X_scaled = self.standardize(X_new, fit=False)
        session = ort.InferenceSession(path)
        input_name = session.get_inputs()[0].name

        # The model is exported with a batch size of 1, so predict row by row
        logits = np.vstack(
            [session.run(None, {input_name: row[None, :]})[0] for row in X_scaled]
        )

        if self.output_dim == 1:
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int).ravel()
            return preds, probs.ravel()

        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.argmax(axis=1), probs