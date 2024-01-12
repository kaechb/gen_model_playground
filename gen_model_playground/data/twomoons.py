from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import torch
from sklearn.datasets import make_moons

class TwoMoonsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Two Moons dataset.

    Attributes:
        batch_size (int): Size of each data batch.
        num_batches (int): Number of batches.
        noise (float): Noise level for data generation.
        dataset (TensorDataset): The dataset containing the Two Moons data.
        mu (torch.Tensor): Mean of the dataset.
        std (torch.Tensor): Standard deviation of the dataset.
        _min (torch.Tensor): Minimum value in the dataset.
        _max (torch.Tensor): Maximum value in the dataset.
    """

    def __init__(self, batch_size=256, num_batches=1000, noise=0.05):
        """
        Initializes the TwoMoonsDataModule.

        Args:
            batch_size (int): Batch size for the dataloader.
            num_batches (int): Number of batches in the dataset.
            noise (float): Noise level for the Two Moons dataset generation.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.noise = noise
        self.dataset = None

    def setup(self, stage=None):
        """
        Prepares the dataset for training or testing.

        Args:
            stage (str, optional): Stage - either 'fit' or 'test'. Default is None.
        """
        self.generate_dataset()

    def generate_dataset(self):
        """
        Generates the Two Moons dataset.
        Standardscales the dataset.

        """
        # Create the dataset
        X, y = make_moons(n_samples=self.num_batches * self.batch_size, noise=self.noise)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        # Normalize the dataset
        self.mu, self.std = X.mean(axis=0), X.std(axis=0)
        self._min, self._max = X.min(axis=0), X.max(axis=0)
        X = (X - self.mu) / self.std
        self.scaled_min, self.scaled_max = X.min(axis=0), X.max(axis=0)
        self.dataset = TensorDataset(X, y)

    def train_dataloader(self):
        """
        Creates a DataLoader for training.

        Returns:
            DataLoader: The DataLoader for training.
        """
        if self.dataset is None:
            self.generate_dataset()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        """
        Creates a DataLoader for validation.

        Returns:
            DataLoader: The DataLoader for validation.
        """
        if self.dataset is None:
            self.generate_dataset()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def on_epoch_start(self):
        """
        Generates a new dataset at the start of each epoch.
        """
        self.generate_dataset()

# Usage example
if __name__ == "__main__":
    datamodule = TwoMoonsDataModule(batch_size=256)
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(batch)
        break