

from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import torch
from sklearn.datasets import make_moons

class TwoMoonsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256, num_batches=1000, noise=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.noise = noise
        self.dataset = None

    def setup(self, stage=None):
        # This will be called at the beginning of each fit (train) and test
        self.generate_dataset()

    def generate_dataset(self):
        # Generate the two moons dataset
        X, y = make_moons(n_samples=self.num_batches*self.batch_size, noise=self.noise)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        self.mu, self.std = X.mean(axis=0), X.std(axis=0)
        self._min, self._max = X.min(axis=0), X.max(axis=0)
        X = (X - self.mu) / self.std
        self.dataset = TensorDataset(X,y)

    def train_dataloader(self):
        # Ensure the dataset is generated
        if self.dataset is None:
            self.generate_dataset()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        # Ensure the dataset is generated
        if self.dataset is None:
            self.generate_dataset()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def on_epoch_start(self):
        # Generate a new dataset at the start of each epoch
        self.generate_dataset()

# Usage example
if __name__ == "__main__":
    datamodule = TwoMoonsDataModule(batch_size=256)
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(batch)
        break
