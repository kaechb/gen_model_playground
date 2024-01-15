from torch import nn
import lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from gen_model_playground.models.model import Model

class VAE(pl.LightningModule):
    """
    PyTorch Lightning module for a Variational Autoencoder (VAE).

    Attributes:
        encoder: The encoder network.
        decoder: The decoder network.
        beta: Weight for the KL divergence term in the loss.
        eps: Small constant to avoid numerical instability.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VAE model with specified hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Model(in_features=self.hparams.in_features, out_features=2 * self.hparams.encoding_dim if self.hparams.name.find("vae")>-1 else self.hparams.encoding_dim, num_blocks=self.hparams.num_blocks, hidden_features=self.hparams.hidden_features, cond_features=self.hparams.cond_features, spectral=self.hparams.spectral, batch_norm=self.hparams.batch_norm, residual=self.hparams.residual)
        self.decoder = Model(in_features=self.hparams.encoding_dim, out_features=self.hparams.in_features, num_blocks=self.hparams.num_blocks, hidden_features=self.hparams.hidden_features, cond_features=self.hparams.cond_features, spectral=self.hparams.spectral, batch_norm=self.hparams.batch_norm, residual=self.hparams.residual)
        self.beta = 0
        self.name = self.hparams.name
        self.save_name = self.hparams.name
        self.eps = 1e-8
        self.save_name += str(self.hparams.encoding_dim)+"_"+self.hparams.dataset
        self.save_name += "_cond" if self.hparams.cond_features > 0 else ""
        self.save_name += "_spectral" if self.hparams.spectral else ""
        self.save_name += "_residual" if self.hparams.residual else ""
        self.save_name += "_bn" if self.hparams.batch_norm else ""
        self.save_name += "_ema" if self.hparams.ema else ""

        self.automatic_optimization = False  # Disable automatic optimization

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the VAE.

        Returns:
            A tuple of lists containing optimizers and learning rate schedulers.
        """
        optimizer = torch.optim.Adam(nn.Sequential(*[self.encoder, self.decoder]).parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs * self.hparams.num_batches // 10, max_epochs=self.hparams.max_epochs * self.hparams.num_batches)
        return [optimizer], [sched]

    def sample(self, z, cond=None):
        """
        Samples from the decoder.

        Args:
            z: Latent space representation.

        Returns:
            Generated samples.
        """
        return self.decoder(z, cond=cond)

    def training_step(self, batch, batch_idx):
        """
        Training step for the VAE.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Loss for training.
        """
        opt = self.optimizers()
        sched = self.lr_schedulers()
        sched.step()
        x = batch[0]

        # Conditional VAE
        if self.hparams.cond_features > 0:
            cond = batch[1]
            x = torch.cat((x, cond), dim=-1)

        # Encoder step
        z = self.encoder(x)
        kl_div = 0
        if self.name.find("vae")>-1:
            mu, logvar = z[:, :self.hparams.encoding_dim], z[:, self.hparams.encoding_dim:]
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            z = eps.mul(std).add(mu)
            kl_div = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

        # Decoder step
        if self.hparams.cond_features > 0:
            cond = batch[1]
            z = torch.cat((z, cond), dim=-1)

        xhat = self.decoder(z)
        loss = torch.sum((xhat - batch[0]) ** 2, dim=1).mean()


        loss += self.beta * kl_div

        opt.zero_grad()
        loss.backward()
        opt.step()
        self.log("train/beta", self.beta, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train/kl_div", kl_div, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def on_train_epoch_end(self):
        """
        Ramps up the weight of the KL divergence term in the loss.
        """
        self.beta = float(self.current_epoch) / self.hparams.max_epochs  if not self.name == "ae" else self.beta

    def on_validation_epoch_start(self):
        """
        Prepares variables for tracking during the validation epoch.
        """
        self.xhat = []
        self.xrec = []
        self.y = []
        self.z = []
        self.x = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE.

        Args:
            batch: Input batch for validation.
            batch_idx: Batch index.

        Returns:
            Validation loss.
        """
        x = batch[0]
        cond = batch[1] if len(batch) > 1 else None

        if self.hparams.cond_features > 0:
            cond = batch[1]
            x = torch.cat((x, cond), dim=-1)
        z = self.encoder(x)
        if self.name.find("vae")>-1:
            mu, logvar = z[:, :self.hparams.encoding_dim], z[:, self.hparams.encoding_dim:]
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        self.z.append(z)

        if self.hparams.cond_features > 0:
            cond = batch[1]
            z = torch.cat((z, cond), dim=-1)
        xrec = self.decoder(z)

        # Random sampling for validation
        z = torch.normal(0, 1, size=(x.shape[0], self.hparams.encoding_dim), device=self.device)
        xhat = self.decoder(z)

        self.xhat.append(xhat)
        self.xrec.append(xrec)
        self.x.append(x)
        self.y.append(cond)

