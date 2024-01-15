import lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from gen_model_playground.utils import gradient_penalty, least_squares, non_saturating, wasserstein
from gen_model_playground.models.model import Model

class GAN(pl.LightningModule):
    """
    PyTorch Lightning module for Generative Adversarial Networks (GANs).

    Attributes:
        generator: The generator network.
        discriminator: The discriminator network.
        gan_type: The type of GAN loss function to be used.
        loss: The loss function for the GAN.
    """

    def __init__(self, **kwargs):
        """
        Initializes the GAN model with specified hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.generator = Model(in_features=self.hparams.in_features, out_features=self.hparams.encoding_dim, num_blocks=self.hparams.num_blocks, hidden_features=self.hparams.hidden_features, cond_features=self.hparams.cond_features, spectral=False, batch_norm=self.hparams.batch_norm, residual=self.hparams.residual, bias=self.hparams.bias_gen)
        self.discriminator = Model(in_features=self.hparams.in_features, out_features=1, num_blocks=self.hparams.num_blocks, hidden_features=self.hparams.hidden_features, cond_features=self.hparams.cond_features, spectral=self.hparams.spectral, batch_norm=self.hparams.batch_norm, residual=self.hparams.residual, bias=self.hparams.bias_dis)
        self.automatic_optimization = False  # Disable automatic optimization
        self.gan_type = self.hparams.gan_type
        self.loss = least_squares if self.hparams.gan_type == "lsgan" else wasserstein if self.hparams.gan_type == "wgan" else non_saturating

        self.spectral = self.hparams.spectral
        self.save_name = self.hparams.gan_type+"_"+self.hparams.dataset
        self.save_name += "_spectral" if self.hparams.spectral else ""
        self.save_name += "_gp"+str(self.hparams.gp_value) if self.hparams.gp  else ""
        self.save_name += "_residual" if self.hparams.residual else ""
        self.save_name += "_bn" if self.hparams.batch_norm else ""
        self.save_name += "_ema" if self.hparams.bias_gen else ""
        self.name = self.hparams.name if not "fm" in vars(self.hparams) else  "feature_matching" if self.hparams.fm else self.hparams.name

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers for both generator and discriminator.

        Returns:
            A tuple of lists containing optimizers and learning rate schedulers.
        """
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=2 * self.hparams.lr, betas=(0.9 if self.hparams.gan_type != "wgan" else 0.5, 0.999))
        sched_g = LinearWarmupCosineAnnealingLR(opt_g, warmup_epochs=self.hparams.max_epochs * self.hparams.num_batches // 10, max_epochs=self.hparams.max_epochs * self.hparams.num_batches)
        sched_d = LinearWarmupCosineAnnealingLR(opt_d, warmup_epochs=self.hparams.max_epochs * self.hparams.num_batches // 10, max_epochs=self.hparams.max_epochs * self.hparams.num_batches)
        return [opt_g, opt_d], [sched_g, sched_d]

    def sample(self, z, cond=None):
        """
        Samples from the generator.

        Args:
            z: Normal distributed noise
            cond: Condition for the generator, tensor of shape (len(z),*).

        Returns:
            Generated samples.
        """


        return self.generator(z, cond)


    def training_step(self, batch, batch_idx):
        """
        Training step for the GAN.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Discriminator loss for logging.
        """
        x = batch[0]
        cond = batch[1] if self.hparams.cond_features > 0 else None
        opt_g, opt_d = self.optimizers()
        sched_d, sched_g = self.lr_schedulers()
        sched_d.step()
        sched_g.step()

        # Train discriminator
        z = torch.normal(0, 1, size=(x.shape[0], 2), device=self.device)
        xhat = self.generator(z, cond)
        opt_d.zero_grad()
        d_loss = self.loss(self.discriminator(x, cond), self.discriminator(xhat), critic=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=False, logger=True)

        if self.hparams.gp:
            gp = 10 * gradient_penalty(x, xhat, self.discriminator, cond=cond, GP=self.hparams.gp_value)
            d_loss = d_loss + gp
            self.log('gp', gp, on_step=True, on_epoch=False, logger=True)

        self.manual_backward(d_loss)
        opt_d.step()

        # Train generator
        if self.global_step > 100:
            opt_g.zero_grad()
            z = torch.normal(0, 1, size=(x.shape[0], 2), device=self.device)
            xhat,lhat = self.generator(z, cond, feature_matching=True)
            g_loss = self.loss(None, self.discriminator(xhat, cond), critic=False)
            if self.hparams.fm:
                _,l = self.discriminator(x, cond, feature_matching=True)
                g_loss = (lhat-l).pow(2).mean()
            self.manual_backward(g_loss)
            opt_g.step()
            self.log('g_loss', g_loss, on_step=True, on_epoch=False, logger=True)

    def on_validation_epoch_start(self):
        """
        Prepares variables for tracking during the validation epoch.
        """
        self.xhat = []
        self.x = []
        self.z = []
        self.y = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the GAN.

        Args:
            batch: Input batch for validation.
            batch_idx: Batch index.

        Returns:
            Validation loss.
        """
        x = batch[0]
        cond = batch[1] if len(batch) > 1 else None

        z = torch.normal(0, 1, size=(x.shape[0], 2), device=self.device)

        xhat = self.generator(z)
        yhat = self.discriminator(xhat)
        y = self.discriminator(x)
        loss = self.loss(y, yhat, critic=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, logger=True)

        self.xhat.append(xhat)
        self.y.append(cond)
        self.z.append(z)
        self.x.append(x)


