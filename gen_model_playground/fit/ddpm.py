
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from gen_model_playground.models.model import Model
import torch
class torch_wrapper(torch.nn.Module):
    """
    A wrapper class for a PyTorch model to make it compatible with torchdyn library.
    It optionally allows the inclusion of time as an input to the model.

    Args:
        model: The neural network model to be wrapped.
        use_t: Flag to indicate whether to use time as an input parameter.
    """

    def __init__(self, model, use_t=True):
        super().__init__()
        self.model = model
        self.use_t = use_t

    def forward(self, t, x, *args, **kwargs):
        """
        Forward pass through the wrapper.

        Args:
            t: Time parameter.
            x: Input tensor.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            The output from the model.
        """
        if self.use_t:
            return self.model(x, t.repeat(x.shape[0])[:, None])
        else:
            print("not using time, are you sure?")
            return self.model(x=x)


class DDPM(pl.LightningModule):
    """
    A PyTorch Lightning module for Denoising Diffusion Probabilistic Models (DDPM).

    This module sets up the DDPM model, including the noise schedule, and defines the training
    and validation procedures.

    Attributes:
        net: The neural network model.
        baralphas, betas, alphas: Parameters for the noise schedule.
        loss: Loss function for the model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the DDPM model with specified hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.save_name = self.hparams.name+"_"+self.hparams.dataset
        self.save_name += "_cond" if self.hparams.cond_features > 0 else ""
        self.name = self.hparams.name
        self.net = Model(**self.hparams).to(self.device)

        # Setting up the noise schedule for DDPM
        s = 0.008  # Noise variance beta

        self.register_buffer("timesteps", torch.tensor(range(0, self.hparams.num_steps), dtype=torch.float32))
        self.register_buffer("schedule", torch.cos((self.timesteps / self.hparams.num_steps + s) / (1 + s) * torch.pi / 2)**2)
        self.register_buffer("baralphas", self.schedule / self.schedule[0])

        self.register_buffer("betas", 1 - self.baralphas / torch.cat([self.baralphas[0:1], self.baralphas[:-1]]))
        self.register_buffer("alphas",  1 - self.betas)
        self.loss = nn.MSELoss()



    def load_datamodule(self, data_module):
        """
        Sets the data loader for training and validation.

        Args:
            data_module: The data module to be used for training and validation.
        """
        self.data_module = data_module


    def sample(self, batch, return_traj=False, cond=None):
        """
        Generates samples from the DDPM model.

        Args:
            batch: Input batch for generating samples.
            return_traj: Flag to return the entire trajectory of the sampling process.

        Returns:
            The last generated sample or the entire trajectory, depending on `return_traj`.
        """
        with torch.no_grad():
            x = torch.randn(batch.shape).to(self.device)
            xt = [x]
            for t in reversed(range(1, self.hparams.num_steps)):

                predicted_noise = self.net(x, t=torch.full([len(batch), 1], t).to(self.device),cond=cond)
                x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1 - self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    variance = self.betas[t]
                    x += (variance ** 0.5) * torch.randn(size=(len(batch), batch.shape[1]), device=self.device)
                xt.append(x)

            return (xt[-1],xt) if return_traj else xt[-1]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.max_epochs*self.hparams.num_batches)

        return [optimizer], [sched]

    def noise(self, X, t):
        """
        Applies noise to the input data.

        Args:
            X: The input data.
            t: Time step indices.

        Returns:
            The noised data and the noise.
        """
        eps = torch.randn(size=X.shape, device=self.device)
        noised = (self.baralphas[t.int()] ** 0.5).repeat(1, X.shape[1]) * X + ((1 - self.baralphas[t.int()]) ** 0.5).repeat(1, X.shape[1]) * eps
        return noised, eps

    def training_step(self, batch, batch_idx):
        """
        The training step for DDPM.

        Args:
            batch: Input batch.
            batch_idx: Index of the batch.

        Returns:
            The computed loss for the training step.
        """
        opt = self.optimizers()
        sched = self.lr_schedulers()
        sched.step()
        cond = batch[1] if self.hparams.cond_features > 0 else None
        batch = batch[0]
        timesteps = torch.randint(0, self.hparams.num_steps, size=[len(batch), 1], device=self.device)
        noised, eps = self.noise(batch, timesteps)
        predicted_noise = self.net(noised,t= timesteps,cond=cond)
        loss = self.loss(predicted_noise, eps)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

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
        The validation step for DDPM.

        Args:
            batch: Input batch for validation.
            batch_idx: Index of the batch.

        Logs the validation loss and appends the generated samples and their trajectories.
        """
        with torch.no_grad():
            self.y.append(batch[1])
            cond = batch[1] if self.hparams.cond_features > 0 else None
            batch = batch[0]
            self.x.append(batch)
            timesteps = torch.randint(0, self.hparams.num_steps, size=[len(batch), 1], device=self.device)
            noised, eps = self.noise(batch, timesteps)
            predicted_noise = self.net(noised, timesteps)
            loss = self.loss(predicted_noise, eps)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            _,trajectory = self.sample(batch, cond=cond, return_traj=True)
            self.xhat.append(trajectory[-1])
            self.z.append(trajectory[0])

    # End of the DDPM class definition