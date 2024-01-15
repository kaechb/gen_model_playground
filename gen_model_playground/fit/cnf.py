import math
import torch
import lightning.pytorch as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchdyn.core import NeuralODE
from gen_model_playground.models.model import Model

class wrapper(torch.nn.Module):
    """
    A wrapper class for a neural network model. This wrapper is designed to compute the trace of the Jacobian
    and add the 'logpz' to the output, which is crucial in normalizing flow models like CNF.

    Args:
        model: The neural network model to be wrapped.
    """

    def __init__(self, model):
        super(wrapper, self).__init__()
        self.model = model

    def forward(self, t, x_aug, args=None,cond=None):
        """
        Forward pass through the wrapper.

        Args:
            t: Time parameter for the dynamical system.
            x_aug: Augmented input, typically including the state and additional parameters like 'logpz'.
            args: Additional arguments, if any.

        Returns:
            The augmented output, including the computed state and updated 'logpz'.
        """
        x, logpz = x_aug[:, :-1], x_aug[:, -1]
        x.requires_grad_(True)
        dxdt = self.model(x)

        # Compute the trace of the Jacobian using autograd, if gradients are required
        dlogpz = torch.zeros(x.shape[0], device=x.device)
        if dxdt.requires_grad:
            for i in range(x.shape[1]):
                dlogpz += torch.autograd.grad(dxdt[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()

        return torch.cat([dxdt, (logpz + dlogpz).unsqueeze(1)], dim=1)


class CNF(pl.LightningModule):
    """
    A PyTorch Lightning module for Continuous Normalizing Flows (CNF).

    This module sets up the model, training, and validation procedures for a CNF model using NeuralODE from TorchDyn.

    Attributes:
        vf: The vector field defined by the CNF model, wrapped in the 'wrapper' class.
        node: The Neural Ordinary Differential Equation (NeuralODE) model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the CNF model with specified hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.save_name = "cnf"+"_"+self.hparams.dataset
        self.name = self.hparams.name

        self.vf = wrapper(Model(**self.hparams).to("cuda"))
        self.node = NeuralODE(self.vf, sensitivity='adjoint', solver='euler')


    def compute_loss(self, z, logpz):
        """
        Computes the loss for the CNF model.

        Args:
            z: The latent variable.
            logpz: Log probability.

        Returns:
            The computed loss.
        """
        # Assuming a standard normal prior for simplicity
        logp = -0.5 * z.pow(2).sum(1) - 0.5 * math.log(2 * math.pi) * z.size(1)
        return -(logp + logpz).mean()

    def load_datamodule(self, data_module):
        """
        Sets the data loader for training and validation.

        Args:
            data_module: The data module to be used for training and validation.
        """
        self.data_module = data_module

    def sample(self, batch, cond=None, t_stop=1):
        """
        Samples from the CNF model.

        Args:
            batch: The input batch for sampling.
            cond: Conditional inputs, if any.
            t_stop: End time for the ODE solver.


        Returns:
            The generated samples.
        """
        x0 = torch.randn(batch.shape[0], batch.shape[1]).to(self.device)
        logpz = torch.zeros(x0.shape[0], 1, device=self.device)
        x_aug = torch.cat([x0, logpz], dim=1)
        x_aug.requires_grad_(True)
        samples = self.node.trajectory(x_aug, t_span=torch.linspace(1, 0, self.hparams.num_steps).to(self.device)).detach()
        return samples[-1, :, :2]

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers for training.

        Returns:
            A tuple of lists containing optimizers and LR schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs * self.hparams.num_batches // 10, max_epochs=self.hparams.max_epochs * self.hparams.num_batches)
        return [optimizer], [sched]

    def training_step(self, batch, batch_idx):
        """
        Defines a single training step.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            The training loss.
        """
        optimizer = self.optimizers()
        optimizer.zero_grad()
        sched = self.lr_schedulers()
        sched.step()
        x = batch[0]
        logpz = torch.zeros(x.shape[0], 1, device=self.device)
        x_aug = torch.cat([x, logpz], dim=1)
        x_aug.requires_grad_(True)

        augmented_traj = self.node.trajectory(x_aug, t_span=torch.linspace(0, 1, self.hparams.num_steps).float().to(self.device))
        x, logpz = augmented_traj[-1][:, :-1], augmented_traj[-1][:, -1]
        loss = self.compute_loss(x, logpz)
        self.manual_backward(loss)
        optimizer.step()
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def on_validation_epoch_start(self):
        """
        Prepares variables for tracking during the validation epoch.
        """
        self.xhat = []
        self.x = []
        self.z = []
        self.y = []
        self.prob = []


    def validation_step(self, batch, batch_idx):
        """
        Defines a single validation step.

        Args:
            batch: The input batch for validation.
            batch_idx: Index of the batch.

        Returns:
            The validation loss.
        """
        with torch.enable_grad():
            x = batch[0]
            self.x.append(x)
            logpz = torch.zeros(x.shape[0], 1, device=self.device)
            x_aug = torch.cat([x, logpz], dim=1)
            x_aug.requires_grad_(True)

            # Forward pass through Neural ODE
            augmented_traj = self.node.trajectory(x_aug, t_span=torch.linspace(0, 1, self.hparams.num_steps).float().to(self.device))
            z, logpz = augmented_traj[-1][:, :-1], augmented_traj[-1][:, -1]

            # Compute loss
            loss = self.compute_loss(z, logpz)
            self.log("val/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Sampling for validation
        xhat = self.sample(x)
        self.xhat.append(xhat.detach())
        self.z.append(z.detach())
        self.y.append(batch[1])

        # End of the CNF class definition