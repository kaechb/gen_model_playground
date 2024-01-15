from nflows.flows.base import Flow as BaseFlow
from nflows.utils.torchutils import create_random_binary_mask
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
import nflows as nf
from gen_model_playground.models.model import Model
import torch
import lightning.pytorch as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Flow(pl.LightningModule):
    """
    PyTorch Lightning module for flow-based models.

    Attributes:
        flow: The constructed flow model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Flow model with specified hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.save_name = self.hparams.name + "_" + self.hparams.flow_type+"_"+self.hparams.dataset
        self.save_name += "_cond" if self.hparams.cond_features > 0 else ""
        self.flow = self.construct_flow()
        self.automatic_optimization = False  # Disable automatic optimization
        self.name = self.hparams.name

    def construct_flow(self):
        """
        Constructs a flow model based on the specified hyperparameters.

        Returns:
            The constructed flow model.
        """
        flows = []
        for i in range(self.hparams.n_coupling):
            mask = create_random_binary_mask(self.hparams.in_features)
            layer_networks = lambda x, y: Model(x, y, hidden_features=self.hparams.hidden_features, cond_features=self.hparams.cond_features, num_blocks=self.hparams.num_blocks, spectral=self.hparams.spectral, batch_norm=self.hparams.batch_norm, residual=self.hparams.residual)
            if self.hparams.flow_type != "affine":
                flows.append(PiecewiseRationalQuadraticCouplingTransform(mask=mask, transform_net_create_fn=layer_networks, tails='linear', tail_bound=self.hparams.tail_bound, num_bins=self.hparams.num_bins))
            else:
                flows.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=layer_networks))
        q0 = nf.distributions.normal.StandardNormal([self.hparams.in_features])
        flows = CompositeTransform(flows)
        return BaseFlow(distribution=q0, transform=flows)

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers.

        Returns:
            A tuple of lists containing optimizers and learning rate schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs * self.hparams.num_batches // 10, max_epochs=self.hparams.num_batches * self.hparams.max_epochs)
        return [optimizer], [sched]

    def forward(self, x, t=None, cond=None, feature_matching=False):
        """
        Forward pass through the flow model for sampling.

        Args:
            x: Input tensor.
            t: Time parameter, unused in this context.
            cond: Conditional inputs.
            feature_matching: Feature matching flag, unused in this context.

        Returns:
            Sampled output.
        """
        return self.flow.sample(x.shape[0] if cond is None else 1, context=cond)

    def sample(self, batch, cond=None, t_stop=1, return_latent=False):
        """
        Samples from the flow model.

        Args:
            batch: Batch size.
            cond: Conditional inputs.
            t_stop: Time parameter, unused in this context.
            return_latent: Flag to return latent space variables.

        Returns:
            Sampled output and optionally latent space variables.
        """
        with torch.no_grad():
            x0 = torch.randn(len(batch), self.hparams.in_features).to(self.device)
            if cond is not None:
                z = self.flow.sample(1, context=cond).squeeze(1)
            else:
                z = self.flow.sample(len(batch), context=cond)
            if return_latent:
                return z, x0
            else:
                return z
    def on_validation_epoch_start(self):
        """
        Prepares variables for tracking during the validation epoch.
        """
        self.xhat = []
        self.x = []
        self.y = []
        self.z = []
        self.prob = []

    def training_step(self, batch, batch_idx):
        """
        Training step for the flow model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Computed loss for the training step.
        """
        opt = self.optimizers()
        sched = self.lr_schedulers()
        sched.step()
        x = batch[0]
        cond = batch[1] if self.hparams.cond_features > 0 else None
        log_likelihood = self.flow.log_prob(inputs=x, context=cond)
        opt.zero_grad()
        loss = -torch.mean(log_likelihood)
        loss.backward()
        opt.step()
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the flow model.

        Args:
            batch: Input batch for validation.
            batch_idx: Batch index.

        Returns:
            Computed validation loss.
        """
        x = batch[0]
        cond = batch[1] if self.hparams.cond_features > 0 else None
        log_likelihood = self.flow.log_prob(inputs=x, context=cond)
        loss = -torch.mean(log_likelihood)
        z = self.flow.transform_to_noise(x, context=cond).squeeze(1)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.xhat.append(self(x, cond=cond).squeeze(1))
        self.y.append(batch[1])
        self.x.append(x)
        self.z.append(z)
