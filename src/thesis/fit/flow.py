from nflows.flows.base import Flow as BaseFlow
from nflows.utils.torchutils import create_random_binary_mask
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.transforms import *
from nflows.nn import nets

import nflows as nf
import torch.nn.functional as F
from ..models.model import Model
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
import torch
import lightning.pytorch as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Flow(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.save_name=self.hparams.name+"_"+self.hparams.flow_type
        self.flow = self.construct_flow()
        self.automatic_optimization = False  # Disable automatic optimization
        self.name=self.hparams.name


    def construct_flow(self,):
        flows = []


        for i in range(self.hparams.ncoupling):
            '''This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not'''
            mask=create_random_binary_mask(self.hparams.in_features)
            layer_networks=lambda x,y: Model(x,y,hidden_features=self.hparams.hidden_features,cond_features=self.hparams.cond_features,num_blocks=self.hparams.num_blocks,spectral=self.hparams.spectral,batch_norm=self.hparams.batch_norm,residual=self.hparams.residual)
            if not self.hparams.flow_type=="affine":
                flows += [PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=layer_networks,
                tails='linear',
                tail_bound=self.hparams.tail_bound,
                num_bins=self.hparams.num_bins)]
            else:
                flows+=[ AffineCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=layer_networks)]
        q0 = nf.distributions.normal.StandardNormal([self.hparams.in_features])
        flows=CompositeTransform(flows)
        # Construct flow model
        return BaseFlow(distribution=q0, transform=flows)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.num_batches*self.hparams.max_epochs)

        return [optimizer], [sched]


    def forward(self, x, t=None,cond=None,feature_matching=False):
        return self.flow.sample(x.shape[0] if cond is None else 1, context=cond)

    def on_validation_epoch_start(self):
        self.xhat=[]
        self.x=[]
        self.y=[]
        self.z=[]
        self.prob=[]
        self.xgrid=[]
        self.ygrid=[]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sched = self.lr_schedulers()
        sched.step()
        x=batch[0]
        if self.hparams.cond_features>0:
            cond=batch[1]
        else:
            cond=None
        log_likelihood = self.flow.log_prob(inputs=x, context=cond)
        opt.zero_grad()
        loss = -torch.mean(log_likelihood)
        loss.backward()
        opt.step()
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x=batch[0]
        if self.hparams.cond_features>0:
            cond=batch[1]
        else:
            cond=None
        log_likelihood = self.flow.log_prob(inputs=x, context=cond)
        loss = -torch.mean(log_likelihood)
        z=self.flow.transform_to_noise(x).squeeze(1)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False,logger=True)
        self.xhat.append(self(x,cond))
        self.y.append(batch[1])
        self.x.append(x)
        self.z.append(z)

        return loss

