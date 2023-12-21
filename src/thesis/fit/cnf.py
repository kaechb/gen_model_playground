
import matplotlib.pyplot as plt
import mplhep as hep
import nflows as nf
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hist import Hist
from jetnet.evaluation import fpd, get_fpd_kpd_jet_features, kpd, w1m
#from src.models.coimport torchdiffeqmponents.diffusion import VPDiffusionSchedule
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import (ExponentialLR, OneCycleLR,
                                      ReduceLROnPlateau)
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher, TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher)
from torchdyn.core import NeuralODE
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from ..models.model import Model
from copy import deepcopy
from torch.cuda import amp
import torch
from torchdyn.models import NeuralODE
from torch.distributions import Normal
from torchdiffeq import odeint
from torch.distributions import MultivariateNormal
from torchdyn.nn import Augmenter
from torchdyn.models import NeuralODE
from torch.autograd import Variable
from torchdyn.models import CNF as CNF_module
from torchdiffeq import odeint_adjoint as odeint
import math
from torchdiffeq import odeint
from torchdyn.nn import Augmenter
from torchdyn.core import NeuralODE




class wrapper(torch.nn.Module):
    #this is a wrapper class that is needed to calculate the trace of the jacobian and to add the logpz to the output
    def __init__(self, model):
        super(wrapper, self).__init__()
        self.model = model

    def forward(self, t, x_aug,args=None):

        x, logpz = x_aug[:, :-1], x_aug[:, -1]
        x.requires_grad_(True)
        dxdt = self.model(x)

        # Compute trace of Jacobian using autograd
        if dxdt.requires_grad:
            dlogpz = torch.zeros(x.shape[0], device=x.device)
            #calculating the trace of the jacobian
            for i in range(x.shape[1]):
                dlogpz += torch.autograd.grad(dxdt[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()
        else:
            dlogpz = torch.zeros(x.shape[0], device=x.device)
        # Compute trace of Jacobian using autograd

        return torch.cat([dxdt, (logpz + dlogpz).unsqueeze(1)], dim=1)





class CNF(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.save_name="cnf"
        self.name=self.hparams.name

        self.vf= wrapper(Model(**self.hparams).to("cuda"))
        self.node=NeuralODE(self.vf, sensitivity='adjoint', solver='euler')
        self.automatic_optimization = False

    def forward(self, x):
        return self.sample(x)
    def compute_loss(self,z, logpz):
        # Assuming a standard normal prior for simplicity
        logp = -0.5 * z.pow(2).sum(1) - 0.5 * math.log(2 * math.pi) * z.size(1)
        return -(logp + logpz).mean()




    def load_datamodule(self,data_module):
        '''needed for lightning training to work, it just sets the dataloader for training and validation'''
        self.data_module=data_module





    def sample(self,batch,cond=None,t_stop=1,ema=False):
        '''This is a helper function that samples from the flow (i.e. generates a new sample)
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation'''

        x0 = torch.randn(batch.shape[0],batch.shape[1]).to(self.device)
        logpz = torch.zeros(x0.shape[0], 1, device=self.device)
        x_aug = torch.cat([x0, logpz], dim=1)
        x_aug.requires_grad_(True)
        samples = self.node.trajectory(x_aug, t_span=torch.linspace(1, 0, self.hparams.num_steps).to(self.device),).detach()
        return samples[-1,:,:2]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.max_epochs*self.hparams.num_batches)

        return [optimizer], [sched]

    def training_step(self,batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        sched = self.lr_schedulers()
        sched.step()
        x = batch[0]
        logpz = torch.zeros(x.shape[0], 1, device=self.device)
        x_aug = torch.cat([x, logpz], dim=1)  # Augment x with logpz
        x_aug.requires_grad_(True)  # Ensure x requires gradients

        # Forward pass through Neural ODE
        augmented_traj = self.node.trajectory(x_aug, t_span=torch.linspace(0,1,self.hparams.num_steps).float().to(self.device))

        # Extract state and logpz
        x, logpz = augmented_traj[-1][:, :-1], augmented_traj[-1,:, -1]

        # Compute loss and backpropagate
        loss = self.compute_loss(x, logpz)
        self.manual_backward(loss)
        optimizer.step()
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False,logger=True)


    def on_validation_epoch_start(self):
        self.xhat=[]
        self.x=[]
        self.z=[]
        self.y=[]
        self.prob=[]
        self.vf.train()
        self.node.train()

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x = batch[0]
            self.x.append(x)
            logpz = torch.zeros(x.shape[0], 1, device=self.device)
            x_aug = torch.cat([x, logpz], dim=1)  # Augment x with logpz
            x_aug.requires_grad_(True)  # Ensure x requires gradients

            # Forward pass through Neural ODE
            augmented_traj = self.node.trajectory(x_aug, t_span=torch.linspace(0,1,self.hparams.num_steps).float().to(self.device))

                # Extract state and logpz
            z, logpz = augmented_traj[-1][:, :-1], augmented_traj[-1,:, -1]

            # Compute loss and backpropagate
            loss = self.compute_loss(z, logpz)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False,logger=True)

            xhat=self.sample(x)



        self.xhat.append(xhat.detach())

        self.z.append(z.detach())
        self.y.append(batch[1])