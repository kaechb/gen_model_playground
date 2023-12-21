
import lightning.pytorch as pl
import torch
import torch.nn as nn
#from src.models.coimport torchdiffeqmponents.diffusion import VPDiffusionSchedule
from torch import nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from ..models.model import Model
import torch

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model,use_t=True):
        super().__init__()
        self.model = model
        self.use_t = use_t

    def forward(self, t, x, *args, **kwargs):
        if self.use_t:
            return self.model( x,t.repeat(x.shape[0])[:, None],)
        else:
            print("not using time, are you sure?")
            return self.model(x=x)




class DDPM(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.save_name=self.hparams.name
        self.name=self.hparams.name
        #This is the Normalizing flow model to be used later, it uses as many
        #coupling_layers as given in the config

        self.net = Model(**self.hparams).to("cuda")
        s = 0.008 #noise variance beta
        self.timesteps = torch.tensor(range(0, self.hparams.num_steps), dtype=torch.float32)
        self.schedule = torch.cos((self.timesteps / self.hparams.num_steps + s) / (1 + s) * torch.pi / 2)**2
        self.baralphas =self.schedule/self.schedule[0]
        self.betas=1-self.baralphas/ torch.cat([self.baralphas[0:1], self.baralphas[:-1]])
        self.alphas=1-self.betas
        self.loss=nn.MSELoss()



    def load_datamodule(self,data_module):
        '''needed for lightning training to work, it just sets the dataloader for training and validation'''
        self.data_module=data_module


    def forward(self,x):
        return self.sample(x)


    def sample(self,batch,cond=None,t_stop=1,num_steps=100,ema=False):
        '''This is a helper function that samples from the flow (i.e. generates a new sample)
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation'''
        with torch.no_grad():
            x0 = torch.randn(batch.shape).to(self.device)

            xt = [x0]
            for t in range(self.hparams.num_steps-1, 0, -1):
                predicted_noise = self.net(x, torch.full([len(batch), 1], t))
                # See DDPM paper between equations 11 and 12
                x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1-self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = self.betas[t]
                    std = variance ** (0.5)
                    x += std * torch.randn(size=(len(batch), batch.shape[1]))
                xt += [x]
                return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.max_epochs*self.hparams.num_batches)

        return [optimizer], [sched]

    def noise(self,X, t):
        eps = torch.randn(size=X.shape)
        noised = (self.baralphas[t] ** 0.5).repeat(1, X.shape[1]) * X + ((1 - self.baralphas[t]) ** 0.5).repeat(1, X.shape[1]) * eps
        return noised, eps

    def training_step(self, batch):
        """training loop of the model, here all the data is passed forward to a gaussian
            This is the important part what is happening here. This is all the training we do """
        opt = self.optimizers()
        sched = self.lr_schedulers()
        sched.step()
        batch= batch[0]

        noised, eps = self.noise(batch, self.timesteps)
        predicted_noise = self.net(noised, self.timesteps)
        loss = self.loss(predicted_noise, eps)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)




    def on_validation_epoch_start(self):
        self.xhat=[]
        self.x=[]
        self.z=[]
        self.y=[]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.y.append(batch[1])
            batch=batch[0]
            self.x.append(batch)
            batch= batch[0]

            noised, eps = self.noise(batch, self.timesteps)
            predicted_noise = self.net(noised, self.timesteps)
            loss = self.loss(predicted_noise, eps)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.xhat=self.sample(batch)

        return loss
