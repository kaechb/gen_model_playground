
import lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from ..utils import gradient_penalty, least_squares, non_saturating, wasserstein
from ..models.model import Model

class GAN(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Model(in_features=self.hparams.in_features, out_features=self.hparams.encoding_dim, num_blocks=self.hparams.num_blocks,hidden_features=self.hparams.hidden_features,cond_features=self.hparams.cond_features,spectral=False,batch_norm=self.hparams.batch_norm,residual=self.hparams.residual,bias=self.hparams.bias_gen)
        self.discriminator = Model(in_features=self.hparams.in_features, out_features=1, num_blocks=self.hparams.num_blocks,hidden_features=self.hparams.hidden_features,cond_features=self.hparams.cond_features,spectral=self.hparams.spectral,batch_norm=self.hparams.batch_norm,residual=self.hparams.residual,bias=self.hparams.bias_dis)
        self.automatic_optimization = False  # Disable automatic optimization
        self.gan_type=self.hparams.gan_type
        self.loss=least_squares if self.hparams.gan_type=="lsgan" else wasserstein if self.hparams.gan_type=="wgan" else non_saturating
        self.spectral=self.hparams.spectral
        self.save_name=self.hparams.gan_type
        self.name=self.hparams.name



    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=2*self.hparams.lr)
        sched_g = LinearWarmupCosineAnnealingLR(opt_g, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.max_epochs*self.hparams.num_batches)
        sched_d = LinearWarmupCosineAnnealingLR(opt_d, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.max_epochs*self.hparams.num_batches)
        return [opt_g, opt_d], [sched_g, sched_d]


    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        batch=batch[0]
        # Get the optimizers

        opt_g, opt_d = self.optimizers()
        sched_d, sched_g = self.lr_schedulers()
        sched_d.step()
        sched_g.step()
        # Sample noise and generate fake data
        with torch.no_grad():
            z = torch.normal(0, 1, size=(batch.shape[0], 2), device=self.device)
            xhat = self.generator(z)


        # Train discriminator
        opt_d.zero_grad()
        d_loss = self.loss(self.discriminator(batch), self.discriminator(xhat),critic=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, logger=True)

        if self.gan_type=="wgan" and not self.spectral:

            gp=10*gradient_penalty(batch,xhat,self.discriminator)
            d_loss=d_loss+gp
            self.log('gp', gp, on_step=True, on_epoch=True, logger=True)

        self.manual_backward(d_loss)
        opt_d.step()


        if self.global_step > 100:
        # Train generator
            opt_g.zero_grad()
            z=torch.normal(0, 1, size=(batch.shape[0], 2), device=self.device)
            xhat = self.generator(z)
            g_loss =self.loss(None,self.discriminator(xhat),critic=False)
            self.manual_backward(g_loss)
            opt_g.step()

            # Log generator loss
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, logger=True)
    def on_validation_epoch_start(self):
        self.xhat=[]
        self.x=[]
        self.z=[]
        self.y=[]
    def validation_step(self, batch, batch_idx):
        # Sample noise and generate fake data
        x=batch[0]
        cond=batch[1]

        with torch.no_grad():
            z = torch.normal(0, 1, size=(x.shape[0], 2), device=self.device)
            xhat = self.generator(z)
            yhat=self.discriminator(xhat)
            y=self.discriminator(x)
            loss=self.loss(y,yhat,critic=True)
            self.log('val/loss', loss, on_step=True, on_epoch=True, logger=True)

        self.xhat.append(xhat)
        self.y.append(cond)
        self.z.append(z)

        self.x.append(x)

    def test_step(self, batch,batch_idx):
        with torch.no_grad():
            z = torch.normal(0, 1, size=(batch[0].shape[0], 2), device=self.device)
            xhat = self.generator(z)
            return xhat

