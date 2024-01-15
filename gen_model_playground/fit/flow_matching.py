
import lightning.pytorch as pl
import torch
from torch.distributions import Normal
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher)
from torchdyn.core import NeuralODE
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from gen_model_playground.models.model import Model
import torch
from torchdyn.models import NeuralODE
from torch.distributions import Normal

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




class FM(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.save_name=self.hparams.matching+"flow_matching"+"_"+self.hparams.dataset
        self.save_name += "_cond" if self.hparams.cond_features > 0 else ""
        self.name=self.hparams.name
        self.net = Model(**self.hparams).to("cuda")
        self.flow_matcher = ConditionalFlowMatcher(sigma=1e-4) if self.hparams.matching=="target" else SchrodingerBridgeConditionalFlowMatcher(sigma=1e-4) if not self.hparams.matching=="exact" else ExactOptimalTransportConditionalFlowMatcher()



    def cnf_likelihood(self, data):
        """
        Calculate the likelihood of data using a trained Continuous Normalizing Flow (CNF)
        implemented with torchdyn's NeuralODE, assuming a standard Gaussian base distribution.

        Args:
        - model (NeuralODE): The trained CNF model.
        - data (torch.Tensor): The data for which to calculate the likelihood.

        Returns:
        - torch.Tensor: The likelihood of the data.
        """
        device = data.device
        # The CNF model's forward pass must return both the transformed data and the log Jacobian determinant
        z, log_jac_det = self(data)
        # Base distribution is a standard Gaussian
        base_dist = Normal(torch.zeros(data.size(1)).to(device), torch.ones(data.size(1)).to(device))
        # Calculate the log probability of the base distribution
        logp_base = base_dist.log_prob(z).sum(dim=1)  # Sum over dimensions for multivariate case
        # Adjust the log probability by the log Jacobian determinant
        logp_data = logp_base - log_jac_det.squeeze()
        return logp_data



    def load_datamodule(self,data_module):
        '''needed for lightning training to work, it just sets the dataloader for training and validation'''
        self.data_module=data_module


    def sample(self,batch,cond=None,t_stop=1,return_traj=False):
        '''This is a helper function that samples from the flow (i.e. generates a new sample)
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation'''
        with torch.no_grad():
            x0 = torch.randn(batch.shape).to(self.device)
            wrapped_cnf = torch_wrapper(model=self.net )
            node=NeuralODE(lambda t,x,args: wrapped_cnf(t,x), solver=self.hparams.solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            samples = node.trajectory(x0, t_span=torch.linspace(0, t_stop, self.hparams.num_steps).to(self.device),)
            if not return_traj:
                return samples[-1,:,:]
            else:
                return samples[-1,:,:],samples

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.max_epochs*self.hparams.num_batches//10, max_epochs=self.hparams.max_epochs*self.hparams.num_batches)

        return [optimizer], [sched]



    def training_step(self, batch):
        """training loop of the model, here all the data is passed forward to a gaussian
            This is the important part what is happening here. This is all the training we do """
        opt = self.optimizers()
        sched = self.lr_schedulers()
        sched.step()
        batch= batch[0]
        x0,x1 =torch.randn_like(batch), batch
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0,x1)
        self.net.train()

        vt = self.net(xt,t).cuda()
        opt.zero_grad()
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        opt.step()


        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mean",vt.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/std",vt.std(), on_step=False, on_epoch=True, prog_bar=False)



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
            x0,x1 =torch.randn_like(batch), batch
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0,x1)
            self.net.eval()

            vt = self.net(xt,t).cuda()
            loss = torch.mean((vt - ut) ** 2)
            xhat,traj=self.sample(batch,return_traj=True)
            self.z.append(traj[0])
            self.xhat.append(xhat)

            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/mean",vt.mean(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/std",vt.std(), on_step=False, on_epoch=True, prog_bar=False)

