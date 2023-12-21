from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger
from ..models.model import Model
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import os
from ..utils import plot, plot_latent, plot_logprob
class PostTrainingCallback(Callback):
    def __init__(self,logger):
        super().__init__()
        self.logger=logger
    def plot_final(self, trainer, pl_module):
        os.makedirs(f"{os.getcwd()}/plots",exist_ok=True)
        os.makedirs(f"{os.getcwd()}/plots/{pl_module.save_name}",exist_ok=True)
        if  hasattr(pl_module,"xrec") and len(pl_module.xrec)>0:

            figrec,axrec=plot(torch.cat(pl_module.xrec).cpu().numpy(),torch.cat(pl_module.x).cpu().numpy())

            self.logger.log_image("reconstruction", [figrec],trainer.global_step)
            figrec.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_final_rec.pdf",format="pdf")

            plt.close()
            upper=torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy()==1]
            lower=torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy()==0]
        if hasattr(pl_module,"z") and len(pl_module.z)>0:
            figlatent,axlatent=plot_latent(lower,upper)
            self.logger.log_image("latent", [figlatent],trainer.global_step)
            figlatent.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_final_latent.pdf",format="pdf")

            plt.close()
        fig,ax=plot(torch.cat(pl_module.xhat).cpu().numpy(),torch.cat(pl_module.x).cpu().numpy())
        fig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_final_samples.pdf",format="pdf")
        plt.close()
        if isinstance(self.logger,WandbLogger):
            self.logger.log_image("samples", [fig],trainer.global_step)
            # Ensure the module is in eval mode for consistent feature extraction

    def on_train_end(self, trainer, pl_module):
        # Get the dataloader
        pl_module.eval()
        pl_module=pl_module.to("cuda" if torch.cuda.is_available() else "cpu")
        best_model_path=trainer.checkpoint_callback.best_model_path
        best_model = pl_module.__class__.load_from_checkpoint(best_model_path)

        # Now use best_model instead of pl_module for further operations
        best_model = best_model.to("cuda" if torch.cuda.is_available() else "cpu")
        best_model.eval()
        if pl_module.save_name.find("nf")>-1:
            pl_module=pl_module.to("cpu")
            xgrid,ygrid=torch.meshgrid(torch.linspace(pl_module.hparams._min[0],pl_module.hparams._max[0],100),torch.linspace(pl_module.hparams._min[1],pl_module.hparams._max[1],100))
            X=torch.cat((xgrid.reshape(-1,1),ygrid.reshape(-1,1)),dim=-1).to(pl_module.device)
            if pl_module.save_name.find("cnf")>-1:
                with torch.enable_grad():

                    logpz = torch.zeros(X.shape[0], 1, device=pl_module.device)
                    x_aug = torch.cat([X, logpz], dim=1)  # Augment x with logpz
                    x_aug.requires_grad_(True)  # Ensure x requires gradients

                    augmented_traj = pl_module.node.trajectory(x_aug, t_span=torch.linspace(0,1,pl_module.hparams.num_steps).float().to(pl_module.device))
                    prob=torch.exp(augmented_traj[-1,:, -1]).reshape(100,100).detach().numpy()

            else:
                prob=torch.exp(pl_module.flow.log_prob(X).reshape(100,100)).detach().numpy()
            fig,ax,logfig,logax=plot_logprob(xgrid,ygrid,prob)
            self.logger.log_image("logprob", [logfig],trainer.global_step)
            logfig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_logprob.pdf",format="pdf")
            fig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_prob.pdf",format="pdf")
            plt.close()
        pl_module=pl_module.to("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = trainer.datamodule.train_dataloader()
        test_discriminator=Model(in_features=2,out_features=1, num_blocks=2,hidden_features=64).to(pl_module.device)
        opt_d = torch.optim.AdamW(test_discriminator.parameters(), lr=0.01)
        print("Training test discriminator")
        i=trainer.global_step

        # Cycle through the dataloader once
        for _ in range(5):
            best_model.xhat=[]
            best_model.x=[]
            best_model.xrec=[]
            best_model.z=[]
            for batch in dataloader:
                i+=1
                opt_d.zero_grad()
                x,y=batch
                best_model.x.append(x)
                with torch.no_grad():
                    z=torch.normal(0, 1, size=(x.shape[0], 2), device=best_model.device)
                    xhat=best_model(z)
                x=x.to(best_model.device)
                x=torch.cat((x,xhat),dim=0)
                y=torch.cat((torch.ones_like(xhat[:,0]),torch.zeros_like(xhat[:,0])),dim=0)
                d_loss=binary_cross_entropy_with_logits(test_discriminator(x).reshape(-1),y.reshape(-1))
                d_loss.backward()
                opt_d.step()
                trainer.logger.experiment.log({"test/d_loss":d_loss},step=i)
                best_model.xhat.append(xhat)
                if pl_module.save_name.find("ae")>-1:
                    z=pl_module.encoder(x)[0]
                    mu,sigma=z[:,:pl_module.hparams.encoding_dim],z[:,pl_module.hparams.encoding_dim:]
                    best_model.z.append(mu+sigma*torch.randn_like(mu))
                    best_model.xrec.append(pl_module.decoder(pl_module.z[-1]))
            test_discriminator.eval()
        with torch.no_grad():
            x=(torch.from_numpy(make_moons(100*x.shape[0],noise=0.05)[0]).float()-trainer.datamodule.mu)/trainer.datamodule.std
            x=x.to(best_model.device)
            z=torch.normal(0, 1, size=(x.shape[0], 2), device=best_model.device)
            xhat=best_model(z)
            x=torch.cat((x,xhat),dim=0).to(best_model.device)
            y=torch.cat((torch.ones_like(xhat[:,0]),torch.zeros_like(xhat[:,0])),dim=0).to(pl_module.device)
            yhat=test_discriminator(x).sigmoid()
            acc=(yhat.round().reshape(-1)==y.reshape(-1)).float().mean()

        _,b,_=plt.hist(yhat[y==0].cpu().numpy(),bins=np.linspace(0,1,101),label="Generated Samples",alpha=1,linewidth=3,histtype="step")
        plt.hist(yhat[y==1].cpu().numpy(),bins=b,label="Groundtruth Samples",alpha=0.5)
        plt.legend()
        trainer.logger.log_image("test/hist",[plt.gcf()])

        trainer.logger.experiment.log({"test/acc":acc})
        plt.close()
        self.plot_final(trainer,best_model)