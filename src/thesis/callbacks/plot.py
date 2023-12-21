import torch
from ..utils import plot, plot_latent
import os
import lightning as pl
from my_cmaps import cmap
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
class PlotCallback(pl.Callback):
    def __init__(self,logger):
        super().__init__()
        self.logger=logger

    def on_validation_epoch_end(self, trainer, pl_module):
        os.makedirs(f"{os.getcwd()}/plots",exist_ok=True)
        os.makedirs(f"{os.getcwd()}/plots/{pl_module.save_name}",exist_ok=True)
        if  hasattr(pl_module,"xrec"):

            figrec,axrec=plot(torch.cat(pl_module.xrec).cpu().numpy(),torch.cat(pl_module.x).cpu().numpy())

            self.logger.log_image("reconstruction", [figrec],trainer.global_step)
            figrec.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_rec.pdf",format="pdf")

            plt.close()

       

        if  hasattr(pl_module,"z") and len(pl_module.z)>0:
            lower=torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy()==1]
            upper=torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy()==0]

            figlatent,axlatent=plot_latent(lower,upper)
            self.logger.log_image("latent", [figlatent],trainer.global_step)
            figlatent.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_latent.pdf",format="pdf")

            plt.close()
        fig,ax=plot(torch.cat(pl_module.xhat).cpu().numpy(),torch.cat(pl_module.x).cpu().numpy())
        fig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_samples.pdf",format="pdf")
        plt.close()
        if isinstance(self.logger,WandbLogger):
            self.logger.log_image("samples", [fig],trainer.global_step)
            # Ensure the module is in eval mode for consistent feature extraction
