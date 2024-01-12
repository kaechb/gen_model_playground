import torch
from gen_model_playground.utils import plot, plot_latent
import os
import lightning as pl
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger

class PlotCallback(pl.Callback):
    """
    A PyTorch Lightning Callback for generating and logging plots during model validation.

    Attributes:
        logger: Logger used to log images, for example, WandbLogger.
    """

    def __init__(self, logger):
        """
        Initializes the PlotCallback with a logger.

        Args:
            logger: An instance of a logger, compatible with PyTorch Lightning.
        """
        super().__init__()
        self.logger = logger

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Callback function that is called at the end of every validation epoch.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model) being trained.
        """
        # Create directories for saving plots
        lims=[[pl_module.hparams.scaled_min[0]*1.1,pl_module.hparams.scaled_max[0]*1.1]
              ,[pl_module.hparams.scaled_min[1]*1.1,pl_module.hparams.scaled_max[1]*1.1]]

        os.makedirs(f"{os.getcwd()}/plots", exist_ok=True)
        os.makedirs(f"{os.getcwd()}/plots/{pl_module.save_name}", exist_ok=True)

        # Plot and save reconstructions if they exist
        if hasattr(pl_module, "xrec"):
            figrec, axrec = plot(torch.cat(pl_module.xrec).cpu().numpy(), torch.cat(pl_module.x).cpu().numpy(),lims=lims)
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image("reconstruction", [figrec], trainer.global_step)
            figrec.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_rec.pdf", format="pdf")
            plt.close()

        # Plot and save latent representations if they exist
        if hasattr(pl_module, "z") and len(pl_module.z) > 0 and (pl_module.save_name.find("nf")>-1 or pl_module.save_name.find("matching")>-1):

            lower = torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy() == 1]
            upper = torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy() == 0]


            figlatent, axlatent = plot_latent(lower, upper)
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image("latent", [figlatent], trainer.global_step)
            figlatent.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_latent.pdf", format="pdf")
            plt.close()

        # Plot and save generated samples
        fig, ax = plot(torch.cat(pl_module.xhat).cpu().numpy(), torch.cat(pl_module.x).cpu().numpy(),lims=lims)
        fig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_samples.pdf", format="pdf")

        # Log the samples image if the logger is WandbLogger
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image("samples", [fig], trainer.global_step)
        plt.close()
