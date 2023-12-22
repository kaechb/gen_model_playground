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
    """
    A PyTorch Lightning Callback for generating and logging plots after training has completed.

    Attributes:
        logger: Logger used to log images, for example, WandbLogger.
    """

    def __init__(self, logger):
        """
        Initializes the PostTrainingCallback with a logger.

        Args:
            logger: An instance of a logger, compatible with PyTorch Lightning.
        """
        super().__init__()
        self.logger = logger

    def plot_final(self, trainer, pl_module):
        """
        Plots and saves the final results of the training.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model) being trained.
        """
        # Create directories for saving plots
        os.makedirs(f"{os.getcwd()}/plots", exist_ok=True)
        os.makedirs(f"{os.getcwd()}/plots/{pl_module.save_name}", exist_ok=True)

        # Plot and save reconstructions if they exist
        if hasattr(pl_module, "xrec") and len(pl_module.xrec) > 0:
            figrec, axrec = plot(torch.cat(pl_module.xrec).cpu().numpy(), torch.cat(pl_module.x).cpu().numpy())
            self.logger.log_image("reconstruction", [figrec], trainer.global_step)
            figrec.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_final_rec.pdf", format="pdf")
            plt.close()

        # Plot and save latent space if available
        if hasattr(pl_module, "z") and len(pl_module.z) > 0:
            upper = torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy() == 1]
            lower = torch.cat(pl_module.z).cpu().numpy()[torch.cat(pl_module.y).cpu().numpy() == 0]
            figlatent, axlatent = plot_latent(lower, upper)
            self.logger.log_image("latent", [figlatent], trainer.global_step)
            figlatent.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_final_latent.pdf", format="pdf")
            plt.close()

        # Plot and save final samples
        fig, ax = plot(torch.cat(pl_module.xhat).cpu().numpy(), torch.cat(pl_module.x).cpu().numpy())
        fig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_final_samples.pdf", format="pdf")
        plt.close()
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image("samples", [fig], trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        """
        Called at the end of the training process. Handles various post-training tasks
        such as evaluating the best model, generating plots, and testing additional components.
        Plots are saved to current working directory in a folder named 'plots'.
        This directory is created automatically if it does not exist.
        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model) being trained.
        """
        # Set the module to evaluation mode and move to the appropriate device
        pl_module.eval()
        pl_module = pl_module.to("cuda" if torch.cuda.is_available() else "cpu")

        # Load the best model from the training checkpoints
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = pl_module.__class__.load_from_checkpoint(best_model_path)
        best_model = best_model.to("cuda" if torch.cuda.is_available() else "cpu")
        best_model.eval()

        # If the model is a normalizing flow, plot and log probability distributions
        if pl_module.save_name.find("nf") > -1:
            pl_module = pl_module.to("cpu")
            xgrid, ygrid = torch.meshgrid(torch.linspace(pl_module.hparams._min[0], pl_module.hparams._max[0], 100),
                                        torch.linspace(pl_module.hparams._min[1], pl_module.hparams._max[1], 100))
            X = torch.cat((xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)), dim=-1).to(pl_module.device)

            # Special handling for continuous normalizing flows
            if pl_module.save_name.find("cnf") > -1:
                with torch.enable_grad():
                    logpz = torch.zeros(X.shape[0], 1, device=pl_module.device)
                    x_aug = torch.cat([X, logpz], dim=1)  # Augment x with logpz
                    x_aug.requires_grad_(True)

                    augmented_traj = pl_module.node.trajectory(x_aug, t_span=torch.linspace(0, 1, pl_module.hparams.num_steps).float().to(pl_module.device))
                    prob = torch.exp(augmented_traj[-1, :, -1]).reshape(100, 100).detach().numpy()
            else:
                prob = torch.exp(pl_module.flow.log_prob(X).reshape(100, 100)).detach().numpy()

            fig, ax, logfig, logax = plot_logprob(xgrid, ygrid, prob)
            self.logger.log_image("logprob", [logfig], trainer.global_step)
            logfig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_logprob.pdf", format="pdf")
            fig.savefig(f"{os.getcwd()}/plots/{pl_module.save_name}/{pl_module.save_name}_prob.pdf", format="pdf")
            plt.close()

        # Move the model back to the appropriate device
        pl_module = pl_module.to("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and train a test discriminator for evaluating generated samples
        dataloader = trainer.datamodule.train_dataloader()
        test_discriminator = Model(in_features=2, out_features=1, num_blocks=2, hidden_features=64).to(pl_module.device)
        opt_d = torch.optim.AdamW(test_discriminator.parameters(), lr=0.01)
        print("Training test discriminator")
        i = trainer.global_step

        # Train the discriminator with both real and generated data
        for _ in range(5):
            best_model.xhat = []
            best_model.x = []
            best_model.xrec = []
            best_model.z = []
            for batch in dataloader:
                i += 1
                opt_d.zero_grad()
                x, y = batch
                best_model.x.append(x)
                with torch.no_grad():
                    z = torch.normal(0, 1, size=(x.shape[0], 2), device=best_model.device)
                    if pl_module.save_name.find("matching"):
                        xhat,z = best_model.sample(z, return_latent=True)

                    else:
                        xhat = best_model(z)
                x = x.to(best_model.device)
                x = torch.cat((x, xhat), dim=0)
                y = torch.cat((torch.ones_like(xhat[:, 0]), torch.zeros_like(xhat[:, 0])), dim=0)
                d_loss = binary_cross_entropy_with_logits(test_discriminator(x).reshape(-1), y.reshape(-1))
                d_loss.backward()
                opt_d.step()
                trainer.logger.experiment.log({"test/d_loss": d_loss}, step=i)
                best_model.xhat.append(xhat)
                if pl_module.save_name.find("ae") > -1:
                    z = pl_module.encoder(x)[0]
                    mu, sigma = z[:, :pl_module.hparams.encoding_dim], z[:, pl_module.hparams.encoding_dim:]
                    best_model.z.append(mu + sigma * torch.randn_like(mu))
                    best_model.xrec.append(pl_module.decoder(best_model.z[-1]))

        # Evaluate the test discriminator and log the results
        test_discriminator.eval()
        with torch.no_grad():
            x = (torch.from_numpy(make_moons(100 * x.shape[0], noise=0.05)[0]).float() - trainer.datamodule.mu) / trainer.datamodule.std
            x = x.to(best_model.device)
            z = torch.normal(0, 1, size=(x.shape[0], 2), device=best_model.device)
            xhat = best_model(z)
            x = torch.cat((x, xhat), dim=0).to(best_model.device)
            y = torch.cat((torch.ones_like(xhat[:, 0]), torch.zeros_like(xhat[:, 0])), dim=0).to(pl_module.device)
            yhat = test_discriminator(x).sigmoid()
            acc = (yhat.round().reshape(-1) == y.reshape(-1)).float().mean()

        # Plot and log histograms of discriminator predictions
        _, bins, _ = plt.hist(yhat[y == 0].cpu().numpy(), bins=np.linspace(0, 1, 101), label="Generated Samples", alpha=1, linewidth=3, histtype="step")
        plt.hist(yhat[y == 1].cpu().numpy(), bins=bins, label="Groundtruth Samples", alpha=0.5)
        plt.legend()
        trainer.logger.log_image("test/hist", [plt.gcf()])
        trainer.logger.experiment.log({"test/acc": acc})
        plt.close()

        # Final plotting and logging using the best model
        self.plot_final(trainer, best_model)
