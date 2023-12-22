import argparse
from gen_model_playground.data import TwoMoonsDataModule
from gen_model_playground.fit.gan import GAN
from gen_model_playground.fit.vae import VAE
from gen_model_playground.fit.flow import Flow
from gen_model_playground.fit.flow_matching import FM
from gen_model_playground.fit.cnf import CNF
from gen_model_playground.fit.ddpm import DDPM
import lightning as pl
from gen_model_playground.callbacks.FID import FrechetDistanceCallback
from gen_model_playground.callbacks.plot import PlotCallback
from gen_model_playground.callbacks.ema import EMA, EMAModelCheckpoint
from gen_model_playground.callbacks.posttrain import PostTrainingCallback

def get_args():
    """
    Parses command line arguments for configuring the model training.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Configuration")

    parser.add_argument('--logger', type=str, default='wandb', help='Logger type, options:["wandb","tensorboard","csv"] (default: wandb)')
    # Model selection argument
    parser.add_argument('--name', type=str, default='ddpm', help='model type, options:["gan","vae","ae","nf","cnf","flow_matching","ddpm"] (default: gan)')

    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--num_batches', type=int, default=1000, help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')

    # Exponential Moving Average (EMA) parameters
    parser.add_argument("--ema", default=False, help="Use EMA")
    parser.add_argument("--decay", default=0.999, help="EMA decay")
    parser.add_argument('--evaluate_ema_weights_instead', default=False, help='Use EMA weights instead of original weights')
    parser.add_argument('--ema_start', type=int, default=10000, help='Step to start EMA (default: 10_000)')

    # GAN-specific parameters
    parser.add_argument('--gan_type', type=str, default='lsgan', help='Type of GAN (default: lsgan)')

    # General model parameters
    parser.add_argument('--spectral', default=False, help='Use spectral normalization')
    parser.add_argument('--residual', default=True, help='Use residual connections')
    parser.add_argument('--batch_norm', default=True, help='Use batch normalization')
    parser.add_argument('--feature_matching', default=False, help='Use feature matching')
    parser.add_argument('--bias_gen', default=False, help='Use bias in generator')
    parser.add_argument('--bias_dis', default=True, help='Use bias in discriminator')

    # Model architecture parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of blocks (default: 2)')
    parser.add_argument('--hidden_features', type=int, default=128, help='Number of units (default: 128)')
    parser.add_argument('--in_features', type=int, default=2, help='Number of features (default: 2)')
    parser.add_argument('--out_features', type=int, default=2, help='Number of features (default: 2)')
    parser.add_argument('--encoding_dim', type=int, default=2, help='Number of features in latent space (default: 2)')
    parser.add_argument('--cond_features', type=int, default=0, help='Number of conditions (default: 0)')

    # Flow model parameters
    parser.add_argument('--num_bins', type=int, default=8, help='Number of bins (default: 8)')
    parser.add_argument('--tail_bound', type=float, default=4, help='Tail bound (default: 4.)')
    parser.add_argument('--ncoupling', type=int, default=16, help='Number of coupling layers (default: 16)')
    parser.add_argument('--flow_type', type=str, default="spline", help='Affine Flows or RQSplines (default: spline)')
    parser.add_argument("--matching", default="exact", help='Flow Matching, options: ["exact","target","schroedinger" (default: "exact")')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps (default: 100)')

    args = parser.parse_args()

    # Adjust the spectral normalization flag based on the model name
    args.spectral = False if args.name not in ["gan", "cnf"] else args.spectral
    return args

if __name__ == "__main__":
    """
    Main execution block for training various machine learning models.
    """
    # Parse command-line arguments
    args = get_args()
    args = args.__dict__

    # Initialize and setup the data module
    datamodule = TwoMoonsDataModule(batch_size=256, num_batches=args["num_batches"])
    datamodule.setup()
    # Update arguments with data-specific parameters

    args["_min"] = datamodule._min[0]
    args["_max"] = datamodule._max[0]

    # Initialize WandbLogger for experiment tracking
    logger = pl.pytorch.loggers.WandbLogger(project="thesis_experiments", save_dir="/beegfs/desy/user/kaechben/thesis_experiments")
    # Update arguments with configurations from the logger if available
    if len(logger.experiment.config.keys()) > 0:
        args.update(**logger.experiment.config)

    # Model selection based on the command-line argument
    if args["name"] == "gan":
        model = GAN(**args)
    elif args["name"] in ["vae", "ae"]:
        model = VAE(**args)
    elif args["name"] == "nf":
        model = Flow(**args)
    elif args["name"] == "flow_matching":
        args["time_features"] = 1
        model = FM(**args)
    elif args["name"] == "cnf":
        model = CNF(**args)
        args["batch_norm"] = False  # CNF does not support batch norm
    elif args["name"] == "ddpm":
        args["time_features"] = 1
        model = DDPM(**args)
    else:
        raise RuntimeError("Unknown model name")

    # Log the code to the logger
    logger.experiment.log_code(".")

    # Setup callbacks for training
    callbacks = [
        PlotCallback(logger),
        EMAModelCheckpoint(save_last=True, save_top_k=2, monitor="step",
                        mode="max", filename="{epoch}_{step}",
                        every_n_train_steps=0, every_n_epochs=args["max_epochs"]//2,
                        save_on_train_epoch_end=False),
        EMA(**args),
        PostTrainingCallback(logger=logger),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    # Additional callback for GANs
    if args["name"] == "gan":
        callbacks.append(FrechetDistanceCallback(num_batches=5))

    # Initialize and configure the trainer
    trainer = pl.Trainer(
        enable_progress_bar=False,
        max_epochs=args["max_epochs"],
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=args["max_epochs"] // 2,
        log_every_n_steps=100,
        inference_mode=False
    )

    # Start the training process
    trainer.fit(model, datamodule)
