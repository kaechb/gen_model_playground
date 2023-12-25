
import lightning.pytorch as pl
import numpy as np
import scipy
class FrechetDistanceCallback(pl.Callback):
    def __init__(self, num_batches=5):
        super().__init__()
        self.num_batches = num_batches

    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure the module is in eval mode
        pl_module.eval()

        real_features = []
        fake_features = []
        fid_scores = []

        # Loop through the validation data
        for i, (real_data, fake_data) in enumerate(zip(pl_module.x, pl_module.xhat)):
            if i >= self.num_batches:
                break

            # Obtain features and convert to NumPy
            _, real_feature = pl_module.discriminator(real_data, feature_matching=True)
            _, fake_feature = pl_module.discriminator(fake_data, feature_matching=True)
            real_features.append(real_feature.cpu().numpy())
            fake_features.append(fake_feature.cpu().numpy())

            # Calculate FID score for the batch
            mu_real, sigma_real = self.calculate_activation_statistics_np(np.vstack(real_features))
            mu_fake, sigma_fake = self.calculate_activation_statistics_np(np.vstack(fake_features))
            fid_score = self.calculate_frechet_distance_np(mu_real, sigma_real, mu_fake, sigma_fake)
            fid_scores.append(fid_score)

        # Average FID score
        avg_fid_score = np.mean(fid_scores)

        # Log the FID score
        self.log("avg_FID_Score",avg_fid_score, on_step=False, on_epoch=True, logger=True)
        # Set model back to train mode
        pl_module.train()

    @staticmethod
    def calculate_activation_statistics_np(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    @staticmethod
    def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fd = np.sum(diff ** 2) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fd