import torch
import torch.nn.functional as F

def least_squares(y_real, y_fake,critic):
    if critic:
        return (F.mse_loss(y_real,torch.ones_like(y_real))+F.mse_loss(y_fake,torch.zeros_like(y_fake)))
    else:
        return 0.5*(F.mse_loss(y_fake,torch.ones_like(y_fake)))

def wasserstein(y_real, y_fake,critic):
    if critic:
        return (-y_real+y_fake).mean()
    else:
        return (-y_fake).mean()

def non_saturating(y_real, y_fake,critic):
    if critic:
        return (F.binary_cross_entropy_with_logits(y_real,torch.ones_like(y_real))+F.binary_cross_entropy_with_logits(y_fake,torch.zeros_like(y_fake))).mean()
    else:
        return (F.binary_cross_entropy_with_logits(y_fake,torch.ones_like(y_fake))).mean()

def gradient_penalty(x, fake,discriminator):
    alpha = torch.rand(x.shape[0], 1, device=x.device)
    alpha = alpha.expand_as(x)
    interpolated = alpha * x + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    y_pred = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.flatten(start_dim=1)
    gradients_norm = gradients.norm(2, dim=1)
    return (torch.nn.functional.relu(gradients_norm - 1) ** 2).mean()