import torch
import torch.nn as nn
from Unet import Unet
from tqdm.auto import tqdm


def extract(a, t, x_shape):
    batch = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch, *((1, ) *(len(x_shape) - 1))).to(t.device)


class DDPM(nn.Module):
    def __init__(self, in_c, timestep):
        super().__init__()
        beta_min = 0.0001
        beta_max = 0.02
        self.beta = torch.linspace(beta_min, beta_max, timestep)
        self.alpha = 1 - self.beta
        self.sqrt_inverse_alpha = torch.sqrt(1.0 / self.alpha)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_pre = nn.functional.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sigma = torch.sqrt(self.beta * (1.0 - self.alpha_bar_pre) / (1.0 - self.alpha_bar))
        self.timestep = timestep
        self.denoise_model = Unet(in_c=in_c, base_c=64, embed_c=128, dim_block=[2, 4, 8], timestep=timestep)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        noisy_image = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return noise, noisy_image

    def forward(self, x, t):
        x = self.denoise_model(x, t)
        return x

    def loss(self, noise, predict):
        B, C, H, W = predict.shape
        l2loss = C * H * W * nn.functional.mse_loss(noise, predict)
        return l2loss

    def sample(self, batch, in_c, image_size):
        xt = torch.randn(size=(batch, in_c, image_size, image_size)).cuda()
        imgs = []
        pbar = tqdm(total=self.timestep)
        for i in reversed(range(0, self.timestep)):
            t = torch.full((batch,), i, device=xt.device)
            sqrt_inverse_alpha_t = extract(self.sqrt_inverse_alpha, t, xt.shape)
            beta_t = extract(self.beta, t, xt.shape)
            sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bar, t, xt.shape)
            mean = sqrt_inverse_alpha_t * (xt - beta_t / sqrt_one_minus_alpha_bar_t * DDPM.forward(self, xt, t))
            if i == 0:
                xt = mean
            else:
                z = torch.randn_like(mean)
                sigma_t = extract(self.sigma, t, mean.shape)
                xt = mean + sigma_t * z
            imgs.append(xt)
            pbar.update(1)
        return imgs
