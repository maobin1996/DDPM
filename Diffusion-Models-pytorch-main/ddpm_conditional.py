import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        # noise_steps: 噪声步骤的数量，表示模型在生成图像时的迭代次数。
        # beta_start 和 beta_end: 控制噪声调度的参数，决定每一步的噪声强度。
        # img_size: 生成图像的尺寸（宽和高都是256）。
        # device: 指定计算设备，通常是“cuda”表示使用GPU。

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # prepare_noise_schedule(): 生成一个从 beta_start 到 beta_end 线性变化的噪声调度。
        # alpha: 计算每一步的α值，表示图像信息的保留程度。
        # alpha_hat: 计算累积的α值，表示从初始图像到当前步骤的图像信息保留程度。
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    # noise_images: 生成带噪声的图像。
    # 输入 x 是当前图像，t 是当前时间步。
    # 通过计算噪声和图像的加权组合，返回带噪声的图像和生成的噪声。
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    # sample_timesteps: 随机选择 n 个时间步用于生成图像。
    # sample: 使用给定的模型生成 n 张新图像。

    # 首先，初始化一个随机噪声图像 x。
    # 反向迭代每一个时间步，从 self.noise_steps 到 1，逐步去噪。
    # predicted_noise: 模型预测的噪声。
    # uncond_predicted_noise: 无条件生成的噪声，用于条件生成的调整。
    # 通过线性插值调整噪声预测。
    # 更新图像 x，结合当前的噪声和模型的预测。
    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            #在一个从 self.noise_steps -1 到1 的反向循环中，使用 tqdm 为该循环提供进度条显示。
            # 这通常用于需要迭代多个步骤的任务，尤其是在图像处理、信号处理或其他需要噪声步骤的算法中，
            # 目的是给用户提供运行进度的可视化反馈。
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # torch.ones(n) 创建一个包含 n 个元素的张量，所有元素的值均为1。
                # torch.ones(n) * i,这会将每个元素乘以 i，结果是一个大小为 n 的张量，所有元素的值均为 i。
                t = (torch.ones(n) * i).long().to(self.device)
                #这里调用一个名为 model 的深度学习模型，将输入 x、时间步张量 t 和标签 labels作为输入传递给模型
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    #torch.lerp 是一个线性插值函数。它接受三个参数：起始点 start，终点 end 和权重 weight。函数的计算方式是：
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                #是一个张量或数组，其中存储了不同时间步的参数。在很多扩散模型中，通常表示在每个时间步骤下的某种系数，可能与噪声的处理或图像的生成有关
                alpha = self.alpha[t][:, None, None, None] 
                #
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    

    # 将生成的图像 x 归一化到[0, 255]范围，并转换为无符号8位整数格式，以便保存或显示。

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:  #随机丢弃标签
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


#普通模型（model）：

# 这是在每个训练轮次中直接更新的模型。它是当前训练过程中的主要模型，随着每次迭代不断调整参数，以减少训练数据上的损失。保存这个模型可以用于后续的推理或评估。
# EMA模型（ema_model）：

# EMA（Exponential Moving Average）模型是通过对普通模型的权重进行指数滑动平均计算得到的。EMA模型通常能提供更平滑和稳定的预测，因为它在训练过程中对参数变动进行了平滑处理。
# 保存这个模型常用于生成更高质量的样本，因为它通常比普通模型表现更好。
# 优化器状态（optimizer）：

# 保存优化器的状态是为了在训练过程中能够恢复到当前的训练状态。这包括学习率、动量等信息，确保在中断后可以从上次的状态继续训练，而不需要重新开始。

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 14
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = "/data/common/maobin/datasets/cifar10-32/train/"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

