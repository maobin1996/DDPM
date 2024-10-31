import torch
import os
from torchvision import transforms
from PIL import Image
import copy
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from ddpm_conditional import Diffusion
from modules import UNet_conditional

def load_model(model_class, checkpoint_path, device):
    model = model_class(num_classes=10)  # 根据你的模型定义调整num_classes
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model

def generate_noise_image(img_size, device):
    # 生成噪声图像
    noise_image = torch.randn(10, 3, img_size, img_size).to(device)  # 生成随机噪声
    return noise_image

def predict(model, diffusion, noise_image, labels, device):
    # 生成时间步
    t = diffusion.sample_timesteps(noise_image.shape[0]).to(device)

    # 进行预测
    with torch.no_grad():
        predicted_noise = model(noise_image, t, labels.to(device))
    print(predicted_noise.shape)
    # predicted_noise = predicted_noise.unsqueeze(0).expand(labels.size(0), -1)
    
   
    # 反向去噪（如果需要）
    denoised_image = diffusion.sample(model, n=len(labels), labels=labels)

    return denoised_image

def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.image_size = 64
    # 假设你有一个args对象，包含运行参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_path = os.path.join("models", args.run_name, "ckpt.pt")  # 模型路径
    ema_model_path ="models/DDPM_conditional/ema_ckpt.pt" # EMA模型路径

    # 加载模型
    # model = load_model(UNet_conditional, model_path, device)
    ema_model = load_model(UNet_conditional, ema_model_path, device)

    # 生成噪声图像
    img_size = args.image_size  # 或者根据你的设置
    noise_image = generate_noise_image(img_size, device)

    # 进行预测
    diffusion = Diffusion(img_size=args.image_size, device=device)
    labels = torch.arange(10).long().to(device)  # 假设你有10个类别
    predicted_image = predict(ema_model, diffusion, noise_image, labels, device)
    # 保存或展示预测结果
    save_images(predicted_image, "predicted_image.jpg")

if __name__ == "__main__":
    main()