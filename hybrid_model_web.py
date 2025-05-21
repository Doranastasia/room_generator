import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from diffusers import StableDiffusionInpaintPipeline
from PIL import ImageEnhance
import numpy as np
import logging

logger = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


def denormalize(t: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    t = t.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.in_conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )
        self.maxpool = base_model.maxpool
        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)

        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        x0 = self.in_conv(x)
        x1 = self.maxpool(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x0)

        x = self.final_upsample(x)
        x = torch.sigmoid(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

class HybridFurnitureModelInpaint(nn.Module):
    def __init__(self, resnet_unet: ResNetUNet, sd_model_id: str, device: str):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resnet_unet = resnet_unet.to(self.device).eval()
        self.sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16 if self.device.type=="cuda" else torch.float32,
            safety_checker=None,
        ).to(self.device)
        self.sd_pipe.enable_attention_slicing()
        self.resize = transforms.Resize((256,256), interpolation=Image.LANCZOS)

    def forward(self,
                x: torch.Tensor,
                prompt: str="modern furniture in realistic interior",
                mask_threshold: float=0.4,
                strength: float=0.7,
                guidance_scale: float=12.0,
                steps: int=150) -> list[Image.Image]:

        pil_outputs: list[Image.Image] = []
        for i in range(x.size(0)):
            inp = x[i:i+1].to(self.device)
            with torch.no_grad():
                mask_pred = self.resnet_unet(inp)
            mask = (mask_pred > mask_threshold).float()
            if mask.mean() < 0.05:
                # возвращаем исходник
                pil = transforms.ToPILImage()(denormalize(inp.squeeze(0).cpu()))
                pil_outputs.append(self.resize(pil))
                continue

            img_tensor = denormalize(inp.squeeze(0).cpu()).clamp(0,1)

            pil_img = self.resize(transforms.ToPILImage()(img_tensor)) 
                    
            mask_np = mask.cpu().numpy()
            if mask_np.ndim == 4:
                # Если размерность (B, C, H, W), берем первый батч и первый канал
                mask_np = mask_np[0, 0, :, :]
            elif mask_np.ndim == 3:
                # Если размерность (B, H, W) или (C, H, W), тоже выбираем правильно
                mask_np = mask_np[0, :, :]  # либо mask_np[0] если это (B,H,W)
            mask_np = (mask_np * 255).astype(np.uint8)

            pil_mask = Image.fromarray(mask_np).resize((256, 256), Image.LANCZOS).convert("L")

            result = self.sd_pipe(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
            ).images[0]

            pil_outputs.append(result)

        return pil_outputs

def load_hybrid_model(weights_path: str, sd_model_id: str, device: str):
    resnet = ResNetUNet(n_classes=1)
    state = torch.load(weights_path, map_location=device)
    resnet.load_state_dict(state)
    model = HybridFurnitureModelInpaint(resnet, sd_model_id, device)
    return model.eval()