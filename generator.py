import logging
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Загружаем DPT для глубины один раз
proc = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
dpt = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512").to(device).eval()

# Глобальная переменная для ленивой инициализации пайплайна
_pipe = None

def _init_pipe():
    """Создаёт и возвращает пайплайн с ControlNet (Stable Diffusion XL)."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    logger.info("Инициализация ControlNet и StableDiffusionXLControlNetPipeline...")

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True
    ).to(device)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    pipe.enable_attention_slicing()
    logger.info("Пайплайн инициализирован.")
    return pipe

def get_pipe():
    """Возвращает пайплайн, инициализируя его при необходимости."""
    global _pipe
    if _pipe is None:
        _pipe = _init_pipe()
    return _pipe

def get_depth(img: Image.Image) -> Image.Image:
    """Вычисляет карту глубины из RGB изображения, возвращает в оттенках серого."""
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = dpt(**inputs).predicted_depth

    prediction = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    depth_map = (normalized * 255).astype(np.uint8)

    # Сглаживаем карту глубины
    depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)
    return Image.fromarray(depth_map).convert("L")  # важный момент — серый режим

def generate_design(
    img_path: str,
    prompt: str,
    neg_prompt: str = "low quality, blurry, distortions",
    steps: int = 100,
    g_scale: float = 13,
    c_scale: float = 0.5,
    seed: int | None = 42
):
    src = Image.open(img_path).convert("RGB")
    depth = get_depth(src)

    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None

    pipe = get_pipe()

    out = pipe(
        prompt=[prompt],
        negative_prompt=[neg_prompt],
        image=depth,
        num_inference_steps=steps,
        guidance_scale=g_scale,
        controlnet_conditioning_scale=c_scale,
        generator=gen
    )
    logger.info(f"[DEBUG generate_design] output type: {type(out)}, images count: {len(out.images)}")
    return out.images[0]
