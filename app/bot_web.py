import os
import logging
from deep_translator import GoogleTranslator, exceptions
import asyncio
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота 
TOKEN   = "7378475396:AAHI8eSFvJVl4BjaEVQ_kAGJ-AB75O9wE8Q"

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

BASE_DIR = Path(__file__).parent
weights_path = BASE_DIR / "resnetunet_mask_weights.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Генераторы
from generator import generate_design
from hybrid_model_web import load_hybrid_model, transform, denormalize
# generate_custom определяем здесь, используя загруженную гибридную модель

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

def generate_custom(img_path: str, prompt: str) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(hybrid.device)
    with torch.no_grad():
        pil_list = hybrid(x, prompt=prompt)
    return pil_list[0]

# Инициализация гибридной модели
hybrid = load_hybrid_model(
    weights_path=str(weights_path),
    # sd_model_id="stabilityai/stable-diffusion-2-1-base",
    sd_model_id="stabilityai/stable-diffusion-2-inpainting", 
    device=DEVICE
)

def translate_prompt(prompt: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(prompt)
    except exceptions.DeepTranslatorError as e:
        logger.warning(f"Не смогли перевести, используем оригинал: {e}")
        return prompt

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Пришлите фото пустой комнаты, затем выберите модель генерации."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    path = os.path.join(TMP_DIR, f"{update.effective_user.id}_src.jpg")
    await file.download_to_drive(path)
    context.user_data["photo"] = path

    keyboard = [
        [InlineKeyboardButton("SDXL-ControlNet", callback_data="model_sdxl")],
        [InlineKeyboardButton("Custom-HybridNet", callback_data="model_custom")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Фото получено! Выберите модель:", reply_markup=reply_markup
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data
    context.user_data["model"] = choice
    await query.edit_message_text(
        f"Модель {'SDXL' if choice=='model_sdxl' else 'HNET'} выбрана. Теперь пришлите описание интерьера."
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "photo" not in context.user_data:
        return await update.message.reply_text("Сначала пришлите фото комнаты.")
    if "model" not in context.user_data:
        return await update.message.reply_text("Сначала выберите модель ControlNet или Custom.")

    prompt = update.message.text
    src_path = context.user_data.pop("photo")
    model_id = context.user_data.pop("model")
    await update.message.reply_text("Генерирую, подождите…")

    prompt = translate_prompt(prompt)
    loop = asyncio.get_running_loop()
    try:
        if model_id == "model_sdxl":
            func = generate_design
        else:
            func = generate_custom
        result_img = await loop.run_in_executor(None, func, src_path, prompt)
    except Exception as e:
        logger.exception("Ошибка генерации:")
        await update.message.reply_text(f"Ошибка во время генерации: {e}")
        return

    out_path = os.path.join(TMP_DIR, f"{update.effective_user.id}_res.jpg")
    result_img.save(out_path)
    with open(out_path, "rb") as f:
        await update.message.reply_photo(f)

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(button, pattern="^model_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()

if __name__ == "__main__":
    main()
