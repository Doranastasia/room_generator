import os
import logging
import asyncio
import time
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from deep_translator import GoogleTranslator
from langdetect import detect

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)


# Токен бота 
TOKEN = os.environ["BOT_TOKEN"]
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)
BASE_DIR = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_instances = {
    "model_custom": None,
    "model_sdxl": None
}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def translate_prompt(prompt: str) -> str:
    try:
        lang = detect(prompt)
        return prompt if lang == "en" else GoogleTranslator(source='auto', target='en').translate(prompt)
    except Exception as e:
        logger.warning(f"Ошибка перевода: {e}")
        return prompt

def generate_custom(img_path: str, prompt: str) -> Image.Image:
    from hybrid_model_web import load_hybrid_model
    if model_instances["model_custom"] is None:
        weights_path = str(BASE_DIR / "best_resnetunet_mask_weights_20_new.pth")
        model_instances["model_custom"] = load_hybrid_model(
            weights_path=weights_path,
            sd_model_id="stabilityai/stable-diffusion-2-inpainting",
            device=DEVICE
        )
        logger.info("Custom-HybridNet загружена.")
    model = model_instances["model_custom"]

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pil_list = model(x, prompt=prompt)
    return pil_list[0]

def generate_sdxl(img_path: str, prompt: str) -> Image.Image:
    from generator import generate_design
    if model_instances["model_sdxl"] is None:
        model_instances["model_sdxl"] = generate_design
        logger.info("SDXL-ControlNet готов к использованию.")
    return model_instances["model_sdxl"](img_path, prompt)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "Добро пожаловать! Вот как работает бот:\n"
        "1. Пришлите фото комнаты\n"
        "2. Пришлите описание интерьера\n"
        "3. Выберите модель генерации\n"
        "4. Получите результат!\n\n"
        "Начните с отправки фото комнаты."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.reply_to_message and update.message.reply_to_message.reply_markup:
        return

    file = await update.message.photo[-1].get_file()
    path = os.path.join(TMP_DIR, f"{update.effective_user.id}_src.jpg")
    await file.download_to_drive(path)
    context.user_data["photo"] = path

    if "prompt" in context.user_data:
        if "model" not in context.user_data:
            keyboard = [
                [InlineKeyboardButton("SDXL-ControlNet", callback_data="model_sdxl")],
                [InlineKeyboardButton("Custom-HybridNet", callback_data="model_custom")]
            ]
            await update.message.reply_text(
                "Фото и описание получены! Выберите модель:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.message.reply_text("Идет генерация... Подождите, пожалуйста!")
            await generate_image_and_send(update, context)
    else:
        await update.message.reply_text("Фото получено! Теперь пришлите описание интерьера.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.reply_to_message and update.message.reply_to_message.reply_markup:
        return

    context.user_data["prompt"] = translate_prompt(update.message.text)

    if "photo" in context.user_data:
        if "model" not in context.user_data:
            keyboard = [
                [InlineKeyboardButton("SDXL-ControlNet", callback_data="model_sdxl")],
                [InlineKeyboardButton("Custom-HybridNet", callback_data="model_custom")]
            ]
            await update.message.reply_text(
                "Фото и описание получены! Выберите модель:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.message.reply_text("Идет генерация... Подождите, пожалуйста!")
            await generate_image_and_send(update, context)
    else:
        await update.message.reply_text("Описание получено! Теперь пришлите фото комнаты.")

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data
    chat_id = query.message.chat_id

    if choice in ["new_prompt", "new_photo", "new_both"]:
        if choice == "new_prompt":
            context.user_data.pop("prompt", None)
            await query.edit_message_text(text="Пришлите новое описание интерьера.")
        elif choice == "new_photo":
            context.user_data.pop("photo", None)
            await query.edit_message_text(text="Пришлите новое фото комнаты.")
        elif choice == "new_both":
            context.user_data.pop("photo", None)
            context.user_data.pop("prompt", None)
            await query.edit_message_text(text="Пришлите новое фото и описание интерьера.")
        return

    if choice == "change_model":
        context.user_data.pop("model", None)
        keyboard = [
            [InlineKeyboardButton("SDXL-ControlNet", callback_data="model_sdxl")],
            [InlineKeyboardButton("Custom-HybridNet", callback_data="model_custom")]
        ]
        await query.edit_message_text(
            text="Выберите новую модель:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    if choice.startswith("model_"):
        context.user_data["model"] = choice
        model_name = "SDXL-ControlNet" if choice == "model_sdxl" else "Custom-HybridNet"
        
        await query.edit_message_text(text=f"Выбрана модель: {model_name}\nИдет генерация... Подождите, пожалуйста!")
        
        await generate_image_and_send(update, context)

async def generate_image_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        chat_id = update.message.chat_id
    elif update.callback_query:
        chat_id = update.callback_query.message.chat_id
    else:
        logger.error("Не удалось определить chat_id")
        return

    prompt = context.user_data.get("prompt")
    src_path = context.user_data.get("photo")
    model_id = context.user_data.get("model")

    if not all([prompt, src_path, model_id]):
        await context.bot.send_message(chat_id=chat_id, text="Ошибка: недостаточно данных для генерации.")
        return

    try:
        start_time = time.time()
        loop = asyncio.get_running_loop()
        func = generate_sdxl if model_id == "model_sdxl" else generate_custom
        result_img = await loop.run_in_executor(None, func, src_path, prompt)
        elapsed = round(time.time() - start_time, 2)
    except Exception as e:
        logger.exception("Ошибка генерации:")
        await context.bot.send_message(chat_id=chat_id, text=f"Произошла ошибка генерации: {e}")
        return

    out_path = os.path.join(TMP_DIR, f"{update.effective_user.id}_res.jpg")
    result_img.save(out_path)
    with open(out_path, "rb") as f:
        await context.bot.send_photo(
            chat_id=chat_id, 
            photo=f, 
            caption=f"Готово! Время генерации: {elapsed} сек."
        )

    keyboard = [
        [InlineKeyboardButton("Новое описание", callback_data="new_prompt")],
        [InlineKeyboardButton("Другое фото", callback_data="new_photo")],
        [InlineKeyboardButton("Новое фото и описание", callback_data="new_both")],
        [InlineKeyboardButton("Сменить модель", callback_data="change_model")]
    ]
    await context.bot.send_message(
        chat_id=chat_id,
        text="Что хотите сделать дальше?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(button))
    
    app.run_polling()

if __name__ == "__main__":
    main()
