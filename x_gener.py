import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

# Папка для сохранения
os.makedirs("XDataset/X", exist_ok=True)

# Можно     скачать .ttf шрифты и добавить сюда
FONTS = [
    "arial.ttf",
    "times.ttf",
    "calibri.ttf",
    "Handlee-Regular.ttf"
]

def generate_x_image(idx, size=28):
    # создаем белый фон
    img = Image.new("L", (size*2, size*2), 255)  # запас, чтобы потом обрезать
    draw = ImageDraw.Draw(img)

    # выбираем случайный шрифт
    font_path = random.choice(FONTS)
    try:
        font = ImageFont.truetype(font_path, random.randint(18, 26))
    except:
        font = ImageFont.load_default()

    # рисуем X
    draw.text((size//2, size//2), random.choice(["x","X","х","Х"]), fill=0, font=font)

    # рандомные повороты
    angle = random.randint(-30, 30)
    img = img.rotate(angle, expand=True, fillcolor=255)

    # обрезка по содержимому
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    img = img.crop(bbox)
    img = ImageOps.invert(img)

    # ресайз в 28x28
    img = img.resize((size, size))

    # иногда добавляем шум
    if random.random() < 0.3:
        arr = np.array(img)
        noise = np.random.randint(0, 50, (size, size))
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # фильтр (размытие или резкость)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(0.5))

    # сохраняем
    img.save(f"XDataset/X/x_{idx}.png")


# Генерация 1000 примеров
for i in range(1000):
    generate_x_image(i)

print("✅ Сгенерировано 1000 картинок X в dataset/X/")