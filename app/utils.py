# utils.py
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_image_with_metadata(image, prompt, params, filename=None, watermark_text="AI-generated"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = (filename or f"{ts}").replace(" ", "_")
    png_path = os.path.join(IMAGES_DIR, f"{safe_name}.png")
    jpeg_path = os.path.join(IMAGES_DIR, f"{safe_name}.jpg")

    # Add watermark before saving PNG
    image_wm = add_text_watermark(image.copy(), watermark_text)
    image_wm.save(png_path, format="PNG")
    # Save JPEG (without alpha)
    image.convert("RGB").save(jpeg_path, format="JPEG", quality=95)

    # Save metadata
    meta = {
        "filename": safe_name,
        "prompt": prompt,
        "params": params,
        "timestamp_utc": ts,
        "png_path": os.path.abspath(png_path),
        "jpeg_path": os.path.abspath(jpeg_path),
    }
    with open(META_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return meta

def add_text_watermark(image, text="AI-generated", opacity=200, fontsize=20, margin=10):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    x = w - text_w - margin
    y = h - text_h - margin
    # semi-transparent rectangle
    rect_xy = [x - 6, y - 6, x + text_w + 6, y + text_h + 6]
    draw.rectangle(rect_xy, fill=(0,0,0,100))
    draw.text((x, y), text, fill=(255,255,255,opacity), font=font)
    return image
