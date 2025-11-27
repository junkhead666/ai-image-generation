import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
import os
import zipfile

# -----------------------------
# Device Selection
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Stable Diffusion
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    return pipe

pipe = load_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎨 AI Image Generator – Stable Diffusion")
st.write("Generate AI images from text prompts using open-source models.")

prompt = st.text_input(
    "Enter your prompt",
    placeholder="a futuristic city at sunset, highly detailed, 4K"
)

negative_prompt = st.text_input(
    "Negative prompt (optional)",
    placeholder="low quality, blur, watermark"
)

style = st.selectbox(
    "Choose a style",
    ["None", "Photorealistic", "Cinematic", "Artistic", "Cartoon", "Anime"]
)

num_images = st.slider("Number of images", 1, 4, 1)

enhance = st.checkbox("Enhance quality", value=True)

generate = st.button("Generate Images")


# -----------------------------
# Prompt Enhancers
# -----------------------------
def enhance_prompt(p):
    return f"{p}, ultra realistic, highly detailed, 4K, professional lighting"


def apply_style(p, style):
    styles = {
        "Photorealistic": "photorealistic, DSLR, ultra sharp, 8K",
        "Cinematic": "cinematic lighting, dramatic shadows, film look",
        "Artistic": "digital art, illustration, concept art",
        "Cartoon": "cartoon style, bold outlines, vibrant",
        "Anime": "anime style, clean lineart, vibrant colors"
    }
    if style == "None":
        return p
    return f"{p}, {styles[style]}"


# -----------------------------
# Generate Images
# -----------------------------
if generate and prompt:

    final_prompt = prompt

    if enhance:
        final_prompt = enhance_prompt(final_prompt)

    if style != "None":
        final_prompt = apply_style(final_prompt, style)

    st.info("⏳ Generating images… please wait")

    images = []
    progress = st.progress(0)

    for i in range(num_images):
        img = pipe(
            final_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25
        ).images[0]

        images.append(img)
        progress.progress((i + 1) / num_images)

    # Save images
    save_dir = "generated_images"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, img in enumerate(images):
        img.save(f"{save_dir}/img_{timestamp}_{idx+1}.png")

    st.success("✅ Generation complete!")

    # Show images
    for img in images:
        st.image(img, use_column_width=True)

    # Create ZIP for download
    zip_path = f"{save_dir}/images_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, 'w') as z:
        for idx in range(num_images):
            z.write(f"{save_dir}/img_{timestamp}_{idx+1}.png")

    with open(zip_path, "rb") as f:
        st.download_button(
            "⬇️ Download All Images (ZIP)",
            f,
            file_name=f"images_{timestamp}.zip"
        )
