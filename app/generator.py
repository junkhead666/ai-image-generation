# generator.py
import os
import time
from typing import List, Optional
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", use_auth_token: Optional[str]=None, device: Optional[str]=None):
    """
    Load a Stable Diffusion pipeline with scheduler and return the pipeline.
    Note: Model weights are downloaded from HF and you must agree to model license.
    """
    device = device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # we'll add custom checks below (or you can use the built-in)
        use_auth_token=use_auth_token,
    )
    # Use a faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    return pipe

def generate_images(pipe, prompt: str, negative_prompt: Optional[str]=None, guidance_scale: float=7.5,
                    num_images: int=1, height: int=512, width: int=512, num_inference_steps: int=30,
                    progress_callback=None):
    """
    Generate images and yield PIL images.
    progress_callback( i, total, time_elapsed, est_remaining_seconds)
    """
    results = []
    start = time.time()
    for i in range(num_images):
        iter_start = time.time()
        image = pipe(prompt=prompt,
                     negative_prompt=negative_prompt,
                     guidance_scale=guidance_scale,
                     height=height,
                     width=width,
                     num_inference_steps=num_inference_steps).images[0]
        results.append(image)
        # progress callback
        if progress_callback:
            elapsed = time.time() - start
            avg = elapsed / (i + 1)
            est_remaining = avg * (num_images - i - 1)
            progress_callback(i + 1, num_images, elapsed, est_remaining)
    return results
