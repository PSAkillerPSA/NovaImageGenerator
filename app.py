import io
import os
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionPipeline

app = FastAPI(
    title="Free AI Image Generator",
    description="Generate images from a prompt in the URL.",
    version="1.0.0"
)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Render CPU instances don't have CUDA.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Starting image generator on {DEVICE}...")
print("Loading Stable Diffusion model...")

if DEVICE == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True
    )

    # Reduces RAM usage on CPU, at the cost of speed.
    pipe.enable_attention_slicing()

print("Model loaded successfully.")


@app.get("/")
def home():
    return {
        "status": "online",
        "usage": "GET /your image prompt here",
        "example": "/a cute cat riding a dragon"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE
    }


@app.get("/{prompt:path}")
def generate_image(prompt: str):
    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide an image prompt."
        )

    # Don't allow absurdly large URLs/prompts.
    if len(prompt) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Prompt is too long. Maximum 1000 characters."
        )

    prompt = prompt.strip()

    print(f"Generating image for: {prompt}")

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )

        image = result.images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="generated.png"',
                "Cache-Control": "no-store"
            }
        )

    except Exception as error:
        print(f"Generation error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Image generation failed."
        )
