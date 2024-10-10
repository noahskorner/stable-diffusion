import uuid
import torch
from diffusers import StableDiffusionPipeline
import os

# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
seed=None
pretrained_model_name_or_path = './models/stable-diffusion-v1-5'
prompt = "A serene lake at sunset, with mountains in the background, digital art"
negative_prompt = "blur, distortion, low quality, text, watermark"
output_dir = "output"

def main():
    # Initialize the generator
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipeline.to(device)

    # Enable attention slicing for lower memory usage
    # pipeline.enable_attention_slicing()

    # Generate images based on the prompt
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=50,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator
    ).images

    # Save the generated images
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        filename = f"{uuid.uuid4()}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"Saved image to {filepath}")

if __name__ == "__main__":
    main()