import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import os

class ImageGenerator:
    def __init__(self, llm_model="gpt2", sd_model="runwayml/stable-diffusion-v1-5", device=None):
        self.llm = pipeline("text-generation", model=llm_model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.sd_pipe = self.sd_pipe.to(self.device)

    def generate_prompt(self, user_input, max_length=100):
        refined = self.llm(user_input, max_length=max_length)[0]['generated_text']
        return refined

    def generate_image(self, prompt, output_path="output.png", guidance_scale=7.5, num_inference_steps=30):
        image = self.sd_pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        image.save(output_path)
        return output_path

    def from_user_input(self, user_input, output_path="output.png"):
        prompt = self.generate_prompt(user_input)
        path = self.generate_image(prompt, output_path=output_path)
        return path, prompt

# Example usage
if __name__ == "__main__":
    ig = ImageGenerator()
    user_input = "A futuristic cityscape at sunset, highly detailed"
    image_path, refined_prompt = ig.from_user_input(user_input, "generated_image.png")
    print(f"Refined prompt: {refined_prompt}")
    print(f"Saved image to: {image_path}")