"""Serving StableDiffusion with FastAPI
https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion
"""
from fastapi import FastAPI
from fastapi.responses import Response
from io import BytesIO
from keras_cv import models
from PIL import Image
from tensorflow import config
from tensorflow import keras


# Use half precision if GPU is available
if config.list_physical_devices("GPU"):
    keras.mixed_precision.set_global_policy("mixed_float16")

# Load the model
model = models.StableDiffusion(jit_compile=True)

# Create the FastAPI app
app = FastAPI()


@app.get("/generate_img")
async def generate_img(prompt: str):
    """Uses a stable diffusion to generate and return an image based on the prompt."""

    # Generate the image based on the prompt
    image = model.text_to_image(prompt, num_steps=25)[0]

    # Convert the PIL Image to bytes in memory
    image_bytes_io = BytesIO()
    Image.fromarray(image).save(image_bytes_io, format="JPEG")
    image_data = image_bytes_io.getvalue()

    # Return the image in a Response object
    # This works with a simple `curl -o myimage.jpg ...`
    return Response(content=image_data, media_type="image/jpeg")
