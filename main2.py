from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("a_model.h5")

CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def resize_image(image, size=(128, 128)):
    """Resize the image to the given size"""
    img = Image.fromarray(image).convert('RGB')  # convert to RGB
    img = img.resize(size)
    img_arr = np.array(img).reshape(1, size[0], size[1], 3)
    return img_arr

@app.post("/alzheimer/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = resize_image(image)
    
    predictions = MODEL.predict(image)


    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
