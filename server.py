import io
import cv2
from fastapi import FastAPI, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file, load_img, img_to_array
from tensorflow import expand_dims
from tensorflow.image import resize as image_resize
from tensorflow.nn import softmax
from class_predictions import class_predictions
import numpy as np
import uvicorn

# Dessert Classification Model
model = load_model("food-vision-model.h5")

# Server
app = FastAPI(title="Dessert Classification Server")


@app.get("/")
def home():
    return "Welcome to the Dessert Classification Website! You can head to http://localhost:8000/docs for testing."


@app.post("/predict")
def predict(image_link: str):
    # Check image extension
    file_extension = image_link.split(".")[-1].lower()
    if file_extension not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=415, detail=f"file extension ({file_extension}) not supported.")

    # Read image
    image = img_to_array(load_img(get_file(origin=image_link), target_size=(224, 224)))

    # Perform classification
    probs = softmax(model.predict(expand_dims(image, 0))[0])
    prediction = class_predictions[np.argmax(probs)]

    # Output result
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
