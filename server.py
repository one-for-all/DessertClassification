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
    # filename = image_file.filename
    # fileExtension = filename.split(".")[-1].lower()
    # if not fileExtension in ["jpg", "jpeg", "png"]:
    #     raise HTTPException(status_code=415, detail=f"file extension ({fileExtension}) not supported.")
    #
    # image_stream = io.BytesIO(image_file.file.read())
    # image_stream.seek(0)
    # file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    # print(file_bytes)
    # image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = img_to_array(load_img(get_file(origin=image_link), target_size=(224, 224)))

    probs = softmax(model.predict(expand_dims(image, 0))[0])
    print(probs)
    prediction = class_predictions[np.argmax(probs)]

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
