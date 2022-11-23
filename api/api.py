from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import tensorflow as tf
import base64

model_path = "../my_model"
model = tf.keras.models.load_model(model_path)

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.post("/predict/upload")
async def predict_from_file(file: UploadFile):
    content = await file.read()

    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    img = img[None, ...]

    prediction = model.predict(img)
    pred_list = np.ndarray.tolist(prediction)

    pred_dict = {}

    index = 0
    for pred in pred_list[0]: 
        pred_dict[index] = pred
        index += 1

    return pred_dict

@app.post("/predict/canvas")
async def predict_from_canvas(image_data: str  = File()):
    encoded_data = image_data.split(',')[1]

    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    dim = (28, 28)
    
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


    cv2.imwrite('image.png', img)

    img = img[None, ...]

    prediction = model.predict(img)
    pred_list = np.ndarray.tolist(prediction)

    pred_dict = {}

    index = 0
    for pred in pred_list[0]: 
        pred_dict[index] = pred
        index += 1

    return pred_dict

