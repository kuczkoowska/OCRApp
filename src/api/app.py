from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
from ..train.mjs import MJSynthDataLoader
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = "/home/jagoda/studia/inteloblicz/OCRAppModels/src/results/crnn_best.keras"
DATA_PATH = "/home/jagoda/studia/inteloblicz/OCRAppModels/mjsynth"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
data_loader = MJSynthDataLoader(DATA_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((128, 32)) 
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)  # (wys, szer, 1)
    img = np.expand_dims(img, axis=0)   # (1, wys, szer, 1)
    return img.astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    preds = model.predict(img)
    pred_text = data_loader.decode_predictions(preds)[0]
    return {"prediction ": pred_text}