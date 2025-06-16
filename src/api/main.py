from fastapi import FastAPI

app = FastAPI()

class InputData(BaseModel):
    features:

@app.post("/predict")