import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from utility import load_config

from prepare import imagepath_to_roi
from predict import draw_predictions

config = load_config("config.yml")
MODEL_PATH = config["model_path"]

model = load_model(MODEL_PATH)

app = FastAPI()

@app.post("/prediction")
async def predict_api(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    contents = await file.read()
    r, i, c = imagepath_to_roi(io.BytesIO(contents), path=False)
    img, pred_string = draw_predictions(model, r, i, c)
    return pred_string

@app.get('/')
def root_route():
  return { 'error': 'Use GET /prediction instead of the root route!' }