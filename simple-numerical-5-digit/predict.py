import os

from keras.models import load_model
import cv2
import numpy as np

from train import parseDataset
from prepare import imagepath_to_roi
from utility import preprocess, load_config

from tqdm import tqdm

config = load_config("config.yml")

MODEL_PATH = config["model_path"]
TEST_DATA_PATH = config["predict"]["test_data_path"]
OUT_PATH = config["predict"]["out_path"]

def draw_predictions(model, roi_list, img, cnts):
    predictions = []
    for roi, cnt in zip(roi_list, cnts):
        (x, y, w, h) = cv2.boundingRect(cnt)

        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(np.array(roi)[:, :, None], axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))
        
        cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
        sub = -(12 + h) if (y - 15 < 0) else 4
        cv2.putText(img, predictions[-1], (x, y - sub), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return img, "".join(predictions)
    
def predict_one(img_path, model):
    r, i, c = imagepath_to_roi(img_path)
    img, pred_string = draw_predictions(model, r, i, c)
    return img, pred_string

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    test_ds = parseDataset(TEST_DATA_PATH)
    test_img_paths = list(test_ds.image_paths)
    print(f"[INFO] found {len(list(test_img_paths))}")
    
    for img_path in tqdm(test_img_paths):
        img, pred_string = predict_one(img_path, model)
        cv2.imwrite(os.path.join(OUT_PATH, pred_string + ".jpg"), img)