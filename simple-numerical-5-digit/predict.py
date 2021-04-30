import os

from keras.models import load_model
import cv2
import numpy as np

from train import parseDataset
from prepare import imagepath_to_roi
from utility import preprocess

MODEL_PATH = "./model/captcha_lenet"
TEST_DATA_PATH = "./Captcha Solver/unsolved-captchas/electoral-tagged"
OUT_PATH = "./data/predicted"

def draw_predictions(img, roi_list, cnts):
    predictions = []
    for roi, cnt in zip(roi_list, cnts):
        (x, y, w, h) = cv2.boundingRect(cnt)

        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(np.array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
        sub = -(12 + h) if (y - 15 < 0) else 4
        cv2.putText(img, pred, (x, y - sub), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return img, "".join(predictions)
    
model = load_model(MODEL_PATH)
test_img_paths = parseDataset(TEST_DATA_PATH).image_paths

for img_path in test_img_paths:
    i, r, c = imagepath_to_roi(img_path)
    img, pred_string = draw_predictions(i, r, c)
    cv2.imwrite(os.path.join(OUT_PATH, pred_string + ".jpg", img))