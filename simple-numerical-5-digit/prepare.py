import os

from imutils import paths
import cv2
import matplotlib.pyplot as plt # TODO
from utility import load_config

config = load_config("config.yml")

SOLVED_PATH = config["prepare"]["solved_path"]
OUTPUT_PATH = config["prepare"]["output_path"]

imagepaths = list(paths.list_images(SOLVED_PATH))
counts = {}
n_digits = 5

def imagepath_to_roi(imagepath, n_digits=5):
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold to invert colours
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # extract four contours with largest areas
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])[:n_digits]

    roi_list = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        roi = 255 - gray[y - 5:y + h + 5, x - 5:x + w + 5]
        roi_list.append(roi)

    assert len(roi_list) == n_digits
    return roi_list, image, cnts

def save_roi(roi_list, OUTPUT_PATH):
    for i, roi in enumerate(roi_list):
        digit = digits[i] 
        dirpath = os.path.sep.join([OUTPUT_PATH, digit])
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
  
        count = counts.get(digit, 1)
        p = os.path.sep.join([dirpath, "{}.png".format(str(count).zfill(6))])
        cv2.imwrite(p, roi)
        counts[digit] = count + 1

def simshow(im):
    if len(im.shape) < 3:
        plt.imshow(im, cmap='gray')
    elif len(im.shape[2] == 1):
        plt.imshow(np.squeeze(im), cmap='gray')
    else:
        plt.imshow(im)

if __name__ == "__main__":
    for (i, imagepath) in enumerate(imagepaths):
        print(f"processing image {i+1}/{len(imagepaths)}")
        
        digits = os.path.split(imagepath)[1][:n_digits]
        image_roi_list, _, _ = imagepath_to_roi(imagepath, n_digits)
        save_roi(image_roi_list, OUTPUT_PATH)
        
        print(counts)
