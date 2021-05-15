import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_config(config_name, CONFIG_PATH="./"):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def preprocess(image, width, height):
    (h, w) = image.shape[:2]

    padW = int((width - image.shape[1])/ 2.0)
    padH = int((height - image.shape[0])/ 2.0)

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

def plot_graph(h_dict, epochs): # TODO: needs testing
    fig, ax = plt.figure()
    fig.style.use('ggplot')

    ax.plot(np.arange(0, epochs), h_dict.history["loss"], label="train_loss")
    ax.plot(np.arange(0, epochs), h_dict.history["val_loss"], label="val_loss")
    ax.plot(np.arange(0, epochs), h_dict.history["accuracy"], label="acc")
    ax.plot(np.arange(0, epochs), h_dict.history["val_accuracy"], label="val_acc")
    fig.title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    fig.legend()
    fig.show()