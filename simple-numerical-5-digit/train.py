# imports
import os

from imutils import paths
import cv2
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD

from utility import preprocess
from model import LeNet

DATASET_PATH = "<add here>"
EPOCHS = 20

class parseDataset:
    def __init__(self, folder_path):
        self.image_paths = paths.list_images(folder_path)
        self.data = []
        self.labels = []
    
    def __len__(self):
        return len(self.image_paths)
        
    def get_data(self):
        # loop over the input images
        for image_path in self.image_paths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = preprocess(image, 28, 28)
            image = img_to_array(image)
            self.data.append(image)

            label = image_path.split(os.path.sep)[-2]
            self.labels.append(label)
  
        self.data = np.array(self.data, dtype="float") / 255.0
        self.labels = np.array(self.labels)
        return self.data, self.labels

if __name__ == "__main__":
    ds = parseDataset(DATASET_PATH)
    data, labels = ds.get_data()
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    model = LeNet.build(width=28, height=28, depth=1, classes=9)
    opt = SGD(lr=0.1)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    H = model.fit(X_train, y_train, batch_size=128, 
                                    epochs=EPOCHS, 
                                    verbose=1, 
                                    validation_data=(X_test, y_test))