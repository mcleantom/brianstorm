from keras.models import Sequential
from keras import layers, Input, models
import numpy as np
import pandas as pd
from data_loader import load_data, load_test_data
from keras.callbacks import EarlyStopping
from plotting import plot_training
from sklearn.metrics import classification_report
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
import PIL
from PIL import Image, ImageFilter
import tensorflow as tf

class Network:
    def __init__(self):
        self.CreateNetwork()

    def CreateNetwork(self):
        self.model = Sequential([
                layers.Conv2D(64, kernel_size=(8,8), strides=(4,4), activation='relu', input_shape=(256,256, 1)),
                layers.Conv2D(32, kernel_size=(4,4), strides=(2,2), activation='relu'),
                layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(250, activation="relu"),
                layers.Dense(6, activation="softmax")
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def Train(self, X, y, X_v, y_v):
        callback = []  # EarlyStopping(patience=2)
        self.history = self.model.fit(X, y, epochs=100, batch_size=80, verbose = True ,callbacks=callback, validation_data = (X_v, y_v))

def loadImages():
    path = '../Data/Chess'
    from os import listdir
    from os.path import isfile, join
    
    allImageFiles = [[path+"/"+folder+"/"+f for f in listdir(path+"/"+folder)] for folder in listdir(path)]
    allImageTypes = [[folder for f in listdir(path+"/"+folder)] for folder in listdir(path)]
    allImageFiles = [val for sublist in allImageFiles for val in sublist]
    allImageTypes = [val for sublist in allImageTypes for val in sublist]
    print(len(allImageFiles))
    images = list(map(lambda im : Image.open(im).convert("1").filter(ImageFilter.MedianFilter(3)), allImageFiles))
    images = list(map(lambda im : im.resize((256,256)), images))
    imageList = zip(allImageTypes, images)
    return list(imageList)
 
def splitData(imageList):
    length = len(imageList)
    
    trainList = imageList[:int(length*0.8)]
    valList = imageList[int(length*0.8):int(length*0.9)]
    testList = imageList[int(length*0.9):]
    
    return trainList, valList, testList

def Uniques(DataList):
    y = ([y for (y, _) in DataList])
    uniques = []
    for name in y:
        if not name in uniques:
            uniques.append(name)
    return uniques

def vectorize(DataList, uniques):
    X = np.array([np.array(x) for (_, x) in DataList])
    X = X.reshape([*X.shape, 1])
    y = ([y for (y, _) in DataList])
   
    y = np.array(list(map(lambda n : uniques.index(n), y)))
    y = to_categorical(y, num_classes=len(uniques))
    #X = pd.DataFrame (X, columns=['Image'])
    #y = pd.DataFrame (y, columns=['Type'])
    return X, y

imageList = loadImages()

uniques = Uniques(imageList)

trainList, valList, testList = splitData(imageList)

X, y = vectorize(trainList, uniques)
X_v, y_v = vectorize(valList, uniques)
X_t, y_t = vectorize(testList, uniques)

network = Network()
network.Train(X, y, X_v, y_v)
plot_training(network.history)