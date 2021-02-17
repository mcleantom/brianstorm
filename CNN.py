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
from PIL import Image
import tensorflow as tf

class Network:
    def __init__(self):
        self.CreateNetwork()

    def CreateNetwork(self):
        self.model = Sequential([
                layers.Conv2D(64,kernel_size=3, activation='relu', input_shape=(600,600, 1)),
                layers.Conv2D(32, kernel_size=3, activation='relu'),
                layers.Flatten(),
                layers.Dense(10, activation="softmax")
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def Train(self, X, y, X_v, y_v):
        callback = EarlyStopping(patience=2)
        self.model.fit(X, y, epochs=15, batch_size=10, verbose = True ,callbacks=[callback], validation_data = (X_v, y_v))

def loadImages():
    path = '../Data/Chess'
    from os import listdir
    from os.path import isfile, join
    
    allImageFiles = [[path+"/"+folder+"/"+f for f in listdir(path+"/"+folder)] for folder in listdir(path)]
    allImageTypes = [[folder for f in listdir(path+"/"+folder)] for folder in listdir(path)]
    allImageFiles = [val for sublist in allImageFiles for val in sublist]
    allImageTypes = [val for sublist in allImageTypes for val in sublist]
    print(len(allImageFiles))
    images = list(map(lambda im : Image.open(im).convert("L"), allImageFiles))
    images = list(map(lambda im : im.resize((600,600)), images))
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
    y = ([y for (y, _) in DataList])
   
    y = np.array(list(map(lambda n : uniques.index(n), y)))
    y = to_categorical(y)
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