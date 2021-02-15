# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:29:23 2021

@author: mclea
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_data():
    data = pd.read_csv("data/Video_games_esrb_rating.csv")
    ESRB_rating = data.esrb_rating.unique().tolist()
    data.esrb_rating = data.esrb_rating.map(lambda x: ESRB_rating.index(x))
    data = data.drop("title", axis=1)

    data_y = data["esrb_rating"]
    data_x = data.drop("esrb_rating", axis=1)

    #data_y = to_categorical(data_y)
    
    train_x, val_x, train_y, val_y = split_data(data_x, data_y)

    return train_x, val_x, train_y, val_y


def split_data(data_x, data_y):
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y,
                                                      test_size=0.2,
                                                      random_state=42)

    return train_x, val_x, train_y, val_y


def load_test_data():
    data = pd.read_csv("data/test_esrb.csv")
    ESRB_rating = data.esrb_rating.unique().tolist()
    data.esrb_rating = data.esrb_rating.map(lambda x: ESRB_rating.index(x))
    data = data.drop("title", axis=1)
    data_y = data["esrb_rating"]
    data_x = data.drop("esrb_rating", axis=1)
    data_y = to_categorical(data_y)
    return data_x, data_y