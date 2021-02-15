# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:36:25 2021

@author: mclea
"""

import matplotlib.pyplot as plt

def plot_training(history):
    
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    fig, ax = plt.subplots()
    ax.plot(loss, c="r", label="Loss")
    ax.plot(val_loss, c="g", label="Val loss")
    ax.legend()
    plt.show()