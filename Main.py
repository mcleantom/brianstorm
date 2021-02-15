from keras.models import Sequential
from keras import layers, Input, models
import numpy as np
import pandas as pd
from data_loader import load_data, load_test_data
from keras.callbacks import EarlyStopping
from plotting import plot_training
from sklearn.metrics import classification_report
from keras.metrics import sparse_categorical_accuracy

class SickModel:

    def createModel(self, input_shape, output_shape):
        self.model = Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(100, activation="relu"),
                layers.Dense(50, activation="relu"),
                layers.Dense(25, activation="relu"),
                layers.Dense(output_shape, activation="softmax")
            ])

        self.model.compile(optimizer="sgd",
                           loss="sparse_categorical_crossentropy",
                           metrics=[sparse_categorical_accuracy])

    def trainModel(self, train_x, train_y, val_x, val_y, epochs, batch_size):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        callbacks = [es]
        self.history = self.model.fit(train_x, train_y,
                                      epochs=epochs,
                                      verbose=True,
                                      batch_size=batch_size,
                                      validation_data=(val_x, val_y),
                                      callbacks = callbacks)


EPOCHS = 500
BATCH_SIZE = 50

train_x, val_x, train_y, val_y = load_data()
dim_input_shape = train_x.shape[1]
dim_output_shape = len(train_y.unique())

model = SickModel()
model.createModel(dim_input_shape, dim_output_shape)
model.trainModel(train_x, train_y, val_x, val_y, EPOCHS, BATCH_SIZE)
plot_training(model.history)
test_x, test_y = load_test_data()

y_pred = model.model.predict(test_x)
y_pred = np.argmax(y_pred, axis=1)
test_y = np.argmax(test_y, axis=1)

print(test_y[:10])
print(y_pred[:10])

print(classification_report(test_y, y_pred))