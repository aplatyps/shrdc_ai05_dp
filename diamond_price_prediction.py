# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:57:09 2022

"""
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os

from random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


file_path = r"B:\MSI\Downloads\shrdc\breast_cancer\dataset\diamonds.csv"
save_path = r"B:\MSI\Downloads\shrdc\diamond_price\img"
diamond_data = pd.read_csv(file_path)

diamond_data = diamond_data.drop('Unnamed: 0', axis=1)
diamond_features = diamond_data.copy()
diamond_label = diamond_features.pop('price')

print("------------------Features-------------------------")
print(diamond_features.head())
print("-----------------Label----------------------")
print(diamond_label.head())

cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
colour_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
ordinal_encoder = OrdinalEncoder(categories=[cut_categories, colour_categories, clarity_categories])
diamond_features[['cut', 'color', 'clarity']] = ordinal_encoder.fit_transform(diamond_features[['cut', 'color', 'clarity']])

print("---------------Transformed Features--------------------")
print(diamond_features.head())

SEED = randint(100, 15000)
x_train, x_iter, y_train, y_iter = train_test_split(diamond_features, diamond_label, test_size=0.4, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter, y_iter, test_size=0.5, random_state=SEED)

standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train = standard_scaler.transform(x_train)
x_val = standard_scaler.transform(x_val)
x_test = standard_scaler.transform(x_test)

number_input = x_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input))
model.add(tf.keras.layers.Dense(128, activation='elu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(32, activation='elu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(8, activation='elu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae', 'mse'])

tf.keras.utils.plot_model(model,
                          to_file='model.png',
                          show_shapes=True,
                          show_layer_activations=True)

base_log_path = r"B:\MSI\Downloads\shrdc\tensorboard_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2)
EPOCHS = 100
BATCH_SIZE = 64
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[tb_callback, es_callback])


plt.loglog(history.history['loss'])
plt.loglog(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.grid(True, which="both", ls="-")
plt.savefig(os.path.join(save_path, "loss.png"), bbox_inches='tight')
plt.show()
plt.clf()

plt.loglog(history.history['mae'])
plt.loglog(history.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.grid(True, which="both", ls="-")
plt.savefig(os.path.join(save_path, "mae.png"), bbox_inches='tight')
plt.show()
plt.clf()

test_result = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}\n\n")

predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions, labels, ".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
plt.savefig(os.path.join(save_path, "result.png"), bbox_inches='tight')
plt.show()