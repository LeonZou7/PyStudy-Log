from tensorflow import keras
import numpy as np


model = keras.Sequential()
model.add(keras.layers.Dense(64, input_dim=100, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000, 100))
label = np.random.randint(5, size=(1000, 1))
one_hot_label = keras.utils.to_categorical(label, num_classes=5)

val_data = np.random.random((1000, 100))
val_label = np.random.randint(5, size=(1000, 1))
val_one_hot_label = keras.utils.to_categorical(val_label,
                                               num_classes=5)

model.fit(data, one_hot_label, epochs=70, batch_size=32,
          validation_data=(val_data, val_one_hot_label))

val_data = np.random.random((1000, 100))
val_label = np.random.randint(5, size=(1000, 1))
val_one_hot_label = keras.utils.to_categorical(val_label,
                                               num_classes=5)
model.evaluate(val_data, val_one_hot_label, batch_size=32)