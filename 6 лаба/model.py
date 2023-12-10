from sklearn import datasets
import numpy as np
import keras
from keras.layers import Dense, Flatten

iris = datasets.load_iris()

iris_array = iris.data
iris_target = keras.utils.to_categorical(iris.target, 3)


model = keras.Sequential([
    Flatten(input_shape = (4, ), name = 'input'),
    Dense(100, activation = 'relu'),
    Dense(100, activation = 'relu'),
    Dense(3, activation = 'softmax', name = 'output')
])

model.compile(optimizer = 'adam',
              loss='mean_squared_error',
              metrics = ['accuracy'])

print(model.summary())

model.fit(iris_array, iris_target, batch_size=10, epochs=10, validation_split=0.2)

model.evaluate(iris_array, iris_target)

model.save('models/model.h5')
